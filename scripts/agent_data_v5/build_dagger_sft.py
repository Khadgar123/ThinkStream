#!/usr/bin/env python
"""Build on-policy / DAgger SFT messages from existing trajectories.

The existing pass5 SFT rows are teacher-forced snapshots: every prompt uses
gold memory.  This script rolls a policy checkpoint through the trajectory
chunk-by-chunk, lets the policy write the memory, then swaps in the gold
assistant target for that same chunk.  The resulting rows train the model to
recover from its own closed-loop memory state.

Typical usage:

  CUDA_VISIBLE_DEVICES=0 python -m scripts.agent_data_v5.build_dagger_sft \
    --ckpt output/agent-sft/checkpoint-250 \
    --trajectories data/agent_v5/batch1/final/train_sft_trajectories.jsonl \
    --frames-root data/agent_v5/batch1/frames \
    --out data/agent_v5/batch1/rendered/video_meta/train_sft_dagger_messages.jsonl \
    --frame-protocol video_meta --max-trajectories 20

For multi-GPU construction, launch multiple shards with --num-shards /
--shard-index and concatenate the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from scripts.agent_data_v5.pass5_messages import build_messages, _emit_row
from scripts.eval.ovo.eval_full import detect_model_class, reset_visual_index
from scripts.eval.processor_loader import load_processor_for_checkpoint
from thinkstream.data.agent_protocol import (
    has_compress_trigger,
    normalize_frame_protocol,
)
from thinkstream.model.agent_loop import StreamingAgentLoop, make_generate_fn
from thinkstream.model.retrieval import make_retriever
from thinkstream.sft.argument import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels


def _default_batch_root(path: Optional[str]) -> Path:
    if path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p if p.name != "final" else p.parent
    env = os.environ.get("THINKSTREAM_DATA_ROOT") or os.environ.get("AGENT_DATA_DIR")
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p if p.name != "final" else p.parent
    return ROOT / "data" / "agent_v5"


def _resolve_path(raw: str, *, base: Path = ROOT) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else base / p


def _iter_trajectory_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _propagate_sample_fields(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    video_id = traj.get("video_id", "")
    video_path = traj.get("video_path", "")
    traj_id = traj.get("trajectory_id", "")
    samples = []
    for s in traj.get("samples") or []:
        item = dict(s)
        item.setdefault("video_id", video_id)
        item.setdefault("video_path", video_path)
        item.setdefault("trajectory_id", traj_id)
        samples.append(item)
    samples.sort(key=lambda x: int(x.get("chunk_idx", 0)))
    return samples


def _group_by_chunk(samples: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_chunk: Dict[int, List[Dict[str, Any]]] = {}
    for s in samples:
        try:
            chunk = int(s.get("chunk_idx", 0))
        except (TypeError, ValueError):
            continue
        by_chunk.setdefault(chunk, []).append(s)
    return dict(sorted(by_chunk.items()))


def _resolve_video_path(video_path: str, video_root: Optional[str]) -> Optional[str]:
    if not video_path:
        return None
    p = Path(video_path)
    if p.is_absolute():
        return str(p)
    if video_root:
        return str(Path(video_root) / video_path)
    return str(ROOT / video_path)


def _new_question(sample: Dict[str, Any]) -> Optional[str]:
    inp = sample.get("input") or {}
    ui = inp.get("user_input")
    if not isinstance(ui, str):
        return None
    ui = ui.strip()
    if not ui or has_compress_trigger(ui):
        return None
    return ui


def _question_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = sample.get("metadata") or {}
    return {
        "options": sample.get("options") or meta.get("options") or [],
        "answer_form": sample.get("answer_form") or meta.get("answer_form") or "",
        "answer_style": sample.get("answer_style") or meta.get("answer_style") or "",
        "answer_instruction": (
            sample.get("answer_instruction")
            or meta.get("answer_instruction")
            or ""
        ),
    }


def _choose_control_sample(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    for s in samples:
        if _new_question(s):
            return s
    return samples[0]


def _content_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    return "\n".join(parts)


_USER_INPUT_RE = re.compile(r"<user_input>(.*?)</user_input>", re.DOTALL)


def _prompt_has_compress_trigger(messages: List[Dict[str, Any]]) -> bool:
    """True only when the actual user input carries a compress trigger.

    The system prompt documents the literal string ``<compress_trigger/>``.
    Scanning all message text therefore marks every normal visual step as a
    compress prompt.  DAgger needs the runtime event, which is rendered under
    the user turn's ``<user_input>...</user_input>`` block.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            for match in _USER_INPUT_RE.finditer(text):
                if has_compress_trigger(match.group(1)):
                    return True
    return False


def _target_allowed(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    *,
    sample_types: set[str],
    include_failed_targets: bool,
) -> tuple[bool, str]:
    sample_type = str(sample.get("sample_type", ""))
    if sample_type not in sample_types:
        return False, "sample_type"
    if not include_failed_targets:
        verification = sample.get("verification") or {}
        if not bool(verification.get("passed", True)):
            return False, "verification_failed"

    prompt_has_compress = _prompt_has_compress_trigger(onpolicy_prompt)
    if sample_type == "compress" and not prompt_has_compress:
        return False, "compress_target_without_trigger"
    if sample_type != "compress" and prompt_has_compress:
        return False, "visual_target_on_compress_prompt"
    return True, ""


def _build_dagger_messages(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    *,
    base_path: Path,
    data_dir: Path,
    frame_protocol: str,
) -> List[Dict[str, Any]]:
    """Use model-memory prompt + gold assistant tail."""
    gold_messages = build_messages(
        sample,
        base_path,
        data_dir=data_dir,
        frame_protocol=frame_protocol,
    )
    if len(gold_messages) < 3:
        raise ValueError("gold messages missing assistant target")
    if len(onpolicy_prompt) != 2:
        raise ValueError(f"expected single-step on-policy prompt, got {len(onpolicy_prompt)}")
    return deepcopy(onpolicy_prompt) + deepcopy(gold_messages[2:])


def _emit_dagger_row(
    *,
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    result: Dict[str, Any],
    fout,
    stats: Dict[str, Any],
    ckpt: str,
    data_dir: Path,
    frame_protocol: str,
    include_failed_targets: bool,
    sample_types: set[str],
) -> bool:
    ok, reason = _target_allowed(
        sample,
        onpolicy_prompt,
        sample_types=sample_types,
        include_failed_targets=include_failed_targets,
    )
    if not ok:
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
        return False
    try:
        messages = _build_dagger_messages(
            sample,
            onpolicy_prompt,
            base_path=ROOT,
            data_dir=data_dir,
            frame_protocol=frame_protocol,
        )
    except Exception as exc:
        reason = f"render_error:{type(exc).__name__}"
        stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
        return False

    row = _emit_row(sample, messages, frame_protocol=frame_protocol)
    prompt_is_compress = _prompt_has_compress_trigger(onpolicy_prompt)
    row["dagger"] = {
        "policy_ckpt": ckpt,
        "rollout_action": result.get("action", ""),
        "rollout_final_action": result.get("final_action", ""),
        "rollout_format_ok": bool(result.get("format_ok", True)),
        "rollout_inter_chunk_compress_prompt": bool(prompt_is_compress),
        "memory_token_count": result.get("memory_token_count"),
        "prompt_text_token_count": result.get("prompt_text_token_count"),
    }
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows"] += 1
    st = row.get("sample_type", "")
    stats["by_type"][st] = stats["by_type"].get(st, 0) + 1
    return True


def build_dagger(
    *,
    ckpt: str,
    trajectories: Path,
    out: Path,
    data_dir: Path,
    frames_root: str,
    video_root: Optional[str],
    frame_protocol: str,
    retriever_kind: str,
    max_results: int,
    alpha: float,
    max_new_tokens: int,
    profile: str,
    sample_types: set[str],
    include_failed_targets: bool,
    max_trajectories: int,
    max_rows: int,
    num_shards: int,
    shard_index: int,
    no_bf16: bool,
    max_compress_turns_per_chunk: int,
    log_every_steps: int,
) -> Dict[str, Any]:
    from scripts.eval.eval_profiles import apply_profile, describe_profile

    profile_cfg = apply_profile(profile)
    print(describe_profile(profile))

    cls, model_type = detect_model_class(ckpt)
    print(f"Loading {cls.__name__} from {ckpt}")
    model = cls.from_pretrained(
        ckpt,
        dtype=torch.bfloat16 if not no_bf16 else None,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    processor = load_processor_for_checkpoint(ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(
        processor.video_processor, "do_sample_frames"
    ):
        processor.video_processor.do_sample_frames = False

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt,
        model_max_length=profile_cfg["model_max_length"],
        padding_side="right",
        use_fast=False,
    )
    tokenizer.add_tokens(
        [
            t for t in processor.tokenizer.get_added_vocab().keys()
            if t not in tokenizer.get_vocab()
        ],
        special_tokens=True,
    )

    print(f"Building retriever: kind={retriever_kind}, alpha={alpha}")
    retriever = make_retriever(
        kind=retriever_kind,
        alpha=alpha,
        max_results=max_results,
        device="cuda",
    )
    loop = StreamingAgentLoop(
        generate_fn=make_generate_fn(model, processor, model_type=model_type),
        tokenizer=tokenizer,
        processor=processor,
        model_type=model_type,
        min_pixels=130_000,
        max_pixels=220_000,
        max_new_tokens=max_new_tokens,
        retriever=retriever,
        compress_mode="system",
        frames_root=frames_root,
        video_root=video_root,
        frame_protocol=frame_protocol,
    )

    stats: Dict[str, Any] = {
        "trajectories_seen": 0,
        "trajectories_used": 0,
        "steps": 0,
        "rows": 0,
        "skipped": {},
        "by_type": {},
        "step_errors": 0,
        "policy_compress_turns": 0,
        "visual_retries_after_compress": 0,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w") as fout:
        for traj_i, traj in enumerate(_iter_trajectory_rows(trajectories)):
            if max_trajectories and stats["trajectories_used"] >= max_trajectories:
                break
            stats["trajectories_seen"] += 1
            if num_shards > 1 and (traj_i % num_shards) != shard_index:
                continue

            samples = _propagate_sample_fields(traj)
            if not samples:
                continue
            video_path = _resolve_video_path(samples[0].get("video_path", ""), video_root)
            if not video_path or not Path(video_path).exists():
                stats["skipped"]["missing_video"] = stats["skipped"].get("missing_video", 0) + len(samples)
                continue

            loop.reset()
            reset_visual_index(loop.retriever)
            stats["trajectories_used"] += 1

            for chunk_idx, chunk_samples in _group_by_chunk(samples).items():
                compress_samples = [
                    s for s in chunk_samples
                    if str(s.get("sample_type", "")) == "compress"
                ]
                visual_samples = [
                    s for s in chunk_samples
                    if str(s.get("sample_type", "")) != "compress"
                ]
                control = _choose_control_sample(visual_samples or chunk_samples)
                q = _new_question(control)
                q_meta = _question_meta(control) if q else None

                compress_turns = 0
                while True:
                    try:
                        # Gold compress rows are inter-chunk memory-management
                        # events, so do not inject the visual question on a
                        # compress-only chunk. For visual chunks, keep the
                        # normal question routing.
                        result = loop.step(
                            chunk_idx=chunk_idx,
                            video_path=video_path,
                            user_question=q if visual_samples else None,
                            user_question_meta=q_meta if visual_samples else None,
                        )
                        onpolicy_prompt = deepcopy(loop._last_step_messages)
                        if not onpolicy_prompt:
                            raise RuntimeError("StreamingAgentLoop did not capture step prompt")
                    except Exception as exc:
                        stats["step_errors"] += 1
                        stats["skipped"]["step_error"] = stats["skipped"].get("step_error", 0) + len(chunk_samples)
                        if stats["step_errors"] <= 5:
                            print(
                                f"[warn] step failed traj={traj_i} chunk={chunk_idx}: "
                                f"{type(exc).__name__}: {exc}",
                                flush=True,
                            )
                        break

                    stats["steps"] += 1
                    prompt_is_compress = _prompt_has_compress_trigger(onpolicy_prompt)

                    if prompt_is_compress:
                        stats["policy_compress_turns"] += 1
                        for sample in compress_samples:
                            _emit_dagger_row(
                                sample=sample,
                                onpolicy_prompt=onpolicy_prompt,
                                result=result,
                                fout=fout,
                                stats=stats,
                                ckpt=ckpt,
                                data_dir=data_dir,
                                frame_protocol=frame_protocol,
                                include_failed_targets=include_failed_targets,
                                sample_types=sample_types,
                            )
                            if max_rows and stats["rows"] >= max_rows:
                                break
                        if max_rows and stats["rows"] >= max_rows:
                            break

                        # Critical DAgger alignment with verl/eval rollout:
                        # a system compress turn is between video chunks. If
                        # the policy actually compressed memory, retry the
                        # same chunk and train the visual target on the
                        # post-compress prompt. Do not train visual targets on
                        # the compress prompt.
                        if visual_samples:
                            if result.get("action") == "compress":
                                compress_turns += 1
                                if compress_turns <= max_compress_turns_per_chunk:
                                    stats["visual_retries_after_compress"] += 1
                                    continue
                                stats["skipped"]["too_many_policy_compress_turns"] = (
                                    stats["skipped"].get("too_many_policy_compress_turns", 0)
                                    + len(visual_samples)
                                )
                            else:
                                stats["skipped"]["policy_failed_compress_before_visual"] = (
                                    stats["skipped"].get("policy_failed_compress_before_visual", 0)
                                    + len(visual_samples)
                                )
                        break

                    for sample in visual_samples:
                        _emit_dagger_row(
                            sample=sample,
                            onpolicy_prompt=onpolicy_prompt,
                            result=result,
                            fout=fout,
                            stats=stats,
                            ckpt=ckpt,
                            data_dir=data_dir,
                            frame_protocol=frame_protocol,
                            include_failed_targets=include_failed_targets,
                            sample_types=sample_types,
                        )
                        if max_rows and stats["rows"] >= max_rows:
                            break
                    if compress_samples:
                        stats["skipped"]["compress_target_without_trigger"] = (
                            stats["skipped"].get("compress_target_without_trigger", 0)
                            + len(compress_samples)
                        )
                    break

                if max_rows and stats["rows"] >= max_rows:
                    break

                if log_every_steps and stats["steps"] % log_every_steps == 0:
                    rate = stats["steps"] / max(time.time() - t0, 1e-6)
                    print(
                        f"[steps={stats['steps']}] rows={stats['rows']} "
                        f"traj_used={stats['trajectories_used']} "
                        f"compress_turns={stats['policy_compress_turns']} "
                        f"rate={rate:.3f} step/s skipped={stats['skipped']}",
                        flush=True,
                    )
                    fout.flush()

            if stats["trajectories_used"] % 5 == 0:
                rate = stats["steps"] / max(time.time() - t0, 1e-6)
                print(
                    f"[{stats['trajectories_used']} traj] rows={stats['rows']} "
                    f"steps={stats['steps']} rate={rate:.3f} step/s",
                    flush=True,
                )
            if max_rows and stats["rows"] >= max_rows:
                break

    stats["out"] = str(out)
    stats["elapsed_sec"] = round(time.time() - t0, 3)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    batch_root = _default_batch_root(None)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--trajectories",
        default=str(batch_root / "final" / "train_sft_trajectories.jsonl"),
    )
    p.add_argument(
        "--out",
        default=str(batch_root / "rendered" / "video_meta" / "train_sft_dagger_messages.jsonl"),
    )
    p.add_argument("--data-dir", default=str(batch_root))
    p.add_argument("--frames-root", default=str(batch_root / "frames"))
    p.add_argument("--video-root", default=None)
    p.add_argument("--frame-protocol", default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"))
    p.add_argument("--retriever", default="bm25", choices=["bm25", "hybrid"])
    p.add_argument("--max-results", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--profile", default="16k", choices=["16k", "32k"])
    p.add_argument(
        "--sample-types",
        default="silent,response,recall,compress",
        help="Comma-separated target sample_type list. Compress rows are only emitted when the on-policy prompt has <compress_trigger/>.",
    )
    p.add_argument("--include-failed-targets", action="store_true")
    p.add_argument("--max-trajectories", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument(
        "--max-compress-turns-per-chunk",
        type=int,
        default=2,
        help="Retry the same visual chunk after at most this many policy compress turns.",
    )
    p.add_argument(
        "--log-every-steps",
        type=int,
        default=20,
        help="Print DAgger rollout progress every N policy steps (0 disables).",
    )
    args = p.parse_args()

    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    sample_types = {x.strip() for x in args.sample_types.split(",") if x.strip()}
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    stats = build_dagger(
        ckpt=args.ckpt,
        trajectories=_resolve_path(args.trajectories),
        out=_resolve_path(args.out),
        data_dir=_resolve_path(args.data_dir),
        frames_root=str(_resolve_path(args.frames_root)),
        video_root=str(_resolve_path(args.video_root)) if args.video_root else None,
        frame_protocol=frame_protocol,
        retriever_kind=args.retriever,
        max_results=args.max_results,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        profile=args.profile,
        sample_types=sample_types,
        include_failed_targets=args.include_failed_targets,
        max_trajectories=args.max_trajectories,
        max_rows=args.max_rows,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        no_bf16=args.no_bf16,
        max_compress_turns_per_chunk=args.max_compress_turns_per_chunk,
        log_every_steps=args.log_every_steps,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
