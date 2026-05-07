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
    --frame-protocol video_meta --correction-only --max-trajectories 20

For production-scale construction, prefer build_dagger_sft_vllm.py with
--correction-only and shard/batch its rollout.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from scripts.agent_data_v5.config import AGENT_CHUNK_SEC
from scripts.agent_data_v5.pass5_messages import build_messages, _emit_row
from scripts.eval.ovo.eval_full import detect_model_class, reset_visual_index
from scripts.eval.processor_loader import load_processor_for_checkpoint
from thinkstream.data.agent_protocol import (
    has_compress_trigger,
    normalize_frame_protocol,
    parse_agent_output_v12,
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
    answer_chunks = (
        sample.get("answer_chunks")
        or sample.get("expected_answer_chunks")
        or meta.get("answer_chunks")
        or meta.get("expected_answer_chunks")
        or []
    )
    per_emit_answers = sample.get("per_emit_answers") or meta.get("per_emit_answers") or []
    open_until = sample.get("open_until") or meta.get("open_until")
    if open_until is None and answer_chunks:
        try:
            open_until = max(int(x) for x in answer_chunks) * AGENT_CHUNK_SEC
        except (TypeError, ValueError):
            open_until = None
    return {
        "options": sample.get("options") or meta.get("options") or [],
        "answer_form": sample.get("answer_form") or meta.get("answer_form") or "",
        "answer_style": sample.get("answer_style") or meta.get("answer_style") or "",
        "answer_instruction": (
            sample.get("answer_instruction")
            or meta.get("answer_instruction")
            or ""
        ),
        "answer_chunks": list(answer_chunks),
        "per_emit_answers": list(per_emit_answers),
        "open_until": open_until,
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

    Compression may also appear in turn-local system/tool text. DAgger needs
    the runtime event, which is rendered under the user turn's
    ``<user_input>...</user_input>`` block.
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


DEFAULT_DAGGER_CORRECTION_REASONS = {
    "format_error",
    "repeated_or_stale_think",
    "missed_compress",
    "bad_compress_json",
    "bad_compress_range",
    "missed_recall",
    "missed_response",
    "wrong_response",
    "early_answer",
}


def _parse_reason_set(raw: str) -> set[str]:
    raw = str(raw or "").strip()
    if not raw or raw.lower() in {"default", "defaults"}:
        return set(DEFAULT_DAGGER_CORRECTION_REASONS)
    if raw.lower() == "all":
        return set(DEFAULT_DAGGER_CORRECTION_REASONS) | {
            "recall_answer_visible_in_policy_prompt",
        }
    return {x.strip() for x in raw.split(",") if x.strip()}


def _extract_answer_text(output_text: str) -> str:
    parsed = parse_agent_output_v12(output_text or "")
    if parsed.get("kind") == "answer":
        return str(parsed.get("answer_text") or "").strip()
    return ""


def _gold_output_text(sample: Dict[str, Any]) -> str:
    if sample.get("sample_type") == "recall" and sample.get("v12_assistant_turn_2"):
        return str(sample.get("v12_assistant_turn_2") or "")
    return str(sample.get("output") or sample.get("v12_assistant_turn_1") or "")


def _gold_think(sample: Dict[str, Any]) -> str:
    output = _gold_output_text(sample)
    parsed = parse_agent_output_v12(output)
    return str(parsed.get("think") or "").strip()


def _gold_answer(sample: Dict[str, Any]) -> str:
    output = _gold_output_text(sample)
    answer = _extract_answer_text(output)
    if answer:
        return answer
    meta = sample.get("metadata") or {}
    return str(
        sample.get("gold_answer")
        or sample.get("canonical_answer")
        or meta.get("gold_answer")
        or meta.get("canonical_answer")
        or ""
    ).strip()


def _gold_compress_range(sample: Dict[str, Any]) -> Optional[List[int]]:
    output = str(sample.get("output") or "")
    parsed = parse_agent_output_v12(output)
    tc = parsed.get("tool_call") or {}
    if tc.get("name") != "compress":
        return None
    tr = (tc.get("arguments") or {}).get("time_range")
    if not isinstance(tr, list) or len(tr) != 2:
        return None
    try:
        return [int(tr[0]), int(tr[1])]
    except (TypeError, ValueError):
        return None


def _normalise_range(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return [start, end]


def _word_tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(t) >= 3
    ]


def _token_overlap(a: str, b: str) -> float:
    aa = set(_word_tokens(a))
    if not aa:
        return 0.0
    return len(aa & set(_word_tokens(b))) / max(len(aa), 1)


def _answer_visible_in_prompt(answer: str, messages: List[Dict[str, Any]]) -> bool:
    answer = str(answer or "").strip()
    if not answer:
        return False
    text = _content_text(messages)
    if answer.lower() in text.lower() and len(answer) >= 3:
        return True
    return _token_overlap(answer, text) >= 0.65


def _memory_text_from_prompt(messages: List[Dict[str, Any]]) -> str:
    text = _content_text(messages)
    blocks = re.findall(r"<memory>(.*?)</memory>", text, flags=re.DOTALL)
    return "\n".join(blocks)


def _has_ngram_repetition(tokens: List[str], n: int = 4, threshold: float = 0.22) -> bool:
    if len(tokens) < n * 3:
        return False
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return False
    return 1.0 - (len(set(grams)) / len(grams)) >= threshold


def _answer_chunks(sample: Dict[str, Any]) -> List[int]:
    meta = sample.get("metadata") or {}
    raw = (
        sample.get("answer_chunks")
        or sample.get("expected_answer_chunks")
        or meta.get("answer_chunks")
        or meta.get("expected_answer_chunks")
        or []
    )
    out: List[int] = []
    for x in raw:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    if out:
        return sorted(set(out))
    per_emit = sample.get("per_emit_answers") or meta.get("per_emit_answers") or []
    for item in per_emit:
        if isinstance(item, dict) and item.get("chunk") is not None:
            try:
                out.append(int(item["chunk"]))
            except (TypeError, ValueError):
                pass
    return sorted(set(out))


def _classify_dagger_corrections(
    sample: Dict[str, Any],
    onpolicy_prompt: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Classify why this on-policy state deserves a gold correction row."""
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    sample_type = str(sample.get("sample_type", ""))
    policy_action = str(result.get("final_action") or result.get("action") or "")
    first_action = str(result.get("action") or "")
    prompt_has_compress = _prompt_has_compress_trigger(onpolicy_prompt)
    format_ok = bool(result.get("format_ok", True)) and not result.get("action_space_error")
    if not format_ok or first_action in {"unknown", "invalid"}:
        reasons.append("format_error")

    policy_think = str(result.get("think") or "").strip()
    gold_think = _gold_think(sample)
    if policy_think:
        toks = _word_tokens(policy_think)
        unique_ratio = len(set(toks)) / max(len(toks), 1)
        memory_overlap = _token_overlap(policy_think, _memory_text_from_prompt(onpolicy_prompt))
        gold_overlap = _token_overlap(policy_think, gold_think)
        too_long_vs_gold = bool(gold_think and len(toks) > max(120, 3 * len(_word_tokens(gold_think))))
        if (
            unique_ratio < 0.42
            or _has_ngram_repetition(toks)
            or too_long_vs_gold
            or (memory_overlap >= 0.72 and gold_overlap < 0.35)
            or (gold_think and gold_overlap < 0.12 and len(toks) >= 18)
        ):
            reasons.append("repeated_or_stale_think")
            detail["think_unique_ratio"] = round(unique_ratio, 3)
            detail["think_memory_overlap"] = round(memory_overlap, 3)
            detail["think_gold_overlap"] = round(gold_overlap, 3)
            detail["policy_think_tokens"] = len(toks)

    if prompt_has_compress or sample_type == "compress":
        if first_action != "compress":
            reasons.append("missed_compress")
        else:
            pred_range = _normalise_range(
                ((result.get("payload") or {}).get("summary") or {}).get("time_range")
            )
            gold_range = _gold_compress_range(sample)
            if pred_range is None:
                reasons.append("bad_compress_json")
            elif gold_range and (pred_range[1] <= gold_range[0] or pred_range[0] >= gold_range[1]):
                reasons.append("bad_compress_range")
                detail["gold_compress_range"] = gold_range
                detail["policy_compress_range"] = pred_range

    gold_answer = _gold_answer(sample)
    policy_answer = str(((result.get("final_payload") or result.get("payload") or {}).get("response")) or "").strip()
    if sample_type == "recall":
        if first_action != "recall":
            if _answer_visible_in_prompt(gold_answer, onpolicy_prompt):
                reasons.append("recall_answer_visible_in_policy_prompt")
            else:
                reasons.append("missed_recall")
        if policy_action == "response" and gold_answer and policy_answer:
            if not (
                policy_answer.lower() == gold_answer.lower()
                or _token_overlap(gold_answer, policy_answer) >= 0.65
            ):
                reasons.append("wrong_response")

    if sample_type == "response":
        if policy_action == "silent":
            reasons.append("missed_response")
        elif policy_action == "response" and gold_answer and policy_answer:
            if not (
                policy_answer.lower() == gold_answer.lower()
                or _token_overlap(gold_answer, policy_answer) >= 0.65
            ):
                reasons.append("wrong_response")

    current_chunk = int(sample.get("chunk_idx", 0) or 0)
    chunks = _answer_chunks(sample)
    if sample_type == "silent" and chunks and current_chunk < min(chunks):
        if policy_action == "response":
            reasons.append("early_answer")
            detail["answer_chunks"] = chunks

    # This is a data-construction warning, not a useful DAgger correction:
    # under the student's memory state recall is no longer minimal.
    if "recall_answer_visible_in_policy_prompt" in reasons and "missed_recall" not in reasons:
        detail["recall_prompt_leak"] = True
    return sorted(set(reasons)), detail


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
    correction_only: bool,
    correction_reasons: set[str],
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

    reasons, correction_detail = _classify_dagger_corrections(
        sample,
        onpolicy_prompt,
        result,
    )
    for r in reasons:
        bucket = stats.setdefault("by_correction_reason", {})
        bucket[r] = bucket.get(r, 0) + 1
    selected_reasons = sorted(set(reasons) & set(correction_reasons))
    if correction_only and not selected_reasons:
        stats["skipped"]["no_selected_correction"] = (
            stats["skipped"].get("no_selected_correction", 0) + 1
        )
        if reasons:
            key = "only_unselected_correction"
            stats["skipped"][key] = stats["skipped"].get(key, 0) + 1
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
        "rollout_action_space_error": result.get("action_space_error", ""),
        "rollout_inter_chunk_compress_prompt": bool(prompt_is_compress),
        "memory_token_count": result.get("memory_token_count"),
        "prompt_text_token_count": result.get("prompt_text_token_count"),
        "correction_only": bool(correction_only),
        "correction_reasons": reasons,
        "selected_correction_reasons": selected_reasons,
        "correction_detail": correction_detail,
    }
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows"] += 1
    st = row.get("sample_type", "")
    stats["by_type"][st] = stats["by_type"].get(st, 0) + 1
    for r in selected_reasons:
        bucket = stats.setdefault("by_selected_correction_reason", {})
        bucket[r] = bucket.get(r, 0) + 1
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
    correction_only: bool,
    correction_reasons: set[str],
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
        "by_correction_reason": {},
        "by_selected_correction_reason": {},
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
                                correction_only=correction_only,
                                correction_reasons=correction_reasons,
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
                            correction_only=correction_only,
                            correction_reasons=correction_reasons,
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
    p.add_argument(
        "--correction-only",
        action="store_true",
        help=(
            "Emit only on-policy states whose rollout matches selected "
            "correction reasons. This is the recommended stage-2 SFT mode."
        ),
    )
    p.add_argument(
        "--correction-reasons",
        default=",".join(sorted(DEFAULT_DAGGER_CORRECTION_REASONS)),
        help=(
            "Comma-separated correction reasons kept under --correction-only. "
            "Use 'default' for the production set or 'all' to include "
            "diagnostic warnings such as recall prompt leaks."
        ),
    )
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
    correction_reasons = _parse_reason_set(args.correction_reasons)
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
        correction_only=args.correction_only,
        correction_reasons=correction_reasons,
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
