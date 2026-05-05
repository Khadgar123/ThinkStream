#!/usr/bin/env python
"""Trajectory-level base VLM probe for ThinkStream data.

This evaluates base Qwen3-VL checkpoints on whole trajectory question sets,
including multi-emit questions. It compares context variants:

- streaming: current 16s visual window only.
- text_memory: historical per-chunk <think> text only.
- streaming_text_memory: current visual window plus historical think text.
- recall_oracle: current visual window plus teacher recalled frames, if present.
- offline_past: uniformly sampled frames from video start to the emit chunk.
- offline_full: uniformly sampled frames from the whole extracted video.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.eval.trajectory_base_probe \\
    --ckpt /home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct \\
    --input train_sft:data/agent_v5/batch1/final/train_sft_trajectories.jsonl \\
    --input train_rl:data/agent_v5/batch1/final/train_rl_trajectories.jsonl \\
    --input val:data/agent_v5/batch1/final/val_trajectories.jsonl \\
    --input test:data/agent_v5/batch1/final/test_trajectories.jsonl \\
    --modes streaming streaming_text_memory recall_oracle \\
    --n-trajectories-per-split 20 \\
    --out output/eval/base_probe/qwen3vl2b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thinkstream.data.agent_protocol import (  # noqa: E402
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    VISUAL_WINDOW_CHUNKS,
    append_visual_frames,
    normalize_frame_protocol,
)
from thinkstream.sft.argument import DataArguments  # noqa: E402
from thinkstream.sft.data_processor import update_processor_pixels  # noqa: E402
from thinkstream.trainer.outcome_match import score_outcome_by_form  # noqa: E402


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
DEFAULT_FPS = float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC)


def detect_model_class(ckpt: str):
    name = ckpt.lower()
    basename = Path(ckpt.rstrip("/")).name.lower()
    if "qwen3" in name and "a" in basename:
        from transformers import Qwen3VLMoeForConditionalGeneration as Cls
        return Cls
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls
    if "qwen2.5" in name or "qwen-2.5" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        return Cls
    from transformers import Qwen3VLForConditionalGeneration as Cls
    return Cls


def chat_template_supports_thinking(processor) -> bool:
    tmpl = getattr(processor, "chat_template", None)
    if not tmpl and hasattr(processor, "tokenizer"):
        tmpl = getattr(processor.tokenizer, "chat_template", None)
    return "enable_thinking" in (tmpl or "")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_inputs(values: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for value in values:
        if ":" not in value:
            raise SystemExit(f"--input must be split:path, got {value!r}")
        name, path = value.split(":", 1)
        out.append((name, Path(path)))
    return out


def question_text(q: Dict[str, Any]) -> str:
    lines = [str(q.get("question") or "").strip()]
    options = list(q.get("options") or [])
    if options:
        lines.append("Options:")
        lines.extend(str(opt) for opt in options)
    instr = str(q.get("answer_instruction") or "").strip()
    if instr:
        lines.append(instr)
    return "\n".join(line for line in lines if line)


def sample_map(traj: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    by_chunk: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in traj.get("samples") or []:
        try:
            by_chunk[int(sample.get("chunk_idx", -1))].append(sample)
        except Exception:
            continue
    return by_chunk


def frame_dir_for(traj: Dict[str, Any], frames_root: Optional[Path]) -> Path:
    if frames_root:
        return frames_root / str(traj.get("video_id"))
    return ROOT / "data" / "agent_v5" / "batch1" / "frames" / str(traj.get("video_id"))


def frame_time(path: Path, fps: float) -> float:
    stem = path.stem
    if stem.startswith("frame_"):
        stem = stem[len("frame_"):]
    try:
        return max(0, int(stem)) / fps
    except ValueError:
        return 0.0


def sample_frames_from_dir(
    frame_dir: Path,
    *,
    start_t: float,
    end_t: float,
    max_frames: int,
    fps: float,
) -> List[str]:
    frames = []
    for fp in sorted(frame_dir.glob("*.jpg")):
        t = frame_time(fp, fps)
        if start_t <= t <= end_t:
            frames.append(fp)
    if not frames:
        return []
    if max_frames > 0 and len(frames) > max_frames:
        idxs = [int(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        frames = [frames[i] for i in idxs]
    return [str(fp) for fp in frames]


def cap_evenly(paths: List[str], max_items: int) -> List[str]:
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    if max_items == 1:
        return [paths[len(paths) // 2]]
    idxs = [int(i * (len(paths) - 1) / (max_items - 1)) for i in range(max_items)]
    return [paths[i] for i in idxs]


def visual_window_frames(
    samples_by_chunk: Dict[int, List[Dict[str, Any]]],
    chunk: int,
    *,
    frame_dir: Optional[Path] = None,
    fps: float = DEFAULT_FPS,
) -> List[str]:
    for sample in samples_by_chunk.get(chunk, []):
        window = ((sample.get("input") or {}).get("visual_window") or {})
        paths = window.get("frame_paths") or []
        if paths:
            return [str(Path(p) if Path(p).is_absolute() else ROOT / p) for p in paths]
        frames = window.get("frames")
        if isinstance(frames, list) and frames:
            return [str(Path(p) if Path(p).is_absolute() else ROOT / p) for p in frames]
        if frame_dir and frame_dir.exists():
            try:
                start_t = float(window.get("video_start", chunk * AGENT_CHUNK_SEC))
                end_t = float(window.get("video_end", (chunk + 1) * AGENT_CHUNK_SEC))
            except Exception:
                start_t = chunk * AGENT_CHUNK_SEC
                end_t = (chunk + 1) * AGENT_CHUNK_SEC
            paths = sample_frames_from_dir(
                frame_dir,
                start_t=start_t,
                end_t=end_t,
                max_frames=0,
                fps=fps,
            )
            if paths:
                return paths
    return []


def extract_think(sample: Dict[str, Any]) -> str:
    for key in ("output", "v12_assistant_turn_1"):
        text = sample.get(key) or ""
        m = THINK_RE.search(text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def memory_text(
    samples_by_chunk: Dict[int, List[Dict[str, Any]]],
    *,
    chunk: int,
    max_chars: int,
) -> str:
    rows = []
    for c in sorted(samples_by_chunk):
        if c >= chunk:
            continue
        think = ""
        for sample in samples_by_chunk[c]:
            think = extract_think(sample)
            if think:
                break
        if think:
            rows.append(f"[{c}-{c + 1}] {think}")
    text = "\n".join(rows)
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]
    return text


def recalled_frame_paths(
    samples_by_chunk: Dict[int, List[Dict[str, Any]]],
    *,
    chunk: int,
    card_id: str,
    max_frames: int,
    frame_dir: Optional[Path] = None,
    fps: float = DEFAULT_FPS,
) -> List[str]:
    candidates = []
    for sample in samples_by_chunk.get(chunk, []):
        if card_id and sample.get("card_id") not in ("", card_id):
            continue
        candidates.append(sample)
    for sample in candidates:
        containers = [
            sample.get("recalled_frames") or {},
            ((sample.get("input") or {}).get("recall_result") or {}),
        ]
        for obj in containers:
            paths = obj.get("frame_paths") or []
            if paths:
                out = [str(Path(p) if Path(p).is_absolute() else ROOT / p) for p in paths]
                return out[:max_frames] if max_frames > 0 else out
            if not frame_dir or not frame_dir.exists():
                continue
            returned_chunks = obj.get("returned_chunks") or []
            if returned_chunks:
                out = []
                for c in returned_chunks:
                    try:
                        ci = int(c)
                    except Exception:
                        continue
                    out.extend(sample_frames_from_dir(
                        frame_dir,
                        start_t=ci * AGENT_CHUNK_SEC,
                        end_t=(ci + 1) * AGENT_CHUNK_SEC,
                        max_frames=0,
                        fps=fps,
                    ))
                if out:
                    return cap_evenly(out, max_frames)
            time_range = str(obj.get("time") or "")
            m = re.search(r"(\d+(?:\.\d+)?)\s*[-,]\s*(\d+(?:\.\d+)?)", time_range)
            if m:
                start_t, end_t = float(m.group(1)), float(m.group(2))
                paths = sample_frames_from_dir(
                    frame_dir,
                    start_t=start_t,
                    end_t=end_t,
                    max_frames=max_frames,
                    fps=fps,
                )
                if paths:
                    return paths
    return []


def build_events(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    for q in traj.get("questions") or []:
        emits = q.get("per_emit_answers") or []
        if not emits:
            emits = [{"chunk": c, "value": q.get("gold_answer") or q.get("correct_answer_text") or ""}
                     for c in (q.get("answer_chunks") or [])]
        if not emits:
            continue
        multi_emit = len(emits) > 1
        for emit_idx, emit in enumerate(emits):
            try:
                chunk = int(emit.get("chunk"))
            except Exception:
                continue
            events.append({
                "card_id": q.get("card_id", ""),
                "family": q.get("family", ""),
                "answer_form": q.get("answer_form", ""),
                "options": list(q.get("options") or []),
                "correct_option": q.get("correct_option", ""),
                "gold_answer": str(emit.get("value") or q.get("gold_answer") or q.get("correct_answer_text") or ""),
                "question": question_text(q),
                "ask_chunk": int(q.get("ask_chunk", chunk)),
                "emit_chunk": chunk,
                "emit_idx": emit_idx,
                "multi_emit": multi_emit,
            })
    return events


def system_prompt(frame_protocol: str) -> str:
    if normalize_frame_protocol(frame_protocol) == "video_meta":
        visual = (
            "You receive Qwen video_metadata with fps/frame indices carrying timestamps. "
            "Use it as timing metadata, but do not copy metadata."
        )
    else:
        visual = (
            "You receive timestamp-tagged images. Use frame timestamps as video time, "
            "but never copy frame tags."
        )
    return (
        "You are a video understanding assistant. "
        f"{visual} Answer the user question directly and obey any requested answer format."
    )


def build_messages(
    *,
    question: str,
    frames: List[str],
    memory: str,
    recall_frames: List[str],
    frame_protocol: str,
    fps: float,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    frame_protocol = normalize_frame_protocol(frame_protocol)
    if memory:
        user_content.append({
            "type": "text",
            "text": "<memory>\n" + memory + "\n</memory>",
        })
    if frames:
        append_visual_frames(
            user_content,
            frames,
            frame_protocol=frame_protocol,
            fps=fps,
            context_label="visual frame",
        )
    if recall_frames:
        append_visual_frames(
            user_content,
            recall_frames,
            frame_protocol=frame_protocol,
            fps=fps,
            context_label="recalled frame",
        )
    user_content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(frame_protocol)}]},
        {"role": "user", "content": user_content},
    ]


def generate_one(model, processor, messages, *, max_new_tokens: int, enable_thinking: Optional[bool]) -> str:
    kwargs = dict(
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        do_sample_frames=False,
    )
    video_metadata = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "video":
                meta = item.get("video_metadata")
                if isinstance(meta, dict):
                    video_metadata.append({k: v for k, v in meta.items() if k != "do_sample_frames"})
    if video_metadata:
        kwargs["video_metadata"] = video_metadata
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    inputs = processor.apply_chat_template(messages, **kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    return processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()


def score_event(pred: str, ev: Dict[str, Any]) -> bool:
    return score_outcome_by_form(
        pred,
        options=ev.get("options") or [],
        correct_option=ev.get("correct_option", ""),
        gold_answer=ev.get("gold_answer", ""),
        answer_form=ev.get("answer_form", ""),
    ) >= 0.5


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def bucket():
        return {"n": 0, "correct": 0}

    by_mode = defaultdict(bucket)
    by_split_mode = defaultdict(bucket)
    by_form = defaultdict(bucket)
    by_family = defaultdict(bucket)
    by_multi = defaultdict(bucket)
    question_hits = defaultdict(lambda: {"n": 0, "correct": 0})

    for r in rows:
        keys = [
            (by_mode, r["mode"]),
            (by_split_mode, f"{r['split']}::{r['mode']}"),
            (by_form, f"{r['mode']}::{r.get('answer_form') or '?'}"),
            (by_family, f"{r['mode']}::{r.get('family') or '?'}"),
            (by_multi, f"{r['mode']}::multi={int(bool(r.get('multi_emit')))}"),
        ]
        for mapping, key in keys:
            mapping[key]["n"] += 1
            mapping[key]["correct"] += int(r["correct"])
        qkey = f"{r['mode']}::{r['split']}::{r['trajectory_id']}::{r['card_id']}"
        question_hits[qkey]["n"] += 1
        question_hits[qkey]["correct"] += int(r["correct"])

    question_by_mode = defaultdict(bucket)
    for key, val in question_hits.items():
        mode = key.split("::", 1)[0]
        question_by_mode[mode]["n"] += 1
        question_by_mode[mode]["correct"] += int(val["n"] > 0 and val["correct"] == val["n"])

    def finalize(mapping):
        out = {}
        for k, v in sorted(mapping.items()):
            out[k] = {
                **v,
                "accuracy": v["correct"] / v["n"] if v["n"] else 0.0,
            }
        return out

    return {
        "by_mode_emit": finalize(by_mode),
        "by_split_mode_emit": finalize(by_split_mode),
        "by_answer_form_emit": finalize(by_form),
        "by_family_emit": finalize(by_family),
        "by_multi_emit": finalize(by_multi),
        "by_mode_question_all_emit": finalize(question_by_mode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", action="append", required=True, help="split:path, repeatable")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["streaming", "text_memory", "streaming_text_memory", "recall_oracle"],
        choices=[
            "streaming", "text_memory", "streaming_text_memory",
            "recall_oracle", "offline_past", "offline_full",
        ],
    )
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--frame-protocol", default="video_meta", choices=["ts_image", "video_meta"])
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--n-trajectories-per-split", type=int, default=0)
    parser.add_argument("--max-events-per-split", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-visual-frames", type=int, default=32)
    parser.add_argument("--max-recall-frames", type=int, default=8)
    parser.add_argument("--max-memory-chars", type=int, default=12000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    Cls = detect_model_class(args.ckpt)
    print(f"Loading {Cls.__name__} from {args.ckpt}")
    model = Cls.from_pretrained(
        args.ckpt,
        dtype=torch.bfloat16 if not args.no_bf16 else None,
        attn_implementation="flash_attention_2",
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    processor = update_processor_pixels(processor, DataArguments())
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    enable_thinking = False if chat_template_supports_thinking(processor) else None
    if enable_thinking is False:
        print("[chat_template] Disabling thinking mode.")

    frames_root = Path(args.frames_root) if args.frames_root else None
    rows: List[Dict[str, Any]] = []
    skipped = Counter()
    t0 = time.time()

    for split, path in parse_inputs(args.input):
        trajectories = list(iter_jsonl(path))
        if args.n_trajectories_per_split > 0:
            trajectories = trajectories[: args.n_trajectories_per_split]
        split_events = 0
        print(f"[{split}] trajectories={len(trajectories)} path={path}")
        for traj_idx, traj in enumerate(trajectories):
            by_chunk = sample_map(traj)
            frame_dir = frame_dir_for(traj, frames_root)
            events = build_events(traj)
            for ev in events:
                if args.max_events_per_split and split_events >= args.max_events_per_split:
                    break
                chunk = int(ev["emit_chunk"])
                visual_frames = visual_window_frames(
                    by_chunk,
                    chunk,
                    frame_dir=frame_dir,
                    fps=args.fps,
                )
                if args.max_visual_frames > 0 and len(visual_frames) > args.max_visual_frames:
                    visual_frames = visual_frames[-args.max_visual_frames:]
                text_mem = memory_text(by_chunk, chunk=chunk, max_chars=args.max_memory_chars)
                oracle_frames = recalled_frame_paths(
                    by_chunk,
                    chunk=chunk,
                    card_id=ev.get("card_id", ""),
                    max_frames=args.max_recall_frames,
                    frame_dir=frame_dir,
                    fps=args.fps,
                )
                event_end_t = (chunk + 1) * AGENT_CHUNK_SEC
                offline_past = sample_frames_from_dir(
                    frame_dir,
                    start_t=0.0,
                    end_t=event_end_t,
                    max_frames=args.max_visual_frames,
                    fps=args.fps,
                )
                offline_full = sample_frames_from_dir(
                    frame_dir,
                    start_t=0.0,
                    end_t=10**9,
                    max_frames=args.max_visual_frames,
                    fps=args.fps,
                )
                contexts = {
                    "streaming": (visual_frames, "", []),
                    "text_memory": ([], text_mem, []),
                    "streaming_text_memory": (visual_frames, text_mem, []),
                    "recall_oracle": (visual_frames, "", oracle_frames),
                    "offline_past": (offline_past, "", []),
                    "offline_full": (offline_full, "", []),
                }
                for mode in args.modes:
                    frames, mem, recall = contexts[mode]
                    if mode in {"streaming", "streaming_text_memory", "recall_oracle"} and not frames:
                        skipped[f"{mode}:no_visual"] += 1
                        continue
                    if mode == "text_memory" and not mem:
                        skipped[f"{mode}:no_memory"] += 1
                        continue
                    if mode == "recall_oracle" and not recall:
                        skipped[f"{mode}:no_recall"] += 1
                        continue
                    if mode.startswith("offline") and not frames:
                        skipped[f"{mode}:no_frames"] += 1
                        continue
                    messages = build_messages(
                        question=ev["question"],
                        frames=frames,
                        memory=mem,
                        recall_frames=recall,
                        frame_protocol=args.frame_protocol,
                        fps=args.fps,
                    )
                    try:
                        pred = generate_one(
                            model,
                            processor,
                            messages,
                            max_new_tokens=args.max_new_tokens,
                            enable_thinking=enable_thinking,
                        )
                        correct = score_event(pred, ev)
                    except Exception as exc:
                        skipped[f"{mode}:{type(exc).__name__}"] += 1
                        continue
                    rows.append({
                        "split": split,
                        "mode": mode,
                        "trajectory_id": traj.get("trajectory_id"),
                        "video_id": traj.get("video_id"),
                        **ev,
                        "n_visual_frames": len(frames),
                        "n_recall_frames": len(recall),
                        "memory_chars": len(mem),
                        "pred": pred[:500],
                        "correct": bool(correct),
                    })
                split_events += 1
            if (traj_idx + 1) % 5 == 0:
                rate = max(1, len(rows)) / max(1e-6, time.time() - t0)
                print(f"[{split}] {traj_idx + 1}/{len(trajectories)} traj rows={len(rows)} rate={rate:.3f}/s")

    report = {
        "ckpt": args.ckpt,
        "inputs": args.input,
        "modes": args.modes,
        "frame_protocol": args.frame_protocol,
        "n_rows": len(rows),
        "skipped": dict(skipped),
        "summary": aggregate(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {out} rows={len(rows)} skipped={dict(skipped)}")
    print(json.dumps(report["summary"]["by_mode_emit"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
