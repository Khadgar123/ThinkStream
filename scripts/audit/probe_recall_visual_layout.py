#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    SYSTEM_PROMPT,
    append_timestamped_image_list,
    build_assistant_content,
    build_user_content,
    canonical_answer_instruction,
    format_memory_block,
    infer_video_metadata,
    tools_for_turn,
)


LAYOUTS = ("video_only", "timestamp_video", "timestamp_images")
CURRENT_POSITIONS = ("before", "same", "after")
ANSWER_POSITIONS = ("first", "middle", "last")
NOISE_LEVELS = (0, 1, 2)


def _frame_dir(frames_root: Path, video_path: str) -> Path:
    rel = Path(video_path)
    stem = rel.with_suffix("")
    if len(stem.parts) >= 2 and stem.parts[0] == "Ego4D" and stem.parts[1] == "clips":
        stem = Path("Ego4D") / "video" / stem.name
    return frames_root / stem


def _chunk_frame_paths(frames_root: Path, video_path: str, chunk: int) -> list[str]:
    directory = _frame_dir(frames_root, video_path)
    base = int(chunk) * int(FRAMES_PER_CHUNK)
    out = []
    for offset in range(int(FRAMES_PER_CHUNK)):
        path = directory / f"frame_{base + offset + 1:06d}.jpg"
        if path.exists():
            out.append(str(path))
    return out


def _chunk_exists(frames_root: Path, video_path: str, chunk: int) -> bool:
    return len(_chunk_frame_paths(frames_root, video_path, chunk)) == int(FRAMES_PER_CHUNK)


def _load_cases(path: Path, frames_root: Path, limit: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions = row.get("questions") or []
            if not questions:
                continue
            q = dict(questions[0])
            chunks = [int(x) for x in (q.get("support_chunks") or [])]
            if len(chunks) != 1:
                continue
            video_path = row.get("source_video_path") or row.get("video_path") or ""
            support = chunks[0]
            if not _chunk_exists(frames_root, video_path, support):
                continue
            task = str(q.get("ovo_task") or q.get("family") or row.get("task") or "")
            q.update({
                "source_video_path": video_path,
                "trajectory_id": row.get("trajectory_id"),
                "support_chunk": support,
                "task": task,
                "segment_start_chunk": row.get("segment_start_chunk", 0),
                "segment_end_chunk": row.get("segment_end_chunk", support),
            })
            by_task[task].append(q)

    cases: list[dict[str, Any]] = []
    for task in sorted(by_task):
        if by_task[task]:
            cases.append(by_task[task][0])
            if len(cases) >= limit:
                return cases
    return cases[:limit]


def _gold(q: dict[str, Any]) -> str:
    if q.get("correct_option"):
        return str(q["correct_option"]).strip().upper()
    ga = str(q.get("gold_answer") or "").strip()
    if ga in {"Yes", "No"}:
        return ga
    return ga.upper()[:1]


def _extract_pred(text: str, options: list[Any]) -> str:
    s = str(text or "").strip()
    m = re.search(r"\b([A-D])\b", s, re.I)
    if m:
        return m.group(1).upper()
    if re.search(r"\byes\b", s, re.I):
        return "Yes"
    if re.search(r"\bno\b", s, re.I):
        return "No"
    for i, opt in enumerate(options or []):
        raw = re.sub(r"^[A-D]\)\s*", "", str(opt).strip(), flags=re.I)
        if raw and raw.lower() in s.lower():
            return chr(65 + i)
    return s[:32]


def _query_record(q: dict[str, Any], current_chunk: int) -> dict[str, Any]:
    answer_form = q.get("answer_form") or ("multiple_choice" if q.get("options") else "")
    answer_style = "letter_plus_text" if answer_form == "multiple_choice" else q.get("answer_style", "")
    instruction_source = dict(q)
    instruction_source["answer_form"] = answer_form
    instruction_source["answer_style"] = answer_style
    return {
        "question": q.get("question", ""),
        "options": list(q.get("options") or []),
        "answer_form": answer_form,
        "answer_style": answer_style,
        "answer_instruction": canonical_answer_instruction(instruction_source),
        "ask_time": current_chunk * AGENT_CHUNK_SEC,
        "open_until": current_chunk * AGENT_CHUNK_SEC,
        "status": "open",
        "answers": [],
    }


def _memory_text(q: dict[str, Any], current_chunk: int) -> str:
    support = int(q["support_chunk"])
    recent = []
    for ck in range(max(0, current_chunk - 3), current_chunk):
        if ck == support:
            continue
        recent.append({
            "chunk": ck,
            "text": f"Observed ordinary recent frames around t={ck * AGENT_CHUNK_SEC}-{(ck + 1) * AGENT_CHUNK_SEC}s.",
        })
    compressed = []
    if support > 3:
        compressed.append({
            "time_range": [0, max(0, support - 2)],
            "text": "Earlier video context was summarized; exact recalled visual evidence is needed for the active question.",
        })
    return format_memory_block({
        "compressed_segments": compressed,
        "compressed": compressed,
        "recent_thinks": recent,
    })


def _current_chunk(q: dict[str, Any], pos: str, frames_root: Path) -> int:
    video = str(q["source_video_path"])
    support = int(q["support_chunk"])
    candidates = {
        "before": [support - 3, support - 2, support - 1],
        "same": [support],
        "after": [support + 1, support + 2, support + 3],
    }[pos]
    for ck in candidates:
        if ck >= 0 and _chunk_exists(frames_root, video, ck):
            return ck
    return support


def _noise_chunks(q: dict[str, Any], frames_root: Path, n_noise: int) -> list[int]:
    if n_noise <= 0:
        return []
    video = str(q["source_video_path"])
    support = int(q["support_chunk"])
    candidates = []
    for delta in (6, -6, 10, -10, 14, -14, 20, -20, 3, -3):
        ck = support + delta
        if ck >= 0 and ck != support and _chunk_exists(frames_root, video, ck):
            candidates.append(ck)
    dedup = []
    for ck in candidates:
        if ck not in dedup:
            dedup.append(ck)
    return dedup[:n_noise]


def _returned_chunks(support: int, noise: list[int], answer_pos: str) -> list[int]:
    if not noise:
        return [support]
    if answer_pos == "first":
        return [support] + noise
    if answer_pos == "last":
        return noise + [support]
    left = noise[:1]
    right = noise[1:]
    return left + [support] + right


def _evidence_content(
    q: dict[str, Any],
    returned_chunks: list[int],
    layout: str,
    frames_root: Path,
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    video = str(q["source_video_path"])
    frame_paths: list[str] = []
    for ck in returned_chunks:
        frame_paths.extend(_chunk_frame_paths(frames_root, video, ck))
    start = min(returned_chunks) * AGENT_CHUNK_SEC
    end = max(returned_chunks) * AGENT_CHUNK_SEC
    content: list[dict[str, Any]] = []

    content.append({
        "type": "text",
        "text": f"The recall tool returned historical video frames for t={int(start)}-{int(end)}.",
        "kv_scope": "recall",
    })

    if layout in {"video_only", "timestamp_video"}:
        content.append({
            "type": "video",
            "video": frame_paths,
            "video_metadata": infer_video_metadata(
                frame_paths,
                fps=float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC),
                start_frame_index=returned_chunks[0] * FRAMES_PER_CHUNK,
                total_num_frames=(max(returned_chunks) + 1) * FRAMES_PER_CHUNK,
            ),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "kv_scope": "recall",
        })
    elif layout == "timestamp_images":
        labels = []
        for ck in returned_chunks:
            labels.extend([f"recalled chunk {ck}"] * FRAMES_PER_CHUNK)
        append_timestamped_image_list(
            content,
            frame_paths,
            fps=float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC),
            start_frame_index=returned_chunks[0] * FRAMES_PER_CHUNK,
            total_num_frames=(max(returned_chunks) + 1) * FRAMES_PER_CHUNK,
            timestamp_labels=labels,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        raise ValueError(layout)
    return content


def _messages_for_case(
    q: dict[str, Any],
    *,
    layout: str,
    current_position: str,
    answer_position: str,
    n_noise: int,
    frames_root: Path,
    min_pixels: int,
    max_pixels: int,
    answer_cue: bool,
    system_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support = int(q["support_chunk"])
    current = _current_chunk(q, current_position, frames_root)
    noise = _noise_chunks(q, frames_root, n_noise)
    returned = _returned_chunks(support, noise, answer_position)

    current_paths = _chunk_frame_paths(frames_root, str(q["source_video_path"]), current)
    query = _query_record(q, current)
    user_content = build_user_content(
        _memory_text(q, current),
        current,
        "",
        user_input=q.get("question", ""),
        queries=[query],
        frame_paths=current_paths,
        frame_protocol="video_meta",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        memory_snapshot=None,
    )
    recall_call = build_assistant_content(
        think="The active question needs evidence from earlier video frames, so I should recall the relevant time span.",
        kind="recall",
        recall_query={
            "start_time": support * AGENT_CHUNK_SEC,
            "end_time": support * AGENT_CHUNK_SEC,
        },
    )
    tool_content = _evidence_content(
        q,
        returned,
        layout,
        frames_root,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if answer_cue:
        tool_content.append({
            "type": "text",
            "text": (
                "\nUse the recalled visual evidence above to answer the active query now. "
                "Follow the required answer format exactly."
            ),
            "kv_scope": "recall",
        })
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": recall_call},
        {"role": "tool", "tool_call_id": "recall", "content": tool_content},
    ]
    meta = {
        "current_chunk": current,
        "support_chunk": support,
        "returned_chunks": returned,
        "noise_chunks": noise,
        "current_position": current_position,
        "answer_position": answer_position,
        "noise_level": n_noise,
        "layout": layout,
    }
    return messages, meta


def _prepare_inputs(processor, messages: list[dict[str, Any]], device) -> tuple[dict[str, Any], int]:
    text = processor.apply_chat_template(
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=getattr(processor.image_processor, "patch_size", 16),
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = None
    if video_inputs is not None:
        preferred = []
        for msg in messages:
            for item in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                if isinstance(item, dict) and item.get("type") == "video":
                    preferred.append(item.get("video_metadata"))
        videos = []
        video_metadata = []
        for i, item in enumerate(video_inputs):
            if isinstance(item, tuple) and len(item) == 2:
                tensor, meta = item
            else:
                tensor, meta = item, None
            if i < len(preferred) and isinstance(preferred[i], dict):
                meta = preferred[i]
            videos.append(tensor)
            video_metadata.append({k: v for k, v in dict(meta or {}).items() if k != "do_sample_frames"})
        video_inputs = videos
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,
        **(video_kwargs or {}),
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs, int(inputs["input_ids"].shape[1])


@torch.inference_mode()
def generate_one(model, processor, q: dict[str, Any], spec: dict[str, Any], args) -> dict[str, Any]:
    messages, meta = _messages_for_case(
        q,
        layout=spec["layout"],
        current_position=spec["current_position"],
        answer_position=spec["answer_position"],
        n_noise=int(spec["noise_level"]),
        frames_root=args.frames_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        answer_cue=bool(args.answer_cue),
        system_prompt=args.system_prompt,
    )
    inputs, prompt_len = _prepare_inputs(processor, messages, model.device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    text = processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
    pred = _extract_pred(text, list(q.get("options") or []))
    gold = _gold(q)
    return {
        **meta,
        "output": text,
        "pred": pred,
        "gold": gold,
        "correct": pred == gold,
        "prompt_tokens": prompt_len,
        "new_tokens": int(out.shape[1] - prompt_len),
        "latency_sec": round(time.time() - t0, 3),
    }


def _matrix(args) -> list[dict[str, Any]]:
    specs = []
    for current_position in CURRENT_POSITIONS:
        for answer_position in ANSWER_POSITIONS:
            for noise_level in NOISE_LEVELS:
                for layout in LAYOUTS:
                    specs.append({
                        "layout": layout,
                        "current_position": current_position,
                        "answer_position": answer_position,
                        "noise_level": noise_level,
                    })
    return specs[:args.max_specs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-cases", type=int, default=6)
    parser.add_argument("--max-specs", type=int, default=27)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--answer-cue", action="store_true")
    parser.add_argument(
        "--strict-post-recall-system",
        action="store_true",
        help="Temporarily replace the post-recall sentence in SYSTEM_PROMPT for A/B probing.",
    )
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("--shard-index must be in [0, shard_count)")
    args.system_prompt = SYSTEM_PROMPT
    if args.strict_post_recall_system:
        args.system_prompt = SYSTEM_PROMPT.replace(
            "After a recall tool result: no recall next turn — answer or stay </Silence>.",
            (
                "After a recall tool result: do not describe recalled frames. "
                "Use them only as evidence; output <think>...</think></Response> answer "
                "if the active query is answerable, otherwise <think>...</think></Silence>."
            ),
        )

    cases = _load_cases(args.trajectories, args.frames_root, args.limit_cases)
    if not cases:
        raise RuntimeError("no single-support-chunk cases with frames found")
    specs = _matrix(args)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()

    results = []
    global_run_idx = 0
    for ci, q in enumerate(cases):
        case = {
            "case_idx": ci,
            "trajectory_id": q.get("trajectory_id"),
            "video": q.get("source_video_path"),
            "task": q.get("task"),
            "question": q.get("question"),
            "support_chunk": q.get("support_chunk"),
            "gold": _gold(q),
            "options": q.get("options") or [],
            "runs": [],
        }
        for spec in specs:
            if global_run_idx % args.shard_count != args.shard_index:
                global_run_idx += 1
                continue
            try:
                case["runs"].append(generate_one(model, processor, q, spec, args))
            except Exception as exc:  # noqa: BLE001
                case["runs"].append({**spec, "error": repr(exc), "correct": False})
            global_run_idx += 1
        if case["runs"]:
            results.append(case)

    summary: dict[str, Any] = {"overall": {}}
    all_runs = [r for c in results for r in c["runs"]]
    summary["overall"] = {
        "correct": sum(bool(r.get("correct")) for r in all_runs),
        "total": len(all_runs),
        "accuracy": round(sum(bool(r.get("correct")) for r in all_runs) / max(len(all_runs), 1), 4),
    }
    for key in ("layout", "current_position", "answer_position", "noise_level"):
        bucket: dict[str, dict[str, Any]] = {}
        for r in all_runs:
            val = str(r.get(key))
            bucket.setdefault(val, {"correct": 0, "total": 0})
            bucket[val]["correct"] += int(bool(r.get("correct")))
            bucket[val]["total"] += 1
        for val, obj in bucket.items():
            obj["accuracy"] = round(obj["correct"] / max(obj["total"], 1), 4)
        summary[key] = bucket

    payload = {
        "model": args.model,
        "trajectories": str(args.trajectories),
        "frames_root": str(args.frames_root),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "answer_cue": bool(args.answer_cue),
        "specs": specs,
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
