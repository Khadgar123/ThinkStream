#!/usr/bin/env python3
"""Probe recall tool-call start_time/end_time localization.

Sources:
- own: use rendered trajectory rows that already contain gold recall tool_calls.
  The prompt is the real dialogue prefix immediately before the gold recall.
- ovo: build a synthetic streaming turn where the active query's support chunk
  is historical relative to the current chunk, so the correct action is recall.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.local_hf_kv_visual_probe import normalize_video_inputs  # noqa: E402
from scripts.audit.probe_recall_visual_layout import (  # noqa: E402
    _chunk_exists,
    _chunk_frame_paths,
)
from thinkstream.data.agent_protocol import (  # noqa: E402
    build_user_content,
    canonical_answer_instruction,
    format_memory_block,
    parse_agent_output,
    tools_for_turn,
)
from thinkstream.data.schema import SYSTEM_PROMPT  # noqa: E402
from thinkstream.models import MODEL_CLS  # noqa: E402


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict) and x.get("type") == "text")
    return ""


def _resolve_msg(msg: Dict[str, Any], *, frames_root: Path, video_id: str) -> Dict[str, Any]:
    msg = copy.deepcopy(msg)
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    for item in content:
        if not (isinstance(item, dict) and item.get("type") == "video"):
            continue
        video = item.get("video")
        if not isinstance(video, list):
            continue
        fixed = []
        for frame in video:
            p = Path(str(frame))
            if p.is_absolute():
                out = p
            elif str(frame).startswith("data/"):
                out = (REPO_ROOT / p).resolve()
            else:
                out = (frames_root / video_id / p.name).resolve() if len(p.parts) == 1 else (frames_root / p).resolve()
            if not out.exists():
                m = re.search(r"frame_(\d+)\.jpg$", out.name)
                if m and int(m.group(1)) == 0:
                    alt = out.with_name("frame_000001.jpg")
                    if alt.exists():
                        out = alt
            fixed.append(str(out))
        item["video"] = fixed
    return msg


def _gold_recall_from_assistant(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for tc in msg.get("tool_calls") or []:
        fn = (tc.get("function") or {})
        if fn.get("name") == "recall":
            args = fn.get("arguments") or {}
            return {"start_time": args.get("start_time"), "end_time": args.get("end_time")}
    content = str(msg.get("content") or "")
    parsed = parse_agent_output(content)
    tc = parsed.get("tool_call") or {}
    if tc.get("name") == "recall":
        args = tc.get("arguments") or {}
        return {"start_time": args.get("start_time"), "end_time": args.get("end_time")}
    return None


def _parse_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, dict):
        try:
            a, b = float(value.get("start_time")), float(value.get("end_time"))
            return (a, b) if b >= a else None
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            a, b = float(value[0]), float(value[1])
            return (min(a, b), max(a, b))
        except (TypeError, ValueError):
            return None
    s = str(value or "")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        return (min(a, b), max(a, b))
    if len(nums) == 1:
        a = float(nums[0])
        return (a, a)
    return None


def _range_iou(pred: Optional[Tuple[float, float]], gold: Optional[Tuple[float, float]]) -> float:
    if pred is None or gold is None:
        return 0.0
    ps, pe = pred
    gs, ge = gold
    inter = max(0.0, min(pe, ge) - max(ps, gs) + 1.0)
    union = max(pe, ge) - min(ps, gs) + 1.0
    return inter / union if union > 0 else 0.0


def _covers(pred: Optional[Tuple[float, float]], gold: Optional[Tuple[float, float]]) -> bool:
    if pred is None or gold is None:
        return False
    ps, pe = pred
    gs, ge = gold
    return ps <= gs and pe >= ge


def _overlaps(pred: Optional[Tuple[float, float]], gold: Optional[Tuple[float, float]]) -> bool:
    if pred is None or gold is None:
        return False
    ps, pe = pred
    gs, ge = gold
    return min(pe, ge) >= max(ps, gs)


def _tail_prompt_messages(messages: List[Dict[str, Any]], end: int, tail_messages: int) -> List[Dict[str, Any]]:
    prompt = list(messages[:end])
    if tail_messages <= 0 or len(prompt) <= tail_messages:
        return prompt
    keep = prompt[-tail_messages:]
    if prompt and prompt[0].get("role") == "system" and keep[0].get("role") != "system":
        keep.insert(0, prompt[0])
    # Chat templates expect a valid role sequence. If the tail starts from an
    # assistant/tool message, advance to the first user message in the tail.
    while keep and keep[0].get("role") not in {"system", "user"}:
        keep.pop(0)
    if keep and keep[0].get("role") == "system":
        while len(keep) > 1 and keep[1].get("role") != "user":
            keep.pop(1)
    return keep


def load_own_cases(path: Path, frames_root: Path, *, limit: int, skip: int, tail_messages: int = 0) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    matched = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            row = json.loads(line)
            messages = row.get("messages") or []
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                gold = _gold_recall_from_assistant(msg)
                if not gold:
                    continue
                if i == 0 or messages[i - 1].get("role") != "user":
                    continue
                if matched < skip:
                    matched += 1
                    continue
                video_id = str(row.get("video_id") or "")
                raw_prompt = _tail_prompt_messages(messages, i, tail_messages)
                prompt_messages = [
                    _resolve_msg(m, frames_root=frames_root, video_id=video_id)
                    for m in raw_prompt
                ]
                cases.append({
                    "source": "own",
                    "line_no": line_no,
                    "trajectory_idx": row.get("trajectory_idx"),
                    "trajectory_type": row.get("trajectory_type"),
                    "video_id": video_id,
                    "prompt_messages": prompt_messages,
                    "tools": row.get("tools") or tools_for_turn("streaming"),
                    "gold_start_end": gold,
                    "task": "own_recall",
                    "tail_messages": tail_messages,
                })
                if len(cases) >= limit:
                    return cases
    return cases


def _query_record(q: Dict[str, Any], current_chunk: int) -> Dict[str, Any]:
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
        "ask_time": current_chunk,
        "open_until": current_chunk,
        "answers": [],
        "status": "open",
    }


def _ovo_gold_interval(q: Dict[str, Any], current: int) -> Optional[Tuple[int, int]]:
    intervals = q.get("ovo_support_intervals") or []
    candidates = []
    for it in intervals:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            s, e = int(it[0]), int(it[1])
            if e < current:
                candidates.append((s, e))
    if candidates:
        return max(candidates, key=lambda x: x[1])
    chunks = sorted(int(x) for x in (q.get("support_chunks") or []) if int(x) < current)
    if chunks:
        return (chunks[0], chunks[-1])
    return None


def load_ovo_cases(path: Path, frames_root: Path, *, limit: int, skip: int, offset: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    matched = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            row = json.loads(line)
            video_path = row.get("source_video_path") or row.get("video_path") or ""
            for q_raw in row.get("questions") or []:
                q = dict(q_raw)
                support_all = sorted(int(x) for x in (q.get("support_chunks") or []))
                if not support_all:
                    continue
                base = max(support_all)
                current = base + int(offset)
                if not _chunk_exists(frames_root, video_path, current):
                    # fallback to a nearby chunk after the last historical interval.
                    found = None
                    for c in range(base + 1, base + 16):
                        if _chunk_exists(frames_root, video_path, c):
                            found = c
                            break
                    if found is None:
                        continue
                    current = found
                gold = _ovo_gold_interval(q, current)
                if gold is None:
                    continue
                if matched < skip:
                    matched += 1
                    continue
                frame_paths = _chunk_frame_paths(frames_root, video_path, current)
                memory_text = format_memory_block({
                    "compressed_segments": [{
                        "time_range": [max(0, current - 20), current - 1],
                        "text": "Historical video exists before the current chunk, but exact visual evidence may require recall.",
                        "source_chunks": list(range(max(0, current - 20), current)),
                    }],
                    "recent_thinks": [],
                })
                user_content = build_user_content(
                    memory_text,
                    current,
                    "",
                    user_input=q.get("question", ""),
                    queries=[_query_record(q, current)],
                    frame_paths=frame_paths,
                    frame_protocol="video_meta",
                    memory_snapshot=None,
                )
                cases.append({
                    "source": "ovo",
                    "line_no": line_no,
                    "trajectory_id": row.get("trajectory_id"),
                    "video_path": video_path,
                    "task": str(q.get("ovo_task") or q.get("family") or ""),
                    "question": q.get("question", ""),
                    "current_chunk": current,
                    "gold_start_end": {"start_time": gold[0], "end_time": gold[1]},
                    "gold_support_interval": list(gold),
                    "prompt_messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "tools": tools_for_turn("streaming"),
                })
                if len(cases) >= limit:
                    return cases
    return cases


def _prepare_inputs(processor: Any, messages: List[Dict[str, Any]], tools: Any, device: str) -> Tuple[Dict[str, Any], int]:
    text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=getattr(processor.image_processor, "patch_size", 16),
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos, video_metadata = normalize_video_inputs(video_inputs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        do_resize=False,
        **(video_kwargs or {}),
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs, int(inputs["input_ids"].shape[1])


@torch.inference_mode()
def run_case(model: Any, processor: Any, case: Dict[str, Any], args: argparse.Namespace, device: str) -> Dict[str, Any]:
    inputs, prompt_len = _prepare_inputs(processor, case["prompt_messages"], case.get("tools"), device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    text = processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False).strip()
    parsed = parse_agent_output(text)
    kind = parsed.get("kind")
    args_pred = ((parsed.get("tool_call") or {}).get("arguments") or {})
    pred_range_raw = {
        "start_time": args_pred.get("start_time"),
        "end_time": args_pred.get("end_time"),
    }
    pred_range = _parse_range(pred_range_raw)
    gold_range = _parse_range(case.get("gold_start_end"))
    return {
        **{k: v for k, v in case.items() if k not in {"prompt_messages", "tools"}},
        "output": text,
        "kind": kind,
        "format_error": parsed.get("format_error"),
        "think": parsed.get("think", ""),
        "pred_start_end": pred_range_raw,
        "pred_range_parsed": list(pred_range) if pred_range else None,
        "gold_range_parsed": list(gold_range) if gold_range else None,
        "is_recall": kind == "recall",
        "range_parse_ok": pred_range is not None,
        "range_iou": round(_range_iou(pred_range, gold_range), 4),
        "range_overlaps_gold": _overlaps(pred_range, gold_range),
        "range_covers_gold": _covers(pred_range, gold_range),
        "prompt_tokens": prompt_len,
        "new_tokens": int(out.shape[1] - prompt_len),
        "latency_sec": round(time.time() - t0, 3),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    by_task: Dict[str, Dict[str, Any]] = {}
    for key, rows in _group_by(results, "task").items():
        by_task[key] = _summary_core(rows)
    return {**_summary_core(results), "by_task": by_task}


def _summary_core(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "recall_rate": sum(r.get("is_recall") for r in rows) / max(1, n),
        "range_parse_rate": sum(r.get("range_parse_ok") for r in rows) / max(1, n),
        "overlap_rate": sum(r.get("range_overlaps_gold") for r in rows) / max(1, n),
        "cover_rate": sum(r.get("range_covers_gold") for r in rows) / max(1, n),
        "mean_iou": sum(float(r.get("range_iou") or 0.0) for r in rows) / max(1, n),
        "format_error_rate": sum(bool(r.get("format_error")) for r in rows) / max(1, n),
    }


def _group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key) or "")].append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["own", "ovo"], required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="qwen3vl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--ovo-current-offset", type=int, default=4)
    ap.add_argument("--tail-messages", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    if args.source == "own":
        cases = load_own_cases(
            Path(args.jsonl),
            frames_root,
            limit=args.limit,
            skip=args.skip,
            tail_messages=args.tail_messages,
        )
    else:
        cases = load_ovo_cases(
            Path(args.jsonl),
            frames_root,
            limit=args.limit,
            skip=args.skip,
            offset=args.ovo_current_offset,
        )
    if not cases:
        raise RuntimeError("no recall range cases found")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model_cls = MODEL_CLS[args.model_type]
    model = model_cls.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()

    results = []
    for case in cases:
        results.append(run_case(model, processor, case, args, args.device))
        print(json.dumps({
            "source": args.source,
            "idx": len(results),
            "task": results[-1].get("task"),
            "kind": results[-1].get("kind"),
            "gold": results[-1].get("gold_start_end"),
            "pred": results[-1].get("pred_start_end"),
            "iou": results[-1].get("range_iou"),
            "overlap": results[-1].get("range_overlaps_gold"),
        }, ensure_ascii=False), flush=True)

    payload = {
        "summary": summarize(results),
        "source": args.source,
        "jsonl": args.jsonl,
        "frames_root": args.frames_root,
        "model": args.model,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
