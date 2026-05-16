#!/usr/bin/env python3
"""Probe compact-memory refill under the production true-KV pattern.

The simulated sequence mirrors the current RL streaming backend:

1. ordinary visual turns append only the new user block + current video chunk;
2. compact-memory update is an isolated text-only turn;
3. after compact update the active stream is reset;
4. the next ordinary visual turn refills from system + compact memory +
   current visual chunk, then subsequent visual turns append deltas again.

For each post-refill visual turn, the script also generates a current-only
baseline from a fresh stream.  This lets us check whether the SFT model's
post-refill <think> tracks the latest visual evidence or copies old memory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.local_hf_kv_visual_probe import (  # noqa: E402
    PIXEL_PROFILES,
    keyword_recall,
    keywords,
    load_model_for_stream,
    make_engine,
    normalize_video_inputs,
)
from scripts.audit.probe_recall_visual_layout import (  # noqa: E402
    _chunk_exists,
    _chunk_frame_paths,
)
from thinkstream.data.agent_protocol import (  # noqa: E402
    AGENT_CHUNK_SEC,
    COMPACT_MEMORY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_content,
    format_memory_block,
    parse_agent_output,
    tools_for_turn,
)
from thinkstream.data.stream_data_processor import compute_position_ids  # noqa: E402
from thinkstream.models.agent_loop import MemoryState, _parse_mem_entries  # noqa: E402


_XML_ESCAPE = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
}


def _xml_text(text: str) -> str:
    out = str(text or "")
    for src, dst in _XML_ESCAPE.items():
        out = out.replace(src, dst)
    return out


def _load_cases(
    path: Path,
    frames_root: Path,
    limit: int,
    min_chunks: int,
    *,
    skip: int = 0,
    unique_video: bool = False,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen_videos: set[str] = set()
    matched = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            video_path = row.get("source_video_path") or row.get("video_path") or ""
            start = int(row.get("segment_start_chunk") or 0)
            end = int(row.get("segment_end_chunk") or start)
            # Extend beyond the query span when frames exist, so we can force
            # multiple compression/refill rounds on short OVO rows.
            while _chunk_exists(frames_root, video_path, end + 1) and end - start + 1 < min_chunks:
                end += 1
            if end - start + 1 < min_chunks:
                continue
            needed = list(range(start, start + min_chunks))
            if not all(_chunk_exists(frames_root, video_path, c) for c in needed):
                continue
            if unique_video and video_path in seen_videos:
                continue
            if matched < skip:
                matched += 1
                if unique_video:
                    seen_videos.add(video_path)
                continue
            seen_videos.add(video_path)
            cases.append({
                "trajectory_id": row.get("trajectory_id"),
                "video_path": video_path,
                "start_chunk": start,
                "end_chunk": start + min_chunks - 1,
                "task": ",".join(sorted({str(q.get("ovo_task") or q.get("family") or "") for q in row.get("questions") or []})),
            })
            if len(cases) >= limit:
                break
    if not cases:
        raise RuntimeError(f"no cases with {min_chunks} contiguous chunks found in {path}")
    return cases


def _memory_update_input(state: MemoryState) -> Tuple[str, Tuple[int, int]]:
    old_lines: List[str] = []
    for seg in state.compressed_segments:
        tr = seg.get("time_range") or []
        if isinstance(tr, list) and len(tr) >= 2:
            t = f"{int(tr[0])}-{int(tr[1])}"
        else:
            t = str(seg.get("time") or "")
        text = str(seg.get("text") or "").strip()
        if text:
            old_lines.append(f'  <m t="{t}">{_xml_text(text)}</m>')
    old_memory = "\n".join(old_lines) if old_lines else "(empty)"

    chunks: List[int] = []
    cap_lines = ["<NEW_CAPTIONS>"]
    for item in state.recent_thinks:
        try:
            chunk = int(item.get("chunk"))
        except (TypeError, ValueError):
            continue
        chunks.append(chunk)
        cap_lines.append(f'  <c t="{chunk}">{_xml_text(item.get("text", ""))}</c>')
    cap_lines.append("</NEW_CAPTIONS>")
    start = min(chunks) if chunks else 0
    end = max(chunks) if chunks else start
    prompt = (
        f"OLD_MEMORY:\n{old_memory}\n\n"
        f"NEW_CAPTIONS:\n{chr(10).join(cap_lines)}\n\n"
        f"Covered latest span: t={start}-{end}\n"
        "Coverage check:\n"
        "- If OLD_MEMORY has <m> lines, preserve useful old information in at least one output line.\n"
        "- If NEW_CAPTIONS has <c> lines, cover the latest new caption timestamps in at least one output line.\n\n"
        "Return only XML lines:\n"
        '  <m t="start-end">one concise event or state.</m>\n\n'
        "Replace the placeholder line with 4-6 chronological <m> lines using real input timestamps.\n"
        "Do not output NEW_MEMORY, markdown, prose, analysis, or any text outside the <m> lines."
    )
    return prompt, (start, end)


def _extract_think(text: str) -> str:
    parsed = parse_agent_output(str(text or ""), allow_bare_memory=True)
    think = str(parsed.get("think") or "").strip()
    if think:
        return think
    match = re.search(r"<think>(.*?)(?:</think>|<\|im_end\|>|$)", str(text or ""), flags=re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _compact_lines(text: str) -> str:
    parsed = parse_agent_output(str(text or ""), allow_bare_memory=True)
    return str(parsed.get("memory_text") or "").strip()


def _state_memory_text(state: MemoryState) -> str:
    return format_memory_block({
        "compressed_segments": state.compressed_segments,
        "compressed": state.compressed_segments,
        "recent_thinks": state.recent_thinks,
    })


def _messages_for_stream_turn(
    *,
    state: MemoryState,
    chunk: int,
    frame_paths: List[str],
    first_or_refill: bool,
    min_pixels: int,
    max_pixels: int,
) -> List[Dict[str, Any]]:
    content = build_user_content(
        _state_memory_text(state) if first_or_refill else "",
        chunk,
        "",
        user_input="",
        queries=[],
        frame_paths=frame_paths,
        frame_protocol="video_meta",
        inter_chunk=False,
        memory_snapshot=None,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    messages = [{"role": "user", "content": content}]
    if first_or_refill:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def _messages_for_compress(prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": COMPACT_MEMORY_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def _encode_messages(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]],
    model_type: str,
) -> Dict[str, Any]:
    text = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    _, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos, video_metadata = normalize_video_inputs(video_inputs)
    proc_kwargs = {
        "text": [text],
        "videos": videos,
        "return_tensors": "pt",
        **(video_kwargs or {}),
    }
    if video_metadata is not None:
        proc_kwargs["video_metadata"] = video_metadata
    encoded = processor(**proc_kwargs)
    rope_inputs = dict(encoded)
    rope_inputs["video_chunk_size"] = AGENT_CHUNK_SEC
    encoded["position_ids"] = compute_position_ids(rope_inputs, processor, model_type)
    return encoded


def _to_device(encoded: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "position_ids", "pixel_values_videos", "video_grid_thw"):
        value = encoded.get(key)
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


@torch.inference_mode()
def _generate(
    engine: Any,
    processor: Any,
    encoded: Dict[str, Any],
    *,
    device: str,
    max_new_tokens: int,
    top_k: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    keep = _to_device(encoded, device)
    t0 = time.time()
    generated = engine.generate(
        **keep,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
    )[0]
    text = processor.tokenizer.decode(generated.tolist(), skip_special_tokens=False).strip()
    wc = int(engine._window_count[0].item()) if hasattr(engine, "_window_count") else 0
    diag = {
        "latency_sec": round(time.time() - t0, 3),
        "generated_tokens": int(generated.numel()),
        "cache_len": int(engine.decoder.cache.cache_seqlens[0, 0].item()),
        "window_count": wc,
        "window_starts": engine._window_starts[0, :wc].detach().cpu().tolist() if wc else [],
        "window_ends": engine._window_ends[0, :wc].detach().cpu().tolist() if wc else [],
    }
    return text, diag


def _baseline_current_only(
    *,
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    chunk: int,
    frame_paths: List[str],
    min_pixels: int,
    max_pixels: int,
    device: str,
) -> str:
    base_engine = make_engine(args, model, processor, device)
    base_engine.reset()
    empty_state = MemoryState(processor.tokenizer)
    messages = _messages_for_stream_turn(
        state=empty_state,
        chunk=chunk,
        frame_paths=frame_paths,
        first_or_refill=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    encoded = _encode_messages(processor, messages, tools=tools_for_turn("streaming"), model_type=args.model_type)
    text, _diag = _generate(
        base_engine,
        processor,
        encoded,
        device=device,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )
    return _extract_think(text) or text


def _copy_risk(think: str, memory_text: str, baseline_think: str) -> Dict[str, Any]:
    cur = keyword_recall(think, baseline_think)
    mem = keyword_recall(think, memory_text)
    think_norm = " ".join(str(think or "").lower().split())
    mem_norm = " ".join(str(memory_text or "").lower().split())
    return {
        "current_baseline_keyword_recall": round(cur, 4),
        "memory_keyword_recall": round(mem, 4),
        "memory_copy_risk": bool(mem > cur + 0.10 or (think_norm and think_norm in mem_norm)),
        "think_keywords": keywords(think, limit=20),
        "baseline_keywords": keywords(baseline_think, limit=20),
    }


def run_case(
    *,
    args: argparse.Namespace,
    processor: Any,
    model: Any,
    case: Dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    device: str,
) -> Dict[str, Any]:
    engine = make_engine(args, model, processor, device)
    state = MemoryState(processor.tokenizer)
    engine.reset()
    need_refill = True
    events: List[Dict[str, Any]] = []
    post_refill_rows: List[Dict[str, Any]] = []
    chunks_since_compress = 0

    for chunk in range(int(case["start_chunk"]), int(case["end_chunk"]) + 1):
        if chunks_since_compress >= args.compress_every and state.recent_thinks:
            prompt, span = _memory_update_input(state)
            comp_messages = _messages_for_compress(prompt)
            comp_encoded = _encode_messages(processor, comp_messages, tools=None, model_type=args.model_type)
            engine.reset()
            comp_text, comp_diag = _generate(
                engine,
                processor,
                comp_encoded,
                device=device,
                max_new_tokens=args.compress_max_new_tokens,
                top_k=args.top_k,
            )
            mem_text = _compact_lines(comp_text)
            entries = _parse_mem_entries(mem_text)
            parse_ok = bool(entries)
            if parse_ok:
                state.replace_with_compact_memory(entries)
            engine.reset()
            need_refill = True
            chunks_since_compress = 0
            events.append({
                "event": "compress",
                "before_chunk": chunk,
                "span": list(span),
                "raw_output": comp_text,
                "memory_text": mem_text,
                "parse_ok": parse_ok,
                "n_entries": len(entries),
                **comp_diag,
            })

        frame_paths = _chunk_frame_paths(Path(args.frames_root), case["video_path"], chunk)
        messages = _messages_for_stream_turn(
            state=state,
            chunk=chunk,
            frame_paths=frame_paths,
            first_or_refill=need_refill,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        encoded = _encode_messages(processor, messages, tools=tools_for_turn("streaming") if need_refill else None, model_type=args.model_type)
        if need_refill:
            engine.reset()
        text, diag = _generate(
            engine,
            processor,
            encoded,
            device=device,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
        )
        parsed = parse_agent_output(text)
        think = _extract_think(text)
        state.add_think(chunk, think)
        row = {
            "event": "stream",
            "chunk": chunk,
            "refill": bool(need_refill),
            "kind": parsed.get("kind"),
            "format_error": parsed.get("format_error"),
            "think": think,
            "raw_output": text,
            "memory_before_visible": _state_memory_text(state),
            **diag,
        }
        if need_refill and events:
            baseline = _baseline_current_only(
                args=args,
                processor=processor,
                model=model,
                chunk=chunk,
                frame_paths=frame_paths,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                device=device,
            )
            row["current_only_baseline_think"] = baseline
            row.update(_copy_risk(think, _state_memory_text(state), baseline))
            post_refill_rows.append(row)
        events.append(row)
        need_refill = False
        chunks_since_compress += 1

    return {
        **case,
        "events": events,
        "post_refill_rows": post_refill_rows,
        "summary": {
            "compress_events": sum(1 for e in events if e["event"] == "compress"),
            "compress_parse_ok": sum(1 for e in events if e["event"] == "compress" and e.get("parse_ok")),
            "post_refill_turns": len(post_refill_rows),
            "post_refill_copy_risk": sum(1 for e in post_refill_rows if e.get("memory_copy_risk")),
            "avg_post_refill_current_recall": (
                sum(float(e.get("current_baseline_keyword_recall") or 0.0) for e in post_refill_rows)
                / max(1, len(post_refill_rows))
            ),
            "avg_post_refill_memory_recall": (
                sum(float(e.get("memory_keyword_recall") or 0.0) for e in post_refill_rows)
                / max(1, len(post_refill_rows))
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-type", default="qwen3vl")
    ap.add_argument("--pixel-profile", default="runtime", choices=sorted(PIXEL_PROFILES))
    ap.add_argument("--max-len", type=int, default=49152)
    ap.add_argument("--kv-window", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit-cases", type=int, default=2)
    ap.add_argument("--skip-cases", type=int, default=0)
    ap.add_argument("--unique-video", action="store_true")
    ap.add_argument("--chunks-per-case", type=int, default=12)
    ap.add_argument("--compress-every", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--compress-max-new-tokens", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    min_pixels, max_pixels = PIXEL_PROFILES[args.pixel_profile]
    model, processor = load_model_for_stream(args, device)
    processor.tokenizer.padding_side = "right"

    cases = _load_cases(
        Path(args.trajectories),
        Path(args.frames_root),
        limit=args.limit_cases,
        min_chunks=args.chunks_per_case,
        skip=args.skip_cases,
        unique_video=bool(args.unique_video),
    )
    results = []
    for case in cases:
        print(json.dumps({"event": "case_start", **case}, ensure_ascii=False), flush=True)
        results.append(run_case(
            args=args,
            processor=processor,
            model=model,
            case=case,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            device=device,
        ))
        print(json.dumps({"event": "case_done", **results[-1]["summary"]}, ensure_ascii=False), flush=True)

    flat_post = [r for case in results for r in case["post_refill_rows"]]
    flat_comp = [e for case in results for e in case["events"] if e["event"] == "compress"]
    summary = {
        "mode": "sft_compress_refill_true_kv_probe",
        "model": args.model,
        "trajectories": args.trajectories,
        "frames_root": args.frames_root,
        "pixel_profile": args.pixel_profile,
        "kv_window": args.kv_window,
        "chunks_per_case": args.chunks_per_case,
        "compress_every": args.compress_every,
        "cases": len(results),
        "compress_events": len(flat_comp),
        "compress_parse_ok": sum(1 for e in flat_comp if e.get("parse_ok")),
        "post_refill_turns": len(flat_post),
        "post_refill_copy_risk": sum(1 for e in flat_post if e.get("memory_copy_risk")),
        "avg_post_refill_current_recall": (
            sum(float(e.get("current_baseline_keyword_recall") or 0.0) for e in flat_post)
            / max(1, len(flat_post))
        ),
        "avg_post_refill_memory_recall": (
            sum(float(e.get("memory_keyword_recall") or 0.0) for e in flat_post)
            / max(1, len(flat_post))
        ),
    }
    payload = {"summary": summary, "cases_detail": results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
