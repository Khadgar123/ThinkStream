#!/usr/bin/env python3
"""Real-frame caption probes for no-response streaming and prefilled memory.

Two modes are supported:
1. stream: run a no-<response> streaming caption test over a time range.
2. prefill_continue: put selected earlier generated captions plus optional visual
   memory into one initialization user message, then continue streaming later chunks.
3. prefill_history_qa: ask one question about initialized historical memory.
4. prefill_time_range_qa: ask where a historical event happened and require a
   time_range answer.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "also", "with", "without",
    "there", "their", "these", "those", "this", "that", "from", "into", "onto", "over",
    "under", "while", "where", "which", "being", "been", "have", "has", "had", "does",
    "display", "displays", "show", "shows", "shown", "scene", "video", "frame", "frames",
    "current", "visible", "appears", "appearing", "using", "wearing", "holding", "object",
    "person", "people", "image", "close", "view", "background", "foreground", "left",
    "right", "center", "central", "likely", "text", "visible", "caption", "chunk",
}


def data_uri(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def jpeg_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def video_jpeg_data_uri(paths: List[Path]) -> str:
    return "data:video/jpeg;base64," + ",".join(jpeg_b64(path) for path in paths)


def video_mp4_data_uri(path: Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def post_json(url: str, payload: Dict, timeout: int) -> Dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_server(base_url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(3)
    raise RuntimeError(f"server not ready after {timeout_sec}s: {last_err}")


def text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if item.get("type") == "text")
    return ""


def assistant_think(row: Dict) -> str:
    text = text_from_content(row["messages"][-1].get("content", ""))
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def visual_window_and_video(row: Dict) -> tuple[Dict, List[str]]:
    user_content = row["messages"][1]["content"]
    text = text_from_content(user_content)
    match = re.search(r"<visual_window>\s*(\{.*?\})\s*</visual_window>", text, flags=re.DOTALL)
    visual_window = json.loads(match.group(1)) if match else {}
    video = []
    for item in user_content:
        if item.get("type") == "video":
            video = item.get("video") or []
            break
    return visual_window, video


def load_rows(path: Path, video_id: str, start_time: int, end_time: int) -> List[Dict]:
    by_time: Dict[int, Dict] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if row.get("video_id") != video_id:
                continue
            if row.get("sample_type") not in {"silent", "response"}:
                continue
            visual_window, video = visual_window_and_video(row)
            current_time = visual_window.get("current_time")
            if current_time is None or not video:
                continue
            if not (start_time <= current_time <= end_time):
                continue
            think = assistant_think(row)
            if not think or current_time in by_time:
                continue
            by_time[current_time] = {
                "line_no": line_no,
                "video_id": video_id,
                "current_time": current_time,
                "video": video,
                "gold_caption": think,
                "sample_type": row.get("sample_type"),
                "chunk_idx": row.get("chunk_idx"),
            }
    missing = [t for t in range(start_time, end_time + 1) if t not in by_time]
    if missing:
        raise RuntimeError(f"missing current_time values: {missing[:30]}")
    return [by_time[t] for t in sorted(by_time)]


def keywords(text: str, limit: int = 50) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", text.lower())
    seen = set()
    result = []
    for word in words:
        word = word.strip("'")
        if word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        result.append(word)
        if len(result) >= limit:
            break
    return result


def keyword_recall(prediction: str, reference: str) -> float:
    ref = keywords(reference)
    if not ref:
        return 0.0
    pred = set(keywords(prediction, limit=140))
    return len([w for w in ref if w in pred]) / len(ref)


def extract_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def build_memory(generated: List[Dict], lookback: int) -> str:
    if lookback <= 0:
        return "\n".join([
            "<memory>",
            '  <m t="none">No historical memory is available.</m>',
            "</memory>",
        ])
    lines = ["<memory>"]
    if not generated:
        lines.append('  <m t="0-32">Compressed summary: no previous generated caption is available.</m>')
        lines.append('  <m t="33">Past think / observation text: none.</m>')
        lines.append('  <m t="34">Past think / observation text: none.</m>')
    else:
        older = generated[:-lookback] if len(generated) > lookback else []
        if older:
            bits = []
            for item in older[-5:]:
                bits.append(f'{item["current_time"]}s: {", ".join(keywords(item["pred_caption"], 8))}')
            lines.append(f'  <m t="{older[0]["current_time"]}-{older[-1]["current_time"]}">Compressed summary: {"; ".join(bits)}</m>')
        else:
            lines.append('  <m t="0-32">Compressed summary: only short recent history is available.</m>')
        for item in generated[-lookback:]:
            caption = item["pred_caption"].replace("\n", " ")[:700]
            lines.append(f'  <m t="{item["current_time"]}">Past think / observation text: {caption}</m>')
    lines.append("</memory>")
    return "\n".join(lines)


def select_turn_frames(row: Dict, frames_per_turn: int) -> List[str]:
    frames = list(row["video"][-max(1, frames_per_turn):])
    if not frames:
        raise ValueError(f"row at t={row.get('current_time')} has no frames")
    return frames


def make_mp4_chunk(row: Dict, frames_per_turn: int, source_fps: float = 2.0) -> Path:
    frames = [Path(p) for p in select_turn_frames(row, frames_per_turn)]
    if len(frames) == 1:
        # Keep the visual content at fps=1 while satisfying video processors
        # that require at least two temporal frames.
        frames = [frames[0], frames[0]]
    video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("video_id", "video")))[:120]
    t = int(row["current_time"])
    cache_root = REPO_ROOT / "output" / "vllm_stream_visual_matrix" / "mp4_chunk_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{video_id}_t{t}_n{len(frames)}_fps{source_fps:g}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    seq_dir = cache_root / f"{out_path.stem}_frames"
    seq_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        link = seq_dir / f"frame_{idx:06d}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            os.symlink(frame.resolve(), link)
        except OSError:
            link.write_bytes(frame.read_bytes())
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(source_fps),
        "-i",
        str(seq_dir / "frame_%06d.jpg"),
        "-frames:v",
        str(len(frames)),
        "-r",
        str(source_fps),
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def add_vision(
    content: List[Dict],
    row: Dict,
    frames_per_turn: int,
    label: str,
    visual_format: str = "image",
) -> None:
    frames = select_turn_frames(row, frames_per_turn)
    t = row["current_time"]
    if visual_format == "image":
        for frame in frames:
            content.append({"type": "text", "text": f"  <t={t}><videochunk>{label}:"})
            content.append({"type": "image_url", "image_url": {"url": data_uri(Path(frame))}})
            content.append({"type": "text", "text": "  </videochunk></t>"})
        return

    if visual_format == "video":
        video_frames = [Path(frame) for frame in frames]
        if len(video_frames) == 1:
            # Qwen3-VL video processor requires at least temporal_factor=2 frames.
            video_frames = [video_frames[0], video_frames[0]]
        content.append({"type": "text", "text": f"  <t={t}><videochunk>{label}:"})
        content.append({"type": "video_url", "video_url": {"url": video_jpeg_data_uri(video_frames)}})
        content.append({"type": "text", "text": "  </videochunk></t>"})
        return

    if visual_format == "mp4":
        video_path = make_mp4_chunk(row, frames_per_turn)
        content.append({"type": "text", "text": f"  <t={t}><videourl_mp4>{label}:"})
        content.append({"type": "video_url", "video_url": {"url": video_mp4_data_uri(video_path)}})
        content.append({"type": "text", "text": "  </videourl_mp4></t>"})
        return

    raise ValueError(f"unknown visual_format: {visual_format}")


def build_stream_user(
    row: Dict,
    generated: List[Dict],
    frames_per_turn: int,
    lookback: int,
    visual_format: str,
) -> List[Dict]:
    content: List[Dict] = [{"type": "text", "text": build_memory(generated, lookback) + "\n\n<current_vision>"}]
    add_vision(
        content,
        row,
        frames_per_turn,
        "CURRENT video chunk; this is the only visual content to caption",
        visual_format,
    )
    content.append({"type": "text", "text": "\n".join([
        "</current_vision>",
        "<active_query>",
        (
            f'  <q t="{row["current_time"]}">Write one concise English natural-language caption / observation '
            "for only the newest current video chunk inside <current_vision>. Describe visible people, objects, "
            "actions, text, and scene state from <current_vision> only. Historical <memory>, "
            "<vision_memory>, or <recent_vision_memory> may be used only for temporal continuity, not as "
            "the visual source for this caption. Do not answer the original QA. Do not output fixed fields. "
            "Do not copy <memory>, historical visual memory, or earlier assistant captions. Output exactly: "
            "<think>English caption of the current chunk</think><answer></answer></q>"
        ),
        "</active_query>",
    ])})
    return content


TIME_RANGE_QUERIES = {
    "flour_measuring": {
        "question": "When did a copper measuring cup or measuring cup scoop or pour white flour near a glass jar?",
        "expected_range": [50, 51],
        "keywords": ["measuring", "cup", "flour", "glass", "jar"],
    },
    "electric_mixer": {
        "question": "When was a black electric mixer used to blend ingredients in a glass bowl?",
        "expected_range": [2, 3],
        "keywords": ["electric", "mixer", "blend", "glass", "bowl"],
    },
    "banana_fork": {
        "question": "When did hands use a fork on a dark banana or chocolate-covered banana on a baking sheet?",
        "expected_range": [1, 2],
        "keywords": ["fork", "banana", "baking", "sheet"],
    },
    "holding_bananas": {
        "question": "When did the man hold a bunch of yellow bananas in the kitchen?",
        "expected_range": [9, 19],
        "keywords": ["bunch", "yellow", "bananas", "kitchen"],
    },
    "sliced_bread_plate": {
        "question": "When were slices of banana bread shown on a decorative or floral plate?",
        "expected_range": [7, 8],
        "keywords": ["slices", "banana", "bread", "plate"],
    },
}


def parse_time_range(text: str) -> List[int] | None:
    json_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            value = str(data.get("time_range", ""))
            nums = [int(x) for x in re.findall(r"\d+", value)]
            if len(nums) >= 2:
                return [min(nums[0], nums[1]), max(nums[0], nums[1])]
            if len(nums) == 1:
                return [nums[0], nums[0]]
        except Exception:
            pass
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if len(nums) >= 2:
        return [min(nums[0], nums[1]), max(nums[0], nums[1])]
    if len(nums) == 1:
        return [nums[0], nums[0]]
    return None


def range_overlap(predicted: List[int] | None, expected: List[int]) -> Dict:
    if predicted is None:
        return {"overlap": 0, "iou": 0.0, "hit": False}
    ps, pe = predicted
    es, ee = expected
    overlap = max(0, min(pe, ee) - max(ps, es) + 1)
    union = max(pe, ee) - min(ps, es) + 1
    return {
        "overlap": overlap,
        "iou": overlap / union if union else 0.0,
        "hit": overlap > 0,
    }


def select_items(items: List[Dict], mode: str, count: int, score_words: List[str] | None = None) -> List[Dict]:
    if mode == "none" or count == 0:
        return []
    if mode == "all" or count < 0 or count >= len(items):
        return list(items)
    if mode == "query_top":
        if not score_words:
            raise ValueError("query_top selection requires score words")
        wanted = {word.lower() for word in score_words}
        scored = []
        for pos, item in enumerate(items):
            item_words = set(keywords(item.get("pred_caption", ""), limit=100))
            score = len(wanted & item_words)
            scored.append((score, pos, item))
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected = [item for score, _, item in scored if score > 0][:count]
        if len(selected) < count:
            selected_times = {item["current_time"] for item in selected}
            for _, _, item in scored:
                if item["current_time"] not in selected_times:
                    selected.append(item)
                    selected_times.add(item["current_time"])
                if len(selected) >= count:
                    break
        return sorted(selected, key=lambda item: item["current_time"])
    if mode == "recent":
        return list(items[-count:])
    if mode == "uniform":
        if count == 1:
            return [items[-1]]
        positions = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
        selected = []
        seen = set()
        for pos in positions:
            if pos not in seen:
                selected.append(items[pos])
                seen.add(pos)
        return selected
    if mode == "uniform_recent":
        recent_count = min(max(1, count // 2), len(items))
        uniform_count = max(0, count - recent_count)
        selected = select_items(items[:-recent_count], "uniform", uniform_count) if uniform_count else []
        selected.extend(items[-recent_count:])
        seen = set()
        deduped = []
        for item in selected:
            key = item["current_time"]
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped
    raise ValueError(f"unknown selection mode: {mode}")


def build_prefill_user(
    prefill_rows: List[Dict],
    prefill_results: List[Dict],
    frames_per_turn: int,
    text_mode: str,
    text_count: int,
    visual_mode: str,
    visual_count: int,
    summary_count: int,
    layout: str,
    summary_start: int | None = None,
    summary_end: int | None = None,
    summary_style: str = "keywords",
    score_words: List[str] | None = None,
    visual_format: str = "image",
) -> List[Dict]:
    content: List[Dict] = []
    text_items = select_items(prefill_results, text_mode, text_count, score_words)
    visual_items = select_items(prefill_results, visual_mode, visual_count, score_words)
    summary_source = [
        item for item in prefill_results
        if (summary_start is None or item["current_time"] >= summary_start)
        and (summary_end is None or item["current_time"] <= summary_end)
    ]
    summary_items = select_items(summary_source, "uniform", summary_count)
    memory_lines = ["<memory>"]
    if summary_style == "keywords":
        summary_bits = [
            f'{item["current_time"]}s: {", ".join(keywords(item["pred_caption"], 8))}'
            for item in summary_items
        ]
    elif summary_style == "snippets":
        summary_bits = [
            f't={item["current_time"]}s: {item["pred_caption"].replace(chr(10), " ")[:220]}'
            for item in summary_items
        ]
    else:
        raise ValueError(f"unknown summary style: {summary_style}")
    if summary_bits:
        summary_time = f'{summary_items[0]["current_time"]}-{summary_items[-1]["current_time"]}'
        memory_lines.append(f'  <m t="{summary_time}">Compressed summary: {"; ".join(summary_bits)}</m>')
    else:
        memory_lines.append('  <m t="none">Compressed summary: none.</m>')
    for item in text_items:
        caption = item["pred_caption"].replace("\n", " ")[:500]
        memory_lines.append(f'  <m t="{item["current_time"]}">Past think / observation text: {caption}</m>')
    memory_lines.append("</memory>")
    by_time = {row["current_time"]: row for row in prefill_rows}

    def append_memory_text() -> None:
        content.append({"type": "text", "text": "\n".join(memory_lines)})

    def append_visual_memory() -> None:
        content.append({"type": "text", "text": "<vision_memory>"})
        for item in visual_items:
            row = by_time[item["current_time"]]
            add_vision(
                content,
                row,
                frames_per_turn,
                "HISTORICAL visual memory only; do not caption as current",
                visual_format,
            )
        content.append({"type": "text", "text": "</vision_memory>"})

    if layout == "text_then_visual":
        append_memory_text()
        append_visual_memory()
    elif layout == "visual_then_text":
        append_visual_memory()
        append_memory_text()
    elif layout == "text_only":
        append_memory_text()
    elif layout == "visual_only":
        append_visual_memory()
    else:
        raise ValueError(f"unknown prefill layout: {layout}")

    content.append({"type": "text", "text": "\n".join([
        "<historical_query>",
        (
            "  <q>Historical initialization only: store this summary, selected text memory, "
            "and selected visual memory for context. Future turns still require captioning only the "
            "newest <current_vision>; do not replay historical captions unless explicitly asked."
            "</q>"
        ),
        "</historical_query>",
    ])})
    return content


def call_model(args: argparse.Namespace, messages: List[Dict]) -> Dict:
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "mm_processor_kwargs": {
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    }
    if args.visual_format == "video":
        frame_indices = [0, 0] if args.frames_per_turn <= 1 else list(range(args.frames_per_turn))
        payload["media_io_kwargs"] = {
            "video": {
                "fps": float(args.visual_fps),
                "frames_indices": frame_indices,
                "total_num_frames": max(len(frame_indices), frame_indices[-1] + 1),
                "do_sample_frames": False,
            }
        }
    elif args.visual_format == "mp4":
        payload["media_io_kwargs"] = {
            "video": {
                "fps": float(args.visual_fps),
                "max_duration": 5,
            }
        }
    return post_json(f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout)


def result_row(row: Dict, raw: str, previous_gold: str, prompt_messages: int, usage: Dict) -> Dict:
    pred = extract_think(raw)
    current = keyword_recall(pred, row["gold_caption"])
    previous = keyword_recall(pred, previous_gold) if previous_gold else 0.0
    return {
        "current_time": row["current_time"],
        "sample_type": row["sample_type"],
        "gold_caption": row["gold_caption"],
        "pred_caption": pred,
        "raw": raw,
        "current_keyword_recall": current,
        "previous_keyword_recall": previous,
        "old_copy_risk": bool(previous_gold and previous > current + 0.10),
        "prompt_messages": prompt_messages,
        "usage": usage,
    }


def summarize(args: argparse.Namespace, out_rows: List[Dict], extra: Dict) -> Dict:
    captions = [r["pred_caption"] for r in out_rows]
    unique_pred = len(set(captions))
    top_repeat_count = max((captions.count(caption) for caption in set(captions)), default=0)
    summary = {
        "mode": args.mode,
        "model": args.model,
        "jsonl": args.jsonl,
        "video_id": args.video_id,
        "frames_per_turn": args.frames_per_turn,
        "visual_format": args.visual_format,
        "visual_fps": args.visual_fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "turns": len(out_rows),
        "avg_current_keyword_recall": sum(r["current_keyword_recall"] for r in out_rows) / len(out_rows),
        "avg_previous_keyword_recall": sum(r["previous_keyword_recall"] for r in out_rows) / len(out_rows),
        "old_copy_risk_count": sum(1 for r in out_rows if r["old_copy_risk"]),
        "unique_pred_count": unique_pred,
        "top_repeat_count": top_repeat_count,
        **extra,
        "results": out_rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_stream(args: argparse.Namespace) -> Dict:
    rows = load_rows(Path(args.jsonl), args.video_id, args.start_time, args.end_time)
    system_message: Dict = {
        "role": "system",
        "content": (
            "You are a streaming video captioning agent. Each current-caption user message has <memory>, "
            "<current_vision>, and <active_query>. There is no <response> block. Caption only the newest "
            "<current_vision> chunk in English. Use <memory>, <vision_memory>, and <recent_vision_memory> "
            "only as historical context; never describe historical visual memory as the current frame."
        ),
    }
    messages: List[Dict] = [system_message]
    out_rows = []
    for idx, row in enumerate(rows):
        generated_memory = [] if args.isolated_turns else out_rows
        user_message = {"role": "user", "content": build_stream_user(
            row,
            generated_memory,
            args.frames_per_turn,
            args.memory_lookback,
            args.visual_format,
        )}
        if args.isolated_turns:
            call_messages = [system_message, user_message]
        else:
            messages.append(user_message)
            call_messages = messages
        response = call_model(args, call_messages)
        raw = response["choices"][0]["message"].get("content", "")
        previous_gold = rows[idx - 1]["gold_caption"] if idx else ""
        out = result_row(row, raw, previous_gold, len(call_messages), response.get("usage", {}))
        out_rows.append(out)
        if not args.isolated_turns:
            messages.append({"role": "assistant", "content": raw})
        print(json.dumps(out, ensure_ascii=False), flush=True)
    return summarize(args, out_rows, {"start_time": args.start_time, "end_time": args.end_time, "isolated_turns": args.isolated_turns})


def run_prefill_continue(args: argparse.Namespace) -> Dict:
    if not args.prefill_summary:
        raise ValueError("--prefill-summary is required for prefill_continue")
    prefill_data = json.loads(Path(args.prefill_summary).read_text(encoding="utf-8"))
    prefill_results = [r for r in prefill_data["results"] if args.prefill_start <= r["current_time"] <= args.prefill_end]
    prefill_rows = load_rows(Path(args.jsonl), args.video_id, args.prefill_start, args.prefill_end)
    rows = load_rows(Path(args.jsonl), args.video_id, args.start_time, args.end_time)
    messages: List[Dict] = [{
        "role": "system",
        "content": (
            "You are a streaming video captioning agent. A first user message may prefill historical "
            "summary, selected text memory, selected visual memory, and historical query. For future turns, use that history "
            "only as context and caption only the newest <current_vision> chunk in English. Historical "
            "<vision_memory> or <recent_vision_memory> must not be treated as current visual input."
        ),
    }]
    messages.append({"role": "user", "content": build_prefill_user(
        prefill_rows,
        prefill_results,
        args.frames_per_turn,
        args.prefill_text_mode,
        args.prefill_text_count,
        args.prefill_visual_mode,
        args.prefill_visual_count,
        args.prefill_summary_count,
        args.prefill_layout,
        args.prefill_summary_start,
        args.prefill_summary_end,
        args.prefill_summary_style,
        visual_format=args.visual_format,
    )})
    out_rows = []
    for idx, row in enumerate(rows):
        generated_for_memory = prefill_results + out_rows
        # Use only generated captions in the current prompt memory; full prefill content remains in chat history.
        messages.append({"role": "user", "content": build_stream_user(
            row,
            generated_for_memory,
            args.frames_per_turn,
            args.memory_lookback,
            args.visual_format,
        )})
        response = call_model(args, messages)
        raw = response["choices"][0]["message"].get("content", "")
        previous_gold = rows[idx - 1]["gold_caption"] if idx else prefill_rows[-1]["gold_caption"]
        out = result_row(row, raw, previous_gold, len(messages), response.get("usage", {}))
        out_rows.append(out)
        messages.append({"role": "assistant", "content": raw})
        print(json.dumps(out, ensure_ascii=False), flush=True)
    return summarize(
        args,
        out_rows,
        {
            "prefill_start": args.prefill_start,
            "prefill_end": args.prefill_end,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "prefill_summary": args.prefill_summary,
            "prefill_items": len(prefill_results),
            "prefill_text_mode": args.prefill_text_mode,
            "prefill_text_count": args.prefill_text_count,
            "prefill_visual_mode": args.prefill_visual_mode,
            "prefill_visual_count": args.prefill_visual_count,
            "prefill_summary_count": args.prefill_summary_count,
            "prefill_summary_start": args.prefill_summary_start,
            "prefill_summary_end": args.prefill_summary_end,
            "prefill_summary_style": args.prefill_summary_style,
            "prefill_layout": args.prefill_layout,
        },
    )


def run_prefill_history_qa(args: argparse.Namespace) -> Dict:
    if not args.prefill_summary:
        raise ValueError("--prefill-summary is required for prefill_history_qa")
    prefill_data = json.loads(Path(args.prefill_summary).read_text(encoding="utf-8"))
    prefill_results = [r for r in prefill_data["results"] if args.prefill_start <= r["current_time"] <= args.prefill_end]
    prefill_rows = load_rows(Path(args.jsonl), args.video_id, args.prefill_start, args.prefill_end)
    target_rows = load_rows(Path(args.jsonl), args.video_id, args.history_check_time, args.history_check_time)
    target = target_rows[0]
    messages: List[Dict] = [{
        "role": "system",
        "content": (
            "You are a streaming video memory agent. A first user message prefilled historical summary, "
            "text memory, and maybe visual memory. Answer later questions from that initialized history. "
            "Do not invent current visual content."
        ),
    }]
    messages.append({"role": "user", "content": build_prefill_user(
        prefill_rows,
        prefill_results,
        args.frames_per_turn,
        args.prefill_text_mode,
        args.prefill_text_count,
        args.prefill_visual_mode,
        args.prefill_visual_count,
        args.prefill_summary_count,
        args.prefill_layout,
        args.prefill_summary_start,
        args.prefill_summary_end,
        args.prefill_summary_style,
        visual_format=args.visual_format,
    )})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": "\n".join([
            "<active_query>",
            (
                f'  <q t="{args.history_check_time}">From the initialized historical memory only, '
                f"describe what was visible around t={args.history_check_time}. "
                "Answer with one concise English caption in <answer>...</answer>.</q>"
            ),
            "</active_query>",
        ])},
    ]})
    response = call_model(args, messages)
    raw = response["choices"][0]["message"].get("content", "")
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else extract_think(raw)
    recall_gold = keyword_recall(answer, target["gold_caption"])
    prefill_item = next((r for r in prefill_results if r["current_time"] == args.history_check_time), None)
    recall_prefill = keyword_recall(answer, prefill_item["pred_caption"]) if prefill_item else 0.0
    row = {
        "history_check_time": args.history_check_time,
        "gold_caption": target["gold_caption"],
        "prefill_caption": prefill_item["pred_caption"] if prefill_item else "",
        "answer": answer,
        "raw": raw,
        "history_gold_keyword_recall": recall_gold,
        "history_prefill_keyword_recall": recall_prefill,
        "usage": response.get("usage", {}),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "model": args.model,
        "jsonl": args.jsonl,
        "video_id": args.video_id,
        "frames_per_turn": args.frames_per_turn,
        "visual_format": args.visual_format,
        "visual_fps": args.visual_fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "prefill_start": args.prefill_start,
        "prefill_end": args.prefill_end,
        "prefill_summary": args.prefill_summary,
        "prefill_items": len(prefill_results),
        "prefill_text_mode": args.prefill_text_mode,
        "prefill_text_count": args.prefill_text_count,
        "prefill_visual_mode": args.prefill_visual_mode,
        "prefill_visual_count": args.prefill_visual_count,
        "prefill_summary_count": args.prefill_summary_count,
        "prefill_summary_start": args.prefill_summary_start,
        "prefill_summary_end": args.prefill_summary_end,
        "prefill_summary_style": args.prefill_summary_style,
        "prefill_layout": args.prefill_layout,
        **row,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return summary


def run_prefill_time_range_qa(args: argparse.Namespace) -> Dict:
    if not args.prefill_summary:
        raise ValueError("--prefill-summary is required for prefill_time_range_qa")
    if args.time_query not in TIME_RANGE_QUERIES:
        raise ValueError(f"unknown --time-query {args.time_query}; choices={sorted(TIME_RANGE_QUERIES)}")
    query_spec = TIME_RANGE_QUERIES[args.time_query]
    prefill_data = json.loads(Path(args.prefill_summary).read_text(encoding="utf-8"))
    prefill_results = [r for r in prefill_data["results"] if args.prefill_start <= r["current_time"] <= args.prefill_end]
    prefill_rows = load_rows(Path(args.jsonl), args.video_id, args.prefill_start, args.prefill_end)
    messages: List[Dict] = [{
        "role": "system",
        "content": (
            "You are a streaming video memory agent. A first user message prefilled historical summary, "
            "text memory, and maybe visual memory. The historical visual chunks are tagged as "
            "<t=x><videochunk>...</videochunk></t>. For time localization questions, answer only from "
            "the initialized history and output exactly one JSON object inside <answer>."
        ),
    }]
    messages.append({"role": "user", "content": build_prefill_user(
        prefill_rows,
        prefill_results,
        args.frames_per_turn,
        args.prefill_text_mode,
        args.prefill_text_count,
        args.prefill_visual_mode,
        args.prefill_visual_count,
        args.prefill_summary_count,
        args.prefill_layout,
        args.prefill_summary_start,
        args.prefill_summary_end,
        args.prefill_summary_style,
        query_spec["keywords"],
        visual_format=args.visual_format,
    )})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": "\n".join([
            "<active_query>",
            (
                f'  <q>From the initialized historical memory only, locate this past event: '
                f"{query_spec['question']} Return the most specific historical time range in seconds. "
                'Use format exactly: <answer>{"time_range":"start-end","evidence":"short reason"}</answer>.'
                "</q>"
            ),
            "</active_query>",
        ])},
    ]})
    response = call_model(args, messages)
    raw = response["choices"][0]["message"].get("content", "")
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else raw.strip()
    predicted = parse_time_range(answer)
    expected = query_spec["expected_range"]
    overlap = range_overlap(predicted, expected)
    row = {
        "time_query": args.time_query,
        "question": query_spec["question"],
        "expected_range": expected,
        "predicted_range": predicted,
        "answer": answer,
        "raw": raw,
        **overlap,
        "usage": response.get("usage", {}),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "model": args.model,
        "jsonl": args.jsonl,
        "video_id": args.video_id,
        "frames_per_turn": args.frames_per_turn,
        "visual_format": args.visual_format,
        "visual_fps": args.visual_fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "prefill_start": args.prefill_start,
        "prefill_end": args.prefill_end,
        "prefill_summary": args.prefill_summary,
        "prefill_items": len(prefill_results),
        "prefill_text_mode": args.prefill_text_mode,
        "prefill_text_count": args.prefill_text_count,
        "prefill_visual_mode": args.prefill_visual_mode,
        "prefill_visual_count": args.prefill_visual_count,
        "prefill_summary_count": args.prefill_summary_count,
        "prefill_summary_start": args.prefill_summary_start,
        "prefill_summary_end": args.prefill_summary_end,
        "prefill_summary_style": args.prefill_summary_style,
        "prefill_layout": args.prefill_layout,
        **row,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stream", "prefill_continue", "prefill_history_qa", "prefill_time_range_qa"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model", default="qwen3vl2b-60chunk-probe")
    parser.add_argument("--jsonl", default="data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl")
    parser.add_argument("--video-id", default="Making_Vanilla_Banana_Bread")
    parser.add_argument("--start-time", type=int, required=True)
    parser.add_argument("--end-time", type=int, required=True)
    parser.add_argument("--prefill-start", type=int, default=0)
    parser.add_argument("--prefill-end", type=int, default=59)
    parser.add_argument("--prefill-summary", default="")
    parser.add_argument("--prefill-text-mode", choices=["none", "all", "recent", "uniform", "uniform_recent", "query_top"], default="all")
    parser.add_argument("--prefill-text-count", type=int, default=-1)
    parser.add_argument("--prefill-visual-mode", choices=["none", "all", "recent", "uniform", "uniform_recent", "query_top"], default="all")
    parser.add_argument("--prefill-visual-count", type=int, default=-1)
    parser.add_argument("--prefill-summary-count", type=int, default=8)
    parser.add_argument("--prefill-summary-start", type=int, default=None)
    parser.add_argument("--prefill-summary-end", type=int, default=None)
    parser.add_argument("--prefill-summary-style", choices=["keywords", "snippets"], default="keywords")
    parser.add_argument("--prefill-layout", choices=["text_then_visual", "visual_then_text", "text_only", "visual_only"], default="text_then_visual")
    parser.add_argument("--history-check-time", type=int, default=50)
    parser.add_argument("--time-query", choices=sorted(TIME_RANGE_QUERIES), default="flour_measuring")
    parser.add_argument("--frames-per-turn", type=int, default=1)
    parser.add_argument("--visual-format", choices=["image", "video", "mp4"], default="image")
    parser.add_argument("--visual-fps", type=float, choices=[1.0, 2.0], default=1.0)
    parser.add_argument("--memory-lookback", type=int, default=2)
    parser.add_argument("--isolated-turns", action="store_true", help="Do not carry previous chat turns in stream mode.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--wait-server-sec", type=int, default=600)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=100000)
    args = parser.parse_args()
    if args.visual_fps == 1.0:
        args.frames_per_turn = 1
    elif args.visual_fps == 2.0:
        args.frames_per_turn = 2
    wait_server(args.base_url.rstrip("/"), args.wait_server_sec)
    if args.mode == "stream":
        summary = run_stream(args)
    elif args.mode == "prefill_continue":
        summary = run_prefill_continue(args)
    elif args.mode == "prefill_history_qa":
        summary = run_prefill_history_qa(args)
    else:
        summary = run_prefill_time_range_qa(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
