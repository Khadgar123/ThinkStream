#!/usr/bin/env python3
"""Probe whether a large teacher can iteratively maintain compact video memory.

The script reads rendered ThinkStream SFT rows, finds real compress-trigger rows,
extracts the per-second ``<memory_think>`` observations selected by
``gold_compress_chunks``, then asks an OpenAI-compatible teacher model to update a
fixed-size memory over multiple compression triggers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"


PROMPT_RULES = """You are updating compact video memory.

Input:
1. OLD_MEMORY: previous compact memory.
2. NEW_CAPTIONS: dense captions from the latest trajectory.

Task:
Output NEW_MEMORY with exactly 5 timestamped lines.

Rules:
- Do not output a paragraph.
- Each line must be one event.
- Preserve concrete objects and actions.
- Merge repeated captions.
- Keep recent events more than old minor events.
- If OLD_MEMORY and NEW_CAPTIONS describe the same continuing event, extend the timestamp.
- Do not mention uncertainty unless the caption is unclear.
- Output only <MEM>...</MEM>."""


BALANCED_PROMPT_RULES = """You are updating compact video memory.

Input:
1. OLD_MEMORY: previous compact memory.
2. NEW_CAPTIONS: dense captions from the latest trajectory.

Task:
Output NEW_MEMORY with exactly 5 timestamped XML lines inside <MEM>.

Hard rules:
- Output only <MEM>...</MEM>.
- Each line must be exactly: <m t="a-b">one concise event.</m>
- Each line must describe one event/state in <=24 words.
- Merge repeated adjacent captions; do not copy raw dense captions.
- Preserve concrete objects and actions.
- If OLD_MEMORY is non-empty, NEW_MEMORY must cover both OLD_MEMORY time ranges and NEW_CAPTIONS time ranges.
- If OLD_MEMORY is non-empty, keep 2 or 3 important old events and use 2 or 3 lines for new/recent events.
- Keep recent events more than old minor events, but never drop all old memory.
- If OLD_MEMORY and NEW_CAPTIONS describe the same continuing event, extend the timestamp.
- Do not mention uncertainty unless the caption is unclear."""


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def assistant_text(row: Dict[str, Any]) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return text_from_content(message.get("content"))
    return ""


def extract_memory_thinks(row: Dict[str, Any]) -> Dict[int, str]:
    user_text = ""
    for message in row.get("messages", []):
        if message.get("role") == "user":
            user_text += "\n" + text_from_content(message.get("content"))
    out: Dict[int, str] = {}
    for match in re.finditer(r"<memory_think>\s*(\{.*?\})\s*</memory_think>", user_text, re.S):
        try:
            item = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if "time" in item and item.get("text"):
            out[int(item["time"])] = str(item["text"]).strip()
    return out


def extract_gold_tool_summary(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = assistant_text(row)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    args = payload.get("arguments") if isinstance(payload, dict) else None
    if isinstance(args, dict) and args.get("text"):
        return args
    return None


def iter_rows(paths: Iterable[Path]) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                yield path, line_no, json.loads(line)


def find_compress_rows(paths: List[Path], video_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected_video = video_id
    for path, line_no, row in iter_rows(paths):
        if row.get("sample_type") != "compress":
            continue
        if selected_video is not None and row.get("video_id") != selected_video:
            continue
        if selected_video is None:
            selected_video = row.get("video_id")
        md = row.get("metadata") or {}
        gold_chunks = md.get("gold_compress_chunks") or []
        thinks = extract_memory_thinks(row)
        selected = [
            {"t": int(t), "text": thinks[int(t)]}
            for t in gold_chunks
            if int(t) in thinks
        ]
        if not selected:
            continue
        rows.append(
            {
                "source": str(path),
                "line_no": line_no,
                "video_id": row.get("video_id"),
                "chunk_idx": row.get("chunk_idx"),
                "gold_compress_chunks": gold_chunks,
                "new_captions": selected,
                "gold_tool_summary": extract_gold_tool_summary(row),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def format_captions(items: List[Dict[str, Any]]) -> str:
    return "\n".join(f'  <c t="{item["t"]}">{item["text"]}</c>' for item in items)


def build_prompt(old_memory: str, captions: List[Dict[str, Any]], prompt_style: str) -> str:
    old = old_memory.strip() or "<MEM>\n</MEM>"
    rules = BALANCED_PROMPT_RULES if prompt_style == "balanced" else PROMPT_RULES
    return f"""{rules}

<OLD_MEMORY>
{old}
</OLD_MEMORY>

<NEW_CAPTIONS>
{format_captions(captions)}
</NEW_CAPTIONS>"""


def post_chat(
    api_base: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout: int,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise video-memory compression teacher."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    parsed = json.loads(body)
    choice = parsed["choices"][0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        usage = parsed.get("usage") or {}
        raise RuntimeError(
            "finish_reason=length "
            f"max_tokens={max_tokens} "
            f"prompt_tokens={usage.get('prompt_tokens', 0)} "
            f"completion_tokens={usage.get('completion_tokens', 0)}"
        )
    message = choice.get("message") or {}
    text = message.get("content") or ""
    if not text and enable_thinking:
        text = message.get("reasoning") or ""
    return {
        "latency_sec": time.time() - started,
        "raw_response": parsed,
        "text": text.strip(),
        "usage": parsed.get("usage"),
    }


def extract_mem_block(text: str) -> str:
    match = re.search(r"<MEM>.*?</MEM>", text, re.S)
    return match.group(0).strip() if match else text.strip()


def validate_memory(text: str) -> Dict[str, Any]:
    block = extract_mem_block(text)
    lines = re.findall(r"<m\b[^>]*>.*?</m>", block, re.S)
    ranges: List[Tuple[int, int]] = []
    for line in lines:
        match = re.search(r't="(\d+)(?:-(\d+))?"', line)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        ranges.append((start, end))
    return {
        "has_mem_block": block.startswith("<MEM>") and block.endswith("</MEM>"),
        "line_count": len(lines),
        "ranges": ranges,
        "closed_line_count": len(lines),
        "char_len": len(block),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://10.16.12.175:8000/v1")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--jsonl", nargs="*", type=Path, default=[
        Path("data/agent_v5/batch1/rendered/video_meta_standard_query_last/train_sft_messages.jsonl"),
        Path("data/agent_v5/batch2/rendered/video_meta_standard_query_last/train_sft_messages.jsonl"),
    ])
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prompt-style", choices=["user", "balanced"], default="user")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("output/teacher_memory_compression/teacher_397b_iterative_mem_probe.json"))
    args = parser.parse_args()

    compress_rows = find_compress_rows(args.jsonl, args.video_id, args.steps)
    if not compress_rows:
        raise SystemExit("No compress rows found.")

    old_memory = ""
    results: List[Dict[str, Any]] = []
    for step_idx, row in enumerate(compress_rows):
        prompt = build_prompt(old_memory, row["new_captions"], args.prompt_style)
        response = post_chat(
            args.api_base,
            args.model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            timeout=args.timeout,
        )
        memory = extract_mem_block(response["text"])
        result = {
            "step": step_idx,
            "source": row["source"],
            "line_no": row["line_no"],
            "video_id": row["video_id"],
            "chunk_idx": row["chunk_idx"],
            "gold_compress_chunks": row["gold_compress_chunks"],
            "new_caption_count": len(row["new_captions"]),
            "new_caption_range": [
                min(item["t"] for item in row["new_captions"]),
                max(item["t"] for item in row["new_captions"]),
            ],
            "old_memory": old_memory,
            "teacher_output": response["text"],
            "parsed_memory": memory,
            "validation": validate_memory(response["text"]),
            "usage": response.get("usage"),
            "latency_sec": response["latency_sec"],
            "gold_tool_summary": row["gold_tool_summary"],
        }
        results.append(result)
        old_memory = memory
        print(
            f"step={step_idx} video={row['video_id']} chunk={row['chunk_idx']} "
            f"new={result['new_caption_range']} lines={result['validation']['line_count']} "
            f"latency={response['latency_sec']:.1f}s"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "api_base": args.api_base,
                "model": args.model,
                "jsonl": [str(p) for p in args.jsonl],
                "video_id": compress_rows[0]["video_id"],
                "prompt_rules": BALANCED_PROMPT_RULES if args.prompt_style == "balanced" else PROMPT_RULES,
                "prompt_style": args.prompt_style,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
