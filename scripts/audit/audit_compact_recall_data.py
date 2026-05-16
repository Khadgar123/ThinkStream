#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MEM_RE = re.compile(r"(?is)^\s*<MEM>.*</MEM>\s*$")
MLINE_RE = re.compile(r'<m\s+t="[^"]+">.*?</m>', re.I | re.S)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return ""


def audit_file(path: Path) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    examples: dict[str, list[Any]] = defaultdict(list)
    with path.open() as fh:
        for ln, line in enumerate(fh, 1):
            try:
                row = json.loads(line)
            except Exception as exc:
                stats["json_error"] += 1
                if len(examples["json_error"]) < 3:
                    examples["json_error"].append((ln, str(exc)))
                continue
            stats["rows"] += 1
            messages = row.get("messages") or []
            trajectory_type = row.get("trajectory_type") or row.get("task_type") or ""
            is_compact = (
                trajectory_type == "compact_memory_update"
                or row.get("sample_type") == "compress"
                or row.get("loss_class") == "compress"
            )
            if is_compact:
                stats["compact_rows"] += 1
                system_text = _text_from_content(messages[0].get("content") if messages else "")
                if "Return only 4-6 chronological XML lines" in system_text:
                    stats["compact_new_prompt"] += 1
                if "Return only one <MEM> block" in system_text:
                    stats["compact_old_prompt"] += 1
                if "# Tools" in system_text or "<tools>" in system_text:
                    stats["compact_system_has_tools_text"] += 1

                user_text = ""
                assistant_text = ""
                for message in messages:
                    if message.get("role") == "user" and not user_text:
                        user_text = _text_from_content(message.get("content"))
                    if message.get("role") == "assistant":
                        assistant_text = str(message.get("content") or "")

                if "OLD_MEMORY:" in user_text and "NEW_CAPTIONS:" in user_text:
                    stats["compact_input_shape_ok"] += 1
                for bad in (
                    "<active_query>",
                    "<active_query>",
                    "<answer_history>",
                    "<response_history>",
                    "</Response>",
                    "<recalled_frames>",
                    "<recall_result>",
                ):
                    if bad in user_text:
                        key = f"compact_user_leak_{bad}"
                        stats[key] += 1
                        if len(examples[key]) < 3:
                            examples[key].append((ln, row.get("video_id"), user_text[:500]))

                if MEM_RE.match(assistant_text):
                    stats["compact_asst_mem_block"] += 1
                else:
                    stats["compact_asst_bad_mem"] += 1
                    starts[(assistant_text.strip()[:40] or "EMPTY").replace("\n", " ")] += 1
                    if len(examples["compact_bad_asst"]) < 5:
                        examples["compact_bad_asst"].append(
                            (ln, row.get("video_id"), assistant_text[:700])
                        )

                n_m = len(MLINE_RE.findall(assistant_text))
                if 4 <= n_m <= 6:
                    stats["compact_mline_count_4_6"] += 1
                else:
                    stats["compact_mline_bad_count"] += 1
                    if len(examples["compact_mline_bad_count"]) < 5:
                        examples["compact_mline_bad_count"].append(
                            (ln, row.get("video_id"), n_m, assistant_text[:500])
                        )

            for message in messages:
                if message.get("role") not in {"tool", "user"}:
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                has_recalled_header = any(
                    isinstance(item, dict) and "recalled_frames" in str(item.get("text", ""))
                    for item in content
                )
                if not has_recalled_header:
                    continue
                stats["recall_payloads"] += 1
                videos = [
                    item
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "video"
                ]
                if len(videos) == 1:
                    stats["recall_packed"] += 1
                elif len(videos) > 1:
                    stats["recall_chunked"] += 1
                else:
                    stats["recall_no_video"] += 1
                if any(
                    item.get("kv_scope") != "recall"
                    for item in content
                    if isinstance(item, dict)
                ):
                    stats["recall_missing_kv_scope"] += 1
                    if len(examples["recall_missing_kv_scope"]) < 3:
                        examples["recall_missing_kv_scope"].append(
                            (ln, row.get("video_id"), content[:4])
                        )
                if len(examples["recall_payload_sample"]) < 3:
                    examples["recall_payload_sample"].append(
                        (
                            ln,
                            row.get("video_id"),
                            [
                                (
                                    item.get("type"),
                                    len(item.get("video") or []),
                                    item.get("kv_scope"),
                                    str(item.get("text", ""))[:120],
                                )
                                for item in content
                                if isinstance(item, dict)
                            ][:5],
                        )
                    )

            flat = json.dumps(messages, ensure_ascii=False)
            if "<|im_start|>system" in flat:
                stats["literal_im_start_system_in_data"] += 1
            if "<tools>" in flat or "# Tools" in flat:
                stats["tools_text_literal_in_messages"] += 1

    return {
        "file": str(path),
        "stats": dict(stats),
        "bad_starts": starts.most_common(10),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        print(json.dumps(audit_file(path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
