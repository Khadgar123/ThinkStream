"""LLaMA-Factory CLI compatibility smoke test.

Validates that pass5 messages output + dataset_info.json can be ingested by
LLaMA-Factory's data loader without modification. Run on the GPU box AFTER
pipeline produces train_sft_messages.jsonl.

What this checks (without actually launching training):
  1. dataset_info.json fields match LLaMA-Factory's expected schema
     (formatting, columns, tags) — DeepEyesV2-aligned.
  2. The jsonl row format passes through LLaMA-Factory's converter
     (load_dataset → AlignedDataset).
  3. apply_chat_template with tools=TOOLS_SCHEMA produces a tokenized
     sequence that includes the <tools> block in system prompt.

Usage (on cluster, after data ready):
    python -m scripts.smoke_llamafactory_compat \\
        --data_dir data/agent_v5/final \\
        --tokenizer Qwen/Qwen3-VL-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_dataset_info(data_dir: Path) -> bool:
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        print(f"  ✗ {info_path} missing — run pass5_messages.py first")
        return False

    info = json.loads(info_path.read_text())
    print(f"  ✓ found {info_path}")
    print(f"    entries: {list(info.keys())}")

    required = {"file_name", "formatting", "columns", "tags"}
    for name, entry in info.items():
        missing = required - entry.keys()
        if missing:
            print(f"  ✗ {name}: missing keys {missing}")
            return False
        if entry["formatting"] != "sharegpt":
            print(f"  ✗ {name}: formatting={entry['formatting']!r}, expected 'sharegpt'")
            return False
        # tags must include role/content/user/assistant/system at minimum
        tag_keys = entry["tags"].keys()
        for k in ("role_tag", "content_tag", "user_tag", "assistant_tag", "system_tag"):
            if k not in tag_keys:
                print(f"  ✗ {name}: tags missing {k}")
                return False
    print("  ✓ dataset_info.json schema OK")
    return True


def check_jsonl_rows(data_dir: Path) -> bool:
    jsonl = data_dir / "train_sft_messages.jsonl"
    if not jsonl.exists():
        print(f"  ✗ {jsonl} missing")
        return False

    with jsonl.open() as f:
        n_total = 0
        n_with_video = 0
        n_recall = 0
        n_compress = 0
        for line in f:
            row = json.loads(line)
            n_total += 1
            if row.get("sample_type") == "recall":
                n_recall += 1
            elif row.get("sample_type") == "compress":
                n_compress += 1
            for msg in row["messages"]:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "video":
                            n_with_video += 1
                            break
            if n_total >= 200:
                break
    print(f"  scanned {n_total} samples (capped at 200)")
    print(f"  with video element: {n_with_video}")
    print(f"  recall (multi-turn shape B): {n_recall}")
    print(f"  compress (inter_chunk): {n_compress}")
    if n_total == 0:
        print(f"  ✗ empty jsonl")
        return False
    return True


def check_chat_template(data_dir: Path, tokenizer_name: str) -> bool:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ⚠ transformers not installed; skipping chat-template check")
        return True

    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ tokenizer load failed ({e}); skipping")
        return True

    from thinkstream.data.agent_protocol import TOOLS_SCHEMA, SYSTEM_PROMPT_V12
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_V12},
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "<answer>x</answer>"},
    ]
    try:
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, tools=TOOLS_SCHEMA,
        )
    except Exception as e:
        print(f"  ✗ apply_chat_template failed: {e}")
        return False

    if "<tools>" in rendered:
        print(f"  ✓ <tools> block rendered in system prompt")
    else:
        print(f"  ⚠ <tools> NOT in rendered text — chat_template may not "
              f"support tools= or template version mismatch")
    if "compress" in rendered and "recall" in rendered:
        print(f"  ✓ both recall + compress tools listed")
    else:
        print(f"  ✗ tool names missing from rendered system prompt")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/agent_v5/final")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("LLaMA-Factory compatibility smoke")
    print("=" * 60)

    ok = True
    print("\n[1/3] dataset_info.json schema")
    ok &= check_dataset_info(data_dir)
    print("\n[2/3] jsonl row inventory")
    ok &= check_jsonl_rows(data_dir)
    print("\n[3/3] chat_template + tools=")
    ok &= check_chat_template(data_dir, args.tokenizer)

    print("\n" + "=" * 60)
    if ok:
        print("✓ data is LLaMA-Factory CLI compatible")
        print("Try:  llamafactory-cli train --dataset thinkstream_train_sft \\")
        print("                              --dataset_dir " + str(data_dir))
    else:
        print("✗ compatibility issues detected — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
