#!/usr/bin/env python3
"""
Context-budget audit for v12.12 RUNTIME profile.

Three layers of verification:
  1. SFT-side: build a synthetic worst-case sample, render via pass5_messages,
     tokenize via Qwen3-VL processor, count actual input tokens. This is what
     SFT training will see — must fit under cutoff_len (default 16384).

  2. vLLM-side (optional): send the same content to a running vLLM with
     mm_processor_kwargs=RUNTIME, read response.usage.prompt_tokens. Confirms
     server-side smart_resize honors the bounds we configured.

  3. Real-batch sweep: read existing pass5 SFT samples, re-tokenize each, and
     report the distribution (max / p99 / p50) of actual input token counts.

Usage:
  # Layer 1 only (no vLLM, requires processor weights locally):
  python scripts/audit/test_context_budget.py --layer 1

  # Layer 1 + 2 (needs vLLM endpoint):
  python scripts/audit/test_context_budget.py --layer 1,2 \\
      --vllm-url http://10.0.0.1:8000/v1 --vllm-model Qwen3-VL-8B-Instruct

  # Layer 3 (sweep existing rendered SFT data):
  python scripts/audit/test_context_budget.py --layer 3 \\
      --jsonl data/agent_v5/final/train_sft_messages.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ctx_budget")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_data.config import (  # noqa: E402
    RUNTIME_MM_PROCESSOR_KWARGS,
    HIRES_MM_PROCESSOR_KWARGS,
    VISUAL_TOKENS_PER_FRAME_RUNTIME,
    VISUAL_WINDOW_TOKENS,
    VISUAL_WINDOW_CHUNKS,
    FRAMES_PER_CHUNK,
    AGENT_CHUNK_SEC,
    RECENT_THINKS_TOKEN_BUDGET,
    SUMMARY_TOKENS_MAX,
    MAX_COMPRESSED_SEGMENTS,
    SYSTEM_PROMPT_TOKENS,
)


# 16K context budget breakdown (design target).
DESIGN_BUDGET_16K = {
    "model_max_len": 16384,
    "obs_input_budget":      11_000,    # leaves ~5K for output + safety
    "obs_recall_input_budget": 12_000,  # ~4K headroom
    "compress_input_budget":   6_000,   # very tight in compress (only memory)
    "obs_output":             1_024,
    "compress_output":        4_096,
}


# ============================================================================
# Layer 1: SFT-side token count via HF processor
# ============================================================================
def build_worst_case_sample(*, with_recall: bool, inter_chunk: bool) -> Dict:
    """Construct a worst-case SFT sample dict matching pass5 schema.

    Worst-case knobs:
      - chunk_idx = 100 (way past visual window saturation at chunk 15
                         and first compression at chunk ~45)
      - 5 summaries × 280 tok each (saturate compressed memory)
      - recent_thinks tokens at trigger threshold 0.8 × 4000 = 3200
      - 4 active queries (max in MAX_ACTIVE_QUERIES)
      - if with_recall: 4 recalled frames + metadata-only recall_result
    """
    chunk_idx = 100
    # Build saturated memory: 5 summaries + recent_thinks at trigger
    SUMMARY_TEXT = "Detailed observation: " + "word " * 70  # ~85 tok / segment
    SUMMARY_LONG = "x " * 280       # exactly cap
    summaries = [
        {
            "type": "summary",
            "time_range": [i * 8, (i + 1) * 8],
            "text": SUMMARY_LONG,   # all at SUMMARY_TOKENS_MAX worst case
            "source_chunks": list(range(i * 8, (i + 1) * 8)),
            "merge_level": 1 + (i // 2),
        }
        for i in range(MAX_COMPRESSED_SEGMENTS)
    ]
    # Recent thinks: ~46 thinks @ 70 tok ≈ 3200 tok (trigger threshold)
    n_recent = 46
    recent = [
        {
            "chunk": 40 + i,
            "time": f"{40+i}-{41+i}",
            # ~70 tok of dense entity description
            "text": (
                "Person in red jacket holds a small ceramic bowl in left hand, "
                "rotates it slowly while looking at the camera, then places "
                "it on the wooden countertop next to the cutting board."
            ),
        }
        for i in range(n_recent)
    ]

    queries = [
        {"question": f"What was on the counter at t={i}?",
         "ask_time": (50 + i * 10),
         "answers": []}
        for i in range(4)
    ]

    visual_window: Dict[str, Any] = {}
    if not inter_chunk:
        # Saturated 32-frame window
        win_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
        n_frames = (chunk_idx - win_start + 1) * FRAMES_PER_CHUNK
        # Use stub paths (won't load — Layer 1 only counts text tok; Layer 2
        # needs real frames, see --frames-dir)
        frame_paths = [f"/tmp/stub_frame_{i:06d}.jpg" for i in range(n_frames)]
        visual_window = {
            "video_start": win_start * AGENT_CHUNK_SEC,
            "video_end": (chunk_idx + 1) * AGENT_CHUNK_SEC,
            "frames": n_frames,
            "frame_paths": frame_paths,
        }

    inp: Dict[str, Any] = {
        "memory": {
            "compressed_segments": [
                {"time_range": s["time_range"], "text": s["text"],
                 "merge_level": s["merge_level"]}
                for s in summaries
            ],
            "recent_thinks": recent,
        },
        "visual_window": visual_window,
        "queries": queries if not inter_chunk else [],
        "recall_result": None,
        "recalled_frames": None,
        "user_input": "" if inter_chunk else "What's happening with the bowl now?",
    }

    if with_recall and not inter_chunk:
        # Worst-case recall: 4 frames + metadata-only recall_result.
        inp["recalled_frames"] = {
            "time_range": [10, 14],
            "n_frames": 4,
            "frame_paths": [f"/tmp/stub_recall_{i}.jpg" for i in range(4)],
            "source": "historical_frames",
        }
        inp["recall_result"] = {
            "source": "historical",
            "time": "10-14",
            "returned_chunks": [10, 11, 12, 13],
        }

    sample = {
        "sample_id": "test_worst",
        "video_id": "vtest",
        "chunk_idx": chunk_idx,
        "sample_type": "compress" if inter_chunk else "response",
        "video_path": "/tmp/test.mp4",
        "input": inp,
        "output": (
            "<think>condense old context</think>"
            "<tool_call>{\"name\":\"compress\",\"arguments\":"
            "{\"time_range\":[0,40],\"text\":\"summary\"}}</tool_call>"
            if inter_chunk else
            "<think>The person is examining the bowl carefully.</think>"
            "<answer>They appear to be inspecting it before use.</answer>"
        ),
        "inter_chunk": inter_chunk,
    }
    return sample


def layer1_sft_side(processor_path: str) -> None:
    """Layer 1: render via pass5, tokenize via processor, count input tokens."""
    print("\n" + "=" * 70)
    print(" Layer 1: SFT-side actual input token count (Qwen3-VL processor)")
    print("=" * 70)

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_path,
                                                   trust_remote_code=True)
        from thinkstream.data.agent_protocol import TOOLS_SCHEMA
    except Exception as exc:
        print(f"❌ Cannot load processor from {processor_path}: {exc}")
        print("   Skipping Layer 1.")
        return

    from scripts.agent_data.pass5_messages import build_messages
    from PIL import Image
    import numpy as np

    # Create stub frame files so process_vision_info can load them.
    stub_dir = Path("/tmp/ctx_budget_stubs")
    stub_dir.mkdir(parents=True, exist_ok=True)
    arr = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Pre-generate stub jpgs at typical 1280x720
    for i in range(80):
        p = stub_dir / f"stub_frame_{i:06d}.jpg"
        if not p.exists():
            img.save(p, quality=85)
    for i in range(8):
        p = stub_dir / f"stub_recall_{i}.jpg"
        if not p.exists():
            img.save(p, quality=85)

    def _retarget(s):
        """Rewrite stub paths to real disk paths."""
        inp = s["input"]
        if inp.get("visual_window", {}).get("frame_paths"):
            inp["visual_window"]["frame_paths"] = [
                str(stub_dir / Path(p).name)
                for p in inp["visual_window"]["frame_paths"]
            ]
        if inp.get("recalled_frames", {}) and inp["recalled_frames"].get("frame_paths"):
            inp["recalled_frames"]["frame_paths"] = [
                str(stub_dir / Path(p).name)
                for p in inp["recalled_frames"]["frame_paths"]
            ]
        return s

    cases = [
        ("普通 obs（无 recall）",      False, False),
        ("obs + recall",              True,  False),
        ("compress turn (inter_chunk)", False, True),
    ]

    print(f"\nRUNTIME profile: {RUNTIME_MM_PROCESSOR_KWARGS}")
    print(f"Design budget: model_max_len = {DESIGN_BUDGET_16K['model_max_len']}\n")

    for label, with_recall, inter_chunk in cases:
        sample = _retarget(build_worst_case_sample(
            with_recall=with_recall, inter_chunk=inter_chunk
        ))
        try:
            messages = build_messages(sample, base_path=Path("/"))
        except Exception as exc:
            print(f"  [{label}] build_messages failed: {exc}")
            continue

        # Tokenize via processor.apply_chat_template (matches SFT exactly)
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                tools=TOOLS_SCHEMA,
                do_sample_frames=False,
            )
        except Exception as exc:
            print(f"  [{label}] tokenize failed: {exc}")
            continue

        n_input = int(inputs["input_ids"].shape[1])
        budget = (DESIGN_BUDGET_16K["compress_input_budget"] if inter_chunk
                  else (DESIGN_BUDGET_16K["obs_recall_input_budget"]
                        if with_recall
                        else DESIGN_BUDGET_16K["obs_input_budget"]))
        out_budget = (DESIGN_BUDGET_16K["compress_output"] if inter_chunk
                      else DESIGN_BUDGET_16K["obs_output"])
        total_with_output = n_input + out_budget
        max_len = DESIGN_BUDGET_16K["model_max_len"]
        ok = total_with_output <= max_len

        print(f"  [{label}]")
        print(f"    input_tokens     = {n_input:>6,}  (budget: {budget:,})")
        print(f"    + output budget  = {total_with_output:>6,}  / {max_len} ({total_with_output/max_len*100:5.1f}%)")
        print(f"    headroom         = {max_len - total_with_output:>6,}")
        print(f"    {'✓ FITS' if ok else '❌ EXCEEDS'} 16K context")
        print()


# ============================================================================
# Layer 2: vLLM-side prompt_tokens via response.usage
# ============================================================================
def layer2_vllm_side(vllm_url: str, vllm_model: str, frame_dir: str) -> None:
    """Layer 2: real vLLM call, read response.usage.prompt_tokens."""
    print("\n" + "=" * 70)
    print(" Layer 2: vLLM-side actual prompt_tokens via response.usage")
    print("=" * 70)

    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai SDK not installed; skipping Layer 2")
        return

    client = OpenAI(base_url=vllm_url, api_key="placeholder")

    # Find 32 real frames in frame_dir
    frame_paths = sorted(Path(frame_dir).glob("*.jpg"))[:32]
    if len(frame_paths) < 32:
        print(f"❌ Need ≥32 frames in {frame_dir}, found {len(frame_paths)}; skipping")
        return

    from scripts.agent_data_pipeline.vllm_client import encode_image_base64
    from thinkstream.data.agent_protocol import append_timestamped_image_list

    content: List[Dict] = []
    # Memory + queries (text portion)
    memory_block = "<memory>\n"
    for i in range(46):
        memory_block += f"[{40+i}-{41+i}] " + "x " * 30 + "\n"
    memory_block += "</memory>"
    content.append({"type": "text", "text": memory_block})

    # Vision frames: project protocol uses frame-tag text + image_url items.
    fps = float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC)
    append_timestamped_image_list(
        content,
        [str(p) for p in frame_paths],
        fps=fps,
        start_frame_index=0,
        total_num_frames=len(frame_paths),
        latest_start_frame_index=max(0, len(frame_paths) - FRAMES_PER_CHUNK),
        image_key="image_url",
        image_url_encoder=encode_image_base64,
    )
    content.append({"type": "text", "text": "What's happening?"})

    print(f"\nProfile: {RUNTIME_MM_PROCESSOR_KWARGS}")
    print(f"Sending obs request with {len(frame_paths)} frames + memory + 4 queries...\n")

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=vllm_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=10,   # we don't care about output, just measure prefill
            temperature=0.0,
            extra_body={
                "mm_processor_kwargs": {
                    **RUNTIME_MM_PROCESSOR_KWARGS,
                    "do_sample_frames": False,
                },
            },
        )
    except Exception as exc:
        print(f"❌ vLLM request failed: {exc}")
        return

    elapsed = time.time() - t0
    prompt_tokens = resp.usage.prompt_tokens
    completion_tokens = resp.usage.completion_tokens
    print(f"  prompt_tokens     = {prompt_tokens:,}")
    print(f"  completion_tokens = {completion_tokens}")
    print(f"  elapsed           = {elapsed:.2f}s")
    print(f"  effective tok/frame = {(prompt_tokens - 800) / 32:.0f}  (rough)")
    expected_visual = VISUAL_TOKENS_PER_FRAME_RUNTIME * 32
    print(f"  expected visual portion ≈ {expected_visual:,} (32 × {VISUAL_TOKENS_PER_FRAME_RUNTIME})")

    if prompt_tokens > 12_000:
        print(f"  ❌ EXCEEDS 12K obs+recall budget")
    else:
        print(f"  ✓ FITS within obs+recall budget (12K)")


# ============================================================================
# Layer 3: real batch sweep (existing rendered SFT data)
# ============================================================================
def layer3_batch_sweep(jsonl_path: str, processor_path: str,
                        sample_limit: int = 200) -> None:
    """Layer 3: tokenize each rendered sample, report distribution."""
    print("\n" + "=" * 70)
    print(" Layer 3: real-batch SFT input token distribution")
    print("=" * 70)

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(processor_path,
                                                   trust_remote_code=True)
        from thinkstream.data.agent_protocol import TOOLS_SCHEMA
    except Exception as exc:
        print(f"❌ Cannot load processor: {exc}")
        return

    p = Path(jsonl_path)
    if not p.exists():
        print(f"❌ JSONL not found: {p}")
        return

    counts: List[Tuple[int, str]] = []   # (n_tokens, sample_id)
    skipped = 0
    with open(p) as f:
        for i, line in enumerate(f):
            if i >= sample_limit:
                break
            try:
                row = json.loads(line)
                msgs = row.get("messages") or []
                inputs = processor.apply_chat_template(
                    msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    tools=TOOLS_SCHEMA,
                    do_sample_frames=False,
                )
                counts.append((int(inputs["input_ids"].shape[1]),
                                row.get("sample_id", f"row_{i}")))
            except Exception:
                skipped += 1

    if not counts:
        print(f"❌ No samples processable (skipped {skipped})")
        return

    counts.sort()
    n = len(counts)
    p50 = counts[n // 2][0]
    p90 = counts[int(n * 0.9)][0]
    p99 = counts[int(n * 0.99)][0]
    max_n, max_id = counts[-1]
    over_16k = sum(1 for c, _ in counts if c > 16384)
    over_12k = sum(1 for c, _ in counts if c > 12_000)

    print(f"\n  Samples processed: {n} ({skipped} skipped)")
    print(f"  Distribution:")
    print(f"    p50  = {p50:>6,}")
    print(f"    p90  = {p90:>6,}")
    print(f"    p99  = {p99:>6,}")
    print(f"    max  = {max_n:>6,}  (sample: {max_id})")
    print(f"    samples > 12K input: {over_12k}")
    print(f"    samples > 16K total: {over_16k}")
    if over_16k > 0:
        print(f"  ❌ {over_16k} sample(s) overflow 16K — investigate {max_id}")
    elif p99 > 12_000:
        print(f"  ⚠ p99 over design budget (12K obs+recall); inspect")
    else:
        print(f"  ✓ All samples within design budget")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", default="1",
                        help="Comma-separated layers to run: 1,2,3 (default: 1)")
    parser.add_argument("--processor", default=os.environ.get(
        "THINKSTREAM_STUDENT_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--frame-dir", default="data/agent_v5/frames",
                        help="Directory with at least 32 .jpg frames for layer 2")
    parser.add_argument("--jsonl", default="data/agent_v5/final/train_sft_messages.jsonl")
    parser.add_argument("--sample-limit", type=int, default=200)
    args = parser.parse_args()

    layers = [s.strip() for s in args.layer.split(",")]

    # Print resolved config first
    print("\n" + "=" * 70)
    print(" v12.12 Context Budget Audit")
    print("=" * 70)
    print(f"  RUNTIME mm_processor_kwargs = {RUNTIME_MM_PROCESSOR_KWARGS}")
    print(f"  HIRES   mm_processor_kwargs = {HIRES_MM_PROCESSOR_KWARGS}")
    print(f"  VISUAL_TOKENS_PER_FRAME    = {VISUAL_TOKENS_PER_FRAME_RUNTIME}")
    print(f"  VISUAL_WINDOW_TOKENS       = {VISUAL_WINDOW_TOKENS}")
    print(f"  RECENT_THINKS_BUDGET       = {RECENT_THINKS_TOKEN_BUDGET}")
    print(f"  Design budget per turn:")
    for k, v in DESIGN_BUDGET_16K.items():
        print(f"    {k:<26} = {v:,}")

    if "1" in layers:
        layer1_sft_side(args.processor)
    if "2" in layers:
        layer2_vllm_side(args.vllm_url, args.vllm_model, args.frame_dir)
    if "3" in layers:
        layer3_batch_sweep(args.jsonl, args.processor, args.sample_limit)


if __name__ == "__main__":
    main()
