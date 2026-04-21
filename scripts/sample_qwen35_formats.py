"""
Sample Qwen3.5-397B responses with different prompt strategies.
Goal: understand output format variations for parse_evidence_result fix.

Strategies:
  A) Original prompt (no prefix) — thinking enabled by default
  B) /no_think prefix — explicitly disable thinking
  C) Original prompt + explicit "wrap thinking in <think>...</think>" instruction

3 videos × 3 strategies × 3 samples = 27 API calls
"""

import base64
import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://10.16.12.175:8000/v1"
MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
FRAMES_ROOT = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_test/frames")
OUTPUT_PATH = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_test/qwen35_format_samples.json")

TEST_VIDEOS = ["0ySyffWMAVI", "xsFSYBJlrqA", "ytb_zgfJuuCbhyw"]
SAMPLES_PER_STRATEGY = 3

EVIDENCE_PROMPT_BASE = """You are annotating a streaming video chunk by chunk for training data construction.

Previous captions for context:
(first chunk)

Current chunk: t=0-2s (frames above).

Output a STRICT JSON object with these fields:
{{
  "time": [0, 2],
  "visible_entities": [
    {{"id": "entity_name", "attributes": ["attr1", "attr2"], "action": "what doing", "position": "where"}}
  ],
  "atomic_facts": [
    {{
      "fact": "precise observable statement",
      "confidence": 0.0-1.0,
      "support_level": "direct_current_chunk | carried_from_previous | inferred",
      "target_resolution_visible": true
    }}
  ],
  "state_changes": ["what changed from previous chunk"],
  "ocr": ["exact text if visible"],
  "spatial": "brief spatial layout description",
  "not_observable": ["any sounds/smells/emotions mentioned that are NOT visible"]
}}

Rules:
- Only include what is VISIBLE in the current chunk frames (t=0-2s)
- support_level: "direct_current_chunk" = visible in current 2s frames
- confidence < 0.7 for uncertain observations
- Maintain consistent entity IDs across chunks
- Do NOT attribute facts from prior chunks to the current chunk

Output JSON only:"""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_content(prompt: str, frame_paths: list) -> list:
    content = []
    for fp in frame_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(fp)},
        })
    content.append({"type": "text", "text": prompt})
    return content


def get_strategies():
    return {
        "A_original": EVIDENCE_PROMPT_BASE,
        "B_no_think": "/no_think\n\n" + EVIDENCE_PROMPT_BASE,
        "C_explicit_think_tags": EVIDENCE_PROMPT_BASE.replace(
            "Output JSON only:",
            "IMPORTANT: If you need to think, wrap your reasoning in <think>...</think> tags, then output JSON.\n\nOutput JSON only:"
        ),
    }


def call_api(messages, temperature=0.3, max_tokens=1024):
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    reasoning = choice["message"].get("reasoning_content", None)
    return {
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def main():
    strategies = get_strategies()
    results = []
    total = len(TEST_VIDEOS) * len(strategies) * SAMPLES_PER_STRATEGY
    done = 0

    for video_id in TEST_VIDEOS:
        frame_dir = FRAMES_ROOT / video_id
        frames = sorted(frame_dir.glob("frame_*.jpg"))[:2]  # first 2 frames = chunk 0
        if len(frames) < 2:
            print(f"SKIP {video_id}: not enough frames")
            continue
        frame_paths = [str(f) for f in frames]

        for strategy_name, prompt in strategies.items():
            for sample_idx in range(SAMPLES_PER_STRATEGY):
                done += 1
                print(f"[{done}/{total}] {video_id} / {strategy_name} / sample {sample_idx+1}")
                t0 = time.time()
                try:
                    messages = [{"role": "user", "content": build_content(prompt, frame_paths)}]
                    resp = call_api(messages, temperature=0.6 if sample_idx > 0 else 0.3)
                    elapsed = time.time() - t0
                    results.append({
                        "video_id": video_id,
                        "strategy": strategy_name,
                        "sample_idx": sample_idx,
                        "temperature": 0.6 if sample_idx > 0 else 0.3,
                        "elapsed_sec": round(elapsed, 1),
                        "content": resp["content"],
                        "reasoning_content": resp["reasoning_content"],
                        "finish_reason": resp["finish_reason"],
                        "usage": resp["usage"],
                    })
                    # Quick preview
                    c = resp["content"] or ""
                    r = resp["reasoning_content"] or ""
                    print(f"  content[:120]: {c[:120]}")
                    print(f"  reasoning[:80]: {r[:80] if r else '(none)'}")
                    print(f"  elapsed: {elapsed:.1f}s")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        "video_id": video_id,
                        "strategy": strategy_name,
                        "sample_idx": sample_idx,
                        "error": str(e),
                    })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(results)} samples saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
