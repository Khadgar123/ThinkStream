"""
采样 Qwen3.5-397B 在造数据 pipeline 各场景下的原始输出格式。

纯文本调用（无视觉），目的是收集输出格式特征用于分析。
每种场景 × 2 次采样（temp=0.3 + temp=0.7）。

场景：
  1. Pass 1 Evidence Graph（thinking=True，无 /no_think）
  2. Pass 2 Observation（/no_think 前缀）
  3. Pass 2 Compress（/no_think 前缀）
  4. Pass 3 Task Question（thinking=True）
  5. Pass 3 Recall Query（/no_think 前缀）
  6. Pass 3 Response（/no_think 前缀）
"""

import json
import time
from pathlib import Path

import requests

API_BASE = "http://10.16.12.175:8000/v1/chat/completions"
MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
OUTPUT = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_test/qwen35_pipeline_format_samples.json")

SCENARIOS = {
    "pass1_evidence": {
        "desc": "Pass 1 Evidence Graph (thinking=True)",
        "max_tokens": 1024,
        "prompt": """You are annotating a streaming video chunk by chunk for training data construction.

Previous captions for context:
(first chunk)

Current chunk: t=0-2s (frames described below since no image input).
Scene description: A woman with blonde hair wearing a white collared shirt and maroon vest is speaking to camera in a news studio. A news ticker is visible at the bottom.

Output a STRICT JSON object with these fields:
{
  "time": [0, 2],
  "visible_entities": [
    {"id": "entity_name", "attributes": ["attr1", "attr2"], "action": "what doing", "position": "where"}
  ],
  "atomic_facts": [
    {
      "fact": "precise observable statement",
      "confidence": 0.0-1.0,
      "support_level": "direct_current_chunk | carried_from_previous | inferred",
      "target_resolution_visible": true
    }
  ],
  "state_changes": ["what changed from previous chunk"],
  "ocr": ["exact text if visible"],
  "spatial": "brief spatial layout description",
  "not_observable": ["any sounds/smells/emotions mentioned that are NOT visible"]
}

Rules:
- Only include what is VISIBLE in the current chunk frames (t=0-2s)
- confidence < 0.7 for uncertain observations
- Maintain consistent entity IDs across chunks

Output JSON only:""",
    },
    "pass2_observation": {
        "desc": "Pass 2 Observation (/no_think prefix)",
        "max_tokens": 128,
        "prompt": """/no_think

You are a streaming video agent generating a think (incremental visual memory note).

Compressed memory:
(none)

Recent thinks:
[0-2] Woman with blonde hair in white shirt and maroon vest speaking to camera in news studio.
[2-4] Same anchor continues speaking. News ticker scrolls at bottom showing weather update.

Visual window: t=0-6s (frames described: news anchor in studio).

Describe what is NEW or CHANGED in the latest 2 seconds (t=4-6s) in 40-60 tokens.

Rules:
- Only observable visual facts
- Maintain entity names from memory
- Focus: entities+attributes, actions, state changes, OCR, spatial
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- If nothing new: brief ongoing state

Output (one paragraph, 40-60 tokens):""",
    },
    "pass2_compress": {
        "desc": "Pass 2 Compress (/no_think prefix)",
        "max_tokens": 512,
        "prompt": """/no_think

Compress these observations into a structured summary.

Observations to compress:
[0-2] Woman with blonde hair in white shirt and maroon vest speaking to camera in news studio.
[2-4] Same anchor continues speaking. News ticker scrolls at bottom showing weather update.
[4-6] Camera angle shifts slightly. Anchor gestures with right hand. Ticker shows "Breaking: Storm warning issued."
[6-8] Anchor looks down at papers briefly. Studio lighting unchanged. Ticker continues scrolling.
[8-10] Anchor resumes eye contact with camera. Small graphic appears in upper right corner showing a map.
[10-12] Map graphic expands. Anchor points toward it. Red zones visible on the map overlay.

Rules:
- Use coarse time sub-ranges: [X-Y]
- Keep ALL named entities with visual attributes
- Keep ALL OCR content verbatim
- Keep state changes as before→after
- Target length: 120 tokens
- DO NOT add information not in the observations

Output JSON: {"time_range": [0, 12], "text": "..."}""",
    },
    "pass3_task_question": {
        "desc": "Pass 3 Task Question (thinking=True)",
        "max_tokens": 512,
        "prompt": """Based on this visual evidence:
Entity: anchor_1
Attributes: blonde hair, white collared shirt, maroon vest
Fact: Anchor gestures with right hand while explaining weather graphic
Time: t=4s

Generate ONE specific, answerable question about this visual detail.
The full fact is: Anchor gestures with right hand while explaining weather graphic

Requirements:
- Natural conversational question
- Answerable from visual observation alone
- Do not include the answer in the question
- "concise_answer" must be a SHORT answer (1-10 words), not the full fact sentence

Output JSON: {"question": "...", "concise_answer": "...", "answer_type": "factoid|procedural|summary"}""",
    },
    "pass3_recall_query": {
        "desc": "Pass 3 Recall Query (/no_think prefix)",
        "max_tokens": 128,
        "prompt": """/no_think

Generate a retrieval query for this scenario:
- Question: "What was the anchor wearing earlier in the broadcast?"
- Visible memory context: [10-12] Map graphic on screen, anchor pointing at red zones.

Based ONLY on the question and the visible memory context, generate 3-5 discriminative
keywords that would help locate the relevant past observation.
NO answer values, NO pronouns, NO articles.
Include entity names + action/attribute anchors from the question and context.

Output JSON (one line): {"query": "keyword1 keyword2 keyword3", "time_range": "0-10"}""",
    },
    "pass3_response": {
        "desc": "Pass 3 Response (/no_think prefix)",
        "max_tokens": 256,
        "prompt": """/no_think

Generate a response for this streaming video agent:
- Question: "What is the anchor doing right now?"
- Available evidence: anchor_1 is gesturing with right hand toward a weather map graphic showing red storm zones
- Answer type: factoid
- Correct answer: gesturing toward weather map with right hand

Requirements:
- Response length: 1-2 sentences
- Base answer ONLY on the provided evidence
- If evidence is insufficient, say "I cannot confirm..."
- Do NOT add information beyond what's in the evidence

Output the response text only:""",
    },
}


def call_api(prompt, max_tokens, temperature):
    resp = requests.post(
        API_BASE,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return {
        "content": choice["message"].get("content", ""),
        "reasoning_content": choice["message"].get("reasoning_content")
                             or choice["message"].get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def main():
    results = []
    temps = [0.3, 0.7]
    total = len(SCENARIOS) * len(temps)
    done = 0

    for scenario_name, cfg in SCENARIOS.items():
        for temp in temps:
            done += 1
            print(f"[{done}/{total}] {scenario_name} temp={temp}")
            t0 = time.time()
            try:
                resp = call_api(cfg["prompt"], cfg["max_tokens"], temp)
                elapsed = time.time() - t0
                content = resp["content"] or ""
                reasoning = resp["reasoning_content"]

                results.append({
                    "scenario": scenario_name,
                    "description": cfg["desc"],
                    "temperature": temp,
                    "max_tokens": cfg["max_tokens"],
                    "elapsed_sec": round(elapsed, 1),
                    "content": content,
                    "reasoning_content": reasoning,
                    "finish_reason": resp["finish_reason"],
                    "usage": resp["usage"],
                    "content_length": len(content),
                    "starts_with_think_tag": content.strip().startswith("<think>"),
                    "starts_with_json": content.strip().startswith("{"),
                    "starts_with_thinking_text": any(
                        content.strip().startswith(p) for p in
                        ["The user", "Thinking", "Let me", "I need", "Okay", "Alright"]
                    ),
                    "has_no_think_prefix": "/no_think" in cfg["prompt"][:20],
                })

                print(f"  elapsed: {elapsed:.1f}s | finish: {resp['finish_reason']}")
                print(f"  reasoning: {str(reasoning)[:100] if reasoning else '(null)'}")
                print(f"  content[:200]: {content[:200]}")
                print()
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ERROR after {elapsed:.1f}s: {e}")
                results.append({
                    "scenario": scenario_name,
                    "description": cfg["desc"],
                    "temperature": temp,
                    "error": str(e),
                    "elapsed_sec": round(elapsed, 1),
                    "has_no_think_prefix": "/no_think" in cfg["prompt"][:20],
                })
                print()

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. {len(results)} samples saved to {OUTPUT}")


if __name__ == "__main__":
    main()
