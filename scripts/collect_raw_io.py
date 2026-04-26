"""
采集 Qwen3.5 在各种条件下的原始输入输出。
只记录原始数据，不做任何解析。
"""
import json, time, sys, base64, requests
from pathlib import Path

API = "http://10.16.12.175:8000/v1/chat/completions"
MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
FRAMES = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_test/frames/0ySyffWMAVI")
OUT = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5_test/raw_io_samples.json")

def img_b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def stream(messages, max_tokens, temp=0.3):
    t0 = time.time()
    r = requests.post(API, json={
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temp, "stream": True,
    }, timeout=600, stream=True)
    reasoning, content, finish = [], [], None
    for line in r.iter_lines():
        if not line: continue
        s = line.decode("utf-8")
        if not s.startswith("data: "): continue
        if s[6:].strip() == "[DONE]": break
        try:
            d = json.loads(s[6:])
            delta = d["choices"][0].get("delta", {})
            if delta.get("reasoning"): reasoning.append(delta["reasoning"])
            if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
            if delta.get("content"): content.append(delta["content"])
            f = d["choices"][0].get("finish_reason")
            if f: finish = f
        except: pass
    return {
        "reasoning": "".join(reasoning),
        "content": "".join(content),
        "finish_reason": finish,
        "elapsed_sec": round(time.time() - t0, 1),
    }

# 准备视觉输入
frame1 = str(sorted(FRAMES.glob("frame_*.jpg"))[0])
frame2 = str(sorted(FRAMES.glob("frame_*.jpg"))[1])

# ============================================================
# 定义所有测试用例：(名称, messages, max_tokens)
# ============================================================
EVIDENCE_PROMPT = """You are annotating a streaming video chunk by chunk for training data construction.

Previous captions for context:
(first chunk)

Current chunk: t=0-2s (frames above).

Output a STRICT JSON object with these fields:
{
  "time": [0, 2],
  "visible_entities": [
    {"id": "entity_name", "attributes": ["attr1"], "action": "what doing", "position": "where"}
  ],
  "atomic_facts": [
    {"fact": "statement", "confidence": 0.9, "support_level": "direct_current_chunk", "target_resolution_visible": true}
  ],
  "state_changes": [],
  "ocr": [],
  "spatial": "",
  "not_observable": []
}

Output JSON only:"""

OBS_PROMPT = """/no_think

You are a streaming video agent generating a think (incremental visual memory note).

Compressed memory:
(none)

Recent thinks:
[0-2] Woman with blonde hair in white shirt speaking to camera in news studio.

Visual window: t=0-4s.

Describe what is NEW or CHANGED in the latest 2 seconds (t=2-4s) in 40-60 tokens.
Output (one paragraph, 40-60 tokens):"""

COMPRESS_PROMPT = """/no_think

Compress these observations into a structured summary.

Observations to compress:
[0-2] Woman with blonde hair in white shirt speaking to camera in news studio.
[2-4] Same anchor continues. Ticker shows weather.
[4-6] Camera shifts. Anchor gestures. Ticker: "Breaking: Storm warning."
[6-8] Anchor looks at papers. Lighting unchanged.

Output JSON: {"time_range": [0, 8], "text": "..."}"""

TASK_Q_PROMPT = """Based on this visual evidence:
Entity: anchor_1
Attributes: blonde hair, white collared shirt, maroon vest
Fact: Anchor gestures with right hand while explaining weather graphic
Time: t=4s

Generate ONE specific, answerable question about this visual detail.
Output JSON: {"question": "...", "concise_answer": "...", "answer_type": "factoid|procedural|summary"}"""

cases = [
    # --- 纯文本，不同 max_tokens ---
    ("text_simple_mt64", [{"role":"user","content":"Say OK"}], 64),
    ("text_simple_mt256", [{"role":"user","content":"Say OK"}], 256),
    ("text_simple_mt512", [{"role":"user","content":"Say OK"}], 512),

    # --- /no_think 测试 ---
    ("text_no_think_mt256", [{"role":"user","content":"/no_think\n\nSay OK"}], 256),

    # --- 纯文本 pipeline prompts ---
    ("text_evidence_mt512", [{"role":"user","content":EVIDENCE_PROMPT}], 512),
    ("text_evidence_mt1024", [{"role":"user","content":EVIDENCE_PROMPT}], 1024),
    ("text_evidence_mt2048", [{"role":"user","content":EVIDENCE_PROMPT}], 2048),
    ("text_obs_mt256", [{"role":"user","content":OBS_PROMPT}], 256),
    ("text_obs_mt512", [{"role":"user","content":OBS_PROMPT}], 512),
    ("text_compress_mt512", [{"role":"user","content":COMPRESS_PROMPT}], 512),
    ("text_taskq_mt512", [{"role":"user","content":TASK_Q_PROMPT}], 512),
    ("text_taskq_mt1024", [{"role":"user","content":TASK_Q_PROMPT}], 1024),

    # --- 视觉输入 (2帧) + evidence prompt ---
    ("vision_evidence_mt1024", [{"role":"user","content":[
        {"type":"image_url","image_url":{"url":img_b64(frame1)}},
        {"type":"image_url","image_url":{"url":img_b64(frame2)}},
        {"type":"text","text":EVIDENCE_PROMPT},
    ]}], 1024),
    ("vision_evidence_mt2048", [{"role":"user","content":[
        {"type":"image_url","image_url":{"url":img_b64(frame1)}},
        {"type":"image_url","image_url":{"url":img_b64(frame2)}},
        {"type":"text","text":EVIDENCE_PROMPT},
    ]}], 2048),

    # --- 视觉 + /no_think observation ---
    ("vision_obs_no_think_mt512", [{"role":"user","content":[
        {"type":"image_url","image_url":{"url":img_b64(frame1)}},
        {"type":"image_url","image_url":{"url":img_b64(frame2)}},
        {"type":"text","text":OBS_PROMPT},
    ]}], 512),
]

results = []
total = len(cases)
for i, (name, messages, mt) in enumerate(cases):
    print(f"[{i+1}/{total}] {name} (max_tokens={mt})", flush=True)
    try:
        r = stream(messages, mt)
        # 记录原始输入（去掉 base64 图片数据，太大）
        input_record = []
        for msg in messages:
            if isinstance(msg["content"], str):
                input_record.append({"type": "text", "text": msg["content"]})
            elif isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item["type"] == "text":
                        input_record.append(item)
                    else:
                        input_record.append({"type": "image_url", "url": "(base64 omitted)"})
        results.append({
            "case": name,
            "max_tokens": mt,
            "input": input_record,
            "output": r,
        })
        print(f"  done: {r['elapsed_sec']}s | finish={r['finish_reason']} | "
              f"reasoning={len(r['reasoning'])}c | content={len(r['content'])}c", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results.append({"case": name, "max_tokens": mt, "error": str(e)})

with open(OUT, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(results)} samples to {OUT}", flush=True)
