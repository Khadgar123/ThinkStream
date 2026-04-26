"""
Qwen3.5-397B 造数据 pipeline 真实场景原始输入输出采集。
用 0ySyffWMAVI 真实帧和 evidence 数据。
不做任何解析，完整记录原始 reasoning + content。

测试矩阵：
  单请求: evidence × {text, vision} × {4k, 8k, 16k} = 6
  单请求: observation(/no_think) × {text, vision} × 4k = 2
  并发:   evidence × text × 4k × {32, 64} = 2
  并发:   evidence × vision × 4k × {32, 64} = 2
  共 12 cases
"""
import asyncio
import aiohttp
import base64
import json
import time
from pathlib import Path

API = "http://10.16.12.175:8000/v1/chat/completions"
MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"
FRAMES_DIR = Path("data/agent_v5_test/frames/0ySyffWMAVI")
EVIDENCE_FILE = Path("data/agent_v5_test/evidence_graph/0ySyffWMAVI.json")
OUT = Path("data/agent_v5_test/raw_io_full_test.json")

# ============================================================
# 加载真实数据
# ============================================================
evidence = json.load(open(EVIDENCE_FILE))
frames_sorted = sorted(FRAMES_DIR.glob("frame_*.jpg"))

def img_b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

# chunk 20 的 24 帧滑动窗口 (chunk 9-20, 每 chunk 2 帧)
TARGET_CHUNK = 20
WINDOW_START = max(0, TARGET_CHUNK - 12 + 1)  # chunk 9
vision_frames_b64 = []
for c in range(WINDOW_START, TARGET_CHUNK + 1):
    for fi in range(2):
        idx = c * 2 + fi
        if idx < len(frames_sorted):
            vision_frames_b64.append(img_b64(str(frames_sorted[idx])))

# 前 20 条真实 caption context
prev_lines = []
for cap in evidence[:TARGET_CHUNK]:
    t = cap["time"]
    ents = [e["id"] for e in cap.get("visible_entities", []) if isinstance(e, dict)]
    prev_lines.append(f"[{t[0]}-{t[1]}] entities: {ents}")
PREV_TEXT = "\n".join(prev_lines[-30:])

# 真实 observation 历史
real_obs = []
for cap in evidence[10:TARGET_CHUNK]:
    raw = cap.get("_raw", "")
    lines = raw.split("\n")
    desc_parts = []
    for line in lines:
        line = line.strip()
        if line.startswith("- **Frame") or line.startswith("- The frames") or line.startswith("- I have"):
            desc_parts.append(line.lstrip("- ").strip("*").strip())
    desc = " ".join(desc_parts)[:150] if desc_parts else f"Scene continues at t={cap['time'][0]}-{cap['time'][1]}s."
    real_obs.append({"chunk": cap["chunk_idx"], "time": f"{cap['time'][0]}-{cap['time'][1]}", "text": desc})

# ============================================================
# 真实 pipeline prompts
# ============================================================
START, END = TARGET_CHUNK * 2, TARGET_CHUNK * 2 + 2

EVIDENCE_PROMPT = f"""You are annotating a streaming video chunk by chunk for training data construction.

Previous captions for context:
{PREV_TEXT}

Current chunk: t={START}-{END}s (frames above).

Output a STRICT JSON object with these fields:
{{
  "time": [{START}, {END}],
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
- Only include what is VISIBLE in the current chunk frames (t={START}-{END}s)
- support_level: "direct_current_chunk" = visible in current 2s frames;
  "carried_from_previous" = entity/attribute inherited from prior context;
  "inferred" = not directly visible, deduced from context
- target_resolution_visible: false if the detail would be too small/blurry at training resolution
- confidence < 0.7 for uncertain observations (small text, fast motion, partial occlusion)
- not_observable: list anything you'd normally say but can't actually SEE
- Maintain consistent entity IDs across chunks (chef_1, pot_1, etc.)
- Do NOT attribute facts from prior chunks to the current chunk

Output JSON only:"""

compressed_mem = '<compressed>{{"time_range":[0,20],"text":"' + \
    " ".join(o["text"][:50] for o in real_obs[:5]) + '"}}</compressed>'
recent_thinks = "\n".join(f'[{o["time"]}] {o["text"]}' for o in real_obs[-10:])
W_START = WINDOW_START * 2
W_END = (TARGET_CHUNK + 1) * 2

OBSERVATION_PROMPT = f"""/no_think

You are a streaming video agent generating a think (incremental visual memory note).

Compressed memory:
{compressed_mem}

Recent thinks:
{recent_thinks}

Visual window: t={W_START}-{W_END}s (frames above).

Describe what is NEW or CHANGED in the latest 2 seconds (t={START}-{END}s) in 40-60 tokens.

Rules:
- Only observable visual facts
- Maintain entity names from memory (e.g., keep "chef_1" consistent)
- Focus: entities+attributes, actions, state changes, OCR, spatial
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- If nothing new: brief ongoing state

Output (one paragraph, 40-60 tokens):"""

# ============================================================
# 工具函数
# ============================================================

def make_messages(prompt, vision):
    if vision:
        content = [{"type": "image_url", "image_url": {"url": b}} for b in vision_frames_b64]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": prompt}]

def input_log(messages):
    out = []
    for msg in messages:
        if isinstance(msg["content"], str):
            out.append({"role": msg["role"], "content": msg["content"]})
        else:
            items = []
            for item in msg["content"]:
                if item["type"] == "text":
                    items.append(item)
                else:
                    items.append({"type": "image_url", "note": f"(base64 omitted, {len(vision_frames_b64)} frames)"})
            out.append({"role": msg["role"], "content": items})
    return out

async def stream_one(session, messages, max_tokens, temp=0.3):
    t0 = time.time()
    reasoning_parts, content_parts, finish = [], [], None
    try:
        async with session.post(API, json={
            "model": MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": temp, "stream": True,
        }, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
            async for line in resp.content:
                s = line.decode("utf-8").strip()
                if not s.startswith("data: "): continue
                d_str = s[6:]
                if d_str == "[DONE]": break
                try:
                    d = json.loads(d_str)
                    delta = d["choices"][0].get("delta", {})
                    r = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                    c = delta.get("content", "")
                    if r: reasoning_parts.append(r)
                    if c: content_parts.append(c)
                    f = d["choices"][0].get("finish_reason")
                    if f: finish = f
                except: pass
    except Exception as e:
        return {"error": str(e), "elapsed_sec": round(time.time() - t0, 1)}
    return {
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "finish_reason": finish,
        "elapsed_sec": round(time.time() - t0, 1),
    }

async def run_batch(messages, max_tokens, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def bounded(idx):
            await asyncio.sleep(idx * 0.5)  # 每个请求间隔 0.5s 逐步发出
            async with sem:
                return await stream_one(session, messages, max_tokens)
        return await asyncio.gather(*[bounded(i) for i in range(concurrency)])

# ============================================================
# 主流程
# ============================================================

async def check_api():
    """确认 API 可用"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"http://10.16.12.175:8000/v1/models", timeout=aiohttp.ClientTimeout(total=10)) as r:
                d = await r.json()
                print(f"API OK: {d['data'][0]['id']}", flush=True)
                return True
    except Exception as e:
        print(f"API DOWN: {e}", flush=True)
        return False

async def main():
    if not await check_api():
        print("Aborting: API not available", flush=True)
        return

    results = []

    print(f"Video: 0ySyffWMAVI | chunk: {TARGET_CHUNK} | vision frames: {len(vision_frames_b64)}", flush=True)
    print(f"Evidence prompt: {len(EVIDENCE_PROMPT)} chars | Observation prompt: {len(OBSERVATION_PROMPT)} chars", flush=True)
    print(flush=True)

    # Part 1: 并发优先 (4 cases) — 最有价值的数据先拿
    concurrent = [
        ("concurrent_vision_mt4096_c64", EVIDENCE_PROMPT, True,  4096, 64),
        ("concurrent_vision_mt4096_c32", EVIDENCE_PROMPT, True,  4096, 32),
        ("concurrent_text_mt4096_c64",   EVIDENCE_PROMPT, False, 4096, 64),
        ("concurrent_text_mt4096_c32",   EVIDENCE_PROMPT, False, 4096, 32),
    ]

    print("=" * 60, flush=True)
    print(f"PART 1: 并发测试 ({len(concurrent)} cases)", flush=True)
    print("=" * 60, flush=True)

    for name, prompt, vision, mt, conc in concurrent:
        print(f"  [{name}] {conc} reqs ...", flush=True)
        msgs = make_messages(prompt, vision)
        t0 = time.time()
        batch = await run_batch(msgs, mt, conc)
        total = round(time.time() - t0, 1)

        ok = [r for r in batch if "error" not in r]
        errs = [r for r in batch if "error" in r]
        fmap = {}
        for r in ok:
            f = r.get("finish_reason", "unknown")
            fmap[f] = fmap.get(f, 0) + 1

        results.append({
            "case": name, "max_tokens": mt, "concurrency": conc,
            "mode": "vision" if vision else "text",
            "num_frames": len(vision_frames_b64) if vision else 0,
            "total_time_sec": total,
            "num_success": len(ok), "num_error": len(errs),
            "finish_counts": fmap,
            "avg_reasoning_chars": round(sum(len(r["reasoning"]) for r in ok) / max(len(ok), 1)),
            "avg_content_chars": round(sum(len(r["content"]) for r in ok) / max(len(ok), 1)),
            "has_content_count": sum(1 for r in ok if r["content"]),
            "input": input_log(msgs),
            "sample_outputs": [batch[0], batch[len(batch)//2], batch[-1]],
            "errors_sample": [r["error"][:200] for r in errs][:3] if errs else [],
        })
        print(f"    {total}s | ok={len(ok)} err={len(errs)} | finish={fmap} | "
              f"avg_reasoning={results[-1]['avg_reasoning_chars']}c | "
              f"has_content={results[-1]['has_content_count']}/{len(ok)}", flush=True)

    # Part 2: 单请求 — 长 thinking 场景优先 (8 cases)
    single = [
        ("evidence_text_mt16384",     EVIDENCE_PROMPT,    False, 16384),
        ("evidence_vision_mt16384",   EVIDENCE_PROMPT,    True,  16384),
        ("evidence_text_mt8192",      EVIDENCE_PROMPT,    False, 8192),
        ("evidence_vision_mt8192",    EVIDENCE_PROMPT,    True,  8192),
        ("evidence_text_mt4096",      EVIDENCE_PROMPT,    False, 4096),
        ("evidence_vision_mt4096",    EVIDENCE_PROMPT,    True,  4096),
        ("observation_text_mt4096",   OBSERVATION_PROMPT, False, 4096),
        ("observation_vision_mt4096", OBSERVATION_PROMPT, True,  4096),
    ]

    print(f"\n{'=' * 60}", flush=True)
    print(f"PART 2: 单请求 ({len(single)} cases)", flush=True)
    print("=" * 60, flush=True)

    async with aiohttp.ClientSession() as session:
        for name, prompt, vision, mt in single:
            print(f"  [{name}] ...", end=" ", flush=True)
            msgs = make_messages(prompt, vision)
            r = await stream_one(session, msgs, mt)
            results.append({
                "case": name, "max_tokens": mt, "concurrency": 1,
                "mode": "vision" if vision else "text",
                "num_frames": len(vision_frames_b64) if vision else 0,
                "input": input_log(msgs), "output": r,
            })
            if "error" in r:
                print(f"ERROR: {r['error'][:80]}", flush=True)
            else:
                print(f"{r['elapsed_sec']}s | finish={r['finish_reason']} | "
                      f"reasoning={len(r['reasoning'])}c | content={len(r['content'])}c", flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(results)} entries -> {OUT}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
