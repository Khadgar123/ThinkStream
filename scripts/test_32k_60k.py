"""32k和60k × 64并发纯文本测试"""
import asyncio, aiohttp, json, time

API = "http://10.16.12.175:8000/v1/chat/completions"
MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8"

evidence = json.load(open("data/agent_v5_test/evidence_graph/0ySyffWMAVI.json"))
prev_lines = []
for cap in evidence[:20]:
    t = cap["time"]
    ents = [e["id"] for e in cap.get("visible_entities", []) if isinstance(e, dict)]
    prev_lines.append(f"[{t[0]}-{t[1]}] entities: {ents}")
PREV_TEXT = "\n".join(prev_lines[-30:])

PROMPT = f"""You are annotating a streaming video chunk by chunk for training data construction.

Previous captions for context:
{PREV_TEXT}

Current chunk: t=40-42s (frames above).

Output a STRICT JSON object with these fields:
{{
  "time": [40, 42],
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
- Only include what is VISIBLE in the current chunk frames (t=40-42s)
- Maintain consistent entity IDs across chunks
- Output JSON only:"""

async def stream_one(session, max_tokens):
    t0 = time.time()
    reasoning_parts, content_parts, finish = [], [], None
    try:
        async with session.post(API, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0.3, "stream": True,
        }, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
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

async def run_test(mt, conc, label):
    print(f"\n[{label}] mt={mt} c={conc} ...", flush=True)
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(conc)
        async def bounded(idx):
            await asyncio.sleep(idx * 0.3)
            async with sem:
                return await stream_one(session, mt)
        t0 = time.time()
        batch = await asyncio.gather(*[bounded(i) for i in range(conc)])
        total = round(time.time() - t0, 1)

    ok = [r for r in batch if "error" not in r]
    errs = [r for r in batch if "error" in r]
    fmap = {}
    for r in ok:
        f = r.get("finish_reason", "unknown")
        fmap[f] = fmap.get(f, 0) + 1
    avg_r = round(sum(len(r["reasoning"]) for r in ok) / max(len(ok), 1))
    avg_c = round(sum(len(r["content"]) for r in ok) / max(len(ok), 1))
    has_c = sum(1 for r in ok if r["content"])

    print(f"  total: {total}s | ok={len(ok)} err={len(errs)}", flush=True)
    print(f"  finish: {fmap}", flush=True)
    print(f"  avg_reasoning={avg_r}c | avg_content={avg_c}c | has_content={has_c}/{len(ok)}", flush=True)
    if errs:
        print(f"  errors: {[r['error'][:100] for r in errs[:3]]}", flush=True)

    return {
        "case": label, "max_tokens": mt, "concurrency": conc, "mode": "text",
        "total_time_sec": total,
        "num_success": len(ok), "num_error": len(errs),
        "finish_counts": fmap,
        "avg_reasoning_chars": avg_r, "avg_content_chars": avg_c,
        "has_content_count": has_c,
        "sample_outputs": [batch[0], batch[len(batch)//2], batch[-1]],
        "errors_sample": [r["error"][:200] for r in errs][:3] if errs else [],
    }

async def main():
    results = []
    results.append(await run_test(32768, 64, "text_mt32768_c64"))
    results.append(await run_test(60000, 64, "text_mt60000_c64"))

    with open("data/agent_v5_test/raw_io_text_32k_60k_c64.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Saved to data/agent_v5_test/raw_io_text_32k_60k_c64.json", flush=True)

asyncio.run(main())
