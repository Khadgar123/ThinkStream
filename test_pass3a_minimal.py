"""Minimal reproduction of pass3a stuck issue."""
import asyncio
import json
import sys
import time

sys.path.insert(0, ".")

from scripts.agent_data_v5.config import EVIDENCE_1B_DIR
from scripts.agent_data_pipeline.vllm_client import VLLMClient

async def main():
    print("=== Start minimal test ===", flush=True)

    # 1. Load evidence for first video
    evidence_files = sorted(EVIDENCE_1B_DIR.glob("*.json"))
    evidence_files = [f for f in evidence_files if f.name != "_version"]
    print(f"Evidence files: {len(evidence_files)}", flush=True)

    with open(evidence_files[0], "r") as f:
        evidence = json.load(f)
    print(f"Loaded evidence for {evidence_files[0].stem}, chunks: {len(evidence)}", flush=True)

    # 2. Create client (same as pipeline)
    client = VLLMClient(
        api_base="http://10.16.12.175:8000/v1",
        model="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8",
        max_concurrent=512,
    )
    print("Client created", flush=True)

    # 3. Test single API call
    t0 = time.time()
    print("About to call API...", flush=True)
    result = await client._call_one(
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=5,
        request_id="test_0",
    )
    print(f"API call done in {time.time()-t0:.1f}s: {result!r}", flush=True)

    # 4. Import classify_chunks and test
    from scripts.agent_data_v5.pass3a_cards import classify_chunks
    t0 = time.time()
    print("About to classify_chunks...", flush=True)
    fc = classify_chunks(evidence)
    print(f"classify_chunks done in {time.time()-t0:.1f}s, families: {list(fc.keys())}", flush=True)

    # 5. Import _format_evidence_for_prompt and test
    from scripts.agent_data_v5.pass3a_cards import _format_evidence_for_prompt
    for family, chunks in fc.items():
        if not chunks:
            continue
        t0 = time.time()
        ev_text = _format_evidence_for_prompt(evidence, chunks)
        print(f"_format_evidence_for_prompt {family} done in {time.time()-t0:.1f}s, len={len(ev_text)}", flush=True)
        break  # test one family only

    # 6. Test generate_cards for one video
    from scripts.agent_data_v5.pass3a_cards import generate_cards
    t0 = time.time()
    print("About to generate_cards...", flush=True)
    cards = await generate_cards(evidence_files[0].stem, evidence, client)
    print(f"generate_cards done in {time.time()-t0:.1f}s, cards: {len(cards)}", flush=True)

    print("=== All tests passed ===", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
