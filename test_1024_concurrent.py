"""Test 1024 concurrent requests to vLLM."""
import asyncio
import time
import sys

sys.path.insert(0, ".")
from scripts.agent_data_pipeline.vllm_client import VLLMClient

async def main():
    client = VLLMClient(
        api_base="http://10.16.12.175:8000/v1",
        model="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8",
        max_concurrent=1024,
    )

    N = 1024
    print(f"Sending {N} concurrent requests (max_tokens=5)...", flush=True)
    t0 = time.time()

    tasks = [
        client._call_one(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            request_id=f"test_{i}",
        )
        for i in range(N)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - t0
    success = sum(1 for r in results if isinstance(r, str) and r is not None)
    errors = [r for r in results if isinstance(r, Exception)]
    none_count = sum(1 for r in results if r is None)

    print(f"Total time: {elapsed:.1f}s", flush=True)
    print(f"Success: {success}, Errors: {len(errors)}, None: {none_count}", flush=True)
    print(f"Avg latency: {elapsed/N*1000:.1f}ms per req (wall clock)", flush=True)
    print(f"Throughput: {N/elapsed:.1f} req/s", flush=True)

    # Print first few error details
    if errors:
        print(f"\nFirst 3 exceptions:", flush=True)
        for i, e in enumerate(errors[:3]):
            print(f"  [{i}] {type(e).__name__}: {e}", flush=True)
    if none_count > 0 and not errors:
        print(f"\nAll {none_count} requests returned None (failed after retries).", flush=True)
        # Check client stats
        print(f"Client stats: completed={client.stats.completed}, failed={client.stats.failed}", flush=True)
        if client.stats.errors:
            print(f"First 3 error messages:", flush=True)
            for e in client.stats.errors[:3]:
                print(f"  {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
