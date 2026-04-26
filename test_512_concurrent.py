"""Test 512 concurrent requests to vLLM + check KV cache pressure."""
import asyncio
import time
import sys
import json

sys.path.insert(0, ".")
from scripts.agent_data_pipeline.vllm_client import VLLMClient

async def get_vllm_metrics():
    """Fetch vLLM metrics from /metrics endpoint."""
    import httpx
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get("http://10.16.12.175:8000/metrics", timeout=5)
            text = r.text
            # Extract key metrics
            metrics = {}
            for line in text.splitlines():
                if "vllm:gpu_cache_usage_perc" in line and not line.startswith("#"):
                    metrics["gpu_cache_usage"] = line.split()[-1]
                elif "vllm:cpu_cache_usage_perc" in line and not line.startswith("#"):
                    metrics["cpu_cache_usage"] = line.split()[-1]
                elif "vllm:num_requests_running" in line and not line.startswith("#"):
                    metrics["running"] = line.split()[-1]
                elif "vllm:num_requests_waiting" in line and not line.startswith("#"):
                    metrics["waiting"] = line.split()[-1]
                elif "vllm:prompt_tokens_total" in line and not line.startswith("#"):
                    metrics["prompt_tokens_total"] = line.split()[-1]
                elif "vllm:generation_tokens_total" in line and not line.startswith("#"):
                    metrics["generation_tokens_total"] = line.split()[-1]
            return metrics
    except Exception as e:
        return {"error": str(e)}

async def main():
    client = VLLMClient(
        api_base="http://10.16.12.175:8000/v1",
        model="/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8",
        max_concurrent=512,
    )

    N = 512
    print(f"Sending {N} concurrent requests (max_tokens=5)...", flush=True)

    # Snapshot before
    before = await get_vllm_metrics()
    print(f"Before: {json.dumps(before, indent=2)}", flush=True)

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

    # Snapshot after
    after = await get_vllm_metrics()
    print(f"After:  {json.dumps(after, indent=2)}", flush=True)

    success = sum(1 for r in results if isinstance(r, str))
    errors = [r for r in results if isinstance(r, Exception)]
    none_count = sum(1 for r in results if r is None)

    print(f"\nTotal time: {elapsed:.1f}s", flush=True)
    print(f"Success(content): {success}, Exceptions: {len(errors)}, None: {none_count}", flush=True)
    print(f"Avg wall-clock latency: {elapsed/N*1000:.1f}ms", flush=True)
    print(f"Throughput: {N/elapsed:.1f} req/s", flush=True)
    print(f"Client stats: completed={client.stats.completed}, failed={client.stats.failed}", flush=True)

    if errors:
        for i, e in enumerate(errors[:3]):
            print(f"  Exception[{i}]: {type(e).__name__}: {e}", flush=True)

    # Now test with a larger payload similar to pass3a
    print("\n--- Now testing with pass3a-like large prompt (max_tokens=16384) ---", flush=True)
    large_prompt = "Describe the following video evidence in detail. " * 200  # ~1400 tokens approx

    before2 = await get_vllm_metrics()
    print(f"Before large: {json.dumps(before2, indent=2)}", flush=True)

    t0 = time.time()
    tasks2 = [
        client._call_one(
            messages=[{"role": "user", "content": large_prompt}],
            max_tokens=16384,
            request_id=f"large_{i}",
        )
        for i in range(512)
    ]
    results2 = await asyncio.gather(*tasks2, return_exceptions=True)
    elapsed2 = time.time() - t0

    after2 = await get_vllm_metrics()
    print(f"After large:  {json.dumps(after2, indent=2)}", flush=True)

    success2 = sum(1 for r in results2 if isinstance(r, str))
    errors2 = [r for r in results2 if isinstance(r, Exception)]
    none2 = sum(1 for r in results2 if r is None)

    print(f"\nTotal time (large): {elapsed2:.1f}s", flush=True)
    print(f"Success(content): {success2}, Exceptions: {len(errors2)}, None: {none2}", flush=True)
    print(f"Throughput: {512/elapsed2:.1f} req/s", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
