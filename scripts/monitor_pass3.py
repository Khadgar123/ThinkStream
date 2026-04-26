#!/usr/bin/env python3
"""Monitor pass3 pipeline and vLLM status."""
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

VLLM_METRICS_URL = "http://10.16.12.175:8000/metrics"
LOG_FILE = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5/pipeline_pass3_run.log")
STREAM_FILE = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5/audits/pass3a_stream.jsonl")
SAMPLE_INTERVAL = 10  # seconds

async def get_vllm_metrics():
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(VLLM_METRICS_URL, timeout=5) as resp:
                text = await resp.text()
                metrics = {}
                for line in text.splitlines():
                    if line.startswith("#"):
                        continue
                    if "vllm:gpu_cache_usage_perc" in line:
                        metrics["gpu_cache"] = float(line.split()[-1]) * 100
                    elif "vllm:num_requests_running" in line:
                        metrics["running"] = int(float(line.split()[-1]))
                    elif "vllm:num_requests_waiting" in line:
                        metrics["waiting"] = int(float(line.split()[-1]))
                    elif "vllm:prompt_tokens_total" in line:
                        metrics["prompt_tokens_total"] = line.split()[-1]
                    elif "vllm:generation_tokens_total" in line:
                        metrics["generation_tokens_total"] = line.split()[-1]
                return metrics
    except Exception as e:
        return {"error": str(e)}

def get_pass3_progress():
    ok_count = 0
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            ok_count = sum(1 for line in f if "200 OK" in line)
    stream_count = 0
    if STREAM_FILE.exists():
        with open(STREAM_FILE, "r") as f:
            stream_count = sum(1 for line in f if line.strip())
    return ok_count, stream_count

async def main():
    print("=" * 90)
    print(f"{'Time':<12} {'vLLM running':>12} {'waiting':>10} {'gpu%':>8} | {'pass3a OK':>10} {'stream':>8} | {'rate':>8} {'est_left':>10}")
    print("=" * 90)

    prev_ok = 0
    prev_time = time.time()

    # Estimate total requests for pass3a: 320 videos * ~10 families = 3200
    TOTAL_PASS3A_REQS = 3200

    while True:
        metrics = await get_vllm_metrics()
        ok_count, stream_count = get_pass3_progress()
        now = time.time()

        running = metrics.get("running", "?")
        waiting = metrics.get("waiting", "?")
        gpu = metrics.get("gpu_cache", "?")
        if isinstance(gpu, float):
            gpu = f"{gpu:.1f}"

        # Rate calculation
        elapsed = now - prev_time
        rate = (ok_count - prev_ok) / elapsed if elapsed > 0 else 0

        # Estimate remaining time for pass3a
        if rate > 0:
            est_left_sec = (TOTAL_PASS3A_REQS - ok_count) / rate
            est_left = f"{est_left_sec/60:.0f}m"
        else:
            est_left = "N/A"

        t = datetime.now().strftime("%H:%M:%S")
        print(f"{t:<12} {running:>12} {waiting:>10} {gpu:>8} | {ok_count:>10} {stream_count:>8} | {rate:>7.1f}/s {est_left:>10}", flush=True)

        prev_ok = ok_count
        prev_time = now
        await asyncio.sleep(SAMPLE_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
