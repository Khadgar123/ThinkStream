#!/usr/bin/env python3
"""Monitor pass3 pipeline with per-request timing and progress estimation."""
import asyncio
import json
import time
import re
from pathlib import Path
from datetime import datetime

VLLM_METRICS_URL = "http://10.16.12.175:8000/metrics"
LOG_FILE = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5/pipeline_pass3_run.log")
STREAM_FILE = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5/audits/pass3a_stream.jsonl")
TASK_CARDS_DIR = Path("/home/tione/notebook/gaozhenkun/hzh/ThinkStream/data/agent_v5/task_cards")
SAMPLE_INTERVAL = 15  # seconds

# Estimate total requests for pass3a based on typical family distribution
# 320 videos, each with ~10 families on average
TOTAL_VIDEOS = 320
AVG_FAMILIES_PER_VIDEO = 10
ESTIMATED_TOTAL_REQUESTS = TOTAL_VIDEOS * AVG_FAMILIES_PER_VIDEO

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
                    if "vllm:num_requests_running" in line:
                        metrics["running"] = int(float(line.split()[-1]))
                    elif "vllm:num_requests_waiting" in line:
                        metrics["waiting"] = int(float(line.split()[-1]))
                    elif "vllm:prompt_tokens_total" in line:
                        metrics["prompt_tokens_total"] = float(line.split()[-1])
                    elif "vllm:generation_tokens_total" in line:
                        metrics["generation_tokens_total"] = float(line.split()[-1])
                return metrics
    except Exception as e:
        return {"error": str(e)}

def parse_log_progress():
    """Parse pipeline log to extract per-request completion times."""
    if not LOG_FILE.exists():
        return 0, 0, []

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    ok_count = 0
    ok_times = []
    error_count = 0

    for line in lines:
        # Match httpx 200 OK lines
        if '"HTTP/1.1 200 OK"' in line:
            ok_count += 1
            # Extract timestamp
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),', line)
            if match:
                ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                ok_times.append(ts)
        elif 'failed after' in line:
            error_count += 1

    return ok_count, error_count, ok_times

def get_task_cards_count():
    """Count completed task cards (videos that finished pass3a)."""
    if not TASK_CARDS_DIR.exists():
        return 0
    return len([f for f in TASK_CARDS_DIR.iterdir() if f.suffix == ".json"])

def get_stream_count():
    if not STREAM_FILE.exists():
        return 0
    with open(STREAM_FILE, "r") as f:
        return sum(1 for line in f if line.strip())

async def main():
    print("=" * 110)
    print(f"{'Time':<10} {'vLLM run':>8} {'wait':>6} {'PromptTok':>10} {'GenTok':>10} | {'OK reqs':>8} {'Errors':>6} {'Cards':>6} {'Stream':>6} | {'Rate':>7} {'AvgLat':>8} {'Prog%':>7} {'EstLeft':>10}")
    print("=" * 110)

    prev_ok = 0
    prev_prompt_tokens = 0
    prev_gen_tokens = 0
    prev_time = time.time()
    first_ok_time = None

    while True:
        metrics = await get_vllm_metrics()
        ok_count, error_count, ok_times = parse_log_progress()
        cards_count = get_task_cards_count()
        stream_count = get_stream_count()
        now = time.time()

        running = metrics.get("running", "?")
        waiting = metrics.get("waiting", "?")
        prompt_tokens = metrics.get("prompt_tokens_total", 0)
        gen_tokens = metrics.get("generation_tokens_total", 0)

        # Rate calculations
        elapsed = now - prev_time
        ok_rate = (ok_count - prev_ok) / elapsed if elapsed > 0 else 0
        prompt_rate = (prompt_tokens - prev_prompt_tokens) / elapsed if elapsed > 0 else 0
        gen_rate = (gen_tokens - prev_gen_tokens) / elapsed if elapsed > 0 else 0

        # Average latency per request (based on time window)
        avg_latency = None
        if ok_times and len(ok_times) >= 2:
            # Look at the last batch of completions in this window
            recent_times = [t for t in ok_times if (now - t.timestamp()) < elapsed * 2]
            if len(recent_times) >= 2:
                time_span = (recent_times[-1] - recent_times[0]).total_seconds()
                avg_latency = time_span / max(len(recent_times) - 1, 1)

        # Progress percentage
        progress_pct = (ok_count / ESTIMATED_TOTAL_REQUESTS) * 100 if ESTIMATED_TOTAL_REQUESTS > 0 else 0

        # Estimate remaining time
        if ok_rate > 0 and ok_count > 0:
            est_left_sec = (ESTIMATED_TOTAL_REQUESTS - ok_count) / ok_rate
            est_left = f"{est_left_sec/60:.0f}m"
        elif ok_count == 0:
            est_left = "warming"
        else:
            est_left = "N/A"

        t = datetime.now().strftime("%H:%M:%S")
        avg_lat_str = f"{avg_latency:.1f}s" if avg_latency else "N/A"

        print(f"{t:<10} {running:>8} {waiting:>6} {prompt_tokens:>10.0f} {gen_tokens:>10.0f} | {ok_count:>8} {error_count:>6} {cards_count:>6} {stream_count:>6} | {ok_rate:>6.1f}/s {avg_lat_str:>8} {progress_pct:>6.1f}% {est_left:>10}", flush=True)

        prev_ok = ok_count
        prev_prompt_tokens = prompt_tokens
        prev_gen_tokens = gen_tokens
        prev_time = now

        await asyncio.sleep(SAMPLE_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
