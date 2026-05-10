#!/usr/bin/env python
"""Hold CUDA memory after long jobs finish.

This is intentionally simple: allocate a configurable fraction of currently
free memory on each requested GPU, then sleep forever with heartbeat logs.
"""

import argparse
import time

import torch


def _parse_gpus(value: str):
    if value.strip().lower() in {"", "all"}:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="all")
    ap.add_argument("--fraction", type=float, default=0.90)
    ap.add_argument("--reserve-gb", type=float, default=6.0)
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--heartbeat-sec", type=int, default=300)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gpus = _parse_gpus(args.gpus)
    allocations = {}
    reserve = int(args.reserve_gb * 1024**3)
    chunk = int(args.chunk_mb * 1024**2)

    print(
        f"[hold_gpus] gpus={gpus} fraction={args.fraction} "
        f"reserve_gb={args.reserve_gb} chunk_mb={args.chunk_mb}",
        flush=True,
    )
    for gpu in gpus:
        torch.cuda.set_device(gpu)
        free, total = torch.cuda.mem_get_info(gpu)
        used = total - free
        target_used = int(total * args.fraction)
        alloc_bytes = max(0, min(free - reserve, target_used - used))
        chunks = []
        allocated = 0
        while allocated + chunk <= alloc_bytes:
            try:
                chunks.append(torch.empty(chunk, dtype=torch.uint8, device=f"cuda:{gpu}"))
                allocated += chunk
            except torch.cuda.OutOfMemoryError:
                break
        allocations[gpu] = chunks
        free_after, total_after = torch.cuda.mem_get_info(gpu)
        print(
            f"[hold_gpus] gpu={gpu} allocated={allocated / 1024**3:.2f}GiB "
            f"free_after={free_after / 1024**3:.2f}GiB total={total_after / 1024**3:.2f}GiB",
            flush=True,
        )

    while True:
        status = []
        for gpu in gpus:
            free, total = torch.cuda.mem_get_info(gpu)
            status.append(f"gpu{gpu}:free={free / 1024**3:.2f}GiB")
        print("[hold_gpus] heartbeat " + " ".join(status), flush=True)
        time.sleep(args.heartbeat_sec)


if __name__ == "__main__":
    main()
