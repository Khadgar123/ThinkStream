#!/usr/bin/env python3
"""Hold CUDA GPUs after a long eval matrix finishes.

This is intentionally simple: one child process per requested GPU allocates a
large CUDA tensor and then sleeps until interrupted.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def _parse_gpus(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def _child_main(args: argparse.Namespace) -> int:
    import torch

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    target = int(free_bytes * args.fraction)
    chunk_bytes = args.chunk_mib * 1024 * 1024

    tensors = []
    allocated = 0
    while allocated + chunk_bytes <= target:
        tensors.append(torch.empty((chunk_bytes,), dtype=torch.uint8, device="cuda"))
        allocated += chunk_bytes

    if allocated == 0 and target > 0:
        tensors.append(torch.empty((target,), dtype=torch.uint8, device="cuda"))
        allocated = target

    print(
        f"[hold] gpu={gpu} allocated={allocated / (1024 ** 3):.1f}GiB "
        f"free_before={free_bytes / (1024 ** 3):.1f}GiB "
        f"total={total_bytes / (1024 ** 3):.1f}GiB",
        flush=True,
    )
    while True:
        time.sleep(args.sleep_sec)


def _parent_main(args: argparse.Namespace) -> int:
    procs: list[subprocess.Popen] = []
    for gpu in _parse_gpus(args.gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    "--fraction",
                    str(args.fraction),
                    "--chunk-mib",
                    str(args.chunk_mib),
                    "--sleep-sec",
                    str(args.sleep_sec),
                ],
                env=env,
            )
        )

    def stop(_signum, _frame):
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(f"[hold] started {len(procs)} GPU holder processes: {args.gpus}", flush=True)
    while True:
        for proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"GPU holder exited early with code {proc.returncode}")
        time.sleep(args.sleep_sec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0 1 2 3 4 5 6 7")
    parser.add_argument("--fraction", type=float, default=0.85)
    parser.add_argument("--chunk-mib", type=int, default=512)
    parser.add_argument("--sleep-sec", type=int, default=60)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        return _child_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
