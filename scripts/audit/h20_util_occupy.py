#!/usr/bin/env python3
"""Keep selected GPUs busy with simple repeated GEMM workloads."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time


def parse_devices(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


GIB = 1024**3


def adjust_memory_filler(
    filler: list[object],
    target_bytes: int,
    chunk_bytes: int,
) -> int:
    import torch

    current_bytes = sum(tensor.numel() * tensor.element_size() for tensor in filler)
    while current_bytes + chunk_bytes <= target_bytes:
        filler.append(torch.empty((chunk_bytes,), device="cuda", dtype=torch.uint8))
        current_bytes += chunk_bytes

    while filler and current_bytes - filler[-1].numel() * filler[-1].element_size() >= target_bytes:
        tensor = filler.pop()
        current_bytes -= tensor.numel() * tensor.element_size()
        del tensor
    torch.cuda.empty_cache()
    return current_bytes


def worker(
    device: int,
    size: int,
    dtype_name: str,
    sync_every: int,
    mem_min_gib: float,
    mem_max_gib: float,
    mem_step_gib: float,
    mem_period_seconds: float,
    mem_chunk_gib: float,
    stop_event: mp.Event,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[dtype_name]

    a = torch.randn((size, size), device="cuda", dtype=dtype)
    b = torch.randn((size, size), device="cuda", dtype=dtype)
    c = torch.empty((size, size), device="cuda", dtype=dtype)
    torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    filler: list[object] = []
    target_gib = mem_min_gib
    direction = 1.0
    chunk_bytes = max(1, int(mem_chunk_gib * GIB))
    filled_bytes = adjust_memory_filler(filler, int(target_gib * GIB), chunk_bytes)
    next_memory_update = time.time() + mem_period_seconds
    print(
        f"gpu {device}: running size={size} dtype={dtype_name} filler={filled_bytes / GIB:.1f}GiB",
        flush=True,
    )

    steps = 0
    while not stop_event.is_set():
        torch.matmul(a, b, out=c)
        steps += 1
        if steps % sync_every == 0:
            torch.cuda.synchronize()
        if mem_period_seconds > 0 and time.time() >= next_memory_update:
            target_gib += direction * mem_step_gib
            if target_gib >= mem_max_gib:
                target_gib = mem_max_gib
                direction = -1.0
            elif target_gib <= mem_min_gib:
                target_gib = mem_min_gib
                direction = 1.0
            filled_bytes = adjust_memory_filler(filler, int(target_gib * GIB), chunk_bytes)
            print(
                f"gpu {device}: target={target_gib:.1f}GiB filler={filled_bytes / GIB:.1f}GiB",
                flush=True,
            )
            next_memory_update = time.time() + mem_period_seconds
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("0,1,2,3,4,5,6,7"))
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32", "tf32"), default="bf16")
    parser.add_argument("--sync-every", type=int, default=16)
    parser.add_argument("--mem-min-gib", type=float, default=70.0)
    parser.add_argument("--mem-max-gib", type=float, default=90.0)
    parser.add_argument("--mem-step-gib", type=float, default=5.0)
    parser.add_argument("--mem-period-seconds", type=float, default=30.0)
    parser.add_argument("--mem-chunk-gib", type=float, default=1.0)
    args = parser.parse_args()

    stop_event = mp.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(
        "occupying devices={devices} size={size} dtype={dtype} mem={mem_min:.1f}-{mem_max:.1f}GiB".format(
            devices=",".join(str(device) for device in args.devices),
            size=args.size,
            dtype=args.dtype,
            mem_min=args.mem_min_gib,
            mem_max=args.mem_max_gib,
        ),
        flush=True,
    )
    processes = [
        mp.Process(
            target=worker,
            args=(
                device,
                args.size,
                args.dtype,
                args.sync_every,
                args.mem_min_gib,
                args.mem_max_gib,
                args.mem_step_gib,
                args.mem_period_seconds,
                args.mem_chunk_gib,
                stop_event,
            ),
        )
        for device in args.devices
    ]
    for process in processes:
        process.start()

    try:
        while not stop_event.is_set():
            alive = sum(1 for process in processes if process.is_alive())
            print(f"heartbeat: {alive}/{len(processes)} workers alive", flush=True)
            time.sleep(60)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
