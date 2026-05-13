#!/usr/bin/env python3
"""Probe 8-GPU H20 power draw with repeated GEMM workloads."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PowerSample:
    timestamp: float
    watts: list[float]
    utils: list[float]


def parse_devices(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_sizes(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def query_power(devices: list[int]) -> PowerSample:
    cmd = [
        "nvidia-smi",
        f"--id={','.join(str(device) for device in devices)}",
        "--query-gpu=power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    watts: list[float] = []
    utils: list[float] = []
    for line in output.strip().splitlines():
        power_text, util_text = [item.strip() for item in line.split(",")]
        watts.append(float(power_text))
        utils.append(float(util_text))
    return PowerSample(timestamp=time.time(), watts=watts, utils=utils)


def summarize(samples: Iterable[PowerSample]) -> dict[str, float]:
    rows = list(samples)
    if not rows:
        return {
            "avg_min": 0.0,
            "avg_mean": 0.0,
            "avg_max": 0.0,
            "peak_min": 0.0,
            "peak_mean": 0.0,
            "peak_max": 0.0,
            "util_mean": 0.0,
        }

    per_sample_avg = [sum(row.watts) / len(row.watts) for row in rows]
    per_gpu_peak = [max(row.watts[index] for row in rows) for index in range(len(rows[0].watts))]
    per_sample_util = [sum(row.utils) / len(row.utils) for row in rows]
    return {
        "avg_min": min(per_sample_avg),
        "avg_mean": sum(per_sample_avg) / len(per_sample_avg),
        "avg_max": max(per_sample_avg),
        "peak_min": min(per_gpu_peak),
        "peak_mean": sum(per_gpu_peak) / len(per_gpu_peak),
        "peak_max": max(per_gpu_peak),
        "util_mean": sum(per_sample_util) / len(per_sample_util),
    }


def worker(
    device: int,
    size: int,
    dtype_name: str,
    start_event: mp.Event,
    stop_event: mp.Event,
    status_queue: mp.Queue,
    duty_cycle: float,
    period: float,
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
    # Touch the tensors before the timed section so allocation is not counted.
    torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    status_queue.put((device, "ready", size))
    start_event.wait()

    active_window = max(0.0, min(1.0, duty_cycle)) * period
    while not stop_event.is_set():
        window_start = time.time()
        active_until = window_start + active_window
        while time.time() < active_until and not stop_event.is_set():
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        sleep_time = period - (time.time() - window_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def run_size(args: argparse.Namespace, size: int) -> dict[str, float]:
    start_event = mp.Event()
    stop_event = mp.Event()
    status_queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(
            target=worker,
            args=(
                device,
                size,
                args.dtype,
                start_event,
                stop_event,
                status_queue,
                args.duty_cycle,
                args.period,
            ),
        )
        for device in args.devices
    ]

    for process in processes:
        process.start()

    ready: set[int] = set()
    deadline = time.time() + args.startup_timeout
    try:
        while len(ready) < len(args.devices):
            if time.time() > deadline:
                raise TimeoutError(f"workers not ready after {args.startup_timeout}s: {sorted(ready)}")
            try:
                device, state, _ = status_queue.get(timeout=1.0)
            except queue.Empty:
                for process in processes:
                    if process.exitcode is not None and process.exitcode != 0:
                        raise RuntimeError(f"worker exited early with code {process.exitcode}")
                continue
            if state == "ready":
                ready.add(int(device))

        samples: list[PowerSample] = []
        start_event.set()
        end_time = time.time() + args.seconds
        while time.time() < end_time:
            samples.append(query_power(args.devices))
            time.sleep(args.sample_interval)
        return summarize(samples)
    finally:
        stop_event.set()
        start_event.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("0,1,2,3,4,5,6,7"))
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("8192,12288,16384,20480,24576"))
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--target-watts", type=float, default=370.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32", "tf32"), default="bf16")
    parser.add_argument("--duty-cycle", type=float, default=1.0)
    parser.add_argument("--period", type=float, default=1.0)
    args = parser.parse_args()

    print(
        "devices={devices} dtype={dtype} seconds={seconds} duty_cycle={duty_cycle}".format(
            devices=",".join(str(device) for device in args.devices),
            dtype=args.dtype,
            seconds=args.seconds,
            duty_cycle=args.duty_cycle,
        ),
        flush=True,
    )
    print(
        "size,avg_min_w,avg_mean_w,avg_max_w,peak_min_w,peak_mean_w,peak_max_w,util_mean_pct",
        flush=True,
    )
    for size in args.sizes:
        stats = run_size(args, size)
        print(
            "{size},{avg_min:.1f},{avg_mean:.1f},{avg_max:.1f},{peak_min:.1f},{peak_mean:.1f},{peak_max:.1f},{util_mean:.1f}".format(
                size=size,
                avg_min=stats["avg_min"],
                avg_mean=stats["avg_mean"],
                avg_max=stats["avg_max"],
                peak_min=stats["peak_min"],
                peak_mean=stats["peak_mean"],
                peak_max=stats["peak_max"],
                util_mean=stats["util_mean"],
            ),
            flush=True,
        )
        if stats["avg_mean"] >= args.target_watts:
            break


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
