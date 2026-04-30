#!/usr/bin/env python3
"""波动动态占卡程序 — 占满6卡显存 + 持续计算，防止他人抢占。

运行方式:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/occupy_6gpu.py
"""

import time
import random
import torch
import threading

DEVICES = [int(x) for x in ("0,1,2,3,4,5".split(","))]
COMPUTE_CYCLE = 60       # 秒，每轮持续计算时长
COMPUTE_INTENSITY = 0.3  # 计算占空比（0.3 = 70%时间在做GEMM，30% sleep）
MEM_LOW = 0.70
MEM_HIGH = 0.95


def occupy_gpu(dev: int):
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    total_bytes = props.total_memory
    while True:
        ratio = random.uniform(MEM_LOW, MEM_HIGH)
        alloc_bytes = int(total_bytes * ratio)
        n_elems = alloc_bytes // 4
        # 显存块1: 大buffer（占显存）
        buf = torch.empty(n_elems, dtype=torch.float32, device=f"cuda:{dev}")
        buf.fill_(float(dev))
        used = torch.cuda.memory_allocated(dev) / 1024**3
        print(f"[GPU {dev}] occupied {used:.1f} GB", flush=True)

        # 显存块2: 矩阵乘法workspace（用一小部分显存做实际计算）
        # 约 8K x 8K 的float32矩阵 ~ 256MB，足够让utilization飙高
        dim = 8192
        a = torch.randn(dim, dim, dtype=torch.float32, device=f"cuda:{dev}")
        b = torch.randn(dim, dim, dtype=torch.float32, device=f"cuda:{dev}")
        c = torch.empty(dim, dim, dtype=torch.float32, device=f"cuda:{dev}")

        t0 = time.time()
        while time.time() - t0 < COMPUTE_CYCLE:
            # 连续做几次GEMM，让GPU忙起来
            for _ in range(8):
                torch.matmul(a, b, out=c)
            # 小sleep制造波动感，同时让utilization不是100%固定
            time.sleep(0.05)

        del buf, a, b, c
        torch.cuda.empty_cache()
        print(f"[GPU {dev}] released, next cycle soon", flush=True)
        time.sleep(random.uniform(3, 10))


def main():
    print(f"Occupying GPUs: {DEVICES}")
    threads = []
    for d in DEVICES:
        t = threading.Thread(target=occupy_gpu, args=(d,), daemon=True)
        t.start()
        threads.append(t)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
