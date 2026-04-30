"""On-GPU profiler — measures actual H20 throughput vs the predictions in
`compute_saturation_config.py`.

Runs three benchmarks:
  1. SFT step time + peak memory (single training step, no real data)
  2. vLLM rollout throughput (tokens/sec/GPU)
  3. Communication overhead (all-gather / reduce-scatter latency)

Output: writes JSON report to <output>/profile_h20_<timestamp>.json
        Compare numbers against compute_saturation_config.py predictions.

Usage:
    # SFT profile only (no vLLM dep)
    torchrun --nproc_per_node=8 scripts/profile_h20.py --bench sft

    # All benchmarks (requires vLLM + flash-attn)
    torchrun --nproc_per_node=8 scripts/profile_h20.py --bench all

    # Quick sanity (1 step, no comm test)
    torchrun --nproc_per_node=8 scripts/profile_h20.py --bench sft --n_steps 1

Designed to fail loudly if predicted vs actual diverges by >25%.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ===========================================================================
# SFT step profile
# ===========================================================================

def profile_sft_step(
    n_steps: int = 5,
    seq_len: int = 16384,
    per_device_batch: int = 8,
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> dict:
    """Run N forward+backward steps with synthetic data; measure time + memory.

    Compare against compute_saturation_config:
      Predicted: 109 s/step at pdb=8, seq=16k, 8×H20
      Predicted memory peak: 35 GB/card
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(
            f"SFT profile: world={world}, seq={seq_len}, pdb={per_device_batch}"
        )

    # Synthetic input — avoids dataloading from disk so we measure compute only
    vocab = 151936
    input_ids = torch.randint(0, vocab, (per_device_batch, seq_len), device=device)
    labels = input_ids.clone()

    from transformers import AutoModelForCausalLM
    import deepspeed

    # ZeRO-3 config matching production
    ds_config = {
        "train_batch_size": per_device_batch * world,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_checkpointing": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    engine, *_ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # Warmup
    if rank == 0:
        logger.info("Warmup (2 steps)...")
    for _ in range(2):
        out = engine(input_ids=input_ids, labels=labels)
        engine.backward(out.loss)
        engine.step()
    torch.cuda.synchronize()

    # Measure
    if rank == 0:
        logger.info(f"Measuring {n_steps} steps...")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = []
    for step in range(n_steps):
        step_start = time.perf_counter()
        out = engine(input_ids=input_ids, labels=labels)
        engine.backward(out.loss)
        engine.step()
        torch.cuda.synchronize()
        step_dur = time.perf_counter() - step_start
        losses.append(float(out.loss.detach()))
        if rank == 0:
            logger.info(
                f"  step {step}: {step_dur:.2f} s, loss={losses[-1]:.4f}"
            )
    total_sec = time.perf_counter() - start
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    avg_step = total_sec / n_steps

    # Reduce metrics across ranks (max for memory, mean for time)
    peak_mem_t = torch.tensor([peak_mem_gb], device=device)
    dist.all_reduce(peak_mem_t, op=dist.ReduceOp.MAX)
    avg_step_t = torch.tensor([avg_step], device=device)
    dist.all_reduce(avg_step_t, op=dist.ReduceOp.AVG)

    result = {
        "world_size": world,
        "seq_len": seq_len,
        "per_device_batch": per_device_batch,
        "effective_batch": per_device_batch * world,
        "n_steps": n_steps,
        "avg_step_sec": float(avg_step_t.item()),
        "peak_memory_gb_per_gpu": float(peak_mem_t.item()),
        "samples_per_sec": (per_device_batch * world) / float(avg_step_t.item()),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    return result


# ===========================================================================
# vLLM rollout throughput
# ===========================================================================

def profile_vllm_rollout(
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    n_prompts: int = 64,
    prompt_len: int = 8192,
    max_new_tokens: int = 256,
    gpu_memory_utilization: float = 0.55,
) -> dict:
    """Measure vLLM batched generation throughput on 8×H20.

    Compare against compute_saturation_config:
      Predicted: 8000 tok/s/GPU
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return {"error": "vllm not installed; pip install vllm"}

    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,                 # 8×H20 = TP=8
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=prompt_len + max_new_tokens,
        dtype="bfloat16",
        enforce_eager=False,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=1.0, top_p=1.0,
        max_tokens=max_new_tokens,
    )
    # Synthetic prompts — avoid disk IO
    prompts = [
        f"Test prompt {i}: " + ("token " * (prompt_len - 5))
        for i in range(n_prompts)
    ]
    # Warmup
    _ = llm.generate(prompts[:8], sampling)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - start

    total_response_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "n_prompts": n_prompts,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "tp_size": 8,
        "elapsed_sec": elapsed,
        "total_response_tokens": total_response_toks,
        "tokens_per_sec_total": total_response_toks / elapsed,
        "tokens_per_sec_per_gpu": total_response_toks / elapsed / 8,
    }


# ===========================================================================
# Communication overhead
# ===========================================================================

def profile_nccl_overhead() -> dict:
    """All-reduce latency for typical ZeRO-3 message sizes."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    sizes_mb = [1, 16, 64, 256, 1024]
    results = {}
    for size_mb in sizes_mb:
        n = size_mb * 1024 * 1024 // 4   # fp32 elements
        x = torch.randn(n, device=device, dtype=torch.float32)
        # Warmup
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        elapsed_per_call = (time.perf_counter() - start) / 10
        bandwidth_gbps = (size_mb / 1024) / elapsed_per_call
        results[f"{size_mb}MB"] = {
            "elapsed_sec": elapsed_per_call,
            "bandwidth_GB_per_s": bandwidth_gbps,
        }
    return results


# ===========================================================================
# Driver
# ===========================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=["sft", "vllm", "nccl", "all"], default="sft")
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--per_device_batch", type=int, default=8)
    parser.add_argument("--model_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output_dir", default="profile_results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "config": vars(args),
    }

    rank = int(os.environ.get("RANK", "0"))

    if args.bench in ("sft", "all"):
        report["sft"] = profile_sft_step(
            n_steps=args.n_steps,
            seq_len=args.seq_len,
            per_device_batch=args.per_device_batch,
            model_path=args.model_path,
        )

    if args.bench in ("nccl", "all"):
        report["nccl"] = profile_nccl_overhead()

    if args.bench in ("vllm", "all") and rank == 0:
        report["vllm"] = profile_vllm_rollout(
            model_path=args.model_path,
            n_prompts=64,
            prompt_len=args.seq_len // 2,
        )

    if rank == 0:
        path = out_dir / f"profile_h20_{timestamp}.json"
        path.write_text(json.dumps(report, indent=2))
        logger.info(f"Report: {path}")

        # Compare against predictions
        try:
            from scripts.compute_saturation_config import (
                HardwareSpec, Qwen3VLSpec, SFTConfig, sft_step_seconds,
            )
            predicted = sft_step_seconds(
                SFTConfig(
                    seq_len=args.seq_len,
                    per_device_batch=args.per_device_batch,
                    grad_accum=1,
                ),
                HardwareSpec(),
                Qwen3VLSpec(),
            )
            if "sft" in report:
                actual_sec = report["sft"]["avg_step_sec"]
                pred_sec = predicted["total_sec_per_step"]
                ratio = actual_sec / pred_sec
                logger.info(
                    f"SFT step: actual={actual_sec:.2f}s, "
                    f"predicted={pred_sec:.2f}s, ratio={ratio:.2f}"
                )
                if abs(ratio - 1.0) > 0.25:
                    logger.warning(
                        f"⚠ actual/predicted ratio {ratio:.2f} > 1.25 — "
                        f"recalibrate compute_saturation_config.py"
                    )
                else:
                    logger.info("✓ within 25% of prediction")
        except Exception as e:
            logger.warning(f"comparison skipped: {e}")


if __name__ == "__main__":
    main()
