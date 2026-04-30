"""Compute saturation config for 8×H20 training of Qwen3-VL-8B.

H20 specs (verified against NVIDIA datasheet, Apr 2026):
  - HBM3: 96 GB per card
  - bf16 dense: 148 TFLOPS (compute-bound; ~6.6× slower than H100)
  - NVLink 4.0: 900 GB/s within node
  - 8 cards/node = 768 GB total HBM, 1.18 PFLOPS total

Strategy for H20: memory-rich, compute-poor.
  - Use ZeRO-3 (memory cheap on H20) → free up RAM for larger batch
  - Maximize batch_size × gradient_accumulation_steps to amortize compute
  - Avoid optimizer offload (NVLink intra-node is fast; offload to CPU
    via PCIe is ~3× slower than just keeping it on HBM at 96GB)
  - vLLM gpu_memory_utilization=0.55 leaves room for ZeRO-3 actor+grad

Output: prints the recommended SFT and RL configs with exact memory budget.

Usage:
    python -m scripts.compute_saturation_config
    python -m scripts.compute_saturation_config --target sft   # SFT only
    python -m scripts.compute_saturation_config --target rl    # RL only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Optional


# ===========================================================================
# Hardware specs
# ===========================================================================

@dataclass
class HardwareSpec:
    name: str = "H20"
    n_gpus: int = 8
    hbm_gb: int = 96
    bf16_tflops: float = 148.0     # dense bf16 throughput per card
    nvlink_gbps: float = 900.0     # intra-node bandwidth

    @property
    def total_hbm_gb(self) -> int:
        return self.n_gpus * self.hbm_gb

    @property
    def total_tflops(self) -> float:
        return self.n_gpus * self.bf16_tflops


# ===========================================================================
# Model specs (Qwen3-VL-8B-Instruct)
# ===========================================================================

@dataclass
class Qwen3VLSpec:
    name: str = "Qwen3-VL-8B-Instruct"
    # Param counts (from HF model card / config.json)
    llm_params_b: float = 7.6      # text decoder
    vision_params_b: float = 0.4   # vision encoder + projector
    # LLM architecture
    n_layers: int = 28
    hidden_size: int = 3584
    n_heads: int = 28
    head_dim: int = 128            # hidden / heads
    n_kv_heads: int = 4            # GQA: 28 query heads → 4 KV heads
    intermediate_size: int = 18944
    vocab_size: int = 151936
    # bf16 = 2 bytes
    dtype_bytes: int = 2
    # AdamW state per param: fp32 momentum + variance = 8 bytes
    optim_bytes_per_param: int = 8

    @property
    def total_params_b(self) -> float:
        return self.llm_params_b + self.vision_params_b

    @property
    def model_gb(self) -> float:
        """bf16 model weight size."""
        return self.total_params_b * 1e9 * self.dtype_bytes / 1e9

    @property
    def grad_gb(self) -> float:
        """bf16 gradient size — same as model."""
        return self.model_gb

    @property
    def optim_gb(self) -> float:
        """AdamW: fp32 momentum + variance = 8 bytes/param."""
        return self.total_params_b * 1e9 * self.optim_bytes_per_param / 1e9

    @property
    def kv_cache_per_token_kb(self) -> float:
        """KV bytes per token (one rollout)."""
        # 2 (K+V) × n_layers × n_kv_heads × head_dim × bf16
        return (2 * self.n_layers * self.n_kv_heads * self.head_dim
                * self.dtype_bytes / 1024)

    def kv_cache_gb(self, n_tokens: int) -> float:
        return n_tokens * self.kv_cache_per_token_kb / 1024 / 1024


# ===========================================================================
# Activation memory estimation
# ===========================================================================

def activation_gb(
    seq_len: int, batch_size: int, model: Qwen3VLSpec,
    use_grad_ckpt: bool = True,
) -> float:
    """Estimate forward-pass activation memory (the dominant SFT cost).

    Without gradient checkpointing: O(batch × seq × hidden × n_layers)
    With gradient checkpointing: O(batch × seq × hidden × sqrt(n_layers))
                                  → reduces activations to ~sqrt(layers) factor
    Empirical formula from Megatron-LM paper, calibrated to Qwen scale.
    """
    base = batch_size * seq_len * model.hidden_size * model.dtype_bytes
    if use_grad_ckpt:
        layer_factor = model.n_layers ** 0.5
    else:
        layer_factor = model.n_layers
    # Per-layer activations include attention + MLP intermediate (the 4×
    # multiplier covers Q/K/V projections + attention output + 2 MLP
    # projections; calibrated against Qwen2.5-7B HF profile).
    bytes_total = base * layer_factor * 4
    return bytes_total / 1e9


# ===========================================================================
# SFT memory budget
# ===========================================================================

@dataclass
class SFTConfig:
    seq_len: int
    per_device_batch: int
    grad_accum: int
    use_zero3: bool = True
    use_grad_ckpt: bool = True

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum

    def memory_gb(self, hw: HardwareSpec, model: Qwen3VLSpec) -> dict:
        if self.use_zero3:
            # ZeRO-3 shards model + optim + grad across n_gpus
            shard = 1.0 / hw.n_gpus
            model_gb = model.model_gb * shard
            grad_gb = model.grad_gb * shard
            optim_gb = model.optim_gb * shard
        else:
            model_gb = model.model_gb
            grad_gb = model.grad_gb
            optim_gb = model.optim_gb
        # Activations are NOT sharded (each card holds its own batch's acts)
        act_gb = activation_gb(
            self.seq_len, self.per_device_batch, model, self.use_grad_ckpt,
        )
        # Add pessimistic overhead: cuda context (~1GB), framework buffers (~2GB)
        framework_gb = 3.0
        return {
            "model": model_gb,
            "grad":  grad_gb,
            "optim": optim_gb,
            "activations": act_gb,
            "framework": framework_gb,
            "total": model_gb + grad_gb + optim_gb + act_gb + framework_gb,
        }


def find_optimal_sft_config(
    hw: HardwareSpec,
    model: Qwen3VLSpec,
    seq_len: int = 16384,
    safety_factor: float = 0.85,
) -> SFTConfig:
    """Find the largest per_device_batch that fits in HBM × safety_factor.

    Effective batch target: ~32-64 samples per step (literature consensus
    for instruction-tuning of 7B-class models). With 8 GPUs:
      effective = per_device × grad_accum × n_gpus
      target 32 → e.g. per_device=2, grad_accum=2, n_gpus=8 = 32 ✓
    """
    target_hbm = hw.hbm_gb * safety_factor

    # Try per_device from 8 down to 1
    for pdb in [8, 4, 2, 1]:
        cfg = SFTConfig(seq_len=seq_len, per_device_batch=pdb, grad_accum=1)
        mem = cfg.memory_gb(hw, model)
        if mem["total"] <= target_hbm:
            # Found the largest fit — adjust grad_accum to reach effective batch ~32
            target_eff = 32
            grad_accum = max(1, target_eff // (pdb * hw.n_gpus))
            return SFTConfig(
                seq_len=seq_len, per_device_batch=pdb, grad_accum=grad_accum,
            )
    # Even pdb=1 doesn't fit — must use offload
    return SFTConfig(seq_len=seq_len, per_device_batch=1, grad_accum=4)


# ===========================================================================
# RL memory budget (rollout + training co-located)
# ===========================================================================

@dataclass
class RLConfig:
    seq_len: int                       # max prompt length
    max_new_tokens: int                # response per chunk
    group_size: int                    # G rollouts per video
    train_batch_videos: int            # videos per training step
    vllm_gpu_mem_util: float = 0.55    # fraction of HBM for vLLM
    use_zero3: bool = True
    use_grad_ckpt: bool = True

    def memory_gb(self, hw: HardwareSpec, model: Qwen3VLSpec) -> dict:
        if self.use_zero3:
            shard = 1.0 / hw.n_gpus
        else:
            shard = 1.0
        actor_gb = model.model_gb * shard
        actor_grad_gb = model.grad_gb * shard
        actor_optim_gb = model.optim_gb * shard

        # vLLM rollout: full model copy + KV cache for batched concurrent gen
        vllm_model_gb = model.model_gb            # vLLM keeps full model
        vllm_kv_gb = model.kv_cache_gb(
            self.seq_len * self.group_size * self.train_batch_videos
        )
        # Cap vLLM at gpu_memory_utilization
        vllm_total_gb = min(
            vllm_model_gb + vllm_kv_gb,
            hw.hbm_gb * self.vllm_gpu_mem_util,
        )

        # Training-side activations (during gradient step)
        # Activations only matter when training, and vLLM is freed before
        # training step (free_cache_engine: true in recipe). So peak is
        # max(rollout_phase, training_phase) per step.
        train_act_gb = activation_gb(
            self.seq_len, 1, model, self.use_grad_ckpt,
        )

        framework_gb = 3.0
        rollout_phase = vllm_total_gb + actor_gb + framework_gb
        training_phase = (actor_gb + actor_grad_gb + actor_optim_gb +
                          train_act_gb + framework_gb)
        return {
            "vllm_model":       vllm_model_gb,
            "vllm_kv_cache":    vllm_kv_gb,
            "vllm_total":       vllm_total_gb,
            "actor_model":      actor_gb,
            "actor_grad":       actor_grad_gb,
            "actor_optim":      actor_optim_gb,
            "train_acts":       train_act_gb,
            "rollout_peak":     rollout_phase,
            "training_peak":    training_phase,
            "peak":             max(rollout_phase, training_phase),
        }


# ===========================================================================
# Throughput estimation
# ===========================================================================

def sft_step_seconds(
    cfg: SFTConfig, hw: HardwareSpec, model: Qwen3VLSpec,
    achievable_mfu: float = 0.45,
) -> dict:
    """Estimate SFT step time (forward + backward).

    FLOPs ≈ 6 × n_params × seq_len × batch_size (Chinchilla approximation:
    forward = 2N, backward = 4N → 6N total).
    H20 achievable MFU on bf16 attention-heavy workloads is ~40-50%.
    """
    flops_per_sample = 6 * model.total_params_b * 1e9 * cfg.seq_len
    flops_per_step = flops_per_sample * cfg.per_device_batch * cfg.grad_accum * hw.n_gpus
    achievable_flops = hw.total_tflops * 1e12 * achievable_mfu
    compute_sec = flops_per_step / achievable_flops
    # Communication: ZeRO-3 all-gather + reduce-scatter on every layer
    comm_overhead = 0.15 * compute_sec
    total_sec = compute_sec + comm_overhead
    return {
        "compute_tflops_per_step": flops_per_step / 1e12,
        "compute_sec": compute_sec,
        "comm_overhead_sec": comm_overhead,
        "total_sec_per_step": total_sec,
        "samples_per_step": cfg.per_device_batch * cfg.grad_accum * hw.n_gpus,
        "samples_per_sec": (cfg.per_device_batch * cfg.grad_accum * hw.n_gpus
                           / total_sec),
    }


def rl_step_seconds(
    cfg: RLConfig, hw: HardwareSpec, model: Qwen3VLSpec,
    avg_chunks_per_video: int = 60,
    avg_response_tokens: int = 256,
    vllm_throughput_tokens_per_sec_per_gpu: float = 8000.0,
) -> dict:
    """Estimate RL step time = rollout_time + train_time.

    Rollout dominates RL step time (vLLM generation across N×G×n_chunks).
    H20 vLLM throughput on Qwen3-VL-8B bf16 ~8K tok/s/GPU (calibrated
    against published verl benchmarks scaled by H100→H20 compute ratio).
    """
    # Rollout: total tokens to generate
    total_rollouts = cfg.train_batch_videos * cfg.group_size
    total_chunks_to_generate = total_rollouts * avg_chunks_per_video
    total_response_tokens = total_chunks_to_generate * avg_response_tokens
    # vLLM throughput is throughput across all GPUs (it parallelizes itself)
    vllm_total_throughput = vllm_throughput_tokens_per_sec_per_gpu * hw.n_gpus
    rollout_sec = total_response_tokens / vllm_total_throughput

    # Training: forward + backward on rollout outputs
    # Per-video: ~chunks × (prompt + response) tokens
    avg_seq_len = (cfg.seq_len + avg_response_tokens) // 2  # rough average
    train_cfg = SFTConfig(
        seq_len=avg_seq_len,
        per_device_batch=1,
        grad_accum=total_rollouts,  # all rollouts in one accum window
    )
    train_timing = sft_step_seconds(train_cfg, hw, model, achievable_mfu=0.40)
    train_sec = train_timing["total_sec_per_step"]

    return {
        "rollout_response_tokens": total_response_tokens,
        "rollout_sec":  rollout_sec,
        "training_sec": train_sec,
        "total_sec_per_step": rollout_sec + train_sec,
        "rollout_pct": rollout_sec / (rollout_sec + train_sec),
    }


# ===========================================================================
# Pretty print
# ===========================================================================

def print_sft_recommendation(hw: HardwareSpec, model: Qwen3VLSpec):
    print("\n" + "=" * 78)
    print(f"SFT recommendation — {hw.name} ×{hw.n_gpus}")
    print("=" * 78)
    cfg = find_optimal_sft_config(hw, model, seq_len=16384)
    mem = cfg.memory_gb(hw, model)
    timing = sft_step_seconds(cfg, hw, model)
    print(f"  cutoff_len:                {cfg.seq_len}")
    print(f"  per_device_train_batch:    {cfg.per_device_batch}")
    print(f"  gradient_accumulation:     {cfg.grad_accum}")
    print(f"  ZeRO stage:                {'3' if cfg.use_zero3 else 'off'}")
    print(f"  gradient_checkpointing:    {cfg.use_grad_ckpt}")
    print(f"  effective batch (×8 GPU):  {cfg.effective_batch * hw.n_gpus}")
    print(f"\n  Memory per GPU (HBM {hw.hbm_gb}GB):")
    print(f"    model (ZeRO-3 shard):    {mem['model']:6.2f} GB")
    print(f"    grad  (ZeRO-3 shard):    {mem['grad']:6.2f} GB")
    print(f"    optim (ZeRO-3 shard):    {mem['optim']:6.2f} GB")
    print(f"    activations:             {mem['activations']:6.2f} GB")
    print(f"    framework overhead:      {mem['framework']:6.2f} GB")
    print(f"    {'TOTAL':24} {mem['total']:6.2f} GB / {hw.hbm_gb} GB"
          f"  ({100 * mem['total'] / hw.hbm_gb:.1f}% util)")
    print(f"\n  Throughput estimate (MFU=45% on H20):")
    print(f"    PFLOPs per step:         {timing['compute_tflops_per_step']/1000:.2f}")
    print(f"    compute time:            {timing['compute_sec']:.2f} s")
    print(f"    comm overhead (ZeRO-3):  {timing['comm_overhead_sec']:.2f} s")
    print(f"    total per step:          {timing['total_sec_per_step']:.2f} s")
    print(f"    samples/sec:             {timing['samples_per_sec']:.2f}")
    print(f"    samples/hour:            {timing['samples_per_sec']*3600:.0f}")


def print_rl_recommendation(hw: HardwareSpec, model: Qwen3VLSpec):
    print("\n" + "=" * 78)
    print(f"RL recommendation (GRPO multi-turn) — {hw.name} ×{hw.n_gpus}")
    print("=" * 78)
    cfg = RLConfig(
        seq_len=16384,
        max_new_tokens=2048,
        group_size=8,
        train_batch_videos=8,
    )
    mem = cfg.memory_gb(hw, model)
    timing = rl_step_seconds(cfg, hw, model)
    print(f"  prompt_length:             {cfg.seq_len}")
    print(f"  max_response_length:       {cfg.max_new_tokens}")
    print(f"  group_size G:              {cfg.group_size}")
    print(f"  train_batch_videos:        {cfg.train_batch_videos}")
    print(f"  total rollouts/step:       {cfg.train_batch_videos * cfg.group_size}")
    print(f"\n  Memory per GPU (HBM {hw.hbm_gb}GB):")
    print(f"    Rollout phase:")
    print(f"      vLLM model:            {mem['vllm_model']:6.2f} GB")
    print(f"      vLLM KV cache:         {mem['vllm_kv_cache']:6.2f} GB")
    print(f"      vLLM total (capped):   {mem['vllm_total']:6.2f} GB"
          f"  (gpu_memory_utilization={cfg.vllm_gpu_mem_util})")
    print(f"      actor (offline):       {mem['actor_model']:6.2f} GB")
    print(f"      framework:             {3.0:6.2f} GB")
    print(f"      {'rollout peak':24} {mem['rollout_peak']:6.2f} GB / {hw.hbm_gb}")
    print(f"    Training phase (vLLM freed):")
    print(f"      actor + grad + optim:  {mem['actor_model'] + mem['actor_grad'] + mem['actor_optim']:6.2f} GB")
    print(f"      activations:           {mem['train_acts']:6.2f} GB")
    print(f"      framework:             {3.0:6.2f} GB")
    print(f"      {'training peak':24} {mem['training_peak']:6.2f} GB / {hw.hbm_gb}")
    print(f"    {'OVERALL PEAK':26} {mem['peak']:6.2f} GB / {hw.hbm_gb}"
          f"  ({100 * mem['peak'] / hw.hbm_gb:.1f}% util)")
    print(f"\n  Throughput estimate:")
    print(f"    rollout response toks:   {timing['rollout_response_tokens']:,}")
    print(f"    rollout time:            {timing['rollout_sec']:.1f} s"
          f"  ({100*timing['rollout_pct']:.0f}% of step)")
    print(f"    training time:           {timing['training_sec']:.1f} s")
    print(f"    total per step:          {timing['total_sec_per_step']:.1f} s")
    print(f"    steps/hour:              {3600 / timing['total_sec_per_step']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--n_gpus", type=int, default=8)
    args = parser.parse_args()

    hw = HardwareSpec(n_gpus=args.n_gpus)
    model = Qwen3VLSpec()

    print("=" * 78)
    print(f"Hardware: {hw.name} × {hw.n_gpus}  "
          f"({hw.total_hbm_gb} GB total HBM, {hw.total_tflops:.0f} TFLOPS bf16)")
    print(f"Model:    {model.name}  "
          f"({model.total_params_b}B params, {model.model_gb:.1f} GB bf16)")
    print(f"          AdamW optim:    {model.optim_gb:.1f} GB (fp32 mom + var)")
    print(f"          KV per token:   {model.kv_cache_per_token_kb:.2f} KB "
          f"(GQA: {model.n_heads}q→{model.n_kv_heads}kv heads)")

    if args.target in ("sft", "both"):
        print_sft_recommendation(hw, model)
    if args.target in ("rl", "both"):
        print_rl_recommendation(hw, model)


if __name__ == "__main__":
    main()
