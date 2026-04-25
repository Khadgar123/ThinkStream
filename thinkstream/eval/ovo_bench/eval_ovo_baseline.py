"""
OVO-Bench evaluation with vanilla (baseline) VLM.

No streaming protocol, no think budget, no special tokens.
Standard video QA: load all frames → prompt → generate → parse answer.

Usage:
    torchrun --nproc_per_node=8 thinkstream/eval/ovo_bench/eval_ovo_baseline.py \
        --benchmark_dir /path/to/ovo_bench \
        --model_path Qwen/Qwen3-VL-8B \
        --model_type qwen3vl
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from eval_baseline import (
    add_baseline_args,
    baseline_predict_mcq,
    build_results,
    load_baseline_model,
    save_results,
    setup_distributed,
    cleanup_distributed,
)
from eval_ovo import evaluate_ovobench_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline OVO-Bench evaluation (no streaming)."
    )
    add_baseline_args(parser)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()

    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")
    model, processor = load_baseline_model(
        args.model_path,
        local_rank,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    options = [
        "No", "Yes",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E",
    ]

    predictions, datums, process_index = baseline_predict_mcq(
        model=model,
        processor=processor,
        benchmark_path=benchmark_path,
        options=options,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        rank=rank,
        world_size=world_size,
        model_type=args.model_type,
        sample=args.sample,
    )

    if process_index == 0:
        results = build_results(datums, predictions, options)
        filename = f"baseline_result_{args.min_pixels}_{args.max_pixels}_{args.max_new_tokens}.json"
        save_path = os.path.join(args.model_path, "eval", "ovo_bench", filename)
        save_results(results, save_path, evaluate_ovobench_results)

    cleanup_distributed(world_size)
