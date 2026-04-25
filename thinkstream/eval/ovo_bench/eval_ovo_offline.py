"""
OVO-Bench offline baseline evaluation.

Loads N frames uniformly sampled from full video (default 64),
standard model.generate, no streaming protocol.

Matches "Open-source Offline Models" row in paper Table 2.

Usage:
    torchrun --nproc_per_node=8 thinkstream/eval/ovo_bench/eval_ovo_offline.py \
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
    add_offline_args,
    load_offline_model,
    offline_predict_mcq,
    setup_distributed,
    cleanup_distributed,
)
from eval_common import build_results, save_results
from eval_ovo import evaluate_ovobench_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline OVO-Bench evaluation (full video, no streaming)."
    )
    add_offline_args(parser)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()

    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")
    model, processor = load_offline_model(
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

    debug_dir = os.path.join(args.model_path, "eval", "ovo_bench", "debug")

    predictions, datums, process_index = offline_predict_mcq(
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
        sample=args.sample,
        debug=args.debug,
        debug_dir=debug_dir,
    )

    if process_index == 0:
        results = build_results(datums, predictions, options)
        filename = (
            f"offline_result_{args.max_frames}frames_"
            f"{args.min_pixels}_{args.max_pixels}_{args.max_new_tokens}.json"
        )
        save_path = os.path.join(args.model_path, "eval", "ovo_bench", filename)
        save_results(results, save_path, evaluate_ovobench_results)

    cleanup_distributed(world_size)
