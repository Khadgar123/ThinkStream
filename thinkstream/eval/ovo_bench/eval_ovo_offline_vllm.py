"""OVO-Bench offline baseline eval via vLLM.

vLLM-batched alternative to eval_ovo_offline.py. No torchrun needed —
vLLM handles tensor parallelism internally.

Usage:
    python thinkstream/eval/ovo_bench/eval_ovo_offline_vllm.py \
        --benchmark_dir /path/to/ovo_bench \
        --model_path Qwen/Qwen3-VL-8B \
        --tensor_parallel_size 8
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoProcessor

from eval_baseline_vllm import offline_predict_mcq_vllm
from eval_common import build_results, save_results
from eval_ovo import evaluate_ovobench_results
from vllm_engine import init_vllm_engine


def add_vllm_offline_args(parser):
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    # vLLM-specific
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=128000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graph capture (slower, easier to debug).")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline OVO-Bench eval via vLLM."
    )
    add_vllm_offline_args(parser)
    args = parser.parse_args()

    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    vp = processor.video_processor
    vp.max_pixels = args.max_pixels
    vp.min_pixels = args.min_pixels
    vp.size["shortest_edge"] = args.min_pixels
    vp.size["longest_edge"] = args.max_pixels

    llm = init_vllm_engine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )

    options = [
        "No", "Yes",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E",
    ]

    debug_dir = os.path.join(args.model_path, "eval", "ovo_bench", "debug_vllm")

    predictions, datums = offline_predict_mcq_vllm(
        llm=llm,
        processor=processor,
        benchmark_path=benchmark_path,
        options=options,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        sample=args.sample,
        temperature=args.temperature,
        debug=args.debug,
        debug_dir=debug_dir,
    )

    results = build_results(datums, predictions, options)
    filename = (
        f"offline_vllm_result_{args.max_frames}frames_"
        f"{args.min_pixels}_{args.max_pixels}_{args.max_new_tokens}.json"
    )
    save_path = os.path.join(args.model_path, "eval", "ovo_bench", filename)
    save_results(results, save_path, evaluate_ovobench_results)
