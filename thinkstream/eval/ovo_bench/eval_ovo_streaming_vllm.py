"""OVO-Bench streaming online eval via vLLM (chunk-lockstep).

Uses StreamingAgentLoop's MemoryState/build_single_step_messages but
batches across samples at each chunk_idx so vLLM dynamic batching can
fill the GPU. No torchrun needed — vLLM handles tensor parallelism.

Usage:
    python thinkstream/eval/ovo_bench/eval_ovo_streaming_vllm.py \
        --benchmark_dir /path/to/ovo_bench \
        --model_path /path/to/agent_ckpt \
        --frames_root /path/to/extracted_frames \
        --tensor_parallel_size 8
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoProcessor

from eval_baseline import OfflineMCQDataset
from eval_common import build_results, save_results
from eval_ovo import evaluate_ovobench_results
from streaming_vllm import streaming_predict_mcq_vllm
from vllm_engine import init_vllm_engine


def add_streaming_vllm_args(parser):
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    # Streaming-specific
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Pre-extracted frames root (frame_NNNNNN.jpg). "
                             "Falls back to online video decode if unset.")
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--max_chunks", type=int, default=30)
    parser.add_argument("--frames_per_chunk", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Suppresses think repetition loops. 1.0 disables; "
                             "1.1 is conservative; 1.2-1.3 if loops persist.")
    # vLLM-specific
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=128000,
                        help="Per-request prompt+gen budget. Streaming "
                             "samples can be long; set generously.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming OVO-Bench eval via vLLM (chunk-lockstep)."
    )
    add_streaming_vllm_args(parser)
    args = parser.parse_args()

    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")
    dataset = OfflineMCQDataset(benchmark_path, sample=args.sample)

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

    debug_dir = os.path.join(args.model_path, "eval", "ovo_bench", "debug_streaming_vllm")

    predictions, datums = streaming_predict_mcq_vllm(
        llm=llm,
        processor=processor,
        dataset=dataset,
        options=options,
        max_new_tokens=args.max_new_tokens,
        frames_per_chunk=args.frames_per_chunk,
        max_chunks=args.max_chunks,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        frames_root=args.frames_root,
        video_root=args.video_root,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        debug=args.debug,
        debug_dir=debug_dir,
    )

    results = build_results(datums, predictions, options)
    filename = (
        f"streaming_vllm_result_{args.max_chunks}chunks_"
        f"{args.frames_per_chunk}fpc_{args.min_pixels}_"
        f"{args.max_pixels}_{args.max_new_tokens}.json"
    )
    save_path = os.path.join(args.model_path, "eval", "ovo_bench", filename)
    save_results(results, save_path, evaluate_ovobench_results)
