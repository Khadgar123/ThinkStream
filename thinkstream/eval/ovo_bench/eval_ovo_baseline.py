"""
OVO-Bench baseline evaluation with streaming video delivery.

Uses the SAME streaming infrastructure as ThinkStream eval (chunk-by-chunk
video, same visual window, same KV cache) but WITHOUT think/action protocol.
The baseline model directly picks the answer from logits at question time.

This is a fair comparison: identical video delivery, only the generation
strategy differs.

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

from eval_common import (
    add_common_args,
    setup_distributed,
    cleanup_distributed,
    load_model_and_processor,
    build_results,
    save_results,
)
from eval_baseline import baseline_sample_restricted
from eval_ovo import evaluate_ovobench_results

from thinkstream.model.inference import streaming_video_chat, StreamingWindowInferenceEngine
from thinkstream.model import DEFAULT_VIDEO_FLEX_WINDOW_SIZE, get_text_config
from thinkstream.data.stream_data_processor import (
    FRAMES_PER_CHUNK,
    DEFAULT_MAX_CHUNKS,
    QWEN_TEMPLATE_WO_SYSTEM,
    preload_video,
    _resolve_vit_patch_size,
)

import gc
import json
import random
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import DataLoader

from eval_common import MCQDataset, NoPadDistributedSampler


# Baseline system prompt: simple instruction, no agent protocol
BASELINE_SYSTEM_PROMPT = (
    "You are a helpful video understanding assistant. "
    "Watch the video carefully and answer questions based on what you observe."
)


@torch.inference_mode()
def baseline_predict_streaming(
    model,
    processor,
    benchmark_path: str,
    options: list,
    question_prefix: str = "",
    question_postfix: str = "\nPlease select the correct answer.",
    max_len: int = 24576,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    max_new_tokens: int = 30,
    remaining_seconds: int = DEFAULT_MAX_CHUNKS,
    rank: int = 0,
    world_size: int = 1,
    model_type: str = "qwen3vl",
    slack_time: float = 3.0,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
):
    """Baseline streaming MCQ prediction.

    Same streaming video delivery as ThinkStream eval, but uses
    baseline_sample_restricted (no think/action, direct answer).
    """
    strict_option_ids = [
        processor.tokenizer(opt, add_special_tokens=False).input_ids[-1]
        for opt in options
    ]
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Baseline sample kwargs: no think budget, no action tokens
    sample_kwargs = {
        "restricted_token_ids": strict_option_ids,
        "eos_token_id": eos_token_id,
    }

    dataset = MCQDataset(
        benchmark_path,
        processor=processor,
        model_type=model_type,
        frames_per_chunk=frames_per_chunk,
        max_chunks=remaining_seconds,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        slack_time=slack_time,
    )

    if world_size > 1:
        sampler = NoPadDistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
    )

    video_token_id = processor.tokenizer.convert_tokens_to_ids(["<|video_pad|>"])[0]
    video_flex_window_size = getattr(
        model.config, "video_flex_window_size", DEFAULT_VIDEO_FLEX_WINDOW_SIZE
    )
    text_cfg = get_text_config(model.config)
    engine = StreamingWindowInferenceEngine(
        model,
        batch_size=1,
        max_len=max_len,
        num_hidden_layers=text_cfg.num_hidden_layers,
        num_key_value_heads=text_cfg.num_key_value_heads,
        head_dim=text_cfg.hidden_size // text_cfg.num_attention_heads,
        vocab_size=text_cfg.vocab_size,
        pad_token_id=model.generation_config.pad_token_id,
        eos_token_ids=model.generation_config.eos_token_id,
        video_token_id=video_token_id,
        video_flex_window_size=video_flex_window_size,
    )

    predictions = []
    datums = []
    local_indices = []

    for idx, datum, preloaded in tqdm.tqdm(
        dataloader, desc="Baseline streaming eval", disable=(rank != 0)
    ):
        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])

            if preloaded is not None and "original_video_end" in preloaded:
                query_ts = preloaded["original_video_end"]
                run_video_end = preloaded["video_end"]
            else:
                base_end = datum.get("video_end")
                if base_end is not None:
                    query_ts = base_end
                    run_video_end = base_end + slack_time if slack_time > 0 else base_end
                else:
                    raise ValueError(f"Sample {idx}: Cannot determine video_end.")

            if "options" in datum and datum["options"]:
                query = (
                    question_prefix
                    + datum["question"]
                    + "\n"
                    + "\n".join(datum["options"])
                    + question_postfix
                )
            else:
                query = datum["question"]

            for result in streaming_video_chat(
                engine=engine,
                processor=processor,
                video_path=video_path,
                queries=[{"content": query, "timestamp": query_ts}],
                video_start=datum.get("video_start", 0.0),
                video_end=run_video_end,
                frames_per_chunk=frames_per_chunk,
                max_chunks=remaining_seconds,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                system_prompt=BASELINE_SYSTEM_PROMPT,
                chat_template_wo_system=QWEN_TEMPLATE_WO_SYSTEM,
                sample=baseline_sample_restricted,
                sample_kwargs=sample_kwargs,
                model_type=model_type,
                preloaded_video=preloaded,
                slack_time=slack_time,
                break_on_answer=True,
            ):
                if result["is_answer"]:
                    gen_tokens = result["generated_tokens"][0]
                    # Baseline outputs: ANSWER <|im_end|> (no <think>/<response>)
                    # First non-special token is the answer
                    try:
                        ans_token = gen_tokens[0].item()
                        predicted_idx = strict_option_ids.index(ans_token)
                    except (IndexError, ValueError):
                        # Fallback: search for any option token
                        predicted_idx = random.randint(0, len(options) - 1)
                        for t in gen_tokens:
                            if t.item() in strict_option_ids:
                                predicted_idx = strict_option_ids.index(t.item())
                                break

                    print(
                        f"[Baseline] {processor.decode(gen_tokens)} -> {options[predicted_idx]}"
                    )
                    predictions.append(predicted_idx)
                    datums.append({**datum, "success": True})
                    local_indices.append(idx)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append(random.randint(0, len(options) - 1))
            datums.append({**datum, "success": False})
            local_indices.append(idx)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if world_size > 1:
        local_data = list(zip(local_indices, predictions, datums))
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)
        if rank == 0:
            all_data = []
            for proc_data in gathered_data:
                all_data.extend(proc_data)
            all_data.sort(key=lambda x: x[0])
            return np.array([d[1] for d in all_data]), [d[2] for d in all_data], 0
        else:
            return np.array(predictions), datums, rank
    else:
        return np.array(predictions), datums, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline OVO-Bench evaluation (streaming video, no agent protocol)."
    )
    add_common_args(parser)
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()

    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")

    # Load model with streaming patches (same as ThinkStream)
    model, processor = load_model_and_processor(
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

    predictions, datums, process_index = baseline_predict_streaming(
        model=model,
        processor=processor,
        benchmark_path=benchmark_path,
        options=options,
        frames_per_chunk=args.frames_per_chunk,
        max_new_tokens=args.max_new_tokens,
        remaining_seconds=args.remaining_seconds,
        rank=rank,
        world_size=world_size,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        slack_time=args.slack_time,
    )

    if process_index == 0:
        results = build_results(datums, predictions, options)
        filename = (
            f"baseline_streaming_result_"
            f"{args.min_pixels}_{args.max_pixels}_{args.max_new_tokens}.json"
        )
        save_path = os.path.join(args.model_path, "eval", "ovo_bench", filename)
        save_results(results, save_path, evaluate_ovobench_results)

    cleanup_distributed(world_size)
