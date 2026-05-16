#!/usr/bin/env python
"""Reference-style OVO streaming baseline for base VLMs.

This follows the CASIA reference eval contract:
- read ``ovo-bench-formatted.jsonl`` from ``--benchmark_dir``;
- stream video chunks through ``streaming_video_chat``;
- inject the question at ``datum["video_end"]`` via
  ``queries=[{"content": query, "timestamp": query_ts}]``;
- constrain the answer to the allowed OVO option tokens.

The script adds shard flags so 8 independent processes can keep 8 GPUs busy
without torch.distributed/NCCL gather overhead.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoProcessor

from thinkstream.data.stream_data_processor import (
    DEFAULT_INFERENCE_MAX_PIXELS,
    DEFAULT_INFERENCE_MIN_PIXELS,
    DEFAULT_MAX_CHUNKS,
    FRAMES_PER_CHUNK,
    QWEN_TEMPLATE_WO_SYSTEM,
    _resolve_vit_patch_size,
    preload_video,
)
from thinkstream.models import DEFAULT_VIDEO_FLEX_WINDOW_SIZE, MODEL_CLS, get_text_config
from thinkstream.models.inference import (
    StreamingWindowInferenceEngine,
    streaming_video_chat,
    think_budget_sample_restricted,
)


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You will see a continuous stream of video chunks. "
    "Based on the user's query and the video content, first output your internal "
    "reasoning enclosed in <think>...</think> tags. "
    "Then, if you determine that a response is needed at this moment, output "
    "</Response> followed by the content. "
    "If no response is needed, output </Silence>. "
    "Your generated thoughts and responses should be continuous and fluent across "
    "the video chunks."
)

BINARY_OPTIONS = ["No", "Yes"]
COUNT_OPTIONS = [str(i) for i in range(11)]
LETTER_OPTIONS = list("ABCDE")
OVO_OPTIONS = BINARY_OPTIONS + COUNT_OPTIONS + LETTER_OPTIONS


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ShardSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, num_shards: int, shard_index: int):
        self.dataset = dataset
        self.num_shards = num_shards
        self.shard_index = shard_index

    def __iter__(self):
        return iter(range(self.shard_index, len(self.dataset), self.num_shards))

    def __len__(self):
        return (len(self.dataset) - self.shard_index + self.num_shards - 1) // self.num_shards


class MCQDataset(Dataset):
    def __init__(
        self,
        path: str,
        *,
        sample: int | None = None,
        processor=None,
        model_type: str = "qwen3vl",
        frames_per_chunk: int = FRAMES_PER_CHUNK,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        min_pixels: int = DEFAULT_INFERENCE_MIN_PIXELS,
        max_pixels: int = DEFAULT_INFERENCE_MAX_PIXELS,
        slack_time: float = 3.0,
    ):
        lines = open(path, encoding="utf-8").readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in lines]
        if self.datums and isinstance(self.datums[0], str):
            self.datums = [json.loads(d) for d in self.datums]
        self.data_dir = os.path.dirname(path)
        self.processor = processor
        self.model_type = model_type
        self.frames_per_chunk = frames_per_chunk
        self.max_chunks = max_chunks
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.slack_time = slack_time
        self.vit_patch_size = _resolve_vit_patch_size(processor)

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        video_path = os.path.join(self.data_dir, datum["video"])
        preloaded = None
        try:
            query_ts = float(datum["video_end"])
            run_video_end = query_ts + self.slack_time if self.slack_time > 0 else query_ts
            preloaded = preload_video(
                video_path,
                video_start=float(datum.get("video_start", 0.0)),
                video_end=run_video_end,
                frames_per_chunk=self.frames_per_chunk,
                max_chunks=self.max_chunks,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                vit_patch_size=self.vit_patch_size,
                model_type=self.model_type,
            )
            preloaded["original_video_end"] = query_ts
        except Exception:
            preloaded = None
        return i, datum, preloaded


def _ensure_control_tokens(model, processor):
    from thinkstream.data.agent_protocol import (
        ensure_agent_special_tokens,
        validate_agent_special_tokens,
    )

    tokenizer = processor.tokenizer
    ensure_agent_special_tokens(tokenizer, model=model)
    validate_agent_special_tokens(tokenizer)
    return {
        "think_end_token_id": tokenizer.convert_tokens_to_ids("</think>"),
        "silent_token_id": tokenizer.convert_tokens_to_ids("</Silence>"),
        "response_token_id": tokenizer.convert_tokens_to_ids("</Response>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "video_token_id": tokenizer.convert_tokens_to_ids("<|video_pad|>"),
    }


def load_model_and_processor(model_path: str, local_rank: int, model_type: str, min_pixels: int, max_pixels: int):
    if model_type not in MODEL_CLS:
        raise ValueError(f"Unsupported model_type={model_type}; choose from {list(MODEL_CLS)}")
    model = MODEL_CLS[model_type].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{local_rank}",
    )
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    vp = processor.video_processor
    vp.max_pixels = max_pixels
    vp.min_pixels = min_pixels
    vp.size["shortest_edge"] = min_pixels
    vp.size["longest_edge"] = max_pixels
    token_ids = _ensure_control_tokens(model, processor)
    model.config.text_config._attn_implementation = "flash_attention_2_infer"
    model.eval()
    return model, processor, token_ids


def build_query(datum: dict, question_prefix: str, question_postfix: str) -> str:
    if datum.get("options"):
        return question_prefix + datum["question"] + "\n" + "\n".join(datum["options"]) + question_postfix
    return question_prefix + datum["question"]


def allowed_options_for_datum(datum: dict) -> list[str]:
    task = str(datum.get("task") or "")
    if task in {"CRR", "SSR"}:
        return list(BINARY_OPTIONS)
    if task == "REC":
        return list(COUNT_OPTIONS)
    options = datum.get("options") or []
    if options:
        return LETTER_OPTIONS[: min(len(options), len(LETTER_OPTIONS))]
    answer = str(datum.get("answer") or "").strip()
    if answer in BINARY_OPTIONS:
        return list(BINARY_OPTIONS)
    if answer.isdigit():
        return list(COUNT_OPTIONS)
    if answer in LETTER_OPTIONS:
        return list(LETTER_OPTIONS)
    return list(OVO_OPTIONS)


def option_token_ids(tokenizer, options: list[str]) -> list[int]:
    ids: list[int] = []
    for opt in options:
        tokenized = tokenizer(opt, add_special_tokens=False).input_ids
        if not tokenized:
            raise ValueError(f"No token id resolved for option={opt!r}")
        ids.append(int(tokenized[-1]))
    return ids


def parse_answer_token(
    gen_tokens: torch.Tensor,
    response_token_id: int,
    strict_option_ids: list[int],
    allowed_options: list[str],
) -> str:
    try:
        resp_pos = (gen_tokens == response_token_id).nonzero(as_tuple=True)[0][0].item()
        ans_token = gen_tokens[resp_pos + 1].item()
        return allowed_options[strict_option_ids.index(ans_token)]
    except Exception:
        return random.choice(allowed_options)


def result_response_for_prediction(pred: str, datum: dict) -> str:
    """Return the value scored by the OVO evaluator.

    Formatted OVO rows use letters for MC tasks. Raw/debug rows sometimes keep
    option text as the answer, so map letter predictions back to option text
    only for that case.
    """
    answer = str(datum.get("answer") or "").strip()
    options = datum.get("options") or []
    if pred in LETTER_OPTIONS and options and answer and answer not in LETTER_OPTIONS:
        idx = LETTER_OPTIONS.index(pred)
        if idx < len(options):
            opt = str(options[idx])
            for prefix in (f"{pred}.", f"{pred})", f"{pred}:"):
                if opt.startswith(prefix):
                    return opt[len(prefix):].strip()
            return opt
    return pred


def is_correct_response(result: dict) -> bool:
    answer = str(result.get("answer") or "").strip()
    response = str(result.get("response") or "").strip()
    if (
        result.get("task") in {"CRR", "SSR", "REC"}
        or answer in BINARY_OPTIONS
        or answer in LETTER_OPTIONS
        or answer.isdigit()
    ):
        return response == answer
    return response[: len(answer)] == answer


@torch.inference_mode()
def predict(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model, processor, token_ids = load_model_and_processor(
        args.model_path,
        local_rank=local_rank,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    benchmark_path = os.path.join(args.benchmark_dir, "ovo-bench-formatted.jsonl")
    dataset = MCQDataset(
        benchmark_path,
        sample=args.sample,
        processor=processor,
        model_type=args.model_type,
        frames_per_chunk=args.frames_per_chunk,
        max_chunks=args.remaining_seconds,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        slack_time=args.slack_time,
    )
    sampler = ShardSampler(dataset, args.num_shards, args.shard_index)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    option_ids_by_tuple: dict[tuple[str, ...], list[int]] = {}
    base_sample_kwargs = {
        "think_end_token_id": token_ids["think_end_token_id"],
        "max_think_tokens": args.think_budget,
        "eos_token_id": token_ids["eos_token_id"],
        "silent_token_id": token_ids["silent_token_id"],
        "response_token_id": token_ids["response_token_id"],
    }
    text_cfg = get_text_config(model.config)
    head_dim = getattr(
        text_cfg,
        "head_dim",
        text_cfg.hidden_size // text_cfg.num_attention_heads,
    )
    engine = StreamingWindowInferenceEngine(
        model,
        batch_size=1,
        max_len=args.max_len,
        num_hidden_layers=text_cfg.num_hidden_layers,
        num_key_value_heads=text_cfg.num_key_value_heads,
        head_dim=head_dim,
        vocab_size=len(processor.tokenizer),
        pad_token_id=model.generation_config.pad_token_id,
        eos_token_ids=model.generation_config.eos_token_id,
        video_token_id=token_ids["video_token_id"],
        video_flex_window_size=getattr(model.config, "video_flex_window_size", DEFAULT_VIDEO_FLEX_WINDOW_SIZE),
    )

    results = []
    for idx, datum, preloaded in tqdm.tqdm(dataloader, desc=f"shard {args.shard_index}/{args.num_shards}"):
        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])
            query_ts = float(datum["video_end"])
            run_video_end = query_ts + args.slack_time if args.slack_time > 0 else query_ts
            if preloaded is not None:
                query_ts = float(preloaded.get("original_video_end", query_ts))
                run_video_end = float(preloaded["video_end"])
            query = build_query(datum, args.question_prefix, args.question_postfix)
            allowed_options = allowed_options_for_datum(datum)
            option_key = tuple(allowed_options)
            strict_option_ids = option_ids_by_tuple.get(option_key)
            if strict_option_ids is None:
                strict_option_ids = option_token_ids(processor.tokenizer, allowed_options)
                option_ids_by_tuple[option_key] = strict_option_ids
            sample_kwargs = {
                **base_sample_kwargs,
                "restricted_token_ids": strict_option_ids,
            }
            pred = None
            decoded = ""
            for result in streaming_video_chat(
                engine=engine,
                processor=processor,
                video_path=video_path,
                queries=[{"content": query, "timestamp": query_ts}],
                video_start=float(datum.get("video_start", 0.0)),
                video_end=run_video_end,
                frames_per_chunk=args.frames_per_chunk,
                max_chunks=args.remaining_seconds,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                system_prompt=SYSTEM_PROMPT,
                chat_template_wo_system=QWEN_TEMPLATE_WO_SYSTEM,
                sample=think_budget_sample_restricted,
                sample_kwargs=sample_kwargs,
                model_type=args.model_type,
                preloaded_video=preloaded,
                slack_time=args.slack_time,
                break_on_answer=True,
            ):
                if result["is_answer"]:
                    gen_tokens = result["generated_tokens"][0]
                    pred = parse_answer_token(
                        gen_tokens,
                        token_ids["response_token_id"],
                        strict_option_ids,
                        allowed_options,
                    )
                    decoded = processor.decode(gen_tokens)
                    break
            if pred is None:
                pred = random.choice(allowed_options)
            results.append({
                "idx": int(idx),
                **datum,
                "response": result_response_for_prediction(pred, datum),
                "response_token": pred,
                "allowed_options": allowed_options,
                "success": True,
                "generated": decoded,
            })
        except Exception as exc:
            allowed_options = allowed_options_for_datum(datum)
            pred = random.choice(allowed_options)
            results.append({
                "idx": int(idx),
                **datum,
                "response": result_response_for_prediction(pred, datum),
                "response_token": pred,
                "allowed_options": allowed_options,
                "success": False,
                "error": repr(exc),
            })
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return results


def evaluate_ovobench_results(results: list[dict]):
    task_to_counts = {}
    for result in results:
        task = result["task"]
        task_to_counts.setdefault(task, {"correct": 0, "total": 0})
        task_to_counts[task]["total"] += 1
        if is_correct_response(result):
            task_to_counts[task]["correct"] += 1

    rt_accs, bt_accs, fr_accs = [], [], []
    for task, counts in sorted(task_to_counts.items()):
        acc = counts["correct"] / counts["total"]
        print(f"{task}: {counts['correct']}/{counts['total']}={acc}")
        if task in ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]:
            rt_accs.append(acc)
        elif task in ["EPM", "ASI", "HLD"]:
            bt_accs.append(acc)
        else:
            fr_accs.append(acc)
    if rt_accs:
        print(f"Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs) / len(rt_accs)}")
    if bt_accs:
        print(f"Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs) / len(bt_accs)}")
    if fr_accs:
        print(f"Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs) / len(fr_accs)}")


def save_json_and_report(results: list[dict], out: str):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results = sorted(results, key=lambda r: r["idx"])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    txt = out.replace(".json", ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            evaluate_ovobench_results(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", default="qwen3vl", choices=list(MODEL_CLS.keys()))
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--frames_per_chunk", type=int, default=FRAMES_PER_CHUNK)
    parser.add_argument("--remaining_seconds", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument("--max_len", type=int, default=24576)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--think_budget", type=int, default=20)
    parser.add_argument("--slack_time", type=float, default=3.0)
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_INFERENCE_MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_INFERENCE_MAX_PIXELS)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--question_prefix", default="")
    parser.add_argument("--question_postfix", default="\nPlease select the correct answer.")
    parser.add_argument("--merge", nargs="+", help="Merge shard JSON files instead of running inference.")
    args = parser.parse_args()

    if args.merge:
        merged = []
        for path in args.merge:
            with open(path, encoding="utf-8") as f:
                merged.extend(json.load(f))
        save_json_and_report(merged, args.out)
        return

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")
    save_json_and_report(predict(args), args.out)


if __name__ == "__main__":
    main()
