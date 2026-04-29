"""vLLM-backed offline baseline MCQ eval.

Drop-in alternative to eval_baseline.offline_predict_mcq that batches the
entire dataset into a single vLLM call. vLLM handles tensor parallelism,
dynamic batching, and paged KV cache internally — no torchrun/DDP needed.

Usage:
    from eval_baseline_vllm import offline_predict_mcq_vllm
    from vllm_engine import init_vllm_engine, make_sampling_params

    llm = init_vllm_engine(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    predictions, datums = offline_predict_mcq_vllm(
        llm, processor, benchmark_path, options, ...
    )
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_baseline import (
    DebugLogger,
    OfflineMCQDataset,
    _load_video_frames,
    parse_answer,
    setup_eval_logging,
)
from vllm_engine import make_sampling_params, prepare_vllm_input


def _build_messages(datum: dict, frames, options: list,
                    question_prefix: str, question_postfix: str) -> tuple:
    if "options" in datum and datum["options"]:
        query = (
            question_prefix + datum["question"] + "\n"
            + "\n".join(datum["options"]) + question_postfix
        )
    else:
        query = datum["question"]
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": query},
        ],
    }]
    return messages, query


def offline_predict_mcq_vllm(
    llm,
    processor,
    benchmark_path: str,
    options: list,
    question_prefix: str = "",
    question_postfix: str = "\nPlease select the correct answer.",
    max_new_tokens: int = 30,
    max_frames: int = 64,
    sample: Optional[int] = None,
    temperature: float = 0.0,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    protocol_version: str = "v11",
):
    """Offline MCQ prediction via vLLM batch generate.

    Loads frames upfront for all samples (single-thread; consider
    ThreadPoolExecutor if frame I/O is the bottleneck), submits one
    `llm.generate()` call covering the whole dataset, then parses outputs.
    """
    dataset = OfflineMCQDataset(benchmark_path, sample=sample)

    if debug_dir is None:
        debug_dir = os.path.join(os.path.dirname(benchmark_path), "debug")
    log = setup_eval_logging(
        os.path.join(debug_dir, "offline_eval_vllm.log"), rank=0
    )
    dbg = DebugLogger(
        os.path.join(debug_dir, "offline_debug_vllm.jsonl"),
        enabled=debug, rank=0,
    )

    log.info(f"vLLM offline eval: {len(dataset)} samples, {max_frames} frames")
    log.info(f"Options: {options}, protocol_version={protocol_version}")

    # v12.0: pass tools=TOOLS_SCHEMA to chat_template so the system prompt
    # auto-renders <tools>...</tools> and the model can emit
    # <tool_call>{...}</tool_call>. Required for protocol_version='v12';
    # leave None for v11 legacy <action>X</action> format.
    tools_for_template = None
    if protocol_version == "v12":
        from thinkstream.data.agent_protocol import TOOLS_SCHEMA
        tools_for_template = TOOLS_SCHEMA

    # ── Phase 1: build all vLLM requests (frame loading + prompt prep) ──
    requests: List[dict] = []
    request_meta: List[dict] = []  # parallel list with idx, datum, frames_loaded, etc.
    skipped: List[dict] = []

    t_prep0 = time.time()
    for i in tqdm.tqdm(range(len(dataset)), desc="Building vLLM inputs"):
        idx, datum = dataset[i]
        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])
            frames, frame_meta = _load_video_frames(
                video_path,
                video_start=datum.get("video_start", 0.0),
                video_end=datum.get("video_end"),
                max_frames=max_frames,
            )
            if not frames:
                raise ValueError(f"No frames loaded from {video_path}")

            messages, query = _build_messages(
                datum, frames, options, question_prefix, question_postfix,
            )
            req = prepare_vllm_input(messages, processor, tools=tools_for_template)

            requests.append(req)
            request_meta.append({
                "idx": idx,
                "datum": datum,
                "frames_loaded": len(frames),
                "frame_meta": frame_meta,
                "query": query,
            })
        except Exception as e:
            log.error(f"[{idx}] PREP ERROR: {e}", exc_info=True)
            skipped.append({"idx": idx, "datum": datum, "error": str(e)})

    log.info(
        f"Prepared {len(requests)} requests in {time.time() - t_prep0:.1f}s "
        f"(skipped {len(skipped)})"
    )

    # ── Phase 2: single batched vLLM generate ──
    sampling_params = make_sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=1 if temperature == 0.0 else -1,
    )
    log.info(
        f"Sampling: max_new_tokens={max_new_tokens}, temperature={temperature}"
    )

    t_gen0 = time.time()
    outputs = llm.generate(requests, sampling_params=sampling_params)
    gen_time = time.time() - t_gen0
    log.info(
        f"vLLM generate done in {gen_time:.1f}s "
        f"({len(requests) / max(gen_time, 1e-6):.2f} samples/s)"
    )

    # ── Phase 3: parse outputs back into predictions ──
    predictions: List[int] = []
    datums_out: List[dict] = []
    indices: List[int] = []
    stats = {"total": 0, "success": 0, "parse_matched": 0,
             "parse_fallback": 0, "errors": 0, "match_types": {},
             "correct": 0}

    # vLLM preserves request order; align by position.
    for meta, output in zip(request_meta, outputs):
        idx = meta["idx"]
        datum = meta["datum"]
        stats["total"] += 1
        debug_record = {
            "idx": idx, "video": datum.get("video", ""),
            "task": datum.get("task", ""),
            "question": datum.get("question", ""),
            "gt_answer": datum.get("answer", ""),
            "frames_loaded": meta["frames_loaded"],
            "frame_meta": meta["frame_meta"],
            "query": meta["query"],
        }

        try:
            generated_text = output.outputs[0].text.strip()
            generated_text_raw = generated_text  # vLLM already returns decoded text
            generated_tokens = len(output.outputs[0].token_ids)

            debug_record["generated_text"] = generated_text
            debug_record["generated_text_raw"] = generated_text_raw
            debug_record["generated_tokens"] = generated_tokens

            parse_result = parse_answer(generated_text, options)
            pred_idx = parse_result["pred_idx"]
            is_correct = options[pred_idx] == datum.get("answer", "")

            debug_record["pred_option"] = options[pred_idx]
            debug_record["parse_matched"] = parse_result["matched"]
            debug_record["match_type"] = parse_result["match_type"]
            debug_record["correct"] = is_correct

            mark = "✓" if is_correct else "✗"
            log.info(
                f"[{idx}] {mark} gen='{generated_text}' -> {options[pred_idx]} "
                f"(gt={datum.get('answer','?')}, "
                f"match={parse_result['match_type']})"
            )

            if not parse_result["matched"]:
                stats["parse_fallback"] += 1
                log.warning(
                    f"[{idx}] PARSE MISS: generated='{generated_text}' "
                    f"gt={datum.get('answer','?')}"
                )
            else:
                stats["parse_matched"] += 1
            mt = parse_result["match_type"]
            stats["match_types"][mt] = stats["match_types"].get(mt, 0) + 1
            if is_correct:
                stats["correct"] += 1

            predictions.append(pred_idx)
            datums_out.append({
                **datum, "success": True,
                "generated_text": generated_text,
                "parse_matched": parse_result["matched"],
                "match_type": parse_result["match_type"],
                "frames_loaded": meta["frames_loaded"],
            })
            indices.append(idx)
            stats["success"] += 1

        except Exception as e:
            stats["errors"] += 1
            debug_record["error"] = str(e)
            log.error(f"[{idx}] PARSE ERROR: {e}", exc_info=True)
            predictions.append(random.randint(0, len(options) - 1))
            datums_out.append({**datum, "success": False, "error": str(e)})
            indices.append(idx)
        finally:
            dbg.log(debug_record)

    # Append skipped samples (frame-prep failures) at the end as random preds.
    for sk in skipped:
        stats["total"] += 1
        stats["errors"] += 1
        predictions.append(random.randint(0, len(options) - 1))
        datums_out.append({**sk["datum"], "success": False, "error": sk["error"]})
        indices.append(sk["idx"])

    # ── Summary ──
    log.info("=" * 50)
    log.info(f"DONE: {stats['total']} samples")
    log.info(f"Success: {stats['success']}, Errors: {stats['errors']}")
    log.info(f"Parse matched: {stats['parse_matched']}, "
             f"Fallback: {stats['parse_fallback']}")
    log.info(f"Match types: {stats['match_types']}")
    if stats["success"] > 0:
        log.info(
            f"Running accuracy: {stats['correct']}/{stats['success']} "
            f"= {stats['correct'] / stats['success']:.1%}"
        )
    log.info("=" * 50)

    dbg.close()

    # Sort by original idx so downstream build_results() matches dataset order.
    paired = sorted(zip(indices, predictions, datums_out), key=lambda t: t[0])
    predictions_sorted = [p for _, p, _ in paired]
    datums_sorted = [d for _, _, d in paired]

    return np.array(predictions_sorted), datums_sorted
