"""
Baseline model evaluation utilities.

Two modes corresponding to two rows in the results table:

1. **Offline baseline** (batch mode):
   Loads N frames uniformly sampled from full video, standard model.generate.
   Matches "Open-source Offline Models" in paper Table 2.

2. **Online baseline** (streaming mode):
   Same streaming engine as ThinkStream but WITHOUT think/action protocol.
   Matches "Open-source Online Models" in paper Table 2.

Debug support:
   All per-sample diagnostics go to a log file (not stdout) so you can
   tail -f during runs. Results JSON includes generated_text, frames_loaded,
   parse diagnostics. --debug enables extra JSONL debug log.
"""

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

BASELINE_MODEL_CLS = {
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl": Qwen3VLForConditionalGeneration,
}

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_common import (
    NoPadDistributedSampler,
    add_common_args,
    build_results,
    load_model_and_processor,
    save_results,
    setup_distributed,
    cleanup_distributed,
)


# ---------------------------------------------------------------------------
# Logging setup — file-based, tail-friendly
# ---------------------------------------------------------------------------


def setup_eval_logging(log_path: str, rank: int = 0) -> logging.Logger:
    """Create a file logger for eval. Only rank 0 writes.

    Usage: tail -f <log_path> to monitor during run.
    """
    log = logging.getLogger(f"baseline_eval_rank{rank}")
    log.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    log.handlers.clear()

    if rank == 0:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        log.addHandler(fh)

        # Also log to stderr for immediate visibility
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(sh)

    return log


# ---------------------------------------------------------------------------
# Debug JSONL logger
# ---------------------------------------------------------------------------


class DebugLogger:
    """Per-sample JSONL debug log. Only writes on rank 0.

    Each line is a full JSON record with input/output/diagnostics.
    tail -f <path> to watch samples as they complete.
    """

    def __init__(self, path: str, enabled: bool = False, rank: int = 0):
        self.enabled = enabled and rank == 0
        self._fh = None
        if self.enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def log(self, record: dict):
        if self._fh:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


# ---------------------------------------------------------------------------
# Answer parsing with diagnostics
# ---------------------------------------------------------------------------


def parse_answer(generated_text: str, options: list) -> dict:
    """Parse generated text into option index with diagnostics.

    Returns dict with pred_idx, matched (bool), match_type (str).
    """
    gen_stripped = generated_text.strip()
    gen_upper = gen_stripped.upper()

    # 1. Exact prefix match
    for i, opt in enumerate(options):
        if gen_upper.startswith(opt.upper()):
            return {"pred_idx": i, "matched": True, "match_type": "prefix"}

    # 2. First character match (e.g., "A." or "A)")
    first_char = gen_upper[:1] if gen_upper else ""
    for i, opt in enumerate(options):
        if opt.upper() == first_char:
            return {"pred_idx": i, "matched": True, "match_type": "first_char"}

    # 3. Contains match (e.g., "The answer is B")
    for i, opt in enumerate(options):
        if opt.upper() in gen_upper:
            return {"pred_idx": i, "matched": True, "match_type": "contains"}

    # 4. No match — random fallback
    return {
        "pred_idx": random.randint(0, len(options) - 1),
        "matched": False,
        "match_type": "random_fallback",
    }


# ---------------------------------------------------------------------------
# Streaming baseline: sample function (no think/action protocol)
# ---------------------------------------------------------------------------


def baseline_sample_restricted(
    next_token: torch.Tensor,
    logits: torch.Tensor,
    step: int,
    generated_tokens: torch.Tensor,
    generated_length: torch.Tensor,
    restricted_token_ids: list,
    eos_token_id: int,
    **kwargs,
) -> torch.Tensor:
    """Baseline restricted sampling: directly pick the best option token.

    Step 0: argmax over restricted_token_ids. Step 1+: force EOS.
    """
    device = next_token.device
    if step == 0:
        restricted_ids = torch.tensor(
            restricted_token_ids, device=device, dtype=torch.long
        )
        restricted_logits = logits[:, restricted_ids]
        top1_local = restricted_logits.argmax(dim=-1)
        top1_token = restricted_ids[top1_local]
        return top1_token.unsqueeze(1)
    else:
        return torch.full_like(next_token, eos_token_id)


# ---------------------------------------------------------------------------
# Offline baseline
# ---------------------------------------------------------------------------


def load_offline_model(
    model_path: str,
    local_rank: int = 0,
    model_type: str = "qwen3vl",
    min_pixels: int = 200704,
    max_pixels: int = 401408,
):
    """Load a vanilla HF VLM (no streaming patches) for offline evaluation."""
    if model_type not in BASELINE_MODEL_CLS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Choose from {list(BASELINE_MODEL_CLS.keys())}"
        )
    model = BASELINE_MODEL_CLS[model_type].from_pretrained(
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
    model.eval()
    return model, processor


def _load_video_frames(
    video_path: str,
    video_start: float = 0.0,
    video_end: float = None,
    max_frames: int = 64,
):
    """Load video frames as PIL Images, uniformly sampled from [start, end].

    Returns (frames, meta_dict) for diagnostics.
    """
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu())
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = int(video_start * fps)
    end_frame = int(video_end * fps) if video_end else total_frames
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    n_available = end_frame - start_frame
    n_sample = min(max_frames, n_available)
    if n_sample <= 0:
        return [], {"sampled": 0, "error": "no_frames_available"}

    indices = np.linspace(start_frame, end_frame - 1, n_sample, dtype=int)
    frames = vr.get_batch(indices.tolist()).asnumpy()

    meta = {
        "fps": round(float(fps), 2),
        "total_frames": total_frames,
        "video_duration_sec": round(total_frames / fps, 1),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "sampled": int(n_sample),
    }
    return [Image.fromarray(f) for f in frames], meta


class OfflineMCQDataset(Dataset):
    def __init__(self, path, sample=None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in lines]
        self.data_dir = os.path.dirname(path)

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        return i, self.datums[i]


def add_offline_args(parser):
    """Add offline baseline CLI arguments."""
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--model_type", type=str, default="qwen3vl",
        choices=list(BASELINE_MODEL_CLS.keys()),
    )
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Enable per-sample JSONL debug log.")
    return parser


@torch.inference_mode()
def offline_predict_mcq(
    model,
    processor,
    benchmark_path: str,
    options: list,
    question_prefix: str = "",
    question_postfix: str = "\nPlease select the correct answer.",
    max_new_tokens: int = 30,
    max_frames: int = 64,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    rank: int = 0,
    world_size: int = 1,
    sample: int = None,
    debug: bool = False,
    debug_dir: str = None,
):
    """Offline MCQ prediction with full diagnostic logging.

    All per-sample info goes to log file (tail -f friendly).
    --debug additionally writes per-sample JSONL with full input/output.
    """
    dataset = OfflineMCQDataset(benchmark_path, sample=sample)

    if world_size > 1:
        sampler = NoPadDistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler,
        collate_fn=lambda batch: batch[0],
    )

    # Setup logging
    if debug_dir is None:
        debug_dir = os.path.join(os.path.dirname(benchmark_path), "debug")
    log = setup_eval_logging(
        os.path.join(debug_dir, "offline_eval.log"), rank=rank
    )
    dbg = DebugLogger(
        os.path.join(debug_dir, "offline_debug.jsonl"),
        enabled=debug, rank=rank,
    )

    log.info(f"Offline eval: {len(dataset)} samples, {max_frames} frames, "
             f"world_size={world_size}")
    log.info(f"Options: {options}")
    log.info(f"Debug JSONL: {'ON' if debug else 'OFF'}")
    log.info(f"Log file: tail -f {debug_dir}/offline_eval.log")

    predictions = []
    datums_out = []
    local_indices = []
    stats = {"total": 0, "success": 0, "parse_matched": 0,
             "parse_fallback": 0, "errors": 0, "match_types": {},
             "correct": 0}

    for idx, datum in tqdm.tqdm(
        dataloader, desc="Offline eval", disable=(rank != 0)
    ):
        stats["total"] += 1
        t0 = time.time()
        debug_record = {
            "idx": idx, "video": datum.get("video", ""),
            "task": datum.get("task", ""),
            "question": datum.get("question", ""),
            "gt_answer": datum.get("answer", ""),
        }

        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])
            video_end = datum.get("video_end")
            video_start = datum.get("video_start", 0.0)

            if "options" in datum and datum["options"]:
                query = (
                    question_prefix + datum["question"] + "\n"
                    + "\n".join(datum["options"]) + question_postfix
                )
            else:
                query = datum["question"]

            frames, frame_meta = _load_video_frames(
                video_path, video_start=video_start,
                video_end=video_end, max_frames=max_frames,
            )
            if not frames:
                raise ValueError(f"No frames loaded from {video_path}")

            log.debug(
                f"[{idx}] video={datum['video']} frames={len(frames)} "
                f"range=[{video_start:.1f}-{video_end}s] "
                f"fps={frame_meta.get('fps')} task={datum.get('task','?')}"
            )

            debug_record["frames_loaded"] = len(frames)
            debug_record["frame_meta"] = frame_meta
            debug_record["query"] = query

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=None, videos=[frames],
                padding=True, return_tensors="pt",
            ).to(model.device)

            debug_record["input_ids_len"] = inputs["input_ids"].shape[1]

            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            generated_text = processor.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()
            generated_text_raw = processor.tokenizer.decode(
                generated, skip_special_tokens=False
            ).strip()

            debug_record["generated_text"] = generated_text
            debug_record["generated_text_raw"] = generated_text_raw
            debug_record["generated_tokens"] = len(generated)

            # Parse answer
            parse_result = parse_answer(generated_text, options)
            pred_idx = parse_result["pred_idx"]
            is_correct = options[pred_idx] == datum.get("answer", "")

            debug_record["pred_option"] = options[pred_idx]
            debug_record["parse_matched"] = parse_result["matched"]
            debug_record["match_type"] = parse_result["match_type"]
            debug_record["correct"] = is_correct

            # Log every sample to file
            mark = "✓" if is_correct else "✗"
            log.info(
                f"[{idx}] {mark} gen='{generated_text}' -> {options[pred_idx]} "
                f"(gt={datum.get('answer','?')}, match={parse_result['match_type']}, "
                f"frames={len(frames)}, {time.time()-t0:.1f}s)"
            )

            if not parse_result["matched"]:
                stats["parse_fallback"] += 1
                log.warning(
                    f"[{idx}] PARSE MISS: generated='{generated_text}' "
                    f"raw='{generated_text_raw}' "
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
                "frames_loaded": len(frames),
            })
            local_indices.append(idx)
            stats["success"] += 1

        except Exception as e:
            stats["errors"] += 1
            debug_record["error"] = str(e)
            log.error(f"[{idx}] ERROR: {e}", exc_info=True)
            predictions.append(random.randint(0, len(options) - 1))
            datums_out.append({**datum, "success": False, "error": str(e)})
            local_indices.append(idx)
        finally:
            debug_record["time_sec"] = round(time.time() - t0, 2)
            dbg.log(debug_record)
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    log.info("=" * 50)
    log.info(f"DONE: {stats['total']} samples")
    log.info(f"Success: {stats['success']}, Errors: {stats['errors']}")
    log.info(f"Parse matched: {stats['parse_matched']}, "
             f"Fallback: {stats['parse_fallback']}")
    log.info(f"Match types: {stats['match_types']}")
    if stats["success"] > 0:
        log.info(f"Running accuracy: {stats['correct']}/{stats['success']} "
                 f"= {stats['correct']/stats['success']:.1%}")
    log.info("=" * 50)

    dbg.close()

    if world_size > 1:
        local_data = list(zip(local_indices, predictions, datums_out))
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)
        if rank == 0:
            all_data = []
            for proc_data in gathered_data:
                all_data.extend(proc_data)
            all_data.sort(key=lambda x: x[0])
            return np.array([d[1] for d in all_data]), [d[2] for d in all_data], 0
        else:
            return np.array(predictions), datums_out, rank
    else:
        return np.array(predictions), datums_out, 0
