import sys
import contextlib
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _EVAL_DIR.parent
_PROJECT_ROOT = _WORKSPACE_ROOT.parent

for _p in (str(_PROJECT_ROOT), str(_WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoProcessor
from thinkstream.model import MODEL_CLS, DEFAULT_VIDEO_FLEX_WINDOW_SIZE, get_text_config
from thinkstream.model.inference import (
    StreamingWindowInferenceEngine,
    streaming_video_chat,
)
from thinkstream.data.stream_data_processor import (
    QWEN_TEMPLATE_WO_SYSTEM,
    FRAMES_PER_CHUNK,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_INFERENCE_MIN_PIXELS,
    DEFAULT_INFERENCE_MAX_PIXELS,
    preload_video,
    _resolve_vit_patch_size,
)

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_NEW_TOKENS = 30
MIN_PIXELS = DEFAULT_INFERENCE_MIN_PIXELS
MAX_PIXELS = DEFAULT_INFERENCE_MAX_PIXELS

# ─── Utility Classes ─────────────────────────────────────────────────────────


class TeeWriter:
    """Write to multiple streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class NoPadDistributedSampler(Sampler):
    """Round-robin sampler that does NOT pad/duplicate samples for even division.
    Each rank gets indices[rank::world_size], so ranks may have different lengths."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return (
            len(self.dataset) - self.rank + self.num_replicas - 1
        ) // self.num_replicas


class MCQDataset(Dataset):
    """Generic MCQ dataset that loads JSONL files.

    When *processor* and *model_type* are provided the dataset pre-loads
    video frames in ``__getitem__`` so that DataLoader ``num_workers`` can
    parallelise the heavy video I/O.
    """

    def __init__(
        self,
        path,
        sample=None,
        *,
        processor=None,
        model_type: str = "",
        frames_per_chunk: int = FRAMES_PER_CHUNK,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        slack_time: float = 0.0,
    ):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)

        self.datums = [
            json.loads(line) for line in tqdm.tqdm(lines, desc="Loading data")
        ]
        if self.datums and isinstance(self.datums[0], str):
            self.datums = [
                json.loads(d) for d in tqdm.tqdm(self.datums, desc="Loading data (x2)")
            ]

        self.data_dir = os.path.dirname(path)

        self._do_preload = processor is not None and model_type
        self._model_type = model_type
        self._frames_per_chunk = frames_per_chunk
        self._max_chunks = max_chunks
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels
        self._slack_time = slack_time
        self._vit_patch_size = (
            _resolve_vit_patch_size(processor) if processor is not None else None
        )

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        if not self._do_preload:
            return i, datum, None

        video_path = os.path.join(self.data_dir, datum["video"])
        try:
            video_end = datum.get("video_end", None)
            if video_end is None:
                raise ValueError(f"video_end is None for datum {datum}")
            original_video_end = video_end
            if self._slack_time > 0:
                original_video_end = video_end
                video_end = video_end + self._slack_time

            preloaded = preload_video(
                video_path,
                video_start=datum.get("video_start", 0.0),
                video_end=video_end,
                frames_per_chunk=self._frames_per_chunk,
                max_chunks=self._max_chunks,
                min_pixels=self._min_pixels,
                max_pixels=self._max_pixels,
                vit_patch_size=self._vit_patch_size,
                model_type=self._model_type,
            )
            if original_video_end is not None:
                preloaded["original_video_end"] = original_video_end
        except Exception:
            preloaded = None
        return i, datum, preloaded


# ─── Setup Helpers ───────────────────────────────────────────────────────────


def setup_distributed():
    """Initialize distributed process group. Returns (local_rank, rank, world_size)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def cleanup_distributed(world_size):
    if world_size > 1:
        dist.destroy_process_group()


def load_model_and_processor(
    model_path,
    local_rank=0,
    *,
    model_type: str,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    """Load model + processor and apply standard configuration."""
    if model_type not in MODEL_CLS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose from {list(MODEL_CLS.keys())}"
        )
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

    model.config.text_config._attn_implementation = "flash_attention_2_infer"
    model.eval()

    return model, processor


def add_common_args(parser):
    """Add benchmark-agnostic CLI arguments."""
    parser.add_argument(
        "--benchmark_dir", type=str, required=True, help="Path to benchmark directory."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory."
    )
    parser.add_argument("--frames_per_chunk", type=int, default=FRAMES_PER_CHUNK)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--sample", type=int, default=None, help="Subsample N items (for debugging)."
    )
    parser.add_argument(
        "--remaining_seconds",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="Max seconds to process.",
    )
    parser.add_argument(
        "--think_budget", type=int, default=20, help="Max thinking tokens budget."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen2.5vl",
        choices=list(MODEL_CLS.keys()),
        help="Model type.",
    )
    parser.add_argument(
        "--slack_time", type=float, default=3.0, help="Slack time window in seconds."
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=MIN_PIXELS,
        help="Minimum number of pixels for video processing.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=MAX_PIXELS,
        help="Maximum number of pixels for video processing.",
    )
    parser.add_argument(
        "--agent_model",
        action="store_true",
        help="Use 3-action agent protocol (think+action{silent|response|recall}).",
    )
    parser.add_argument(
        "--use_agent_loop",
        action="store_true",
        help=(
            "Drive eval through StreamingAgentLoop (per-timestep, system "
            "compress_trigger injection, recall orchestration) — matches "
            "the SFT/RL training format byte-for-byte. Required for v12 "
            "models. Kept as opt-in for legacy reasons; the eval entry "
            "scripts (eval_ovo / eval_rtvu) ignore this flag and always "
            "go through mcq_predict_agent_loop after the v12.5 cleanup."
        ),
    )
    parser.add_argument(
        "--compress_mode",
        default="system",
        choices=["system", "self"],
        help=(
            "How <action>compress</action> is triggered (only used with "
            "--use_agent_loop). 'system' (default, SFT-trained ckpt): when "
            "memory.should_compress() fires, system inserts a "
            "<compress_trigger range='X-Y'/> with a fixed FIFO range and "
            "the model only writes the <summary>. 'self' (post-GDPO RL "
            "ckpt): system never inserts a trigger; the model decides "
            "autonomously when to compress and which range to summarize. "
            "Pure-SFT ckpts under 'self' will likely never compress and "
            "overflow — only use 'self' with an RL-tuned policy."
        ),
    )
    return parser


# ─── Core Prediction ─────────────────────────────────────────────────────────


def preprocess_logits_for_metrics(logits, labels, strict_option_ids):
    """Extract logits for option tokens at the last non-padding position."""
    return torch.stack(
        [
            logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids]
            for logit in logits
        ]
    ).argmax(dim=-1)


# ─── Agent Loop Eval (v3.0: single-step format, matching training) ───────────


def mcq_predict_agent_loop(
    model,
    processor,
    dataset,
    model_type: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    frames_per_chunk: int = FRAMES_PER_CHUNK,
    question_prefix: str = "",
    question_postfix: str = "\nAnswer with a single letter.",
    compress_mode: str = "system",
    rank: int = 0,
    world_size: int = 1,
):
    """MCQ prediction using StreamingAgentLoop (single-step format).

    Ensures eval uses EXACTLY the same input format as SFT training:
    video-first layout, <memory> tags, <visual_window> tags, etc.
    """
    from thinkstream.model.agent_loop import (
        StreamingAgentLoop,
        make_generate_fn,
        AGENT_CHUNK_SEC,
    )

    tokenizer = processor.tokenizer
    generate_fn = make_generate_fn(model, processor, model_type=model_type)

    predictions = []
    datums = []

    sampler = NoPadDistributedSampler(dataset, world_size, rank)
    for idx in tqdm.tqdm(sampler, desc="AgentLoop eval", disable=(rank != 0)):
        datum = dataset.datums[idx]
        video_path = os.path.join(dataset.data_dir, datum["video"])

        if "options" in datum and datum["options"]:
            query = question_prefix + datum["question"] + "\n" + "\n".join(datum["options"]) + question_postfix
        else:
            query = datum["question"]

        # Determine timing
        video_end = datum.get("video_end")
        video_start = datum.get("video_start", 0.0)
        if video_end is None:
            continue

        query_ts = video_end
        num_chunks = int((video_end - video_start) / AGENT_CHUNK_SEC)
        ask_chunk = max(0, num_chunks - 1)

        # Run agent loop
        loop = StreamingAgentLoop(
            generate_fn=generate_fn,
            tokenizer=tokenizer,
            processor=processor,
            model_type=model_type,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            compress_mode=compress_mode,
        )

        answer_text = ""
        for chunk_idx in range(num_chunks):
            q = query if chunk_idx == ask_chunk else None
            result = loop.step(chunk_idx=chunk_idx, video_path=video_path, user_question=q)

            if result["action"] == "response":
                answer_text = result["payload"].get("response", "")
                break
            elif result["action"] == "recall" and result.get("final_action") == "response":
                answer_text = result.get("final_payload", {}).get("response", "")
                break

        # Map answer to option index
        options = datum.get("options", [])
        pred_idx = 0
        if options and answer_text:
            answer_upper = answer_text.strip().upper()
            for i, opt in enumerate(options):
                if opt.startswith(answer_upper[:1]) or answer_upper[:1] == chr(65 + i):
                    pred_idx = i
                    break

        predictions.append(pred_idx)
        datums.append(datum)

    return np.array(predictions), datums, 0


# ─── Results I/O ─────────────────────────────────────────────────────────────


def build_results(datums, predictions, options):
    """Merge all original datum fields with the predicted response."""
    return [
        {**datum, "response": options[pred_idx]}
        for datum, pred_idx in zip(datums, predictions)
    ]


def save_results(results, save_json_path, evaluate_fn):
    """Persist full results to JSON and write evaluation summary to both stdout and a .txt file."""
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    save_txt_path = save_json_path.replace(".json", ".txt")
    with open(save_txt_path, "w", encoding="utf-8") as f:
        tee = TeeWriter(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            evaluate_fn(results)
