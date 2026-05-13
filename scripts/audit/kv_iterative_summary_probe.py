#!/usr/bin/env python3
"""KV-preserving streaming caption + iterative summary compression probe.

The probe models each segment as real multi-turn streaming:
- one user turn per second;
- each user turn contains only the current 1-second visual chunk;
- assistant caption tokens stay in the same KV cache within the segment.

At a segment boundary it compresses the just-finished segment.  For segment 0,
the compression prompt relies on the accumulated KV conversation.  For segment
1+, the compression prompt explicitly includes the previous bounded summary
and the captions generated in the current segment, testing whether the memory
can remain bounded across iterations.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thinkstream.data.stream_data_processor import compute_position_ids  # noqa: E402


STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "also", "with", "without",
    "there", "their", "these", "those", "this", "that", "from", "into", "onto", "over",
    "under", "while", "where", "which", "being", "been", "have", "has", "had", "does",
    "display", "displays", "show", "shows", "shown", "scene", "video", "frame", "frames",
    "current", "visible", "appears", "appearing", "using", "wearing", "holding", "object",
    "person", "people", "image", "close", "view", "background", "foreground", "left",
    "right", "center", "central", "likely", "text", "visible", "caption", "chunk",
    "second", "seconds", "observation", "observations",
}


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def visual_window_and_video(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    user_content = row["messages"][1]["content"]
    text = text_from_content(user_content)
    match = re.search(r"<visual_window>\s*(\{.*?\})\s*</visual_window>", text, flags=re.DOTALL)
    visual_window = json.loads(match.group(1)) if match else {}
    video = []
    for item in user_content:
        if isinstance(item, dict) and item.get("type") == "video":
            video = item.get("video") or []
            break
    return visual_window, video


def load_rows(path: Path, video_id: str, start: int, end: int) -> List[Dict[str, Any]]:
    by_time: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            if row.get("video_id") != video_id:
                continue
            if row.get("sample_type") not in {"silent", "response"}:
                continue
            visual_window, video = visual_window_and_video(row)
            current_time = visual_window.get("current_time")
            if current_time is None or not video:
                continue
            current_time = int(current_time)
            if not (start <= current_time <= end):
                continue
            if current_time in by_time:
                continue
            by_time[current_time] = {
                "line_no": line_no,
                "video_id": video_id,
                "t": current_time,
                "video": video,
                "sample_type": row.get("sample_type"),
                "chunk_idx": row.get("chunk_idx"),
            }
    missing = [t for t in range(start, end + 1) if t not in by_time]
    if missing:
        raise RuntimeError(f"missing current_time values: {missing[:40]}")
    return [by_time[t] for t in range(start, end + 1)]


def set_processor_pixels(processor: Any, min_pixels: int, max_pixels: int, fps: float) -> None:
    for attr in ("image_processor", "video_processor"):
        proc = getattr(processor, attr, None)
        if proc is None:
            continue
        if hasattr(proc, "min_pixels"):
            proc.min_pixels = min_pixels
        if hasattr(proc, "max_pixels"):
            proc.max_pixels = max_pixels
        if hasattr(proc, "size") and isinstance(proc.size, dict):
            proc.size["shortest_edge"] = min_pixels
            proc.size["longest_edge"] = max_pixels
        if attr == "video_processor":
            if hasattr(proc, "fps"):
                proc.fps = fps
            if hasattr(proc, "do_sample_frames"):
                proc.do_sample_frames = False


def keyword_set(text: str, limit: int = 300) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", text.lower())
    out: List[str] = []
    seen = set()
    for word in words:
        word = word.strip("'")
        if word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        out.append(word)
        if len(out) >= limit:
            break
    return out


def keyword_recall(prediction: str, references: List[str]) -> Dict[str, Any]:
    ref = keyword_set("\n".join(references))
    if not ref:
        return {"recall": 0.0, "hits": [], "misses": [], "ref_keyword_count": 0}
    pred = set(keyword_set(prediction))
    hits = [w for w in ref if w in pred]
    misses = [w for w in ref if w not in pred]
    return {
        "recall": len(hits) / len(ref),
        "hits": hits[:80],
        "misses": misses[:80],
        "ref_keyword_count": len(ref),
    }


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({**message, "content": content})
    return normalized


class KVRunner:
    def __init__(
        self,
        model,
        processor,
        *,
        model_type: str,
        video_chunk_size: float,
        eos_ids: List[int],
        device: str,
    ) -> None:
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.video_chunk_size = video_chunk_size
        self.eos_ids = set(int(x) for x in eos_ids if x is not None)
        self.device = torch.device(device)
        self.past_key_values = None
        self.next_position = 0
        self.turn_index = 0

    def reset(self) -> None:
        self.past_key_values = None
        self.next_position = 0
        self.turn_index = 0
        if hasattr(self.model.model, "rope_deltas"):
            self.model.model.rope_deltas = None

    def _prepare_inputs(
        self,
        messages: List[Dict[str, Any]],
        video_metadata: Optional[List[Dict[str, Any]]] = None,
        *,
        add_generation_prompt: bool = True,
    ):
        messages = normalize_messages(messages)
        kwargs = dict(
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            do_sample_frames=False,
        )
        if video_metadata:
            kwargs["video_metadata"] = video_metadata
        inputs = self.processor.apply_chat_template(messages, **kwargs)
        inputs["video_chunk_size"] = self.video_chunk_size
        position_ids = compute_position_ids(inputs, self.processor, self.model_type)
        if self.past_key_values is not None:
            position_ids = position_ids + int(self.next_position)
        inputs["position_ids"] = position_ids
        inputs.pop("video_chunk_size", None)
        return inputs.to(self.device)

    @torch.inference_mode()
    def generate_turn(
        self,
        messages: List[Dict[str, Any]],
        *,
        video_metadata: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        inputs = self._prepare_inputs(messages, video_metadata, add_generation_prompt=True)
        input_len = int(inputs["input_ids"].shape[-1])
        t0 = time.time()
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=None,
            position_ids=inputs["position_ids"],
            past_key_values=self.past_key_values,
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            use_cache=True,
            logits_to_keep=1,
        )
        self.past_key_values = outputs.past_key_values
        self.next_position = int(inputs["position_ids"].max().item()) + 1

        generated: List[int] = []
        logits = outputs.logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            pos = torch.full(
                (3, 1, 1),
                int(self.next_position),
                dtype=torch.long,
                device=self.device,
            )
            step_out = self.model(
                input_ids=next_token,
                attention_mask=None,
                position_ids=pos,
                past_key_values=self.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            self.past_key_values = step_out.past_key_values
            logits = step_out.logits[:, -1, :]
            self.next_position += 1
            if token_id in self.eos_ids:
                break

        decoded = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        self.turn_index += 1
        return {
            "input_tokens": input_len,
            "new_tokens": len(generated),
            "latency_sec": time.time() - t0,
            "next_position": int(self.next_position),
            "cache_seq_len": int(self.past_key_values.get_seq_length()) if self.past_key_values is not None else 0,
            "text": decoded,
        }

    @torch.inference_mode()
    def prefill_context(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = self._prepare_inputs(messages, None, add_generation_prompt=False)
        input_len = int(inputs["input_ids"].shape[-1])
        t0 = time.time()
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=None,
            position_ids=inputs["position_ids"],
            past_key_values=self.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        self.past_key_values = outputs.past_key_values
        self.next_position = int(inputs["position_ids"].max().item()) + 1
        return {
            "input_tokens": input_len,
            "latency_sec": time.time() - t0,
            "next_position": int(self.next_position),
            "cache_seq_len": int(self.past_key_values.get_seq_length()) if self.past_key_values is not None else 0,
        }


def video_metadata_for_time(t: int, frame_count: int, fps: float) -> Dict[str, Any]:
    base = int(round(t * fps))
    indices = [base + i for i in range(frame_count)]
    return {
        "fps": fps,
        "frames_indices": indices,
        "total_num_frames": max(indices[-1] + 1 if indices else 1, 1),
    }


def build_caption_turn(
    row: Dict[str, Any],
    *,
    include_system: bool,
    frames_per_second: int,
    fps: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t = int(row["t"])
    frames = list(row["video"])[-frames_per_second:]
    metadata = video_metadata_for_time(t, len(frames), fps)
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": f'<current_vision t="{t}-{t + 1}" fps="{fps:g}">'},
            {
                "type": "video",
                "video": frames,
                "video_metadata": {**metadata, "do_sample_frames": False},
            },
            {"type": "text", "text": "</current_vision>\nDescribe only this current 1-second chunk in one concise English sentence."},
        ],
    }
    if include_system:
        return [
            {
                "role": "system",
                "content": "You are a streaming video observer. For each turn, describe only the current visual chunk, not earlier chunks.",
            },
            user,
        ], [metadata]
    return [user], [metadata]


def format_caption_memory(captions: List[Dict[str, Any]]) -> str:
    return "\n".join(f'  <m t="{c["t"]}">{c["caption"]}</m>' for c in captions)


def build_summary_turn(
    *,
    segment_start: int,
    segment_end: int,
    previous_summary: str,
    captions: List[Dict[str, Any]],
    use_explicit_captions: bool,
) -> List[Dict[str, str]]:
    prev = previous_summary.strip() or '  <m t="none">No previous summary.</m>'
    caption_block = format_caption_memory(captions)
    if use_explicit_captions:
        input_block = f"""<previous_summary>
{prev}
</previous_summary>

<current_segment_captions t="{segment_start}-{segment_end}">
{caption_block}
</current_segment_captions>"""
    else:
        input_block = (
            "Use the accumulated conversation in KV for the current segment. "
            "No explicit per-second captions are repeated here."
        )
    carry_rule = (
        "If <previous_summary> is not empty, the output must carry forward "
        "2-3 entries from it and add 2-3 entries from <current_segment_captions>. "
        "Do not output a summary that only covers the current segment."
        if previous_summary.strip()
        else "No previous summary is provided, so summarize the current segment only."
    )
    prompt = f"""Instruction:
Compress streaming video memory into one bounded historical state summary.
Target exactly 4 entries. Output at most 6 entries and never more.
Each entry must be no more than 18 words.
Each entry should be an abstract stage/event/state, not a caption.
Cover BOTH previous-summary events and current-segment events.
{carry_rule}
Merge adjacent seconds that describe the same event.
Keep only task progress, object state, and useful time ranges.
Do not copy every caption. Do not repeat the same event in many adjacent entries.
Prefer broad ranges like 0-8, 9-20, or 21-29 over one entry per second.
Do not predict future actions.
Use this format only and stop immediately after </global_state_summary>:
<global_state_summary>
  <m t="a-b">short event/state.</m>
</global_state_summary>

{input_block}"""
    return [{"role": "user", "content": prompt}]


@torch.inference_mode()
def generate_fresh_text(
    model,
    processor,
    messages: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    messages = normalize_messages(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    t0 = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    )
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    decoded = processor.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    return {
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "new_tokens": int(new_ids.shape[-1]),
        "latency_sec": time.time() - t0,
        "fresh_context": True,
        "text": decoded,
    }


def build_memory_init_turn(summary: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a streaming video observer. Keep the initialized historical summary as context, but caption only the current visual chunk in later turns.",
        },
        {
            "role": "user",
            "content": f"""<global_state_summary>
{summary.strip()}
</global_state_summary>

Historical initialization only. Store this bounded summary as context. Future current-caption turns must describe only the attached current visual chunk.""",
        },
        {"role": "assistant", "content": ""},
    ]


def load_model(model_path: str, device: str, attn_implementation: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=attn_implementation,
    ).eval()
    return model, processor


def run_segment(
    runner: KVRunner,
    rows: List[Dict[str, Any]],
    *,
    segment_start: int,
    segment_end: int,
    previous_summary: str,
    frames_per_second: int,
    fps: float,
    caption_max_tokens: int,
    summary_max_tokens: int,
    explicit_summary_input: bool,
    fresh_summary_context: bool,
    device: str,
) -> Dict[str, Any]:
    captions: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    for row in rows:
        messages, metadata = build_caption_turn(
            row,
            include_system=(runner.turn_index == 0),
            frames_per_second=frames_per_second,
            fps=fps,
        )
        out = runner.generate_turn(messages, video_metadata=metadata, max_new_tokens=caption_max_tokens)
        captions.append({"t": int(row["t"]), "caption": out["text"]})
        turns.append({"t": int(row["t"]), **out})

    summary_messages = build_summary_turn(
        segment_start=segment_start,
        segment_end=segment_end,
        previous_summary=previous_summary,
        captions=captions,
        use_explicit_captions=explicit_summary_input,
    )
    if fresh_summary_context:
        summary_out = generate_fresh_text(
            runner.model,
            runner.processor,
            summary_messages,
            max_new_tokens=summary_max_tokens,
            device=device,
        )
    else:
        summary_out = runner.generate_turn(summary_messages, max_new_tokens=summary_max_tokens)
    summary_text = summary_out["text"]
    references = [c["caption"] for c in captions]
    if previous_summary:
        references = [previous_summary] + references
    return {
        "segment_range": [segment_start, segment_end],
        "captions": captions,
        "turns": turns,
        "summary": summary_text,
        "summary_stats": summary_out,
        "summary_keyword_recall_vs_inputs": keyword_recall(summary_text, references),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=Path("data/agent_v5/batch1/rendered/video_meta_standard_query_last/val_messages.jsonl"))
    parser.add_argument("--video-id", default="HxfwLkoj2gs")
    parser.add_argument("--model", default="/home/tione/notebook/gaozhenkun/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--segment-sec", type=int, default=30)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--frames-per-second", type=int, default=2, choices=[1, 2])
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-pixels", type=int, default=130000)
    parser.add_argument("--max-pixels", type=int, default=220000)
    parser.add_argument("--caption-max-tokens", type=int, default=64)
    parser.add_argument("--summary-max-tokens", type=int, default=384)
    parser.add_argument("--explicit-first-summary", action="store_true")
    parser.add_argument("--fresh-summary-context", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/base_summary_iteration_probe/kv_iter30_hxfw.json"))
    args = parser.parse_args()

    rows = load_rows(args.jsonl, args.video_id, 0, args.segment_sec * args.segments - 1)
    model, processor = load_model(args.model, args.device, args.attn_implementation)
    set_processor_pixels(processor, args.min_pixels, args.max_pixels, args.fps)
    eos_ids = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    runner = KVRunner(
        model,
        processor,
        model_type="qwen3vl",
        video_chunk_size=1.0,
        eos_ids=eos_ids,
        device=args.device,
    )

    segments: List[Dict[str, Any]] = []
    previous_summary = ""
    for segment_idx in range(args.segments):
        seg_start = segment_idx * args.segment_sec
        seg_end = seg_start + args.segment_sec - 1
        seg_rows = [r for r in rows if seg_start <= int(r["t"]) <= seg_end]
        if segment_idx > 0:
            runner.reset()
            init = runner.prefill_context(build_memory_init_turn(previous_summary))
        else:
            init = None
        segment_result = run_segment(
            runner,
            seg_rows,
            segment_start=seg_start,
            segment_end=seg_end,
            previous_summary=previous_summary,
            frames_per_second=args.frames_per_second,
            fps=args.fps,
            caption_max_tokens=args.caption_max_tokens,
            summary_max_tokens=args.summary_max_tokens,
            explicit_summary_input=(segment_idx > 0 or args.explicit_first_summary),
            fresh_summary_context=args.fresh_summary_context,
            device=args.device,
        )
        segment_result["memory_init_turn"] = init
        segments.append(segment_result)
        previous_summary = segment_result["summary"]

    result = {
        "video_id": args.video_id,
        "jsonl": str(args.jsonl),
        "model": args.model,
        "segment_sec": args.segment_sec,
        "segments_requested": args.segments,
        "frames_per_second": args.frames_per_second,
        "fps": args.fps,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "segments": segments,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "video_id": args.video_id,
        "segments": [
            {
                "range": s["segment_range"],
                "first_caption": s["captions"][0]["caption"] if s["captions"] else "",
                "last_caption": s["captions"][-1]["caption"] if s["captions"] else "",
                "summary_recall": round(s["summary_keyword_recall_vs_inputs"]["recall"], 3),
                "summary": s["summary"],
            }
            for s in segments
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
