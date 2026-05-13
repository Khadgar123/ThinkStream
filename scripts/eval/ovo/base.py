#!/usr/bin/env python
"""OVO-Bench base eval — VLM-only (no streaming agent, no recall, no compress).

Used to measure "what does a plain Qwen3-VL get on OVO when given N frames of
video?" — the ceiling/baseline a streaming agent must beat.

Video-context modes (--mode):
  streaming       — uniformly sample --max_frames frames from
                    [ask_realtime - visual_window_sec, ask_realtime].
                    Apples-to-apples vs the streaming agent's visual_window.
  offline_prefix  — uniformly sample --max_frames frames from [0, ask_realtime].
                    No future leakage; useful as a prefix-only VLM baseline.
  offline_full    — uniformly sample --max_frames frames from the whole video.
                    Offline upper bound; not timing-comparable to online agents.
  oracle_support  — sample from annotated event/clue windows when OVO exposes
                    them; diagnostic only, not a paper-comparable baseline.
  official_prompt_prefix — official OVO prompts with pre-extracted prefix
                    frames; fast approximation to official chunked videos.
  official_offline — use OVO-Bench official chunked_videos/{id}.mp4 and
                    chunked_videos/{id}_{probe}.mp4 plus official prompts.

The legacy name "offline" is kept as an alias of "offline_prefix".

For FT (REC/SSR/CRR) tasks we evaluate at EACH test_info probe time. In
streaming/offline_prefix modes, the model sees only frames up to the probe
time. In offline_full mode, it sees uniformly sampled frames from the whole
video. For RT/BT (MCQ) tasks we evaluate once at sample.realtime.

Usage:
    python scripts/eval/ovo/base.py \\
        --ckpt Qwen/Qwen3-VL-8B-Instruct \\
        --benchmark_json /path/to/ovo_bench_new.json \\
        --video_root /path/to/videos \\
        --frames_root data/agent_v5/frames \\
        --mode streaming \\
        --max_frames 24
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

from thinkstream.sft.args import DataArguments
from thinkstream.sft.data_processor import update_processor_pixels
from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS
from scripts.eval.ovo.eval_full import (
    RT_TASKS, BT_TASKS, FT_TASKS, ALL_TASKS,
    detect_model_class,
    build_mcq_question, build_rec_question, build_ssr_question, build_crr_question,
    extract_letter, extract_int, is_yes, is_no,
    resolve_video_path,
)


from thinkstream.data.agent_protocol import (  # canonical v12.5 timing
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    VISUAL_WINDOW_CHUNKS,
    append_visual_frames,
    normalize_frame_protocol,
)
DEFAULT_VISUAL_WINDOW_SEC = float(VISUAL_WINDOW_CHUNKS * AGENT_CHUNK_SEC)
DEFAULT_FRAME_FPS = float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC)


def _default_frames_root() -> str:
    root = os.environ.get("THINKSTREAM_FRAMES_ROOT")
    if root:
        return root
    data_root = os.environ.get("THINKSTREAM_DATA_ROOT") or os.environ.get("AGENT_DATA_DIR")
    if data_root:
        p = Path(data_root)
        return str(p.parent / "frames") if p.name == "final" else str(p / "frames")
    return "data/agent_v5/frames"


# ─── Frame sampling ──────────────────────────────────────────────────────────


def sample_frame_paths(frame_dir: Path,
                       t_start: float, t_end: float, n_frames: int,
                       fps: float = DEFAULT_FRAME_FPS):
    """Pick at most n_frames frame paths from [t_start, t_end].

    Frames are pre-extracted at the project runtime FPS. Project frame_*.jpg
    names are 1-based frame indices; plain numeric names are treated as
    zero-based frame indices.
    """
    if not frame_dir.exists():
        return None
    all_frames = sorted(frame_dir.glob("*.jpg"))
    if not all_frames:
        return None
    # Filter to [t_start, t_end] by timestamp reconstructed from frame index.
    # Frame files are named frame_{idx:06d}.jpg (1-based) or {idx:06d}.jpg.
    in_range = []
    for fp in all_frames:
        try:
            stem = fp.stem
            if stem.startswith("frame_"):
                frame_idx = int(stem[6:]) - 1
            else:
                frame_idx = int(stem)
        except ValueError:
            continue
        frame_t = max(0, frame_idx) / float(fps)
        if t_start <= frame_t <= t_end:
            in_range.append(fp)
    if not in_range:
        return None
    if len(in_range) <= n_frames:
        return [str(f) for f in in_range]
    # Even sampling
    step = (len(in_range) - 1) / (n_frames - 1) if n_frames > 1 else 1
    indices = [round(i * step) for i in range(n_frames)]
    return [str(in_range[i]) for i in indices]


def _base_system_prompt(frame_protocol: str) -> str:
    if normalize_frame_protocol(frame_protocol) == "video_meta":
        visual = (
            "You receive a pre-sampled video block whose Qwen video_metadata "
            "carries fps, frame indices, and timestamps. Use those timestamps "
            "as real video time."
        )
    else:
        visual = (
            "You receive timestamp-tagged images. Each frame is preceded by "
            "structural metadata like <frame ts=\"12.5\" role=\"visual frame\" />; "
            "use it as real video time, but never copy it."
        )
    return (
        "You are a helpful video understanding assistant. "
        f"{visual} "
        "Answer using the requested format: yes/no questions -> Yes or No; "
        "counting questions -> an integer; multiple-choice questions -> a "
        "single letter A/B/C/D unless explicitly instructed otherwise."
    )


def _chat_template_supports_thinking(processor) -> bool:
    tmpl = getattr(processor, "chat_template", None)
    if not tmpl and hasattr(processor, "tokenizer"):
        tmpl = getattr(processor.tokenizer, "chat_template", None)
    return "enable_thinking" in (tmpl or "")


def build_messages(frame_paths, question, *, frame_protocol="video_meta",
                   fps: float = DEFAULT_FRAME_FPS, include_system: bool = True,
                   min_pixels=None, max_pixels=None, total_pixels=None):
    frame_protocol = normalize_frame_protocol(frame_protocol)
    frame_list = list(frame_paths)
    user_content = []
    # Fallback: if single element is a video file, use video block directly
    if len(frame_list) == 1 and str(frame_list[0]).endswith(('.mp4', '.avi', '.mov', '.mkv')):
        item = {"type": "video", "video": str(frame_list[0]), "fps": fps}
        if min_pixels is not None:
            item["min_pixels"] = min_pixels
        if max_pixels is not None:
            item["max_pixels"] = max_pixels
        if total_pixels is not None:
            item["total_pixels"] = total_pixels
        user_content.append(item)
    else:
        # Qwen video blocks require at least two temporal frames. Very early
        # probes can legitimately select only one pre-extracted frame.
        if len(frame_list) == 1:
            frame_list = frame_list * 2
        append_visual_frames(
            user_content,
            frame_list,
            frame_protocol=frame_protocol,
            fps=fps,
            context_label="visual frame",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            kv_scope="ordinary",
        )
        if total_pixels is not None and frame_protocol == "video_meta":
            for item in reversed(user_content):
                if isinstance(item, dict) and item.get("type") == "video":
                    item["total_pixels"] = total_pixels
                    break
    user_content.append({"type": "text", "text": question})
    user_msg = {
        "role": "user",
        "content": user_content,
    }
    if not include_system:
        return [user_msg]
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": _base_system_prompt(frame_protocol),
            }],
        },
        user_msg,
    ]


# ─── Per-task evaluators (no agent_loop) ─────────────────────────────────────


def eval_one_probe(model, processor, pad_id,
                   frame_paths, question, max_new_tokens,
                   frame_protocol="video_meta", fps: float = DEFAULT_FRAME_FPS,
                   enable_thinking=None, include_system: bool = True,
                   min_pixels=None, max_pixels=None, total_pixels=None,
                   preprocess: str = "direct"):
    """Single VLM forward. Returns the decoded text."""
    frame_list = list(frame_paths)
    direct_video = (
        len(frame_list) == 1
        and str(frame_list[0]).endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    messages = build_messages(
        frame_list,
        question,
        frame_protocol=frame_protocol,
        fps=fps,
        include_system=include_system,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
    )
    if preprocess == "qwen_vl_utils":
        if process_vision_info is None:
            raise RuntimeError(
                "qwen-vl-utils preprocessing requested, but qwen_vl_utils "
                "is not importable"
            )
        text_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            text_kwargs["enable_thinking"] = bool(enable_thinking)
        text = processor.apply_chat_template(messages, **text_kwargs)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=getattr(processor.image_processor, "patch_size", 16),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        preferred_video_metadata = []
        for msg in messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "video":
                    meta = item.get("video_metadata")
                    if isinstance(meta, dict):
                        preferred_video_metadata.append({
                            k: v for k, v in meta.items()
                            if k != "do_sample_frames"
                        })
                    else:
                        preferred_video_metadata.append(None)
        video_metadatas = None
        if video_inputs is not None:
            normalized_videos = []
            video_metadatas = []
            for i, video_input in enumerate(video_inputs):
                util_meta = None
                if isinstance(video_input, tuple) and len(video_input) == 2:
                    video_tensor, util_meta = video_input
                else:
                    video_tensor = video_input
                meta = (
                    preferred_video_metadata[i]
                    if i < len(preferred_video_metadata)
                    and preferred_video_metadata[i] is not None
                    else util_meta
                )
                if isinstance(meta, dict):
                    meta = {k: v for k, v in meta.items()
                            if k != "do_sample_frames"}
                    if hasattr(video_tensor, "shape") and len(video_tensor.shape) >= 1:
                        n_frames = int(video_tensor.shape[0])
                        indices = list(meta.get("frames_indices") or [])
                        if indices and len(indices) < n_frames:
                            indices.extend([indices[-1]] * (n_frames - len(indices)))
                        elif len(indices) > n_frames:
                            indices = indices[:n_frames]
                        if indices:
                            meta["frames_indices"] = indices
                            meta["total_num_frames"] = int(max(
                                meta.get("total_num_frames") or 0,
                                max(indices) + 1,
                            ))
                normalized_videos.append(video_tensor)
                video_metadatas.append(meta)
            video_inputs = normalized_videos
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **(video_kwargs or {}),
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = gen[0, prompt_len:]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    template_kwargs = dict(
        tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    )
    # Compatibility: transformers >= 5.0 requires do_sample_frames
    # via processor_kwargs; older versions accept it directly.
    import inspect
    sig = inspect.signature(processor.apply_chat_template)
    do_sample_frames = bool(direct_video)
    if "processor_kwargs" in sig.parameters:
        template_kwargs["processor_kwargs"] = {"do_sample_frames": do_sample_frames}
    else:
        template_kwargs["do_sample_frames"] = do_sample_frames
    video_metadata = []
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "video":
                meta = item.get("video_metadata")
                if isinstance(meta, dict):
                    video_metadata.append({k: v for k, v in meta.items()
                                           if k != "do_sample_frames"})
    if video_metadata:
        template_kwargs["video_metadata"] = video_metadata
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)
    vp = getattr(processor, "video_processor", None)
    old_do_sample = getattr(vp, "do_sample_frames", None) if vp is not None else None
    if direct_video and vp is not None and hasattr(vp, "do_sample_frames"):
        vp.do_sample_frames = True
    try:
        inputs = processor.apply_chat_template(
            messages, **template_kwargs,
        )
    finally:
        if old_do_sample is not None:
            vp.do_sample_frames = old_do_sample
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id,
        )
    new_tokens = gen[0, prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _normalize_mode(mode: str) -> str:
    if mode == "offline":
        return "offline_prefix"
    return mode


def _official_mcq_question(sample):
    formatted_options = "; ".join(
        f"{chr(65 + i)}. {option}"
        for i, option in enumerate(sample.get("options", []))
    ) + ";"
    return (
        f"Question: {sample['question']} Options: {formatted_options} "
        "Respond only with the letter corresponding to your chosen option "
        "(e.g., A, B, C). Do not include any additional text or explanation "
        "in your response."
    )


def _official_rec_question(sample):
    activity = sample.get("activity", "perform the action")
    question = "How many times did they " + activity + "?"
    return (
        "You're watching a video in which people may perform a certain type "
        "of action repetively.\n"
        "The person performing this kind of action are referred to as 'they' "
        "in the following statement. You're task is to count how many times "
        "have different people in the video perform this kind of action in "
        f"total. One complete motion counts as one. Now, answer the following "
        f"question: {question} Provide your answer as a single number (e.g., "
        "0, 1, 2, 3...) indicating the total count. Do not include any "
        "additional text or explanation in your response."
    )


def _official_ssr_question(step_text):
    return (
        "You're watching a tutorial video which contain a sequential of steps.\n"
        f"The following is one step from the whole procedures: {step_text} "
        "Your task is to determine if the man or woman in the video is "
        "currently performing this step. Answer only with \"Yes\" or \"No\". "
        "Do not include any additional text or explanation in your response."
    )


def _official_crr_question(sample):
    return (
        "You're responsible of answering questions based on the video content. "
        "The following question are relevant to the latest frames, i.e. the "
        f"end of the video.\n{sample['question']} Decide whether existing "
        "visual content, especially latest frames, i.e. frames that near the "
        "end of the video, provide enough information for answering the "
        "question. Answer only with \"Yes\" or \"No\". Do not include any "
        "additional text or explanation in your response."
    )


def _official_chunk_path(chunked_dir, sample, probe_index=None):
    stem = str(sample.get("id"))
    if sample.get("task") in FT_TASKS and probe_index is not None:
        stem = f"{stem}_{probe_index}"
    return str(Path(chunked_dir) / f"{stem}.mp4")


def _uses_official_prompt(mode: str) -> bool:
    return mode in {"official_offline", "official_prompt_prefix"}


def _frame_sampling_mode(mode: str) -> str:
    return "offline_prefix" if mode == "official_prompt_prefix" else mode


def _oracle_support_window(sample, probe, ask_t, visual_window_sec):
    """Best-effort evidence window from OVO annotations.

    This is intentionally a diagnostic mode: it answers whether the VLM can solve
    the task when the relevant evidence is likely visible, not whether it obeys
    an online/paper protocol.
    """
    sample = sample or {}
    probe = probe or {}
    task = sample.get("task")
    margin = max(2.0, min(8.0, float(visual_window_sec) / 2.0))
    ask_t = float(ask_t)

    if task == "REC":
        starts = sample.get("start_times") or []
        ends = sample.get("end_times") or []
        intervals = []
        for s, e in zip(starts, ends):
            try:
                s_f, e_f = float(s), float(e)
            except (TypeError, ValueError):
                continue
            if s_f <= ask_t:
                intervals.append((s_f, min(e_f, ask_t)))
        if intervals:
            return max(0.0, min(s for s, _ in intervals) - margin), max(e for _, e in intervals) + margin

    if task == "SSR":
        step = str(probe.get("step", ""))
        steps = [str(x) for x in (sample.get("all_steps") or [])]
        starts = sample.get("start_time") or []
        ends = sample.get("end_time") or []
        idx = next((i for i, x in enumerate(steps) if x == step), None)
        if idx is not None and idx < len(starts) and idx < len(ends):
            try:
                s_f, e_f = float(starts[idx]), float(ends[idx])
            except (TypeError, ValueError):
                s_f = e_f = None
            if s_f is not None:
                if probe.get("type") == 1 or s_f <= ask_t:
                    return max(0.0, s_f - margin), max(ask_t, e_f) + margin

    if task == "CRR":
        try:
            clue_t = float(sample.get("clue_time", ask_t))
        except (TypeError, ValueError):
            clue_t = ask_t
        if probe.get("type") == 1 or ask_t >= clue_t:
            return max(0.0, clue_t - margin), ask_t + margin

    return max(0.0, ask_t - float(visual_window_sec)), ask_t


def _frames_for_probe(frames_root, video_root, video_path, mode, ask_t,
                      visual_window_sec, max_frames,
                      sample=None, probe=None,
                      fps: float = DEFAULT_FRAME_FPS):
    """Pick frame range for a probe at time ask_t (seconds)."""
    mode = _normalize_mode(mode)
    vp = Path(video_path)
    if video_root:
        try:
            rel = vp.relative_to(Path(video_root))
            frame_dir = Path(frames_root) / rel.with_suffix("")
        except ValueError:
            frame_dir = Path(frames_root) / vp.with_suffix("")
    else:
        frame_dir = Path(frames_root) / vp.with_suffix("")
    if mode == "streaming":
        t_start = max(0.0, ask_t - visual_window_sec)
        t_end = ask_t
    elif mode == "offline_full":
        t_start = 0.0
        t_end = float("inf")
    elif mode == "oracle_support":
        t_start, t_end = _oracle_support_window(
            sample, probe, ask_t, visual_window_sec)
    else:  # offline_prefix
        t_start = 0.0
        t_end = ask_t
    return sample_frame_paths(frame_dir, t_start, t_end, max_frames, fps=fps)


def _strict_letter(text: str):
    """Strict format check for MC: response.strip() must be exactly one letter."""
    if not text:
        return None
    s = text.strip()
    return s if re.fullmatch(r"[A-Z]", s) else None


def _strict_yes_no(text: str):
    if not text:
        return None
    s = text.strip()
    return s if s in ("Yes", "No") else None


def _strict_int(text: str):
    if not text:
        return None
    s = text.strip()
    return s if s.isdigit() else None


def eval_mcq_base(sample, model, processor, pad_id, video_root, frames_root,
                  chunked_dir,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient", frame_protocol="video_meta",
                  fps: float = DEFAULT_FRAME_FPS, enable_thinking=None,
                  min_pixels=None, max_pixels=None, total_pixels=None,
                  preprocess="direct"):
    realtime = float(sample["realtime"])
    if mode == "official_offline":
        fp = [_official_chunk_path(chunked_dir, sample)]
        question = _official_mcq_question(sample)
        include_system = False
    else:
        video_path = resolve_video_path(sample["video"], video_root)
        if not Path(video_path).exists():
            return None
        fp = _frames_for_probe(frames_root, video_root, video_path,
                               _frame_sampling_mode(mode), realtime,
                               visual_window_sec, max_frames, sample=sample, fps=fps)
        question = (
            _official_mcq_question(sample)
            if _uses_official_prompt(mode)
            else build_mcq_question(sample)
        )
        include_system = not _uses_official_prompt(mode)
    if not fp:
        return None
    if len(fp) == 1 and str(fp[0]).endswith(".mp4") and not Path(fp[0]).exists():
        return None
    text = eval_one_probe(
        model, processor, pad_id, fp, question, max_new_tokens,
        frame_protocol=frame_protocol, fps=fps,
        enable_thinking=enable_thinking, include_system=include_system,
        min_pixels=min_pixels, max_pixels=max_pixels,
        total_pixels=total_pixels, preprocess=preprocess,
    )
    pred = _strict_letter(text) if scoring == "strict" else extract_letter(text)
    gt = chr(65 + sample["gt"])
    return {
        "task": sample["task"], "id": sample.get("id"),
        "probes": [{
            "realtime": realtime, "gt": gt, "pred": pred,
            "raw": text[:200], "correct": pred == gt,
            "targeted_correct": pred == gt,
            "strict_correct": pred == gt,
        }],
    }


def eval_rec_base(sample, model, processor, pad_id, video_root, frames_root,
                  chunked_dir,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient", frame_protocol="video_meta",
                  fps: float = DEFAULT_FRAME_FPS, enable_thinking=None,
                  min_pixels=None, max_pixels=None, total_pixels=None,
                  preprocess="direct"):
    if mode == "official_offline":
        video_path = None
        question = _official_rec_question(sample)
        include_system = False
    else:
        video_path = resolve_video_path(sample["video"], video_root)
        if not Path(video_path).exists():
            return None
        question = (
            _official_rec_question(sample)
            if _uses_official_prompt(mode)
            else build_rec_question(sample)
        )
        include_system = not _uses_official_prompt(mode)
    probes = []
    for probe_index, probe in enumerate(sample["test_info"]):
        ask_t = float(probe["realtime"])
        if mode == "official_offline":
            fp = [_official_chunk_path(chunked_dir, sample, probe_index)]
        else:
            fp = _frames_for_probe(frames_root, video_root, video_path,
                                   _frame_sampling_mode(mode), ask_t,
                                   visual_window_sec, max_frames, sample=sample,
                                   probe=probe, fps=fps)
        if not fp:
            continue
        if len(fp) == 1 and str(fp[0]).endswith(".mp4") and not Path(fp[0]).exists():
            continue
        text = eval_one_probe(
            model, processor, pad_id, fp, question, max_new_tokens,
            frame_protocol=frame_protocol, fps=fps,
            enable_thinking=enable_thinking, include_system=include_system,
            min_pixels=min_pixels, max_pixels=max_pixels,
            total_pixels=total_pixels, preprocess=preprocess,
        )
        if scoring == "strict":
            s = _strict_int(text)
            pred = int(s) if s else None
        else:
            pred = extract_int(text)
        gt = int(probe["count"])
        probes.append({
            "realtime": ask_t, "gt": gt, "pred": pred,
            "raw": text[:200], "correct": pred == gt,
            "targeted_correct": pred == gt,
            "strict_correct": pred == gt,
            "count_abs_error": abs(pred - gt) if pred is not None else None,
        })
    return {"task": sample["task"], "id": sample.get("id"), "probes": probes}


def _yes_no_pred(text, scoring):
    if scoring == "strict":
        return _strict_yes_no(text)
    if is_yes(text):
        return "Yes"
    if is_no(text):
        return "No"
    return None


def eval_ssr_base(sample, model, processor, pad_id, video_root, frames_root,
                  chunked_dir,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient", frame_protocol="video_meta",
                  fps: float = DEFAULT_FRAME_FPS, enable_thinking=None,
                  min_pixels=None, max_pixels=None, total_pixels=None,
                  preprocess="direct"):
    if mode == "official_offline":
        video_path = None
        include_system = False
    else:
        video_path = resolve_video_path(sample["video"], video_root)
        if not Path(video_path).exists():
            return None
        include_system = not _uses_official_prompt(mode)
    probes = []
    for probe_index, probe in enumerate(sample["test_info"]):
        ask_t = float(probe["realtime"])
        if mode == "official_offline":
            fp = [_official_chunk_path(chunked_dir, sample, probe_index)]
            question = _official_ssr_question(probe.get("step", ""))
        else:
            fp = _frames_for_probe(frames_root, video_root, video_path,
                                   _frame_sampling_mode(mode), ask_t,
                                   visual_window_sec, max_frames, sample=sample,
                                   probe=probe, fps=fps)
            question = (
                _official_ssr_question(probe.get("step", ""))
                if _uses_official_prompt(mode)
                else build_ssr_question(probe.get("step", ""))
            )
        if not fp:
            continue
        if len(fp) == 1 and str(fp[0]).endswith(".mp4") and not Path(fp[0]).exists():
            continue
        text = eval_one_probe(
            model, processor, pad_id, fp, question, max_new_tokens,
            frame_protocol=frame_protocol, fps=fps,
            enable_thinking=enable_thinking, include_system=include_system,
            min_pixels=min_pixels, max_pixels=max_pixels,
            total_pixels=total_pixels, preprocess=preprocess,
        )
        gt = "Yes" if probe.get("type") == 1 else "No"
        pred = _yes_no_pred(text, scoring)
        probes.append({
            "realtime": ask_t, "gt": gt, "pred": pred,
            "raw": text[:200], "correct": pred == gt,
            "targeted_correct": pred == gt,
            "strict_correct": pred == gt,
        })
    return {"task": sample["task"], "id": sample.get("id"), "probes": probes}


def eval_crr_base(sample, model, processor, pad_id, video_root, frames_root,
                  chunked_dir,
                  mode, visual_window_sec, max_frames, max_new_tokens,
                  scoring="lenient", frame_protocol="video_meta",
                  fps: float = DEFAULT_FRAME_FPS, enable_thinking=None,
                  min_pixels=None, max_pixels=None, total_pixels=None,
                  preprocess="direct"):
    if mode == "official_offline":
        video_path = None
        question = _official_crr_question(sample)
        include_system = False
    else:
        video_path = resolve_video_path(sample["video"], video_root)
        if not Path(video_path).exists():
            return None
        question = (
            _official_crr_question(sample)
            if _uses_official_prompt(mode)
            else build_crr_question(sample)
        )
        include_system = not _uses_official_prompt(mode)
    probes = []
    for probe_index, probe in enumerate(sample["test_info"]):
        ask_t = float(probe["realtime"])
        if mode == "official_offline":
            fp = [_official_chunk_path(chunked_dir, sample, probe_index)]
        else:
            fp = _frames_for_probe(frames_root, video_root, video_path,
                                   _frame_sampling_mode(mode), ask_t,
                                   visual_window_sec, max_frames, sample=sample,
                                   probe=probe, fps=fps)
        if not fp:
            continue
        if len(fp) == 1 and str(fp[0]).endswith(".mp4") and not Path(fp[0]).exists():
            continue
        text = eval_one_probe(
            model, processor, pad_id, fp, question, max_new_tokens,
            frame_protocol=frame_protocol, fps=fps,
            enable_thinking=enable_thinking, include_system=include_system,
            min_pixels=min_pixels, max_pixels=max_pixels,
            total_pixels=total_pixels, preprocess=preprocess,
        )
        gt = "Yes" if probe.get("type") == 1 else "No"
        pred = _yes_no_pred(text, scoring)
        probes.append({
            "realtime": ask_t, "gt": gt, "pred": pred,
            "type": probe.get("type"),
            "raw": text[:200], "correct": pred == gt,
            "targeted_correct": pred == gt,
            "strict_correct": pred == gt,
        })
    return {"task": sample["task"], "id": sample.get("id"), "probes": probes}


def dispatch(sample, **kw):
    task = sample.get("task")
    if task in RT_TASKS or task in BT_TASKS:
        return eval_mcq_base(sample, **kw)
    if task == "REC":
        return eval_rec_base(sample, **kw)
    if task == "SSR":
        return eval_ssr_base(sample, **kw)
    if task == "CRR":
        return eval_crr_base(sample, **kw)
    return None


def _sample_eval_cost(sample):
    """Approximate number of model forwards needed for load-balanced sharding."""
    test_info = sample.get("test_info")
    if isinstance(test_info, list) and test_info:
        return len(test_info)
    return 1


def _weighted_shard(samples, num_shards, shard_index):
    """Deterministically split samples by estimated model-call cost."""
    if num_shards <= 1:
        return list(samples)
    buckets = [[] for _ in range(num_shards)]
    costs = [0 for _ in range(num_shards)]
    ordered = sorted(
        samples,
        key=lambda s: (
            -_sample_eval_cost(s),
            str(s.get("task", "")),
            str(s.get("id", "")),
        ),
    )
    for sample in ordered:
        idx = min(range(num_shards), key=lambda i: (costs[i], i))
        buckets[idx].append(sample)
        costs[idx] += _sample_eval_cost(sample)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    print(f"[shard] estimated model calls per shard: {costs}")
    return sorted(
        buckets[shard_index],
        key=lambda s: (str(s.get("task", "")), str(s.get("id", ""))),
    )


# ─── Aggregation ─────────────────────────────────────────────────────────────


def aggregate(results):
    by_task = defaultdict(lambda: {
        "n": 0,
        "correct": 0,
        "strict_correct": 0,
        "targeted_correct": 0,
        "count_abs_errors": [],
    })
    for r in results:
        for p in r.get("probes", []):
            by_task[r["task"]]["n"] += 1
            correct = bool(p.get("correct"))
            strict = bool(p.get("strict_correct", correct))
            targeted = bool(p.get("targeted_correct", correct))
            by_task[r["task"]]["correct"] += int(correct)
            by_task[r["task"]]["strict_correct"] += int(strict)
            by_task[r["task"]]["targeted_correct"] += int(targeted)
            if p.get("count_abs_error") is not None:
                by_task[r["task"]]["count_abs_errors"].append(float(p["count_abs_error"]))
    out = {}
    diagnostics = {}
    for t, v in by_task.items():
        acc = v["correct"] / max(v["n"], 1)
        strict_acc = v["strict_correct"] / max(v["n"], 1)
        targeted_acc = v["targeted_correct"] / max(v["n"], 1)
        out[t] = {
            "n": v["n"],
            "acc": acc,
            "strict_acc": strict_acc,
            "targeted_acc": targeted_acc,
        }
        # Base VLM eval answers each probe as an independent call at probe
        # time, so timing-aware accuracies are identical to content accuracy.
        diagnostics[t] = {
            "acc_content": acc,
            "acc_strict": strict_acc,
            "acc_targeted": targeted_acc,
            "count_mae": (
                sum(v["count_abs_errors"]) / len(v["count_abs_errors"])
                if v["count_abs_errors"] else 0.0
            ),
            "acc_no_early": acc,
            "acc_no_late": acc,
            "acc_on_time": acc,
            "response_early_rate": 0.0,
            "response_late_rate": 0.0,
            "response_missing_rate": 0.0,
            "recall_events": 0,
            "recall_support_hit_rate": 0.0,
            "acc_with_recall": 0.0,
            "n_with_recall": 0,
            "acc_without_recall": acc,
            "n_without_recall": v["n"],
            "compress_events": 0,
            "compress_success_rate": 0.0,
            "stable_think_pairs": 0,
        }
    # Category averages (mean of per-task accs in each category)
    def cat_avg(tasks, metric="acc"):
        if metric == "acc":
            accs = [out[t]["acc"] for t in tasks if t in out]
        elif metric in {"strict_acc", "targeted_acc"}:
            accs = [out[t][metric] for t in tasks if t in out]
        else:
            accs = [diagnostics[t][metric] for t in tasks if t in diagnostics]
        return {"avg": sum(accs) / max(len(accs), 1), "n_tasks": len(accs)}
    rt = cat_avg(RT_TASKS)
    bt = cat_avg(BT_TASKS)
    ft = cat_avg(FT_TASKS)
    active_cats = [v for v in (rt, bt, ft) if v["n_tasks"] > 0]
    overall = sum(v["avg"] for v in active_cats) / max(len(active_cats), 1)
    rt_strict = cat_avg(RT_TASKS, "strict_acc")
    bt_strict = cat_avg(BT_TASKS, "strict_acc")
    ft_strict = cat_avg(FT_TASKS, "strict_acc")
    rt_targeted = cat_avg(RT_TASKS, "targeted_acc")
    bt_targeted = cat_avg(BT_TASKS, "targeted_acc")
    ft_targeted = cat_avg(FT_TASKS, "targeted_acc")
    active_strict = [v for v in (rt_strict, bt_strict, ft_strict) if v["n_tasks"] > 0]
    active_targeted = [v for v in (rt_targeted, bt_targeted, ft_targeted) if v["n_tasks"] > 0]
    return {
        "per_task": out,
        "diagnostics": diagnostics,
        "category": {
            "RT": {
                **rt,
                "strict_acc": rt_strict["avg"],
                "targeted_acc": rt_targeted["avg"],
                "acc_no_early": cat_avg(RT_TASKS, "acc_no_early")["avg"],
                "acc_no_late": cat_avg(RT_TASKS, "acc_no_late")["avg"],
                "acc_on_time": cat_avg(RT_TASKS, "acc_on_time")["avg"],
            },
            "BT": {
                **bt,
                "strict_acc": bt_strict["avg"],
                "targeted_acc": bt_targeted["avg"],
                "acc_no_early": cat_avg(BT_TASKS, "acc_no_early")["avg"],
                "acc_no_late": cat_avg(BT_TASKS, "acc_no_late")["avg"],
                "acc_on_time": cat_avg(BT_TASKS, "acc_on_time")["avg"],
            },
            "FT": {
                **ft,
                "strict_acc": ft_strict["avg"],
                "targeted_acc": ft_targeted["avg"],
                "acc_no_early": cat_avg(FT_TASKS, "acc_no_early")["avg"],
                "acc_no_late": cat_avg(FT_TASKS, "acc_no_late")["avg"],
                "acc_on_time": cat_avg(FT_TASKS, "acc_on_time")["avg"],
            },
        },
        "overall": overall,
        "overall_strict": (
            sum(v["avg"] for v in active_strict) / max(len(active_strict), 1)
        ),
        "overall_targeted": (
            sum(v["avg"] for v in active_targeted) / max(len(active_targeted), 1)
        ),
        "overall_no_early": overall,
        "overall_no_late": overall,
        "overall_on_time": overall,
    }


def print_report(agg):
    print()
    print(f"{'task':<6}  {'n':>6}  {'acc':>8}  {'strict':>8}  {'target':>8}")
    print("-" * 47)
    for t in sorted(agg["per_task"].keys()):
        v = agg["per_task"][t]
        print(
            f"{t:<6}  {v['n']:>6}  {v['acc']:>8.3f}  "
            f"{v.get('strict_acc', 0.0):>8.3f}  "
            f"{v.get('targeted_acc', 0.0):>8.3f}"
        )
    print()
    for cat in ("RT", "BT", "FT"):
        v = agg["category"][cat]
        print(
            f"{cat}: acc={v['avg']:.3f} strict={v.get('strict_acc', 0.0):.3f} "
            f"target={v.get('targeted_acc', 0.0):.3f} ({v['n_tasks']} tasks)"
        )
    print(
        f"OVERALL: acc={agg['overall']:.3f} "
        f"strict={agg.get('overall_strict', 0.0):.3f} "
        f"target={agg.get('overall_targeted', 0.0):.3f}"
    )


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--benchmark_json", required=True)
    p.add_argument("--video_root", required=True)
    p.add_argument("--frames_root", default=_default_frames_root(),
                   help="Pre-extracted frame root")
    p.add_argument("--chunked_dir", default=None,
                   help="OVO official chunked_videos root for official_offline")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated subset of OVO tasks (default: all 12)")
    p.add_argument("--n_per_task", type=int, default=None)
    p.add_argument("--mode", default="streaming",
                   choices=["offline", "offline_prefix", "offline_full",
                            "streaming", "oracle_support",
                            "official_prompt_prefix", "official_offline"])
    p.add_argument("--max_frames", type=int, default=24,
                   help="Frame budget. Streaming default: 24 (fixed window). "
                        "Offline sweep: 64 / 128 / 256 / 512 / 1024.")
    p.add_argument("--visual_window_sec", type=float, default=DEFAULT_VISUAL_WINDOW_SEC,
                   help="Window size for streaming mode (canonical runtime default)")
    p.add_argument("--fps", type=float, default=DEFAULT_FRAME_FPS,
                   help="FPS used to map pre-extracted frame indices to seconds. "
                        "Default is the canonical runtime FPS=2.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--scoring", default="lenient", choices=["lenient", "strict"],
                   help="lenient (default): first-matching-token wins. strict: "
                        "response.strip() must EQUAL the expected token "
                        "(option letter / Yes-No / digit). Strict tells you whether "
                        "the model natively follows the requested format.")
    p.add_argument("--profile", default="16k", choices=["16k", "32k"],
                   help="Eval context profile (parity with agent eval). For "
                        "base eval the only effect is metadata stamping in "
                        "the output JSON — base VLM doesn't use queries/recall "
                        "caps. Match the agent profile when comparing.")
    p.add_argument(
        "--frame-protocol",
        default=os.environ.get("THINKSTREAM_FRAME_PROTOCOL", "video_meta"),
        choices=["video_meta", "ts_image"],
        help="Visual carrier for pre-extracted frames in this baseline eval.",
    )
    p.add_argument(
        "--preprocess",
        default=os.environ.get("OVO_PREPROCESS", "auto"),
        choices=["auto", "direct", "qwen_vl_utils"],
        help="Visual preprocessing path. auto uses qwen-vl-utils for Qwen3-VL "
             "and the direct processor path for Qwen2.5-VL.",
    )
    p.add_argument("--min_pixels", type=int, default=DEFAULT_VIDEO_MIN_PIXELS)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    p.add_argument(
        "--total_pixels",
        type=int,
        default=None,
        help="Optional total video pixel budget for qwen-vl-utils video items. "
             "Defaults to max_frames * max_pixels * 2.",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split selected samples across N processes.")
    p.add_argument("--shard_index", type=int, default=0,
                   help="Shard id for this process, in [0, num_shards).")
    args = p.parse_args()
    frame_protocol = normalize_frame_protocol(args.frame_protocol)
    args.mode = _normalize_mode(args.mode)
    if args.chunked_dir is None:
        args.chunked_dir = str(Path(args.benchmark_json).resolve().parent / "chunked_videos")

    Cls, model_family = detect_model_class(args.ckpt)
    preprocess = args.preprocess
    if preprocess == "auto":
        preprocess = "qwen_vl_utils" if model_family == "qwen3vl" else "direct"
    if preprocess == "qwen_vl_utils" and process_vision_info is None:
        raise RuntimeError("qwen-vl-utils preprocessing requested but unavailable")
    total_pixels = args.total_pixels
    if total_pixels is None:
        total_pixels = int(args.max_frames * args.max_pixels * 2)
    print(f"[mode={args.mode} max_frames={args.max_frames}] Loading {Cls.__name__} from {args.ckpt}")
    print(f"[preprocess] family={model_family} frame_protocol={frame_protocol} path={preprocess} total_pixels={total_pixels}")
    load_kwargs = {
        "dtype": torch.bfloat16 if not args.no_bf16 else None,
        "attn_implementation": "flash_attention_2",
    }
    if torch.cuda.is_available():
        # Each sharded worker sees one GPU via CUDA_VISIBLE_DEVICES. Loading
        # directly to that device avoids a slow CPU -> GPU recursive .cuda().
        load_kwargs["device_map"] = {"": "cuda:0"}
    model = Cls.from_pretrained(args.ckpt, **load_kwargs).eval()
    if torch.cuda.is_available() and not getattr(model, "hf_device_map", None):
        model = model.cuda()
    processor = AutoProcessor.from_pretrained(args.ckpt)
    data_args = DataArguments(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    data_args.video_min_pixels = args.min_pixels
    data_args.video_max_pixels = args.max_pixels
    processor = update_processor_pixels(processor, data_args)
    vp = getattr(processor, "video_processor", None)
    if vp is not None and hasattr(vp, "max_frames"):
        vp.max_frames = args.max_frames
    if vp is not None and hasattr(vp, "fps"):
        vp.fps = args.fps
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "do_sample_frames"):
        processor.video_processor.do_sample_frames = False
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    enable_thinking = False if _chat_template_supports_thinking(processor) else None
    if enable_thinking is False:
        print("[chat_template] Disabling thinking mode for answer-only base eval.")

    with open(args.benchmark_json) as f:
        all_samples = json.load(f)
    task_filter = (set(t.strip() for t in args.tasks.split(",")) if args.tasks
                   else ALL_TASKS)
    selected_samples = []
    for s in all_samples:
        if s.get("task") in task_filter:
            selected_samples.append(s)
    by_task = defaultdict(list)
    for s in selected_samples:
        by_task[s["task"]].append(s)
    if args.n_per_task:
        for t in by_task:
            by_task[t] = by_task[t][:args.n_per_task]
    selected_samples = [s for task in sorted(by_task) for s in by_task[task]]
    if args.num_shards > 1:
        selected_samples = _weighted_shard(
            selected_samples, args.num_shards, args.shard_index)
        by_task = defaultdict(list)
        for s in selected_samples:
            by_task[s["task"]].append(s)

    total = sum(len(v) for v in by_task.values())
    print(f"Running base eval on {total} samples across {len(by_task)} tasks: "
          f"{sorted(by_task.keys())}")

    kw = dict(
        model=model, processor=processor, pad_id=pad_id,
        video_root=args.video_root, frames_root=args.frames_root,
        chunked_dir=args.chunked_dir,
        mode=args.mode, visual_window_sec=args.visual_window_sec,
        max_frames=args.max_frames, max_new_tokens=args.max_new_tokens,
        scoring=args.scoring,
        frame_protocol=frame_protocol,
        fps=args.fps,
        enable_thinking=enable_thinking,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        total_pixels=total_pixels,
        preprocess=preprocess,
    )

    results = []
    t0 = time.time()
    done = 0
    for task in sorted(by_task.keys()):
        for sample in by_task[task]:
            try:
                r = dispatch(sample, **kw)
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"[{task} id={sample.get('id')}] {type(e).__name__}: {e}")
            done += 1
            if done % 10 == 0:
                rate = done / max(1e-6, time.time() - t0)
                print(f"[{done}/{total}] {rate*60:.1f} samples/min")

    if not results:
        print("No successful samples.")
        return

    agg = aggregate(results)
    print_report(agg)

    out = args.out or (
        f"{args.ckpt}/eval/ovo_base/"
        f"{args.mode}_{args.max_frames}_{args.scoring}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "ckpt": args.ckpt, "mode": args.mode,
            "benchmark_json": args.benchmark_json,
            "video_root": args.video_root,
            "frames_root": args.frames_root,
            "chunked_dir": args.chunked_dir,
            "max_frames": args.max_frames,
            "visual_window_sec": args.visual_window_sec,
            "fps": args.fps,
            "scoring": args.scoring,
            "profile": args.profile,
            "frame_protocol": frame_protocol,
            "preprocess": preprocess,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "total_pixels": total_pixels,
            "enable_thinking": enable_thinking,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "tasks_evaluated": sorted(by_task.keys()),
            "n_samples": len(results),
            "summary": agg, "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
