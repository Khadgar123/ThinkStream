"""
Stage 0: Video Preprocessing & Asset Creation

Converts raw videos into structured, searchable segment archives with:
- Scene detection & chunk splitting
- Frame extraction & keyframe selection
- Dense captioning (VL model)
- OCR extraction (VL model)
- ASR transcription (Whisper)
- Visual & text embeddings
- Tag extraction & memory key generation
- Salience & recallability scoring

Usage:
    python -m scripts.agent_data_pipeline.stage0_preprocess \
        --video_list data/agent/video_list.json \
        --output_dir data/agent \
        [--caption_model Qwen/Qwen2.5-VL-72B-Instruct] \
        [--num_gpus 8] \
        [--skip_asr] [--skip_embedding]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import (
    AGENT_CHUNK_SEC,
    CAPTION_PROMPT,
    EMBEDDING_DIR,
    KEYFRAME_DIR,
    META_DIR,
    OCR_PROMPT,
    RAW_FPS,
    SCENE_DETECT_THRESHOLD,
    SEGMENT_ARCHIVE_DIR,
    SEGMENT_OVERLAP_SEC,
    SEGMENT_SEC,
    ensure_dirs,
)
from .utils import (
    compute_motion_score,
    compute_recallability,
    compute_salience,
    detect_scenes,
    extract_frames_for_segment,
    extract_tags_from_caption,
    generate_memory_keys,
    get_video_duration_ms,
    make_segment_id,
    save_embedding,
    save_keyframe,
    select_keyframes,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Sub-task T0.1: Scene detection + segment splitting
# ===================================================================


def build_segments(
    video_duration_ms: int,
    segment_sec: int = SEGMENT_SEC,
    overlap_sec: int = SEGMENT_OVERLAP_SEC,
) -> List[Dict]:
    """Generate overlapping segments for the episodic memory archive."""
    step_ms = (segment_sec - overlap_sec) * 1000
    segments = []
    start = 0
    end_limit = video_duration_ms
    while start + segment_sec * 1000 <= end_limit:
        segments.append({
            "start_ms": start,
            "end_ms": start + segment_sec * 1000,
        })
        start += step_ms
    # Handle trailing segment if video doesn't divide evenly
    if segments and segments[-1]["end_ms"] < end_limit:
        remaining = end_limit - segments[-1]["end_ms"]
        if remaining > 1000:  # at least 1s remaining
            segments.append({
                "start_ms": segments[-1]["end_ms"] - overlap_sec * 1000,
                "end_ms": end_limit,
            })
    return segments


def assign_scene_ids(
    segments: List[Dict],
    scene_boundaries: List[Tuple[float, float]],
) -> List[Dict]:
    """Assign scene_id to each segment based on scene boundaries."""
    if not scene_boundaries:
        for seg in segments:
            seg["scene_id"] = "scene_0000"
        return segments

    for seg in segments:
        seg_mid_sec = (seg["start_ms"] + seg["end_ms"]) / 2000.0
        assigned = False
        for i, (s_start, s_end) in enumerate(scene_boundaries):
            if s_start <= seg_mid_sec <= s_end:
                seg["scene_id"] = f"scene_{i:04d}"
                assigned = True
                break
        if not assigned:
            seg["scene_id"] = f"scene_{len(scene_boundaries) - 1:04d}"
    return segments


# ===================================================================
# Sub-task T0.2: Frame extraction
# ===================================================================


def extract_and_save_keyframes(
    video_path: str,
    video_id: str,
    segments: List[Dict],
    fps: int = RAW_FPS,
) -> List[Dict]:
    """Extract frames for each segment and save keyframes."""
    for seg in segments:
        frames = extract_frames_for_segment(
            video_path, seg["start_ms"], seg["end_ms"], fps=fps
        )
        seg["num_frames"] = len(frames)

        # Select and save keyframes
        kf_indices = select_keyframes(frames)
        kf_paths = []
        for idx in kf_indices:
            kf_path = KEYFRAME_DIR / video_id / f"frame_{seg['start_ms']:06d}_{idx}.jpg"
            save_keyframe(frames[idx], kf_path)
            kf_paths.append(str(kf_path))
        seg["keyframe_paths"] = kf_paths

        # Compute motion score
        seg["motion_score"] = compute_motion_score(frames)

    return segments


# ===================================================================
# Sub-task T0.3: ASR transcription
# ===================================================================


def transcribe_segments(
    video_path: str,
    segments: List[Dict],
    model_name: str = "large-v3",
    language: str = "zh",
) -> List[Dict]:
    """Run Whisper ASR on each segment."""
    try:
        import whisper

        model = whisper.load_model(model_name)
        for seg in segments:
            try:
                start_sec = seg["start_ms"] / 1000.0
                end_sec = seg["end_ms"] / 1000.0
                result = model.transcribe(
                    video_path,
                    language=language,
                    clip_timestamps=[start_sec, end_sec],
                )
                text = result.get("text", "").strip()
                seg["asr_text"] = text if text else "无"
            except Exception as exc:
                logger.warning("ASR failed for segment %s: %s", seg.get("segment_id", "?"), exc)
                seg["asr_text"] = "无"
    except ImportError:
        logger.warning("whisper not installed, skipping ASR")
        for seg in segments:
            seg["asr_text"] = "无"

    for seg in segments:
        seg["has_asr"] = seg["asr_text"] != "无"
    return segments


# ===================================================================
# Sub-task T0.4 & T0.5: Dense caption + OCR (VL model)
# ===================================================================


def _load_vl_model(model_name: str):
    """Load a vision-language model for captioning/OCR."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def _vl_generate(model, processor, images, prompt: str, max_tokens: int = 150) -> str:
    """Generate text from images + prompt using VL model."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    # Trim input tokens
    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def caption_segments(
    segments: List[Dict],
    model,
    processor,
) -> List[Dict]:
    """Generate dense captions for each segment using VL model."""
    from PIL import Image

    for seg in segments:
        try:
            images = [Image.open(p) for p in seg["keyframe_paths"] if Path(p).exists()]
            if not images:
                seg["dense_caption"] = ""
                continue
            seg["dense_caption"] = _vl_generate(model, processor, images, CAPTION_PROMPT)
        except Exception as exc:
            logger.warning("Caption failed for segment %s: %s", seg.get("segment_id", "?"), exc)
            seg["dense_caption"] = ""
    return segments


def ocr_segments(
    segments: List[Dict],
    model,
    processor,
) -> List[Dict]:
    """Extract OCR text from each segment's keyframes."""
    from PIL import Image

    for seg in segments:
        try:
            # Use only the first keyframe for OCR
            kf_paths = seg.get("keyframe_paths", [])
            if not kf_paths or not Path(kf_paths[0]).exists():
                seg["ocr_text"] = "无"
                seg["has_ocr"] = False
                continue
            image = Image.open(kf_paths[0])
            result = _vl_generate(model, processor, [image], OCR_PROMPT, max_tokens=100)
            seg["ocr_text"] = result if result and result != "无" else "无"
        except Exception as exc:
            logger.warning("OCR failed for segment %s: %s", seg.get("segment_id", "?"), exc)
            seg["ocr_text"] = "无"

        seg["has_ocr"] = seg["ocr_text"] != "无"
    return segments


# ===================================================================
# Sub-task T0.6: Visual embedding
# ===================================================================


def compute_visual_embeddings(
    segments: List[Dict],
    video_id: str,
    model_name: str = "google/siglip-so400m-patch14-384",
) -> List[Dict]:
    """Compute visual embeddings for each segment's keyframes."""
    try:
        from transformers import AutoModel, AutoProcessor
        from PIL import Image

        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        for seg in segments:
            kf_paths = seg.get("keyframe_paths", [])
            if not kf_paths:
                seg["visual_emb_path"] = ""
                continue

            images = [Image.open(p) for p in kf_paths if Path(p).exists()]
            if not images:
                seg["visual_emb_path"] = ""
                continue

            inputs = processor(images=images, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_image_features(**inputs)

            # Average embeddings if multiple keyframes
            emb_np = emb.mean(dim=0).cpu().numpy()
            emb_path = EMBEDDING_DIR / video_id / f"visual_{seg['start_ms']:06d}.npy"
            save_embedding(emb_np, emb_path)
            seg["visual_emb_path"] = str(emb_path)

    except ImportError:
        logger.warning("SigLIP not available, skipping visual embeddings")
        for seg in segments:
            seg["visual_emb_path"] = ""

    return segments


# ===================================================================
# Sub-task T0.7: Tag extraction + memory keys
# ===================================================================


def extract_tags_and_keys(segments: List[Dict]) -> List[Dict]:
    """Extract entity/action/state tags and generate memory keys."""
    for seg in segments:
        caption = seg.get("dense_caption", "")
        entities, actions, states = extract_tags_from_caption(caption)
        seg["entity_tags"] = entities
        seg["action_tags"] = actions
        seg["state_tags"] = states
        seg["memory_keys"] = generate_memory_keys(
            entities, actions, seg.get("ocr_text", "无"), caption
        )
    return segments


# ===================================================================
# Sub-task T0.8: Text embedding
# ===================================================================


def compute_text_embeddings(
    segments: List[Dict],
    video_id: str,
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
) -> List[Dict]:
    """Compute text embeddings for each segment."""
    try:
        from FlagEmbedding import FlagModel

        model = FlagModel(model_name)

        for seg in segments:
            text = seg.get("dense_caption", "") + " " + " ".join(seg.get("memory_keys", []))
            if not text.strip():
                seg["text_emb_path"] = ""
                continue

            emb = model.encode(text)
            emb_path = EMBEDDING_DIR / video_id / f"text_{seg['start_ms']:06d}.npy"
            save_embedding(np.array(emb), emb_path)
            seg["text_emb_path"] = str(emb_path)

    except ImportError:
        logger.warning("FlagEmbedding not available, trying sentence-transformers")
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct")

            for seg in segments:
                text = seg.get("dense_caption", "") + " " + " ".join(seg.get("memory_keys", []))
                if not text.strip():
                    seg["text_emb_path"] = ""
                    continue
                emb = model.encode(text)
                emb_path = EMBEDDING_DIR / video_id / f"text_{seg['start_ms']:06d}.npy"
                save_embedding(np.array(emb), emb_path)
                seg["text_emb_path"] = str(emb_path)

        except ImportError:
            logger.warning("No text embedding model available, skipping")
            for seg in segments:
                seg["text_emb_path"] = ""

    return segments


# ===================================================================
# Sub-task T0.9: Salience + recallability
# ===================================================================


def compute_scores(
    segments: List[Dict],
) -> Tuple[List[Dict], float]:
    """Compute salience for each segment and recallability for the video."""
    scene_ids = [s.get("scene_id", "") for s in segments]

    for i, seg in enumerate(segments):
        is_boundary = i > 0 and scene_ids[i] != scene_ids[i - 1]
        seg["salience"] = compute_salience(
            motion_score=seg.get("motion_score", 0.0),
            entity_tags=seg.get("entity_tags", []),
            action_tags=seg.get("action_tags", []),
            has_ocr=seg.get("has_ocr", False),
            has_asr=seg.get("has_asr", False),
            is_scene_boundary=is_boundary,
        )

    recallability = compute_recallability(segments)
    return segments, recallability


# ===================================================================
# Main: process a single video end-to-end
# ===================================================================


def process_single_video(
    video_path: str,
    video_id: str,
    vl_model=None,
    vl_processor=None,
    skip_asr: bool = False,
    skip_embedding: bool = False,
) -> Tuple[List[Dict], float]:
    """Run the full Stage 0 pipeline on a single video.

    Returns:
        (segments, recallability_score)
    """
    logger.info("Stage 0: Processing video %s", video_id)

    # T0.1: Scene detection + segment splitting
    duration_ms = get_video_duration_ms(video_path)
    scene_boundaries = detect_scenes(video_path, threshold=SCENE_DETECT_THRESHOLD)
    segments = build_segments(duration_ms)

    # Assign IDs and scene IDs
    for seg in segments:
        seg["video_id"] = video_id
        seg["segment_id"] = make_segment_id(video_id, seg["start_ms"], seg["end_ms"])
        seg["raw_fps"] = RAW_FPS

    segments = assign_scene_ids(segments, scene_boundaries)

    # Save scene & segment metadata
    meta = {
        "video_id": video_id,
        "video_path": video_path,
        "duration_ms": duration_ms,
        "num_scenes": len(scene_boundaries) if scene_boundaries else 1,
        "num_segments": len(segments),
        "scene_boundaries": scene_boundaries,
    }
    meta_path = META_DIR / f"{video_id}_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # T0.2: Frame extraction + keyframes
    segments = extract_and_save_keyframes(video_path, video_id, segments)

    # T0.3: ASR
    if not skip_asr:
        segments = transcribe_segments(video_path, segments)
    else:
        for seg in segments:
            seg["asr_text"] = "无"
            seg["has_asr"] = False

    # T0.4 & T0.5: Dense caption + OCR
    if vl_model is not None and vl_processor is not None:
        segments = caption_segments(segments, vl_model, vl_processor)
        segments = ocr_segments(segments, vl_model, vl_processor)
    else:
        logger.warning("No VL model provided, skipping caption/OCR")
        for seg in segments:
            seg["dense_caption"] = ""
            seg["ocr_text"] = "无"
            seg["has_ocr"] = False

    # T0.7: Tag extraction + memory keys
    segments = extract_tags_and_keys(segments)

    # T0.6 & T0.8: Embeddings
    if not skip_embedding:
        segments = compute_visual_embeddings(segments, video_id)
        segments = compute_text_embeddings(segments, video_id)
    else:
        for seg in segments:
            seg["visual_emb_path"] = ""
            seg["text_emb_path"] = ""

    # T0.9: Salience + recallability
    segments, recallability = compute_scores(segments)

    # Save segment archive
    archive_path = SEGMENT_ARCHIVE_DIR / f"{video_id}.jsonl"
    write_jsonl(segments, archive_path)

    logger.info(
        "Stage 0 complete: %s → %d segments, recallability=%.3f",
        video_id, len(segments), recallability,
    )
    return segments, recallability


# ===================================================================
# Batch processing
# ===================================================================


def process_batch(
    video_list: List[Dict],
    caption_model_name: Optional[str] = None,
    skip_asr: bool = False,
    skip_embedding: bool = False,
) -> Dict[str, float]:
    """Process a batch of videos through Stage 0.

    Args:
        video_list: List of {"video_id": str, "video_path": str}
        caption_model_name: HuggingFace model name for captioning/OCR
        skip_asr: Skip ASR transcription
        skip_embedding: Skip embedding computation

    Returns:
        Dict mapping video_id → recallability_score
    """
    ensure_dirs()

    # Load VL model once
    vl_model, vl_processor = None, None
    if caption_model_name:
        logger.info("Loading VL model: %s", caption_model_name)
        vl_model, vl_processor = _load_vl_model(caption_model_name)

    results = {}
    for item in video_list:
        video_id = item["video_id"]
        video_path = item["video_path"]

        # Skip if already processed
        archive_path = SEGMENT_ARCHIVE_DIR / f"{video_id}.jsonl"
        if archive_path.exists():
            logger.info("Skipping %s (already processed)", video_id)
            continue

        try:
            _, recallability = process_single_video(
                video_path=video_path,
                video_id=video_id,
                vl_model=vl_model,
                vl_processor=vl_processor,
                skip_asr=skip_asr,
                skip_embedding=skip_embedding,
            )
            results[video_id] = recallability
        except Exception as exc:
            logger.error("Failed to process %s: %s", video_id, exc, exc_info=True)
            results[video_id] = -1.0

    return results


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Video preprocessing")
    parser.add_argument("--video_list", required=True, help="JSON file with video list")
    parser.add_argument("--caption_model", default=None, help="VL model for captioning/OCR")
    parser.add_argument("--skip_asr", action="store_true")
    parser.add_argument("--skip_embedding", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.video_list) as f:
        video_list = json.load(f)

    results = process_batch(
        video_list=video_list,
        caption_model_name=args.caption_model,
        skip_asr=args.skip_asr,
        skip_embedding=args.skip_embedding,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Stage 0 Summary: {len(results)} videos processed")
    valid = {k: v for k, v in results.items() if v >= 0}
    if valid:
        scores = list(valid.values())
        print(f"  Recallability: mean={sum(scores)/len(scores):.3f}, "
              f"min={min(scores):.3f}, max={max(scores):.3f}")
    failed = {k: v for k, v in results.items() if v < 0}
    if failed:
        print(f"  Failed: {len(failed)} videos")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
