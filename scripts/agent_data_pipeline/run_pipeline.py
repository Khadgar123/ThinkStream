"""
Agent Data Construction Pipeline — Main Orchestrator

Runs Stage 0 through Stage 6 end-to-end, or individual stages.

Usage:
    # Full pipeline
    python -m scripts.agent_data_pipeline.run_pipeline \
        --video_list data/agent/video_list.json \
        --api_base http://localhost:8000/v1

    # Specific stages
    python -m scripts.agent_data_pipeline.run_pipeline \
        --video_list data/agent/video_list.json \
        --stages 0,1

    # Fast mode (skip VL gates, skip ASR/embeddings)
    python -m scripts.agent_data_pipeline.run_pipeline \
        --video_list data/agent/video_list.json \
        --fast

    # Offline mode (save prompts, no API calls)
    python -m scripts.agent_data_pipeline.run_pipeline \
        --video_list data/agent/video_list.json \
        --offline
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import ensure_dirs

logger = logging.getLogger(__name__)


def run_stage0(
    video_list: List[Dict],
    caption_model: Optional[str] = None,
    skip_asr: bool = False,
    skip_embedding: bool = False,
) -> Dict[str, float]:
    """Stage 0: Video preprocessing & asset creation."""
    from .stage0_preprocess import process_batch

    logger.info("=" * 60)
    logger.info("STAGE 0: Video Preprocessing")
    logger.info("=" * 60)
    t0 = time.time()
    results = process_batch(
        video_list=video_list,
        caption_model_name=caption_model,
        skip_asr=skip_asr,
        skip_embedding=skip_embedding,
    )
    logger.info("Stage 0 completed in %.1f seconds", time.time() - t0)
    return results


def run_stage1(video_ids: Optional[List[str]] = None) -> Dict[str, int]:
    """Stage 1: Event timeline generation."""
    from .stage1_timeline import process_batch

    logger.info("=" * 60)
    logger.info("STAGE 1: Event Timeline Generation")
    logger.info("=" * 60)
    t0 = time.time()
    results = process_batch(video_ids)
    logger.info("Stage 1 completed in %.1f seconds", time.time() - t0)
    return results


def run_stage2(
    video_ids: List[str],
    api_base: str,
    model: str,
    target_tasks: int = 12,
    offline: bool = False,
) -> Dict[str, int]:
    """Stage 2: 397B teacher task pack generation."""
    from .stage2_teacher import generate_task_packs, generate_task_packs_offline

    logger.info("=" * 60)
    logger.info("STAGE 2: Teacher Task Pack Generation")
    logger.info("=" * 60)
    t0 = time.time()
    if offline:
        results = generate_task_packs_offline(video_ids, target_tasks)
    else:
        results = generate_task_packs(video_ids, api_base, model, target_tasks)
    logger.info("Stage 2 completed in %.1f seconds", time.time() - t0)
    return results


def run_stage3() -> int:
    """Stage 3: Sparse → Dense think expansion."""
    from .stage3_expand import expand_all_episodes

    logger.info("=" * 60)
    logger.info("STAGE 3: Think Expansion")
    logger.info("=" * 60)
    t0 = time.time()
    count = expand_all_episodes()
    logger.info("Stage 3 completed in %.1f seconds", time.time() - t0)
    return count


def run_stage4(
    repair_api_base: Optional[str] = None,
    repair_model: str = "Qwen/Qwen3.5-397B-A22B-FP8",
) -> Dict[str, int]:
    """Stage 4: Query verification."""
    from .stage4_verify_query import verify_all_episodes

    logger.info("=" * 60)
    logger.info("STAGE 4: Query Verification")
    logger.info("=" * 60)
    t0 = time.time()
    stats = verify_all_episodes(repair_api_base, repair_model)
    logger.info("Stage 4 completed in %.1f seconds", time.time() - t0)
    return stats


def run_stage5(
    vl_model: Optional[str] = None,
    fast: bool = False,
) -> Dict[str, int]:
    """Stage 5: 6-gate verification."""
    from .stage5_gates import verify_all_episodes

    logger.info("=" * 60)
    logger.info("STAGE 5: 6-Gate Verification")
    logger.info("=" * 60)
    t0 = time.time()
    stats = verify_all_episodes(vl_model_name=vl_model, fast_mode=fast)
    logger.info("Stage 5 completed in %.1f seconds", time.time() - t0)
    return stats


def run_stage6(
    version: str = "v0.1",
    clip_videos: bool = False,
) -> Dict[str, int]:
    """Stage 6: Final sample assembly."""
    from .stage6_assemble import assemble_all

    logger.info("=" * 60)
    logger.info("STAGE 6: Sample Assembly")
    logger.info("=" * 60)
    t0 = time.time()
    stats = assemble_all(version=version, clip_videos=clip_videos)
    logger.info("Stage 6 completed in %.1f seconds", time.time() - t0)
    return stats


# ===================================================================
# Full pipeline
# ===================================================================


def run_full_pipeline(
    video_list: List[Dict],
    stages: Optional[List[int]] = None,
    api_base: str = "http://localhost:8000/v1",
    teacher_model: str = "Qwen/Qwen3.5-397B-A22B-FP8",
    caption_model: Optional[str] = None,
    vl_model: Optional[str] = None,
    target_tasks: int = 12,
    version: str = "v0.1",
    fast: bool = False,
    offline: bool = False,
    clip_videos: bool = False,
) -> Dict:
    """Run the full data construction pipeline or specific stages."""
    ensure_dirs()

    all_stages = stages or list(range(7))
    video_ids = [v["video_id"] for v in video_list]
    results = {}
    pipeline_start = time.time()

    if 0 in all_stages:
        results["stage0"] = run_stage0(
            video_list,
            caption_model=caption_model,
            skip_asr=fast,
            skip_embedding=fast,
        )

    if 1 in all_stages:
        results["stage1"] = run_stage1(video_ids)

    if 2 in all_stages:
        results["stage2"] = run_stage2(
            video_ids, api_base, teacher_model, target_tasks, offline
        )

    if 3 in all_stages:
        results["stage3"] = run_stage3()

    if 4 in all_stages:
        repair_base = api_base if not offline else None
        results["stage4"] = run_stage4(repair_base, teacher_model)

    if 5 in all_stages:
        results["stage5"] = run_stage5(vl_model, fast)

    if 6 in all_stages:
        results["stage6"] = run_stage6(version, clip_videos)

    total_time = time.time() - pipeline_start
    results["total_time_sec"] = total_time

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("=" * 60)

    return results


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Agent Data Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all stages
  python -m scripts.agent_data_pipeline.run_pipeline \\
      --video_list data/agent/video_list.json \\
      --api_base http://localhost:8000/v1

  # Only run preprocessing (Stage 0-1)
  python -m scripts.agent_data_pipeline.run_pipeline \\
      --video_list data/agent/video_list.json \\
      --stages 0,1

  # Fast mode (skip GPU-heavy steps)
  python -m scripts.agent_data_pipeline.run_pipeline \\
      --video_list data/agent/video_list.json \\
      --fast

  # Offline mode (save prompts, no API calls)
  python -m scripts.agent_data_pipeline.run_pipeline \\
      --video_list data/agent/video_list.json \\
      --offline
""",
    )
    parser.add_argument("--video_list", required=True,
                        help="JSON file: [{\"video_id\": \"...\", \"video_path\": \"...\"}]")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated stage numbers (e.g. 0,1,2). Default: all")
    parser.add_argument("--api_base", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base for teacher model")
    parser.add_argument("--teacher_model", default="Qwen/Qwen3.5-397B-A22B-FP8")
    parser.add_argument("--caption_model", default=None,
                        help="VL model for dense captioning/OCR (e.g. Qwen/Qwen2.5-VL-72B-Instruct)")
    parser.add_argument("--vl_model", default=None,
                        help="VL model for Stage 5 gate verification")
    parser.add_argument("--target_tasks", type=int, default=12,
                        help="Target tasks per video for teacher")
    parser.add_argument("--version", default="v0.1",
                        help="Version tag for output files")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: skip ASR, embeddings, VL gates")
    parser.add_argument("--offline", action="store_true",
                        help="Offline mode: save prompts, no API calls")
    parser.add_argument("--clip_videos", action="store_true",
                        help="Actually cut video clips with ffmpeg")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Load video list
    with open(args.video_list) as f:
        video_list = json.load(f)
    logger.info("Loaded %d videos from %s", len(video_list), args.video_list)

    # Parse stages
    stages = None
    if args.stages:
        stages = [int(s.strip()) for s in args.stages.split(",")]

    # Run pipeline
    results = run_full_pipeline(
        video_list=video_list,
        stages=stages,
        api_base=args.api_base,
        teacher_model=args.teacher_model,
        caption_model=args.caption_model,
        vl_model=args.vl_model,
        target_tasks=args.target_tasks,
        version=args.version,
        fast=args.fast,
        offline=args.offline,
        clip_videos=args.clip_videos,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
