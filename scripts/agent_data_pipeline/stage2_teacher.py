"""
Stage 2: 397B Teacher Task Pack Generation

For each video, sends its event timeline + segment archive summary to the
large teacher model (Qwen3.5-397B) to generate 8-20 candidate tasks per video,
including recall-positive and non-recall tasks.

Usage:
    python -m scripts.agent_data_pipeline.stage2_teacher \
        --video_ids vid_001,vid_002 \
        --api_base http://localhost:8000/v1 \
        [--model Qwen/Qwen3.5-397B-A22B-FP8] \
        [--target_tasks 12]
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from .config import (
    EPISODE_RAW_PATH,
    EVENT_TIMELINE_DIR,
    RECENT_WINDOW_SEC,
    SEGMENT_ARCHIVE_DIR,
    TEACHER_TASK_PACK_SYSTEM,
    TEACHER_TASK_PACK_USER_TEMPLATE,
    ensure_dirs,
)
from .utils import (
    append_jsonl,
    infer_video_type,
    load_segment_archive,
    read_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Input preparation
# ===================================================================


def prepare_archive_summary(segments: List[Dict]) -> str:
    """Compress segment archive to a summary suitable for the teacher prompt."""
    summary = []
    for seg in segments:
        entry = {
            "id": seg["segment_id"],
            "t": f"{seg['start_ms']/1000:.1f}-{seg['end_ms']/1000:.1f}s",
            "cap": seg.get("dense_caption", "")[:50],
            "ent": seg.get("entity_tags", [])[:3],
            "act": seg.get("action_tags", [])[:2],
            "ocr": seg.get("ocr_text", "")[:20] if seg.get("has_ocr") else "",
            "asr": seg.get("asr_text", "")[:20] if seg.get("has_asr") else "",
            "sal": round(seg.get("salience", 0), 2),
        }
        summary.append(entry)
    return json.dumps(summary, ensure_ascii=False)


def prepare_teacher_input(
    video_id: str,
    target_task_count: int = 12,
) -> Optional[Dict]:
    """Prepare the full prompt input for the 397B teacher."""
    # Load event timeline
    timeline_path = EVENT_TIMELINE_DIR / f"{video_id}.jsonl"
    if not timeline_path.exists():
        logger.warning("No event timeline for %s", video_id)
        return None

    event_timeline = read_jsonl(timeline_path)
    segments = load_segment_archive(video_id)

    if not event_timeline or not segments:
        logger.warning("Empty timeline or segments for %s", video_id)
        return None

    timeline_json = json.dumps(event_timeline, ensure_ascii=False, indent=2)
    archive_summary = prepare_archive_summary(segments)
    video_type = infer_video_type(event_timeline, segments)

    user_prompt = TEACHER_TASK_PACK_USER_TEMPLATE.format(
        recent_window_sec=RECENT_WINDOW_SEC,
        video_type=video_type,
        target_task_count=target_task_count,
        timeline_json=timeline_json,
        archive_summary=archive_summary,
    )

    return {
        "video_id": video_id,
        "video_type": video_type,
        "system_prompt": TEACHER_TASK_PACK_SYSTEM,
        "user_prompt": user_prompt,
        "num_events": len(event_timeline),
        "num_segments": len(segments),
    }


# ===================================================================
# LLM API call
# ===================================================================


def call_teacher_api(
    system_prompt: str,
    user_prompt: str,
    api_base: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Optional[str]:
    """Call the teacher LLM via OpenAI-compatible API."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=api_base, api_key="placeholder")

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as exc:
                logger.warning(
                    "API call attempt %d/%d failed: %s", attempt + 1, max_retries, exc
                )
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))

    except ImportError:
        logger.error("openai package not installed")

    return None


# ===================================================================
# Output post-processing
# ===================================================================


def assign_difficulty(task: Dict) -> str:
    """Assign difficulty level based on task type and recall requirement."""
    task_type = task.get("task_type", "")
    need_recall = task.get("need_recall", False)

    if not need_recall and task_type in ("current_perception", "short_temporal"):
        return "easy"
    elif not need_recall:
        return "medium"
    elif task_type in ("retrospective_detail", "procedural_state", "compare_past_present"):
        return "hard"
    else:
        return "very_hard"


def infer_source_dataset(video_id: str) -> str:
    """Infer source dataset from video_id convention."""
    if video_id.startswith("streamo_") or video_id.startswith("ytb_"):
        return "streamo"
    return "thinkstream"


def postprocess_teacher_output(
    raw_output: str,
    video_id: str,
    video_type: str,
) -> List[Dict]:
    """Parse and validate the teacher's JSON output."""
    # Try to extract JSON from the response
    try:
        # Handle potential markdown code blocks
        text = raw_output.strip()
        if text.startswith("```"):
            # Remove code block markers
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        import re
        match = re.search(r'\{[\s\S]*\}', raw_output)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Failed to parse teacher output for %s", video_id)
                return []
        else:
            logger.error("No JSON found in teacher output for %s", video_id)
            return []

    tasks = data.get("tasks", [])
    episodes = []

    for task in tasks:
        episode = {
            "episode_id": f"ep_recall_{uuid4().hex[:6]}",
            "source_dataset": infer_source_dataset(video_id),
            "video_id": video_id,
            "video_type": video_type,
            "task_type": task.get("task_type", "current_perception"),
            "question": task.get("question", ""),
            "dialogue_history": [],
            "ask_time_ms": task.get("ask_time_ms", 0),
            "earliest_response_time_ms": task.get("earliest_response_time_ms", 0),
            "recent_window_sec": RECENT_WINDOW_SEC,
            "support_event_ids": task.get("support_event_ids", []),
            "support_segment_ids": task.get("support_segment_ids", []),
            "support_outside_recent": False,  # Will be verified in Stage 5
            "need_recall": task.get("need_recall", False),
            "recall_reason": task.get("recall_reason", ""),
            "canonical_answer": task.get("canonical_answer", {"answer_type": "span", "value": {}}),
            "natural_response": task.get("natural_response", ""),
            "query_candidates": task.get("query_candidates", []),
            "sparse_think_milestones": task.get("sparse_think_milestones", []),
            "difficulty": assign_difficulty(task),
            "verification": {},
            "provenance": {
                "pipeline_version": "v0.1",
                "stage2_task_id": task.get("task_id", ""),
            },
        }

        # Compute support_outside_recent
        if episode["need_recall"] and episode["ask_time_ms"] > 0:
            recent_start = max(0, episode["ask_time_ms"] - RECENT_WINDOW_SEC * 1000)
            recent_end = episode["ask_time_ms"]
            # Check if any support segment is within recent window
            all_outside = True
            for sid in episode["support_segment_ids"]:
                # Parse segment start/end from segment_id: {video_id}_{start:08d}_{end:08d}
                parts = sid.rsplit("_", 2)
                if len(parts) >= 3:
                    try:
                        seg_start = int(parts[-2])
                        seg_end = int(parts[-1])
                        if max(recent_start, seg_start) < min(recent_end, seg_end):
                            all_outside = False
                            break
                    except ValueError:
                        pass
            episode["support_outside_recent"] = all_outside

        # Validate basic fields
        if not episode["question"]:
            logger.warning("Skipping task with empty question in %s", video_id)
            continue

        episodes.append(episode)

    return episodes


# ===================================================================
# Main: generate task packs
# ===================================================================


def generate_task_packs(
    video_ids: List[str],
    api_base: str,
    model: str,
    target_tasks: int = 12,
) -> Dict[str, int]:
    """Generate task packs for multiple videos.

    Returns: Dict mapping video_id → number of tasks generated
    """
    ensure_dirs()
    results = {}

    for video_id in video_ids:
        logger.info("Stage 2: Generating task pack for %s", video_id)

        # Prepare input
        teacher_input = prepare_teacher_input(video_id, target_tasks)
        if teacher_input is None:
            results[video_id] = -1
            continue

        # Call teacher API
        raw_output = call_teacher_api(
            system_prompt=teacher_input["system_prompt"],
            user_prompt=teacher_input["user_prompt"],
            api_base=api_base,
            model=model,
        )

        if raw_output is None:
            logger.error("Teacher API returned None for %s", video_id)
            results[video_id] = -1
            continue

        # Post-process
        episodes = postprocess_teacher_output(
            raw_output, video_id, teacher_input["video_type"]
        )

        # Save episodes
        for ep in episodes:
            append_jsonl(ep, EPISODE_RAW_PATH)

        results[video_id] = len(episodes)
        logger.info("Stage 2: %s → %d tasks", video_id, len(episodes))

    return results


# ===================================================================
# Offline mode: generate task packs from local model or existing output
# ===================================================================


def generate_task_packs_offline(
    video_ids: List[str],
    target_tasks: int = 12,
) -> Dict[str, int]:
    """Generate task pack prompts and save them for manual execution.

    Use this when the 397B API is not available. Saves prompt files
    that can be sent to the model manually.
    """
    ensure_dirs()
    prompts_dir = EPISODE_RAW_PATH.parent / "teacher_prompts"
    prompts_dir.mkdir(exist_ok=True)

    results = {}
    for video_id in video_ids:
        teacher_input = prepare_teacher_input(video_id, target_tasks)
        if teacher_input is None:
            results[video_id] = -1
            continue

        # Save prompt for manual execution
        prompt_path = prompts_dir / f"{video_id}_prompt.json"
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_id": video_id,
                "messages": [
                    {"role": "system", "content": teacher_input["system_prompt"]},
                    {"role": "user", "content": teacher_input["user_prompt"]},
                ],
            }, f, ensure_ascii=False, indent=2)

        results[video_id] = 0  # Prompt saved, not yet executed
        logger.info("Stage 2 (offline): Saved prompt for %s → %s", video_id, prompt_path)

    return results


def import_teacher_responses(responses_dir: Path) -> int:
    """Import manually-obtained teacher responses.

    Reads JSON files from responses_dir, each containing the raw
    teacher output for a video.
    """
    ensure_dirs()
    count = 0

    for resp_file in sorted(responses_dir.glob("*_response.json")):
        with open(resp_file) as f:
            data = json.load(f)

        video_id = data.get("video_id", resp_file.stem.replace("_response", ""))
        raw_output = data.get("output", "")
        video_type = data.get("video_type", "other")

        episodes = postprocess_teacher_output(raw_output, video_id, video_type)
        for ep in episodes:
            append_jsonl(ep, EPISODE_RAW_PATH)
        count += len(episodes)

    logger.info("Imported %d episodes from %s", count, responses_dir)
    return count


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Teacher task pack generation")
    parser.add_argument("--video_ids", required=True, help="Comma-separated video IDs")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A22B-FP8")
    parser.add_argument("--target_tasks", type=int, default=12)
    parser.add_argument("--offline", action="store_true",
                        help="Save prompts for manual execution instead of calling API")
    parser.add_argument("--import_responses", default=None,
                        help="Directory with teacher response JSON files to import")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.import_responses:
        count = import_teacher_responses(Path(args.import_responses))
        print(f"Imported {count} episodes")
        return

    video_ids = [v.strip() for v in args.video_ids.split(",")]

    if args.offline:
        results = generate_task_packs_offline(video_ids, args.target_tasks)
    else:
        results = generate_task_packs(
            video_ids, args.api_base, args.model, args.target_tasks
        )

    # Summary
    print(f"\nStage 2 Summary:")
    total = sum(v for v in results.values() if v > 0)
    failed = sum(1 for v in results.values() if v < 0)
    print(f"  Total tasks: {total}")
    if failed:
        print(f"  Failed videos: {failed}")


if __name__ == "__main__":
    main()
