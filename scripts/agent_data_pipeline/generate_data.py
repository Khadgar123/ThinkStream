"""
流视频 Agent 数据构造主脚本

实现 docs/data_construction.md 中定义的 Step 1-6 全流程。
通过 vLLM API 并发调用 397B 模型完成标注和任务设计。

Usage:
    # 1. AMD 节点启动 vLLM (见文档)
    # 2. 运行压测确定并发上限
    python -m scripts.agent_data_pipeline.generate_data stress_test \
        --api_base http://AMD_IP:8000/v1 --max_concurrent 8

    # 3. 全流程造数据
    python -m scripts.agent_data_pipeline.generate_data run \
        --api_base http://AMD_IP:8000/v1 \
        --streamo_dir /path/to/streamo \
        --video_root /path/to/videos \
        --output_dir data/agent \
        --max_concurrent 40 \
        --num_videos 200 \
        --hours 12
"""

import argparse
import asyncio
import copy
import json
import logging
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .vllm_client import VLLMClient, build_content_with_images, encode_image_base64

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECENT_WINDOW_SEC = 24
AGENT_CHUNK_SEC = 2
SEGMENT_SEC = 4
FPS = 2  # frames per second for extraction
FRAMES_PER_SEGMENT = 4  # keyframes sent to 397B per segment (1fps coverage)

AGENT_SYSTEM_PROMPT = (
    "You are a streaming video QA agent receiving a continuous stream of video chunks.\n\n"
    "Each assistant turn must output exactly one of these three forms:\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    '<query>{"query":"...","time_bias":"...","target":"...","topk":3}</query>\n\n'
    "Rules:\n"
    "- Only one action per turn.\n"
    "- Think only about new observations in the current chunk. Do not restate the whole video.\n"
    "- If the recent window has enough evidence, respond directly without recall.\n"
    "- If the answer depends on content that has left the recent window, use recall.\n"
    "- Recall queries must be short, searchable keywords — not full questions, no pronouns.\n"
    "- After receiving <recall_result>, only output silent or response.\n"
    "- If no speech is needed for the current chunk, output silent."
)

# Prompt templates
SEGMENT_ANNOTATE_PROMPT = """Describe what you see in this video segment (t={start}-{end}s).

Output strict JSON with exactly these keys:
{{"action":"one sentence: who does what","entities":["person_A","knife","cutting_board"],"visual_details":[{{"entity":"person_A","attributes":"red apron, short hair, standing left"}},{{"entity":"knife","attributes":"silver blade, black handle"}}],"ocr":"exact text if visible, otherwise none","change":"new entity appeared / state changed / video start"}}

Rules:
- action: one sentence, subject + verb + object
- entities: flat list of noun phrases, max 6
- visual_details: array of {{"entity":"...","attributes":"color, shape, clothing, position"}} for each important entity
- ocr: copy exact text/numbers verbatim; write "none" if no visible text
- change: compared to previous segment; write "video start" for the first segment
- Use English for all output"""

TASK_DESIGN_PROMPT = """You are a streaming video agent training data constructor.

Below are per-segment annotations for a {duration}s video ({num_segments} segments, 4s each):
{annotations_text}

recent_window = 24 seconds (the model can only see the most recent 24s of frames at ask_time).
For recall tasks: support_segment must be >= 7 segments before ask_segment (gap > 24s).

Design the following training tasks (reference segment_id, do not guess timestamps):

A. Immediate answer (R1) x {n_r1}: ask about content visible in the current segment
   answer_type: entity | yesno | number | slot
B. Delayed answer (S3_R2) x {n_s3r2}: ask about something not yet happened, ask_segment < answer_segment, gap >= 2 segments
   answer_type: entity | yesno
C. Visual detail recall (RC1) x {n_rc1}: ask about specific visual attributes (color/shape/clothing) from visual_details that require seeing the original frame — not memorizable from text
   answer_type: slot | entity
D. Numeric recall (RC2) x {n_rc2}: ask about exact numbers/text from ocr field (skip if no ocr segments)
   answer_type: number | slot
E. Procedural recall (RC3) x {n_rc3}: ask about previous step details/order (skip if no procedural actions)
   answer_type: ordered_steps | entity
F. Comparison recall (RC4) x {n_rc4}: ask what changed between support_segment and ask_segment
   answer_type: slot | entity
G. Causal recall (RC5) x 1: ask why something happened, referencing a cause from much earlier
   answer_type: span
H. Entity re-identification (RC6) x 1: ask whether current entity is the same one from before
   answer_type: yesno
I. Trigger (S4_R4) x {n_trigger}: user sets a monitoring condition, agent waits then fires
   answer_type: entity | yesno
J. Progressive answer (R3) x {n_r3}: ask a question whose answer unfolds over time — the agent responds multiple times as new info appears. E.g. "What ingredients are being added?" → response at each new ingredient. Provide response_segments (2-4 segments where the agent should respond) and partial_answers for each.
   answer_type: entity | slot
K. Multi-turn follow-up recall (RC7) x {n_rc7}: first a base Q&A happens at base_ask_segment, then much later (>24s) a follow-up question references that earlier conversation. The follow-up requires recall to retrieve the base Q&A context. Provide base_question, base_answer, base_ask_segment.
   answer_type: slot | entity | yesno

Example tasks:
{{"task_type":"RC1","question":"What color was the apron the chef was wearing at the beginning?","support_segment":"seg_002","ask_segment":"seg_015","answer_segment":"seg_015","expected_answer":"red","answer_type":"slot","natural_response":"The chef was wearing a red apron.","why_recall":"The chef's apron was visible around t=8s but is now outside the 24s window at t=60s.","query_candidates":[{{"query":"chef apron color","time_bias":"past_far","target":"entity","topk":3}},{{"query":"apron beginning","time_bias":"past_far","target":"entity","topk":3}}]}}

{{"task_type":"R3","question":"What ingredients are being added to the pan?","ask_segment":"seg_005","response_segments":["seg_005","seg_008","seg_011"],"partial_answers":["Adding oil","Now adding garlic","Adding the sliced onions"],"expected_answer":"oil, garlic, onions","answer_type":"entity","natural_response":"Oil, garlic, and sliced onions were added."}}

{{"task_type":"RC7","base_question":"What did the chef add to the soup?","base_answer":"Salt and pepper","base_ask_segment":"seg_003","question":"Earlier you said something was added to the soup — how much salt was it?","support_segment":"seg_003","ask_segment":"seg_015","answer_segment":"seg_015","expected_answer":"two pinches","answer_type":"slot","natural_response":"It was about two pinches of salt.","why_recall":"The base Q&A about soup ingredients happened at t=12s, now out of window at t=60s.","query_candidates":[{{"query":"salt added soup amount","time_bias":"past_far","target":"entity","topk":3}},{{"query":"soup seasoning","time_bias":"past_far","target":"entity","topk":3}}]}}

Output JSON for each task with these exact keys:
{{"task_type":"R1|S3_R2|R3|RC1|RC2|RC3|RC4|RC5|RC6|RC7|S4_R4",
  "question":"natural conversational English question",
  "support_segment":"seg_xx (recall tasks only, otherwise null)",
  "ask_segment":"seg_xx",
  "answer_segment":"seg_xx (same as ask for R1/RC, later for S3_R2/S4_R4, null for R3)",
  "expected_answer":"short verifiable answer",
  "answer_type":"yesno|number|slot|entity|ordered_steps|span",
  "natural_response":"1-2 sentence natural answer",
  "why_recall":"reason (recall tasks only, otherwise null)",
  "query_candidates":[{{"query":"short keywords NOT a question","time_bias":"past_far|past_recent|any","target":"entity|action|ocr|procedure","topk":3}}],
  "response_segments":["seg_xx","seg_xx"] (R3 only, otherwise omit),
  "partial_answers":["answer1","answer2"] (R3 only, otherwise omit),
  "base_question":"..." (RC7 only),
  "base_answer":"..." (RC7 only),
  "base_ask_segment":"seg_xx" (RC7 only)}}

query_candidates rules (recall tasks only, at least 2):
- NOT a question, no pronouns, no articles
- Must contain entity or object anchor words
- Short (2-6 words) but discriminative enough to retrieve the right segment

Output JSON array only, no explanation."""

THINK_PROMPT = """You are a streaming video agent. Below are the current video frames (t={window_start}s to t={window_end}s).
Current chunk: t={chunk_start}s-{chunk_end}s, happening: {chunk_description}

{context}

Write your internal reasoning in 15-48 tokens. Do not write an action. Output English only."""

RECALL_PHRASING = ["earlier", "before", "previously", "a while ago", "at the beginning"]

# ---------------------------------------------------------------------------
# Per-step concurrency & token config (based on GPU memory analysis)
# 8× MI300X 192GB, Qwen3.5-397B-A17B-FP8
# ---------------------------------------------------------------------------

STEP_CONFIG = {
    "2a": {  # Segment annotation
        "max_concurrent": 64,
        "max_tokens": 512,
        "temperature": 0.3,
    },
    "2b": {  # Task design
        "max_concurrent": 64,
        "max_tokens": 4096,
        "temperature": 0.7,
    },
    "2d": {  # Think generation — 24 frames × 1500 = 36K+ per request
        "max_concurrent": 32,
        "max_tokens": 128,
        "temperature": 0.5,
    },
}


# ===================================================================
# Step 1: Video Selection + Frame Extraction
# ===================================================================


def select_videos(
    streamo_dir: str,
    video_root: str,
    num_videos: int = 200,
    min_duration: float = 60.0,
    seed: int = 42,
) -> List[Dict]:
    """Select videos from Streamo for data construction."""
    random.seed(seed)

    # Load Streamo annotations
    anno_path = Path(streamo_dir) / "raw_data.json"
    if not anno_path.exists():
        # Try jsonl
        anno_path = Path(streamo_dir) / "raw_data.jsonl"

    logger.info("Loading annotations from %s", anno_path)
    if anno_path.suffix == ".json":
        with open(anno_path) as f:
            data = json.load(f)
    else:
        with open(anno_path) as f:
            data = [json.loads(l) for l in f if l.strip()]

    # Group by video
    video_map = {}
    for row in data:
        vid_name = row.get("video_name", "")
        if not vid_name:
            continue
        if vid_name not in video_map:
            video_map[vid_name] = {
                "video_name": vid_name,
                "video_path": row.get("video_path", ""),
                "source": row.get("source", "unknown"),
                "max_time": 0.0,
                "task_types": set(),
                "num_annotations": 0,
            }
        entry = video_map[vid_name]
        entry["num_annotations"] += 1
        entry["task_types"].add(row.get("task_type", ""))
        for resp in row.get("response", []):
            for key in ("end_time", "st_time", "time"):
                val = resp.get(key)
                if val and float(val) > entry["max_time"]:
                    entry["max_time"] = float(val)

    # Filter by duration
    candidates = [
        v for v in video_map.values()
        if v["max_time"] >= min_duration
    ]
    logger.info("Found %d videos >= %.0fs (from %d total)", len(candidates), min_duration, len(video_map))

    # Sort by estimated richness
    for v in candidates:
        v["task_types"] = list(v["task_types"])
        v["score"] = (
            min(1.0, v["max_time"] / 180.0) * 0.4
            + min(1.0, len(v["task_types"]) / 5.0) * 0.3
            + min(1.0, v["num_annotations"] / 10.0) * 0.3
        )
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Select top candidates, diversify by source
    selected = candidates[:min(num_videos * 3, len(candidates))]
    if len(selected) > num_videos:
        selected = random.sample(selected, num_videos)

    # Build video list with full paths
    result = []
    for v in selected:
        vpath = v["video_path"]
        if not Path(vpath).is_absolute():
            vpath = str(Path(video_root) / vpath)
        result.append({
            "video_id": v["video_name"].replace(".mp4", ""),
            "video_path": vpath,
            "duration_sec": v["max_time"],
            "source": v["source"],
            "num_annotations": v["num_annotations"],
        })

    return result


def extract_frames(video_path: str, output_dir: Path, fps: int = FPS) -> List[Dict]:
    """Extract frames at given fps, grouped into 4s segments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    duration = float(result.stdout.strip())

    # Extract all frames at fps, resize to 720px short edge
    # 720px → ~1500 vision tokens per frame (good for OCR + visual details)
    # Step 2d with 24 window frames → ~37.5K tokens, requires max-model-len >= 65536
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={fps},scale='if(gt(iw,ih),720,-2)':'if(gt(iw,ih),-2,720)'",
         "-q:v", "2",
         str(output_dir / "frame_%06d.jpg")],
        check=True, capture_output=True,
    )

    # Group into segments
    segments = []
    num_frames = int(duration * fps)
    seg_frames = SEGMENT_SEC * fps  # 4s * 2fps = 8 frames

    for seg_idx in range(0, num_frames, seg_frames):
        start_sec = seg_idx / fps
        end_sec = min(start_sec + SEGMENT_SEC, duration)
        frame_paths = []
        for f_idx in range(seg_idx, min(seg_idx + seg_frames, num_frames)):
            fp = output_dir / f"frame_{f_idx + 1:06d}.jpg"
            if fp.exists():
                frame_paths.append(str(fp))

        if frame_paths:
            # Select evenly spaced keyframes (1fps = 4 frames per 4s segment)
            n = min(FRAMES_PER_SEGMENT, len(frame_paths))
            if len(frame_paths) <= n:
                selected_frames = frame_paths
            else:
                step = len(frame_paths) / n
                indices = [int(i * step) for i in range(n)]
                selected_frames = [frame_paths[i] for i in indices]

            segments.append({
                "segment_id": f"seg_{seg_idx // seg_frames:03d}",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "frame_paths": selected_frames,
                "all_frame_paths": frame_paths,  # keep full set for later use
            })

    return segments


# ===================================================================
# Step 2a: Segment Annotation
# ===================================================================


def build_annotation_requests(
    video_id: str,
    segments: List[Dict],
) -> List[Dict]:
    """Build vLLM requests for segment annotation."""
    requests = []
    for seg in segments:
        prompt = SEGMENT_ANNOTATE_PROMPT.format(
            start=f"{seg['start_sec']:.0f}",
            end=f"{seg['end_sec']:.0f}",
        )
        content = build_content_with_images(prompt, seg["frame_paths"])
        requests.append({
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.3,
            "id": f"{video_id}_{seg['segment_id']}",
            "_meta": {"video_id": video_id, "segment_id": seg["segment_id"], **seg},
        })
    return requests


def parse_annotation_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse 397B annotation output into structured format."""
    default = {
        "segment_id": meta["segment_id"],
        "start_sec": meta["start_sec"],
        "end_sec": meta["end_sec"],
        "action": "",
        "entities": [],
        "visual_details": [],
        "ocr": "none",
        "change": "",
        "frame_paths": meta.get("frame_paths", []),
    }
    if raw is None:
        return default

    try:
        # Try direct parse first, then extract JSON block
        parsed = None
        try:
            parsed = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            # Find the first balanced JSON object
            start = raw.find("{")
            if start >= 0:
                depth = 0
                for i in range(start, len(raw)):
                    if raw[i] == "{":
                        depth += 1
                    elif raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            parsed = json.loads(raw[start:i + 1])
                            break
        if parsed:
            default.update({
                "action": parsed.get("action", ""),
                "entities": parsed.get("entities", []),
                "visual_details": parsed.get("visual_details", []),
                "ocr": parsed.get("ocr", "none"),
                "change": parsed.get("change", ""),
            })
    except (json.JSONDecodeError, ValueError):
        default["action"] = raw[:200]  # Fallback: use raw text

    return default


# ===================================================================
# Step 2b: Task Design
# ===================================================================


def build_task_design_request(
    video_id: str,
    segments: List[Dict],  # annotated segments
    duration_sec: float,
) -> Dict:
    """Build a single request for 397B to design tasks for one video."""
    num_segments = len(segments)

    # Build annotations text
    anno_lines = []
    for seg in segments:
        line = (
            f"{seg['segment_id']} [{seg['start_sec']:.0f}-{seg['end_sec']:.0f}s]: "
            f"{seg.get('action', '')} | "
            f"entities: {seg.get('entities', [])} | "
            f"details: {json.dumps(seg.get('visual_details', {}), ensure_ascii=False)} | "
            f"ocr: {seg.get('ocr', 'none')} | "
            f"change: {seg.get('change', '')}"
        )
        anno_lines.append(line)

    # Calculate task counts based on video length
    has_ocr = any(s.get("ocr", "none") != "none" for s in segments)
    has_steps = num_segments > 10  # Long enough for procedural

    n_r1 = 3
    n_s3r2 = 2
    n_rc1 = min(3, max(1, num_segments // 10))
    n_rc2 = 2 if has_ocr else 0
    n_rc3 = 2 if has_steps else 0
    n_rc4 = 1 if num_segments > 15 else 0
    n_trigger = 1
    n_r3 = 1 if num_segments > 10 else 0  # progressive answer needs enough segments
    n_rc7 = 1 if num_segments > 15 else 0  # follow-up needs base Q&A + 24s gap

    prompt = TASK_DESIGN_PROMPT.format(
        duration=f"{duration_sec:.0f}",
        num_segments=num_segments,
        annotations_text="\n".join(anno_lines),
        n_r1=n_r1, n_s3r2=n_s3r2, n_rc1=n_rc1, n_rc2=n_rc2,
        n_rc3=n_rc3, n_rc4=n_rc4, n_trigger=n_trigger,
        n_r3=n_r3, n_rc7=n_rc7,
    )

    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.7,
        "id": f"design_{video_id}",
        "_meta": {"video_id": video_id, "duration_sec": duration_sec},
    }


def parse_task_design_result(
    raw: Optional[str], video_id: str
) -> List[Dict]:
    """Parse 397B task design output."""
    if raw is None:
        return []

    try:
        # Try direct parse first
        try:
            tasks = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            # Find the first balanced JSON array
            start = raw.find("[")
            if start < 0:
                logger.warning("No JSON array found in task design for %s", video_id)
                return []
            depth = 0
            tasks = None
            for i in range(start, len(raw)):
                if raw[i] == "[":
                    depth += 1
                elif raw[i] == "]":
                    depth -= 1
                    if depth == 0:
                        tasks = json.loads(raw[start:i + 1])
                        break
            if tasks is None:
                logger.warning("Unbalanced JSON array in task design for %s", video_id)
                return []
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse task design for %s", video_id)
        return []

    if not isinstance(tasks, list):
        tasks = [tasks]

    for i, task in enumerate(tasks):
        task["task_id"] = f"{video_id}_task_{i:03d}"
        task["video_id"] = video_id

    return tasks


# ===================================================================
# Step 2c: Rule Validation
# ===================================================================


def validate_task(
    task: Dict,
    segments: List[Dict],
    duration_sec: float,
) -> Optional[Dict]:
    """Validate and fix task timing using segment information."""
    seg_map = {s["segment_id"]: s for s in segments}

    # Resolve segment references
    support_seg = seg_map.get(task.get("support_segment", ""))
    ask_seg = seg_map.get(task.get("ask_segment", ""))
    answer_seg = seg_map.get(task.get("answer_segment", ""))

    if not ask_seg:
        return None  # Can't proceed without ask segment

    task["ask_time_sec"] = ask_seg["start_sec"]

    if answer_seg:
        task["answer_time_sec"] = answer_seg["start_sec"]
    else:
        task["answer_time_sec"] = task["ask_time_sec"]

    task_type = task.get("task_type", "")

    # RC7 must be checked BEFORE the generic RC* handler
    if task_type == "RC7":
        # Multi-turn follow-up recall
        base_ask_seg = seg_map.get(task.get("base_ask_segment", ""))
        if not base_ask_seg:
            return None
        if not task.get("base_question") or not task.get("base_answer"):
            return None
        if not support_seg:
            # support_segment should be base_ask_segment for RC7
            support_seg = base_ask_seg
            task["support_segment"] = base_ask_seg["segment_id"]
        task["support_time_sec"] = support_seg["start_sec"]
        task["base_ask_time_sec"] = base_ask_seg["start_sec"]
        task["need_recall"] = True
        # Constraint: base Q&A + 24s < follow-up ask
        gap = task["ask_time_sec"] - task["base_ask_time_sec"]
        if gap < RECENT_WINDOW_SEC:
            fixed_ask = task["base_ask_time_sec"] + RECENT_WINDOW_SEC + 2
            if fixed_ask > duration_sec:
                return None
            task["ask_time_sec"] = fixed_ask
            task["answer_time_sec"] = fixed_ask
            task["timing_fixed"] = True
            for s in segments:
                if s["start_sec"] <= fixed_ask < s["end_sec"]:
                    task["ask_segment"] = s["segment_id"]
                    task["answer_segment"] = s["segment_id"]
                    break

    elif task_type == "R3":
        task["need_recall"] = False
        # Validate response_segments list
        resp_segs = task.get("response_segments") or []
        partial_answers = task.get("partial_answers") or []
        if len(resp_segs) < 2:
            return None
        # Resolve all response segments to times
        resp_times = []
        for rs_id in resp_segs:
            rs = seg_map.get(rs_id)
            if not rs:
                return None
            resp_times.append(rs["start_sec"])
        # Must be in ascending order
        if resp_times != sorted(resp_times):
            return None
        task["response_times_sec"] = resp_times
        task["answer_time_sec"] = resp_times[-1]  # last response
        # Pad partial_answers if needed
        if len(partial_answers) < len(resp_segs):
            partial_answers.extend([""] * (len(resp_segs) - len(partial_answers)))
        task["partial_answers"] = partial_answers[:len(resp_segs)]

    elif task_type.startswith("RC"):
        # Generic recall handler (RC1-RC6)
        if not support_seg:
            return None
        task["support_time_sec"] = support_seg["start_sec"]
        task["need_recall"] = True
        # Constraint: support + 24s < ask
        gap = task["ask_time_sec"] - task["support_time_sec"]
        if gap < RECENT_WINDOW_SEC:
            fixed_ask = task["support_time_sec"] + RECENT_WINDOW_SEC + 2
            if fixed_ask > duration_sec:
                return None
            task["ask_time_sec"] = fixed_ask
            task["answer_time_sec"] = fixed_ask
            task["timing_fixed"] = True
            for s in segments:
                if s["start_sec"] <= fixed_ask < s["end_sec"]:
                    task["ask_segment"] = s["segment_id"]
                    task["answer_segment"] = s["segment_id"]
                    break

    elif task_type == "S3_R2":
        task["need_recall"] = False
        # Constraint: ask < answer
        if task["ask_time_sec"] >= task["answer_time_sec"]:
            return None
        # Minimum gap
        if task["answer_time_sec"] - task["ask_time_sec"] < AGENT_CHUNK_SEC * 2:
            return None

    elif task_type == "S4_R4":
        task["need_recall"] = False
        if not answer_seg or task["ask_time_sec"] >= task["answer_time_sec"]:
            return None

    else:  # R1 and other response types
        task["need_recall"] = False
        task["answer_time_sec"] = task["ask_time_sec"]

    # General validation
    if not task.get("question"):
        return None
    if not task.get("expected_answer") and not task.get("natural_response"):
        return None
    if task["ask_time_sec"] > duration_sec:
        return None

    task["status"] = "verified"
    return task


# ===================================================================
# Step 2d: Think Generation
# ===================================================================


def _collect_window_frames(
    segments: List[Dict], ask_time: float, window_sec: int = RECENT_WINDOW_SEC,
    max_frames: int = 24,
) -> List[str]:
    """Collect frames from segments within the recent window.

    Returns up to max_frames evenly sampled from the window.
    Default 24 frames → 24 × 1500 = 36K vision tokens, requires max-model-len >= 65536.
    """
    window_start = max(0, ask_time - window_sec)
    window_segs = [
        s for s in segments
        if s["end_sec"] > window_start and s["start_sec"] < ask_time
    ]
    all_frames = []
    for s in window_segs:
        all_frames.extend(s.get("frame_paths", []))

    if len(all_frames) <= max_frames:
        return all_frames

    step = len(all_frames) / max_frames
    return [all_frames[int(i * step)] for i in range(max_frames)]


def build_think_requests(
    tasks: List[Dict],
    video_annotations: Dict[str, List[Dict]],
    max_window_frames: int = 24,
) -> List[Dict]:
    """Build requests for 397B to write think content.

    Sends ACTUAL FRAMES from the recent window (not text annotations)
    to avoid hallucination. Each request includes up to max_window_frames
    frames from the 24s window.

    Token budget per request: ~24 frames × 1500 tok + ~1500 text ≈ 37.5K
    Requires --max-model-len >= 65536.
    """
    requests = []
    for task in tasks:
        video_id = task["video_id"]
        segments = video_annotations.get(video_id, [])
        ask_time = task["ask_time_sec"]
        window_start = max(0, ask_time - RECENT_WINDOW_SEC)

        # Collect actual frames from recent window
        window_frames = _collect_window_frames(
            segments, ask_time, RECENT_WINDOW_SEC, max_window_frames
        )

        # Find current chunk segment
        chunk_seg = None
        for seg in segments:
            if seg["start_sec"] <= ask_time < seg["end_sec"]:
                chunk_seg = seg
                break
        if not chunk_seg:
            chunk_seg = segments[-1] if segments else {
                "action": "", "start_sec": 0, "end_sec": 0,
            }

        if task.get("need_recall"):
            context = (
                f'User just asked: "{task["question"]}"\n'
                f"You are viewing frames from t={window_start:.0f}s to t={ask_time:.0f}s.\n"
                f"Determine whether the current window has enough information to answer. If not, explain what historical information needs to be retrieved."
            )
        else:
            context = (
                f'User just asked: "{task["question"]}"\n'
                f"You are viewing frames from t={window_start:.0f}s to t={ask_time:.0f}s.\n"
                f"Determine whether the current window has enough information to answer."
            )

        prompt = THINK_PROMPT.format(
            window_start=f"{window_start:.0f}",
            window_end=f"{ask_time:.0f}",
            chunk_start=f"{chunk_seg['start_sec']:.0f}",
            chunk_end=f"{chunk_seg['end_sec']:.0f}",
            chunk_description=chunk_seg.get("action", ""),
            context=context,
        )

        # Send actual window frames + prompt
        if window_frames:
            content = build_content_with_images(prompt, window_frames)
        else:
            content = prompt

        requests.append({
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 128,
            "temperature": 0.5,
            "id": f"think_{task['task_id']}",
            "_meta": {"task_id": task["task_id"], "role": "ask"},
        })

        # For recall tasks: post-recall think with support segment frames
        if task.get("need_recall"):
            support_seg = None
            for seg in segments:
                if seg["segment_id"] == task.get("support_segment"):
                    support_seg = seg
                    break
            if support_seg:
                # Recall think sees: support frames (evidence) + current chunk frames (context)
                support_frames = support_seg.get("frame_paths", [])[:FRAMES_PER_SEGMENT]
                current_frames = chunk_seg.get("frame_paths", [])[:2]  # 2 frames for current context
                recall_all_frames = support_frames + current_frames

                recall_prompt = (
                    f"You are a streaming video agent. You just triggered recall and retrieved historical frames (t={support_seg['start_sec']:.0f}s).\n"
                    f"The first {len(support_frames)} frames are retrieved history, the last {len(current_frames)} frames are the current view.\n"
                    f"The user's question is: \"{task['question']}\"\n"
                    f"Combining history and current view, write your reasoning in 15-40 tokens. Output English only."
                )
                if recall_all_frames:
                    recall_content = build_content_with_images(recall_prompt, recall_all_frames)
                else:
                    recall_content = recall_prompt
                requests.append({
                    "messages": [{"role": "user", "content": recall_content}],
                    "max_tokens": 64,
                    "temperature": 0.3,
                    "id": f"think_recall_{task['task_id']}",
                    "_meta": {"task_id": task["task_id"], "role": "after_recall"},
                })

    return requests


# ===================================================================
# Step 3: Triplet Binding
# ===================================================================


def generate_triplets(tasks: List[Dict]) -> List[Dict]:
    """Generate three-way binding for each recall task.

    RC7 (multi-turn follow-up) is excluded because its control would
    reference a base conversation that doesn't exist in the control episode.
    """
    triplets = []
    for task in tasks:
        if not task.get("need_recall"):
            continue
        if task.get("task_type") == "RC7":
            continue  # RC7 control needs base Q&A — skip triplet binding

        # Control: same question, ask when evidence is visible
        control = copy.deepcopy(task)
        control["task_id"] = task["task_id"] + "_ctrl"
        control["task_type"] = "R1"
        control["ask_time_sec"] = task.get("support_time_sec", 0) + 2
        control["answer_time_sec"] = control["ask_time_sec"]
        control["need_recall"] = False
        control["think_at_ask"] = "The answer is visible in the current window, responding directly."
        control["triplet_role"] = "control"

        # False-recall negative: add recall phrasing but evidence is visible
        fn = copy.deepcopy(task)
        fn["task_id"] = task["task_id"] + "_fn"
        fn["task_type"] = "R7"
        prefix = random.choice(RECALL_PHRASING)
        q = fn["question"]
        if not any(p.lower() in q.lower() for p in RECALL_PHRASING):
            # Lowercase the first char of original question after prefix
            q_body = q[0].lower() + q[1:] if q else q
            fn["question"] = f"{prefix.capitalize()}, {q_body}"
        fn["ask_time_sec"] = task.get("support_time_sec", 0) + 2
        fn["answer_time_sec"] = fn["ask_time_sec"]
        fn["need_recall"] = False
        fn["think_at_ask"] = f"Although the user said '{prefix}', the evidence is in the current window — no recall needed."
        fn["triplet_role"] = "false_negative"

        task["triplet_role"] = "recall_positive"
        triplets.append({
            "triplet_id": f"tri_{task['task_id']}",
            "recall": task,
            "control": control,
            "false_negative": fn,
        })

    return triplets


# ===================================================================
# Step 4: Chunk-Level Assembly
# ===================================================================


def assemble_episode(
    task: Dict,
    segments: List[Dict],
    duration_sec: float,
    video_path: str,
) -> Dict:
    """Assemble a task into chunk-level SFT episode."""
    task_type = task.get("task_type", "")

    # ── R3: Progressive answer (multiple response chunks) ──
    if task_type == "R3":
        return _assemble_r3(task, segments, duration_sec, video_path)

    # ── RC7: Multi-turn follow-up recall ──
    if task_type == "RC7":
        return _assemble_rc7(task, segments, duration_sec, video_path)

    ask_time = task["ask_time_sec"]
    answer_time = task["answer_time_sec"]

    # Determine chunk range
    start_time = max(0, ask_time - RECENT_WINDOW_SEC)
    end_time = min(answer_time + AGENT_CHUNK_SEC * 2, duration_sec)

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # Build chunk sequence
    t = start_time
    while t < end_time:
        chunk_end = min(t + AGENT_CHUNK_SEC, duration_sec)
        is_ask_chunk = (t <= ask_time < chunk_end)
        is_answer_chunk = (t <= answer_time < chunk_end) and answer_time != ask_time

        # Find segment for this chunk
        seg = None
        for s in segments:
            if s["start_sec"] <= t < s["end_sec"]:
                seg = s
                break

        # User message
        user_content = "<video>"
        if is_ask_chunk:
            user_content += f"\n<question>\n{task['question']}\n</question>"

        messages.append({"role": "user", "content": user_content})

        # Assistant message
        if task.get("need_recall") and is_ask_chunk:
            # Recall chunk
            think = task.get("think_at_ask", "Cannot find the answer in the current window, need to recall.")
            candidates = task.get("query_candidates") or []
            query = candidates[0] if candidates else {
                "query": task.get("expected_answer", ""),
                "time_bias": "past_far",
                "target": "entity",
                "topk": 3,
            }
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>{think}</think>"
                    f"<action>recall</action>"
                    f"<query>{json.dumps(query, ensure_ascii=False)}</query>"
                ),
            })

            # Recall result (user injection)
            support_seg = None
            for s in segments:
                if s["segment_id"] == task.get("support_segment"):
                    support_seg = s
                    break
            if support_seg:
                recall_text = (
                    f"<recall_result>\n"
                    f'<item rank="1" start="{support_seg["start_sec"]:.1f}" '
                    f'end="{support_seg["end_sec"]:.1f}">'
                    f'caption: {support_seg.get("action", "")} '
                    f'ocr: {support_seg.get("ocr", "none")}'
                    f'</item>\n'
                    f"</recall_result>\nContinue following the protocol to respond."
                )
            else:
                recall_text = "<recall_result>\n<item>Not found</item>\n</recall_result>\nContinue following the protocol to respond."

            messages.append({"role": "user", "content": recall_text})

            # Post-recall response
            think_after = task.get("think_after_recall", "Retrieved relevant information, can now answer.")
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>{think_after}</think>"
                    f"<action>response</action>"
                    f"<response>{task.get('natural_response', task.get('expected_answer', ''))}</response>"
                ),
            })
            break  # End episode after recall+response

        elif is_ask_chunk or is_answer_chunk:
            # Response chunk (immediate or delayed answer)
            think = task.get("think_at_ask", "The current frames have sufficient information to answer.")
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>{think}</think>"
                    f"<action>response</action>"
                    f"<response>{task.get('natural_response', task.get('expected_answer', ''))}</response>"
                ),
            })
            if is_answer_chunk:
                break

        else:
            # Silent chunk
            is_waiting = (
                task.get("task_type") in ("S3_R2", "S4_R4")
                and ask_time <= t < answer_time
            )
            if is_waiting:
                think = f"User asked about this, but the event hasn't occurred yet. Waiting."
            elif seg:
                think = seg.get("action", "Scene in progress.")[:50]
            else:
                think = "Continuing to observe."
            messages.append({
                "role": "assistant",
                "content": f"<think>{think}</think><action>silent</action>",
            })

        t = chunk_end

    return _make_episode(task, messages, video_path)


def _build_canonical(task: Dict) -> Dict:
    """Build canonical answer dict from task."""
    expected = task.get("expected_answer", "")
    answer_type = task.get("answer_type", "")
    if answer_type not in ("yesno", "number", "slot", "entity", "ordered_steps", "span", "multiple_choice"):
        if expected.lower() in ("yes", "no"):
            answer_type = "yesno"
        elif re.match(r'^\d+', expected):
            answer_type = "number"
        else:
            answer_type = "entity"
    return {"answer_type": answer_type, "value": {"answer": expected}}


def _make_episode(task: Dict, messages: List[Dict], video_path: str) -> Dict:
    """Build the final episode dict."""
    return {
        "episode_id": f"ep_{task['task_id']}",
        "task_id": task["task_id"],
        "task_type": task.get("task_type", ""),
        "sample_type": task.get("triplet_role", "simple"),
        "video_id": task["video_id"],
        "video_path": video_path,
        "messages": messages,
        "canonical_answer": _build_canonical(task),
        "need_recall": task.get("need_recall", False),
        "difficulty": "hard" if task.get("need_recall") else "medium",
        "protocol_version": "3action",
    }


def _assemble_r3(
    task: Dict, segments: List[Dict], duration_sec: float, video_path: str,
) -> Dict:
    """Assemble R3 (progressive answer) episode.

    User asks once, agent responds multiple times as new info appears.
    Between responses the agent outputs silent (observing, waiting for more).
    """
    ask_time = task["ask_time_sec"]
    resp_times = task.get("response_times_sec", [])
    partial_answers = task.get("partial_answers", [])
    if not resp_times:
        resp_times = [ask_time]
        partial_answers = [task.get("natural_response", task.get("expected_answer", ""))]

    start_time = max(0, ask_time - AGENT_CHUNK_SEC * 2)
    end_time = min(resp_times[-1] + AGENT_CHUNK_SEC * 2, duration_sec)

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    resp_idx = 0  # which response we're on
    question_asked = False

    t = start_time
    while t < end_time:
        chunk_end = min(t + AGENT_CHUNK_SEC, duration_sec)
        is_ask_chunk = (t <= ask_time < chunk_end)

        seg = None
        for s in segments:
            if s["start_sec"] <= t < s["end_sec"]:
                seg = s
                break

        # User message
        user_content = "<video>"
        if is_ask_chunk:
            user_content += f"\n<question>\n{task['question']}\n</question>"
            question_asked = True
        messages.append({"role": "user", "content": user_content})

        # Check if this chunk matches a response time
        is_resp_chunk = (
            resp_idx < len(resp_times)
            and t <= resp_times[resp_idx] < chunk_end
        )

        if is_resp_chunk:
            answer_text = partial_answers[resp_idx] if resp_idx < len(partial_answers) else ""
            if resp_idx == 0:
                think = task.get("think_at_ask", "New information visible, responding.")
            else:
                think = f"New information appeared, updating answer ({resp_idx + 1}/{len(resp_times)})."
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>{think}</think>"
                    f"<action>response</action>"
                    f"<response>{answer_text}</response>"
                ),
            })
            resp_idx += 1
            if resp_idx >= len(resp_times):
                break  # all responses done
        elif question_asked and resp_idx < len(resp_times):
            # Between responses: silent, waiting for more info
            think = "Watching for more information to appear."
            messages.append({
                "role": "assistant",
                "content": f"<think>{think}</think><action>silent</action>",
            })
        else:
            # Before question
            think = seg.get("action", "Scene in progress.")[:50] if seg else "Continuing to observe."
            messages.append({
                "role": "assistant",
                "content": f"<think>{think}</think><action>silent</action>",
            })

        t = chunk_end

    return _make_episode(task, messages, video_path)


_RC7_CONTEXT_CHUNKS = 3  # silent chunks to keep around each key event


def _assemble_rc7(
    task: Dict, segments: List[Dict], duration_sec: float, video_path: str,
) -> Dict:
    """Assemble RC7 (multi-turn follow-up recall) episode.

    1. Base Q&A happens at base_ask_time (within window at that point)
    2. Gap is truncated to _RC7_CONTEXT_CHUNKS silent chunks (avoids 30+ filler silents)
    3. Follow-up question at ask_time requires recall to retrieve base Q&A context
    """
    base_ask_time = task.get("base_ask_time_sec", 0)
    ask_time = task["ask_time_sec"]

    # Build two windows: around base Q&A and around follow-up, skip the gap
    base_start = max(0, base_ask_time - AGENT_CHUNK_SEC * _RC7_CONTEXT_CHUNKS)
    base_end = base_ask_time + AGENT_CHUNK_SEC * (_RC7_CONTEXT_CHUNKS + 1)
    followup_start = max(base_end, ask_time - AGENT_CHUNK_SEC * _RC7_CONTEXT_CHUNKS)
    followup_end = min(ask_time + AGENT_CHUNK_SEC * 2, duration_sec)

    # Collect chunk times from both windows
    chunk_times = []
    t = base_start
    while t < base_end:
        chunk_times.append(t)
        t += AGENT_CHUNK_SEC
    t = followup_start
    while t < followup_end:
        chunk_times.append(t)
        t += AGENT_CHUNK_SEC

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    base_done = False

    for t in chunk_times:
        chunk_end = t + AGENT_CHUNK_SEC
        is_base_ask = (t <= base_ask_time < chunk_end) and not base_done
        is_followup_ask = (t <= ask_time < chunk_end) and base_done

        seg = None
        for s in segments:
            if s["start_sec"] <= t < s["end_sec"]:
                seg = s
                break

        # User message
        user_content = "<video>"
        if is_base_ask:
            user_content += f"\n<question>\n{task.get('base_question', '')}\n</question>"
        elif is_followup_ask:
            user_content += f"\n<question>\n{task['question']}\n</question>"
        messages.append({"role": "user", "content": user_content})

        # Assistant message
        if is_base_ask:
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>The answer is visible in the current frames.</think>"
                    f"<action>response</action>"
                    f"<response>{task.get('base_answer', '')}</response>"
                ),
            })
            base_done = True

        elif is_followup_ask:
            think = task.get("think_at_ask", "The earlier conversation is outside my window, need to recall.")
            candidates = task.get("query_candidates") or []
            query = candidates[0] if candidates else {
                "query": task.get("base_question", ""),
                "time_bias": "past_far",
                "target": "dialogue",
                "topk": 3,
            }
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>{think}</think>"
                    f"<action>recall</action>"
                    f"<query>{json.dumps(query, ensure_ascii=False)}</query>"
                ),
            })

            # Recall result
            support_seg = None
            for s in segments:
                if s["segment_id"] == task.get("support_segment"):
                    support_seg = s
                    break
            base_q = task.get("base_question", "")
            base_a = task.get("base_answer", "")
            if support_seg:
                recall_text = (
                    f"<recall_result>\n"
                    f'<item rank="1" start="{support_seg["start_sec"]:.1f}" '
                    f'end="{support_seg["end_sec"]:.1f}">'
                    f'caption: {support_seg.get("action", "")} '
                    f'Q: {base_q} A: {base_a}'
                    f'</item>\n'
                    f"</recall_result>\nContinue following the protocol to respond."
                )
            else:
                recall_text = "<recall_result>\n<item>Not found</item>\n</recall_result>\nContinue following the protocol to respond."

            messages.append({"role": "user", "content": recall_text})

            think_after = task.get("think_after_recall", "Retrieved the earlier conversation, can now answer the follow-up.")
            messages.append({
                "role": "assistant",
                "content": (
                    f"<think>{think_after}</think>"
                    f"<action>response</action>"
                    f"<response>{task.get('natural_response', task.get('expected_answer', ''))}</response>"
                ),
            })
            break

        else:
            if seg:
                think = seg.get("action", "Scene in progress.")[:50]
            else:
                think = "Continuing to observe."
            messages.append({
                "role": "assistant",
                "content": f"<think>{think}</think><action>silent</action>",
            })

    return _make_episode(task, messages, video_path)


# ===================================================================
# Step 5: Filtering
# ===================================================================


_FORMAT_RE = re.compile(
    r"<think>.*?</think><action>(?:silent|response|recall)</action>"
    r"(?:<response>.*?</response>|<query>\{.*?\}</query>)?",
    re.DOTALL,
)


def filter_episode(episode: Dict) -> Tuple[bool, str]:
    """Apply hard-rule filtering. Returns (pass, reason)."""
    messages = episode.get("messages", [])
    if len(messages) < 3:
        return False, "too_few_messages"

    # Check format compliance
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"] if isinstance(msg["content"], str) else ""
            if not _FORMAT_RE.search(content):
                return False, f"format_mismatch: {content[:80]}"

    # Check recall timing
    if episode.get("need_recall"):
        task_id = episode.get("task_id", "")
        # Timing is already validated in Step 2c
        pass

    # Check non-empty answer
    ca = episode.get("canonical_answer", {})
    if not ca.get("value", {}).get("answer"):
        return False, "empty_answer"

    return True, "ok"


# ===================================================================
# Step 6: Sampling + Final Assembly
# ===================================================================


def assemble_final(
    episodes: List[Dict],
    output_dir: Path,
) -> Dict[str, int]:
    """Sample by target ratios and write final training data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by type
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep["sample_type"]].append(ep)

    recall_pos = by_type.get("recall_positive", [])
    control = by_type.get("control", [])
    false_neg = by_type.get("false_negative", [])
    simple = by_type.get("simple", [])

    # SFT-A: protocol warmstart (recall 0-5%)
    sft_a = (
        simple[:]
        + recall_pos[:len(simple) // 4]
        + control[:len(simple) // 4]
    )
    random.shuffle(sft_a)

    # SFT-B: recall heavy
    sft_b = (
        recall_pos
        + control
        + false_neg
        + random.sample(simple, min(len(simple), len(recall_pos)))
    )
    random.shuffle(sft_b)

    # RL: verifiable answers only
    rl = [
        ep for ep in episodes
        if ep.get("canonical_answer", {}).get("answer_type") in
           ("yesno", "number", "entity", "slot", "multiple_choice")
    ]

    def write_jsonl(items, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(sft_a, output_dir / "sft_a.jsonl")
    write_jsonl(sft_b, output_dir / "sft_b.jsonl")
    write_jsonl(rl, output_dir / "rl_pool.jsonl")

    stats = {
        "sft_a": len(sft_a),
        "sft_b": len(sft_b),
        "rl": len(rl),
        "recall_positive": len(recall_pos),
        "control": len(control),
        "false_negative": len(false_neg),
        "simple": len(simple),
        "total_episodes": len(episodes),
    }
    return stats


# ===================================================================
# Statistics
# ===================================================================


def print_statistics(
    videos: List[Dict],
    annotations: Dict[str, List[Dict]],
    tasks: List[Dict],
    episodes: List[Dict],
):
    """Print comprehensive statistics."""
    print(f"\n{'='*70}")
    print("DATA CONSTRUCTION STATISTICS")
    print(f"{'='*70}")

    # Video stats
    print(f"\n--- Videos ---")
    print(f"  Total: {len(videos)}")
    durations = [v["duration_sec"] for v in videos]
    print(f"  Duration: mean={sum(durations)/len(durations):.0f}s, "
          f"min={min(durations):.0f}s, max={max(durations):.0f}s")
    source_dist = Counter(v.get("source", "?") for v in videos)
    for src, cnt in source_dist.most_common():
        print(f"    {src}: {cnt}")

    # Segment stats
    total_segs = sum(len(segs) for segs in annotations.values())
    print(f"\n--- Segments ---")
    print(f"  Total: {total_segs}")
    ocr_count = sum(
        1 for segs in annotations.values() for s in segs
        if s.get("ocr", "none") != "none"
    )
    print(f"  With OCR: {ocr_count} ({100*ocr_count/max(total_segs,1):.1f}%)")

    # Task stats
    print(f"\n--- Tasks ---")
    print(f"  Total: {len(tasks)}")
    type_dist = Counter(t.get("task_type", "?") for t in tasks)
    for tt, cnt in type_dist.most_common():
        print(f"    {tt}: {cnt} ({100*cnt/max(len(tasks),1):.1f}%)")
    recall_count = sum(1 for t in tasks if t.get("need_recall"))
    print(f"  Need recall: {recall_count} ({100*recall_count/max(len(tasks),1):.1f}%)")

    # Episode stats
    print(f"\n--- Episodes ---")
    print(f"  Total: {len(episodes)}")
    sample_dist = Counter(e.get("sample_type", "?") for e in episodes)
    for st, cnt in sample_dist.most_common():
        print(f"    {st}: {cnt}")

    # Action distribution
    action_counts = Counter()
    for ep in episodes:
        for msg in ep.get("messages", []):
            if msg["role"] == "assistant":
                c = msg["content"] if isinstance(msg["content"], str) else ""
                if "<action>silent</action>" in c:
                    action_counts["silent"] += 1
                elif "<action>response</action>" in c:
                    action_counts["response"] += 1
                elif "<action>recall</action>" in c:
                    action_counts["recall"] += 1

    total_actions = sum(action_counts.values())
    if total_actions:
        print(f"\n--- Action Distribution ---")
        for action in ["silent", "response", "recall"]:
            cnt = action_counts.get(action, 0)
            print(f"    {action}: {cnt} ({100*cnt/total_actions:.1f}%)")

    print(f"{'='*70}")


# ===================================================================
# Main Pipeline
# ===================================================================


async def run_pipeline(
    api_base: str,
    model: str,
    streamo_dir: str,
    video_root: str,
    output_dir: str,
    max_concurrent: int = 40,
    num_videos: int = 200,
    seed: int = 42,
):
    """Run the full data construction pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = VLLMClient(api_base, model, max_concurrent=max_concurrent)

    # ── Step 1: Select videos + Extract frames ──
    logger.info("Step 1: Selecting videos and extracting frames...")
    registry_path = out / "video_registry.jsonl"

    if registry_path.exists():
        logger.info("Loading existing video registry")
        with open(registry_path) as f:
            videos = [json.loads(l) for l in f if l.strip()]
    else:
        videos = select_videos(streamo_dir, video_root, num_videos, seed=seed)
        with open(registry_path, "w") as f:
            for v in videos:
                f.write(json.dumps(v, ensure_ascii=False) + "\n")

    logger.info("Selected %d videos", len(videos))

    # Extract frames
    frames_dir = out / "frames"
    all_segments = {}
    failed_videos = []
    for v in videos:
        vid = v["video_id"]
        seg_cache = out / "segments" / f"{vid}.json"
        if seg_cache.exists():
            with open(seg_cache) as f:
                all_segments[vid] = json.load(f)
            # Update duration from actual segments
            if all_segments[vid]:
                v["duration_sec"] = max(v["duration_sec"], all_segments[vid][-1]["end_sec"])
        else:
            try:
                segs = extract_frames(v["video_path"], frames_dir / vid)
                all_segments[vid] = segs
                seg_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(seg_cache, "w") as f:
                    json.dump(segs, f, ensure_ascii=False)
                # Update duration from ffprobe-based segments
                if segs:
                    v["duration_sec"] = max(v["duration_sec"], segs[-1]["end_sec"])
            except Exception as exc:
                logger.warning("Failed to extract frames for %s: %s", vid, exc)
                failed_videos.append(vid)
    if failed_videos:
        logger.warning("Skipped %d videos due to extraction failures", len(failed_videos))
        videos = [v for v in videos if v["video_id"] not in failed_videos]

    total_segs = sum(len(s) for s in all_segments.values())
    logger.info("Total segments: %d", total_segs)

    # ── Step 2a: Segment annotation ──
    anno_path = out / "segment_annotations.jsonl"
    if anno_path.exists():
        logger.info("Loading existing segment annotations")
        with open(anno_path) as f:
            all_annos_flat = [json.loads(l) for l in f if l.strip()]
        video_annotations = defaultdict(list)
        for a in all_annos_flat:
            video_annotations[a.get("video_id", "")].append(a)
    else:
        logger.info("Step 2a: Annotating %d segments (concurrent=%d)...",
                     total_segs, STEP_CONFIG["2a"]["max_concurrent"])
        anno_requests = []
        for vid, segs in all_segments.items():
            anno_requests.extend(build_annotation_requests(vid, segs))

        client.semaphore = asyncio.Semaphore(STEP_CONFIG["2a"]["max_concurrent"])
        results = await client.batch_chat(anno_requests, max_tokens=512, temperature=0.3)
        client.print_stats()

        video_annotations = defaultdict(list)
        all_annos_flat = []
        for req, result in zip(anno_requests, results):
            meta = req["_meta"]
            anno = parse_annotation_result(result, meta)
            anno["video_id"] = meta["video_id"]
            video_annotations[meta["video_id"]].append(anno)
            all_annos_flat.append(anno)

        with open(anno_path, "w") as f:
            for a in all_annos_flat:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # ── Step 2b: Task design ──
    task_raw_path = out / "task_candidates_raw.jsonl"
    if task_raw_path.exists():
        logger.info("Loading existing task candidates")
        with open(task_raw_path) as f:
            all_tasks_raw = [json.loads(l) for l in f if l.strip()]
    else:
        logger.info("Step 2b: Designing tasks for %d videos (concurrent=%d)...",
                     len(videos), STEP_CONFIG["2b"]["max_concurrent"])
        design_requests = []
        for v in videos:
            vid = v["video_id"]
            annos = video_annotations.get(vid, [])
            if annos:
                req = build_task_design_request(vid, annos, v["duration_sec"])
                design_requests.append(req)

        client.semaphore = asyncio.Semaphore(STEP_CONFIG["2b"]["max_concurrent"])
        results = await client.batch_chat(design_requests, max_tokens=4096, temperature=0.7)
        client.print_stats()

        all_tasks_raw = []
        for req, result in zip(design_requests, results):
            vid = req["_meta"]["video_id"]
            tasks = parse_task_design_result(result, vid)
            all_tasks_raw.extend(tasks)

        with open(task_raw_path, "w") as f:
            for t in all_tasks_raw:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    logger.info("Raw tasks: %d", len(all_tasks_raw))

    # ── Step 2c: Rule validation ──
    logger.info("Step 2c: Validating tasks...")
    video_dur = {v["video_id"]: v["duration_sec"] for v in videos}
    verified_tasks = []
    rejected = 0
    for task in all_tasks_raw:
        vid = task["video_id"]
        segs = video_annotations.get(vid, [])
        dur = video_dur.get(vid, 0)
        result = validate_task(task, segs, dur)
        if result:
            verified_tasks.append(result)
        else:
            rejected += 1

    logger.info("Verified: %d, Rejected: %d", len(verified_tasks), rejected)

    with open(out / "task_candidates_verified.jsonl", "w") as f:
        for t in verified_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ── Step 2d: Think generation ──
    task_pool_path = out / "task_pool.jsonl"
    if task_pool_path.exists():
        logger.info("Loading existing task pool")
        with open(task_pool_path) as f:
            verified_tasks = [json.loads(l) for l in f if l.strip()]
    else:
        logger.info("Step 2d: Generating think content (concurrent=%d)...",
                     STEP_CONFIG["2d"]["max_concurrent"])
        think_requests = build_think_requests(verified_tasks, video_annotations)
        client.semaphore = asyncio.Semaphore(STEP_CONFIG["2d"]["max_concurrent"])
        results = await client.batch_chat(think_requests, max_tokens=128, temperature=0.5)
        client.print_stats()

        # Map think results back to tasks
        task_map = {t["task_id"]: t for t in verified_tasks}
        for req, result in zip(think_requests, results):
            meta = req["_meta"]
            task_id = meta["task_id"]
            if task_id in task_map and result:
                if meta["role"] == "ask":
                    task_map[task_id]["think_at_ask"] = result.strip()
                elif meta["role"] == "after_recall":
                    task_map[task_id]["think_after_recall"] = result.strip()

        verified_tasks = list(task_map.values())
        with open(task_pool_path, "w") as f:
            for t in verified_tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # ── Step 3: Triplet binding ──
    logger.info("Step 3: Generating triplets...")
    triplets = generate_triplets(verified_tasks)
    with open(out / "task_triplets.jsonl", "w") as f:
        for tri in triplets:
            f.write(json.dumps(tri, ensure_ascii=False) + "\n")
    logger.info("Triplets: %d (from %d recall tasks)", len(triplets),
                sum(1 for t in verified_tasks if t.get("need_recall")))

    # ── Step 4: Assembly ──
    logger.info("Step 4: Assembling episodes...")
    all_tasks_for_assembly = list(verified_tasks)
    for tri in triplets:
        all_tasks_for_assembly.append(tri["control"])
        all_tasks_for_assembly.append(tri["false_negative"])

    video_path_map = {v["video_id"]: v["video_path"] for v in videos}
    episodes = []
    for task in all_tasks_for_assembly:
        vid = task["video_id"]
        segs = video_annotations.get(vid, [])
        dur = video_dur.get(vid, 0)
        vpath = video_path_map.get(vid, "")
        ep = assemble_episode(task, segs, dur, vpath)
        episodes.append(ep)

    with open(out / "sft_episodes_raw.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # ── Step 5: Filtering ──
    logger.info("Step 5: Filtering...")
    filtered = []
    filter_stats = Counter()
    for ep in episodes:
        passed, reason = filter_episode(ep)
        filter_stats[reason] += 1
        if passed:
            filtered.append(ep)

    logger.info("Filter results: %s", dict(filter_stats))

    # ── Step 6: Final assembly ──
    logger.info("Step 6: Final assembly...")
    stats = assemble_final(filtered, out)

    # ── Statistics ──
    print_statistics(videos, video_annotations, verified_tasks, filtered)

    print(f"\n{'='*70}")
    print("FINAL OUTPUT")
    print(f"{'='*70}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput directory: {out}")
    print(f"{'='*70}")

    # Save stats
    with open(out / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Agent Data Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--api_base", required=True, help="vLLM API base URL")
    run_parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    run_parser.add_argument("--streamo_dir", required=True, help="Streamo data directory")
    run_parser.add_argument("--video_root", required=True, help="Video files root")
    run_parser.add_argument("--output_dir", default="data/agent")
    run_parser.add_argument("--max_concurrent", type=int, default=40)
    run_parser.add_argument("--num_videos", type=int, default=200)
    run_parser.add_argument("--seed", type=int, default=42)

    # Stress test command
    st_parser = subparsers.add_parser("stress_test", help="Test vLLM throughput")
    st_parser.add_argument("--api_base", required=True)
    st_parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    st_parser.add_argument("--max_concurrent", type=int, default=8)
    st_parser.add_argument("--num_requests", type=int, default=20)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.command == "run":
        asyncio.run(run_pipeline(
            api_base=args.api_base,
            model=args.model,
            streamo_dir=args.streamo_dir,
            video_root=args.video_root,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            num_videos=args.num_videos,
            seed=args.seed,
        ))
    elif args.command == "stress_test":
        from .vllm_client import stress_test
        asyncio.run(stress_test(
            api_base=args.api_base,
            model=args.model,
            num_requests=args.num_requests,
            max_concurrent=args.max_concurrent,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
