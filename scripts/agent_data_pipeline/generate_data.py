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
import math
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "你是流视频问答 agent。你会持续接收视频片段流。\n\n"
    "每次 assistant turn 必须严格输出以下三种形式之一：\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    '<query>{"query":"...","time_bias":"...","target":"...","topk":3}</query>\n\n'
    "规则：\n"
    "- 每个 turn 只允许一个 action。\n"
    "- think 只写当前新增判断，不复述整段视频。\n"
    "- 若当前 recent window 证据已足够，直接 response，不要 recall。\n"
    "- 若问题依赖已离开 recent window 的历史内容，优先 recall。\n"
    "- recall query 必须短、可检索、非完整问句、避免代词和泛词。\n"
    "- 收到 <recall_result> 后，只能再输出 silent 或 response。\n"
    "- 若当前 chunk 不该说话，输出 silent。"
)

# Prompt templates
SEGMENT_ANNOTATE_PROMPT = """请描述这个视频片段 (t={start}-{end}s) 中看到的内容：
1. 主要动作（谁在做什么）
2. 出现的实体（人/物/场景，列出名称）
3. 视觉细节（颜色、形状、文字、数字、穿着、位置关系）
4. OCR文字（如果有可见文字/数字，完整列出；没有写"无"）
5. 与前一段的变化（新出现/消失/状态改变，如果是第一段写"视频开始"）

输出严格 JSON：
{{"action":"...","entities":["..."],"visual_details":{{"实体名":"描述"}},"ocr":"...","change":"..."}}"""

TASK_DESIGN_PROMPT = """你是流视频 agent 训练数据构造器。

以下是一个 {duration}s 视频的逐段标注（共 {num_segments} 段，每段 4 秒）：
{annotations_text}

recent_window = 24 秒（模型在 ask_time 时只能看到最近 24 秒的帧）

请设计以下训练任务（引用 segment_id，不要猜测时间）：

A. 即时回答 (R1) × {n_r1}：问当前段可见的内容
B. 延迟回答 (S3_R2) × {n_s3r2}：问还没发生的事，ask_segment < answer_segment，间隔 ≥2 段
C. 视觉细节 recall (RC1) × {n_rc1}：问 visual_details 中的具体属性（颜色/外观/穿着），support_segment 和 ask_segment 间隔 ≥7 段（>24s），必须是文字描述记不住的视觉细节
D. 数值 recall (RC2) × {n_rc2}：问 ocr 中的精确数字/文字（如果有）
E. 步骤 recall (RC3) × {n_rc3}：问之前步骤的细节/顺序（如果有操作步骤）
F. 比较 recall (RC4) × {n_rc4}：问"和之前比有什么变化"
G. Trigger (S4_R4) × {n_trigger}：设定监控条件

每个任务输出 JSON：
{{"task_type":"R1|S3_R2|RC1|RC2|RC3|RC4|RC5|RC6|RC7|S4_R4","question":"自然口语化","support_segment":"seg_xx","ask_segment":"seg_xx","answer_segment":"seg_xx","expected_answer":"简短可验证","natural_response":"自然回答","why_recall":"为什么需要recall(仅recall类)"}}

输出 JSON 数组，不要解释。"""

THINK_PROMPT = """你是流视频 agent。你正在观看视频，当前看到的是 t={window_start}s 到 t={window_end}s 的内容。

当前段 (t={chunk_start}-{chunk_end}s) 的内容: {chunk_description}

{context}

请写出你此刻的内部推理 (think)，只写一句话(20-40字)，描述你的判断。不要写 action。"""

RECALL_PHRASING = ["之前", "前面", "刚才", "早些时候", "一开始"]


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
    duration = float(result.stdout.strip())

    # Extract all frames at fps
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={fps}", "-q:v", "2",
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
        "visual_details": {},
        "ocr": "无",
        "change": "",
        "frame_paths": meta.get("frame_paths", []),
    }
    if raw is None:
        return default

    try:
        # Extract JSON from response
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            parsed = json.loads(match.group())
            default.update({
                "action": parsed.get("action", ""),
                "entities": parsed.get("entities", []),
                "visual_details": parsed.get("visual_details", {}),
                "ocr": parsed.get("ocr", "无"),
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
            f"ocr: {seg.get('ocr', '无')} | "
            f"change: {seg.get('change', '')}"
        )
        anno_lines.append(line)

    # Calculate task counts based on video length
    has_ocr = any(s.get("ocr", "无") != "无" for s in segments)
    has_steps = num_segments > 10  # Long enough for procedural

    n_r1 = 3
    n_s3r2 = 2
    n_rc1 = min(3, max(1, num_segments // 10))
    n_rc2 = 2 if has_ocr else 0
    n_rc3 = 2 if has_steps else 0
    n_rc4 = 1 if num_segments > 15 else 0
    n_trigger = 1

    prompt = TASK_DESIGN_PROMPT.format(
        duration=f"{duration_sec:.0f}",
        num_segments=num_segments,
        annotations_text="\n".join(anno_lines),
        n_r1=n_r1, n_s3r2=n_s3r2, n_rc1=n_rc1, n_rc2=n_rc2,
        n_rc3=n_rc3, n_rc4=n_rc4, n_trigger=n_trigger,
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
        # Try to extract JSON array
        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            tasks = json.loads(match.group())
        else:
            tasks = json.loads(raw)
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

    # Recall tasks: validate support timing
    if task_type.startswith("RC"):
        if not support_seg:
            return None
        task["support_time_sec"] = support_seg["start_sec"]
        task["need_recall"] = True

        # Constraint: support + 24s < ask
        gap = task["ask_time_sec"] - task["support_time_sec"]
        if gap < RECENT_WINDOW_SEC:
            # Try to fix: push ask_time forward
            fixed_ask = task["support_time_sec"] + RECENT_WINDOW_SEC + 2
            if fixed_ask > duration_sec:
                return None  # Video too short
            task["ask_time_sec"] = fixed_ask
            task["answer_time_sec"] = fixed_ask
            task["timing_fixed"] = True

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


def build_think_requests(
    tasks: List[Dict],
    video_annotations: Dict[str, List[Dict]],
) -> List[Dict]:
    """Build requests for 397B to write think content."""
    requests = []
    for task in tasks:
        video_id = task["video_id"]
        segments = video_annotations.get(video_id, [])
        ask_time = task["ask_time_sec"]

        # Find chunk description at ask time
        chunk_seg = None
        for seg in segments:
            if seg["start_sec"] <= ask_time < seg["end_sec"]:
                chunk_seg = seg
                break
        if not chunk_seg:
            chunk_seg = segments[-1] if segments else {"action": "", "start_sec": 0, "end_sec": 0}

        window_start = max(0, ask_time - RECENT_WINDOW_SEC)

        if task.get("need_recall"):
            context = f'用户刚问: "{task["question"]}"\n你判断是否需要检索历史片段来回答。'
        else:
            context = f'用户刚问: "{task["question"]}"\n你判断当前窗口内是否有足够信息回答。'

        prompt = THINK_PROMPT.format(
            window_start=f"{window_start:.0f}",
            window_end=f"{ask_time:.0f}",
            chunk_start=f"{chunk_seg['start_sec']:.0f}",
            chunk_end=f"{chunk_seg['end_sec']:.0f}",
            chunk_description=chunk_seg.get("action", ""),
            context=context,
        )

        requests.append({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.5,
            "id": f"think_{task['task_id']}",
            "_meta": {"task_id": task["task_id"], "role": "ask"},
        })

        # For recall tasks: also generate post-recall think
        if task.get("need_recall"):
            support_seg = None
            for seg in segments:
                if seg["segment_id"] == task.get("support_segment"):
                    support_seg = seg
                    break
            if support_seg:
                recall_context = (
                    f"刚才检索到了历史片段 (t={support_seg['start_sec']:.0f}s): "
                    f"{support_seg.get('action', '')}\n"
                    f"用户的问题是: \"{task['question']}\"\n"
                    f"请写出获得检索结果后的推理。"
                )
                requests.append({
                    "messages": [{"role": "user", "content": recall_context}],
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
    """Generate three-way binding for each recall task."""
    triplets = []
    for task in tasks:
        if not task.get("need_recall"):
            continue

        # Control: same question, ask when evidence is visible
        control = copy.deepcopy(task)
        control["task_id"] = task["task_id"] + "_ctrl"
        control["task_type"] = "R1"
        control["ask_time_sec"] = task.get("support_time_sec", 0) + 2
        control["answer_time_sec"] = control["ask_time_sec"]
        control["need_recall"] = False
        control["think_at_ask"] = "当前窗口内就能看到答案，直接回答。"
        control["triplet_role"] = "control"

        # False-recall negative: add recall phrasing but evidence is visible
        fn = copy.deepcopy(task)
        fn["task_id"] = task["task_id"] + "_fn"
        fn["task_type"] = "R7"
        prefix = random.choice(RECALL_PHRASING)
        q = fn["question"]
        if not any(p in q for p in RECALL_PHRASING):
            fn["question"] = f"{prefix}{q}"
        fn["ask_time_sec"] = task.get("support_time_sec", 0) + 2
        fn["answer_time_sec"] = fn["ask_time_sec"]
        fn["need_recall"] = False
        fn["think_at_ask"] = f"虽然用户说了'{prefix}'，但证据就在当前窗口里，不需要检索。"
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
            think = task.get("think_at_ask", "当前窗口内找不到答案，需要检索。")
            query = task.get("query_candidates", [{}])[0] if task.get("query_candidates") else {
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
                    f'ocr: {support_seg.get("ocr", "无")}'
                    f'</item>\n'
                    f"</recall_result>\n继续按协议回答。"
                )
            else:
                recall_text = "<recall_result>\n<item>未找到</item>\n</recall_result>\n继续按协议回答。"

            messages.append({"role": "user", "content": recall_text})

            # Post-recall response
            think_after = task.get("think_after_recall", "检索到相关信息，可以回答。")
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
            think = task.get("think_at_ask", "当前画面有足够信息回答。")
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
            if seg:
                think = seg.get("action", "场景进行中。")[:50]
            else:
                think = "继续观察。"
            messages.append({
                "role": "assistant",
                "content": f"<think>{think}</think><action>silent</action>",
            })

        t = chunk_end

    # Build canonical answer
    expected = task.get("expected_answer", "")
    canonical = {"answer_type": "entity", "value": {"answer": expected}}
    if expected.lower() in ("是", "否", "yes", "no"):
        canonical["answer_type"] = "yesno"
    elif re.match(r'^\d+', expected):
        canonical["answer_type"] = "number"

    return {
        "episode_id": f"ep_{task['task_id']}",
        "task_id": task["task_id"],
        "task_type": task.get("task_type", ""),
        "sample_type": task.get("triplet_role", "simple"),
        "video_id": task["video_id"],
        "video_path": video_path,
        "messages": messages,
        "canonical_answer": canonical,
        "need_recall": task.get("need_recall", False),
        "difficulty": "hard" if task.get("need_recall") else "medium",
        "protocol_version": "3action",
    }


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
           ("yesno", "number", "entity", "multiple_choice")
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
        if s.get("ocr", "无") != "无"
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
    for v in videos:
        vid = v["video_id"]
        seg_cache = out / "segments" / f"{vid}.json"
        if seg_cache.exists():
            with open(seg_cache) as f:
                all_segments[vid] = json.load(f)
        else:
            segs = extract_frames(v["video_path"], frames_dir / vid)
            all_segments[vid] = segs
            seg_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(seg_cache, "w") as f:
                json.dump(segs, f, ensure_ascii=False)

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
        logger.info("Step 2a: Annotating %d segments...", total_segs)
        anno_requests = []
        for vid, segs in all_segments.items():
            anno_requests.extend(build_annotation_requests(vid, segs))

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
        logger.info("Step 2b: Designing tasks for %d videos...", len(videos))
        design_requests = []
        for v in videos:
            vid = v["video_id"]
            annos = video_annotations.get(vid, [])
            if annos:
                req = build_task_design_request(vid, annos, v["duration_sec"])
                design_requests.append(req)

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
        logger.info("Step 2d: Generating think content...")
        think_requests = build_think_requests(verified_tasks, video_annotations)
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
