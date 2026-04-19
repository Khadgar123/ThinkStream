"""
Shared configuration for the Agent Data Construction Pipeline.

All paths, constants, prompts, and schema definitions live here so that
every stage script imports from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Directory layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ThinkStream/
DATA_ROOT = PROJECT_ROOT / "data" / "agent"

# Stage 0 outputs
SEGMENT_ARCHIVE_DIR = DATA_ROOT / "segment_archive"
KEYFRAME_DIR = DATA_ROOT / "keyframes"
EMBEDDING_DIR = DATA_ROOT / "embeddings"
META_DIR = DATA_ROOT / "meta"

# Stage 1 outputs
EVENT_TIMELINE_DIR = DATA_ROOT / "event_timeline"

# Stage 2-5 intermediate & final episode tables
EPISODE_RAW_PATH = DATA_ROOT / "episode_recall_raw.jsonl"
EPISODE_DENSE_PATH = DATA_ROOT / "episode_recall_dense.jsonl"
EPISODE_VERIFIED_PATH = DATA_ROOT / "episode_recall_verified.jsonl"
EPISODE_FINAL_PATH = DATA_ROOT / "episode_recall.jsonl"

# Stage 6 outputs
SFT_DIR = DATA_ROOT / "sft_final"
RETRIEVER_TRAIN_PATH = DATA_ROOT / "retriever_train.jsonl"
RL_POOL_PATH = DATA_ROOT / "rl_pool.jsonl"
CLIP_DIR = DATA_ROOT / "clips"

ALL_DIRS = [
    DATA_ROOT, SEGMENT_ARCHIVE_DIR, KEYFRAME_DIR, EMBEDDING_DIR, META_DIR,
    EVENT_TIMELINE_DIR, SFT_DIR, CLIP_DIR,
]

# ---------------------------------------------------------------------------
# 2. Video chunking & segment parameters
# ---------------------------------------------------------------------------

AGENT_CHUNK_SEC = 2.0          # Fixed 2-second chunk for agent data
SEGMENT_SEC = 4                # Segment length for episodic memory archive
SEGMENT_OVERLAP_SEC = 2        # Overlap between segments
RAW_FPS = 2                    # Frames-per-second for extraction
FRAMES_PER_CHUNK = 2           # Frames per 2s chunk (= 1 fps)
KEYFRAMES_PER_SEGMENT = 2      # Representative keyframes per segment

# ---------------------------------------------------------------------------
# 3. Recent window & recall
# ---------------------------------------------------------------------------

RECENT_WINDOW_SEC = 24         # Recent visual buffer in seconds
MAX_RECALL_PER_CHUNK = 1       # V1: at most one recall per chunk
RECALL_TOPK = 3                # Retrieval top-k

# ---------------------------------------------------------------------------
# 4. Token budgets
# ---------------------------------------------------------------------------

THINK_TOKENS_SILENT = (10, 30)       # (min, max) for silent chunks
THINK_TOKENS_RESPONSE = (15, 48)     # for response chunks
THINK_TOKENS_RECALL = (15, 48)       # for recall chunks
RESPONSE_TOKENS_BRIEF = (10, 80)     # brief immediate answer
RESPONSE_TOKENS_EXPLAIN = (40, 200)  # explanatory answer
QUERY_TOKENS = (30, 100)             # recall query JSON

# ---------------------------------------------------------------------------
# 5. Action distribution targets (chunk-level)
# ---------------------------------------------------------------------------

ACTION_DIST_TARGET = {
    "silent": (0.58, 0.65),
    "response": (0.23, 0.30),
    "recall": (0.10, 0.15),
}

# ---------------------------------------------------------------------------
# 6. Difficulty distribution targets
# ---------------------------------------------------------------------------

DIFFICULTY_DIST_TARGET = {
    "easy": 0.25,
    "medium": 0.35,
    "hard": 0.30,
    "very_hard": 0.10,
}

# ---------------------------------------------------------------------------
# 7. Task type enum
# ---------------------------------------------------------------------------

TASK_TYPES = [
    "current_perception",
    "short_temporal",
    "retrospective_detail",
    "procedural_state",
    "compare_past_present",
    "long_causal",
    "ocr_history",
    "multi_turn_followup",
    "delayed_trigger",
    "continuous_narration",
]

# ---------------------------------------------------------------------------
# 8. Event type enum
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    "procedure_step",
    "state_change",
    "entity_action",
    "dialogue",
    "ocr_event",
    "scene_transition",
]

# ---------------------------------------------------------------------------
# 9. Answer type enum (canonical_answer.answer_type)
# ---------------------------------------------------------------------------

VERIFIABLE_ANSWER_TYPES = [
    "multiple_choice",
    "yesno",
    "number",
    "slot",
    "entity",
]

ALL_ANSWER_TYPES = VERIFIABLE_ANSWER_TYPES + ["ordered_steps", "span"]

# ---------------------------------------------------------------------------
# 10. Verification gate thresholds
# ---------------------------------------------------------------------------

GATE_THRESHOLDS = {
    "gate1_overlap_max": 0,        # support & recent overlap must be 0
    "gate2_hit_at_k": True,        # at least 1 hit in top-k
    "gate3_coverage_min": 0.5,     # temporal overlap ≥ 50 %
    "gate4_no_recall_score_max": 0.4,   # no-recall baseline must fail
    "gate5_with_recall_score_min": 0.8,  # with-recall must pass
    "gate6_score_drop_min": 0.5,   # counterfactual score drop ≥ 0.5
}

# ---------------------------------------------------------------------------
# 11. System prompts (agent protocol)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT_ZH = (
    "你是流视频问答 agent。你会持续接收视频片段流。\n\n"
    "每次 assistant turn 必须严格输出以下三种形式之一：\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    "<query>{\"query\":\"...\",\"time_bias\":\"...\",\"target\":\"...\",\"topk\":3}</query>\n\n"
    "规则：\n"
    "- 每个 turn 只允许一个 action。\n"
    "- think 只写当前新增判断，不复述整段视频。\n"
    "- 若当前 recent window 证据已足够，直接 response，不要 recall。\n"
    "- 若问题依赖已离开 recent window 的历史内容，优先 recall。\n"
    "- recall query 必须短、可检索、非完整问句、避免代词和泛词。\n"
    "- 收到 <recall_result> 后，只能再输出 silent 或 response。\n"
    "- 若当前 chunk 不该说话，输出 silent。"
)

AGENT_SYSTEM_PROMPT_EN = (
    "You are a streaming video QA agent receiving a continuous stream of video chunks.\n\n"
    "Each assistant turn must output exactly one of these three forms:\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    "<query>{\"query\":\"...\",\"time_bias\":\"...\",\"target\":\"...\",\"topk\":3}</query>\n\n"
    "Rules:\n"
    "- Only one action per turn.\n"
    "- Think only about new observations in the current chunk. Do not restate the whole video.\n"
    "- If the recent window has enough evidence, respond directly without recall.\n"
    "- If the answer depends on content that has left the recent window, use recall.\n"
    "- Recall queries must be short, searchable keywords — not full questions, no pronouns.\n"
    "- After receiving <recall_result>, only output silent or response.\n"
    "- If no speech is needed for the current chunk, output silent."
)

# ---------------------------------------------------------------------------
# 12. Teacher prompts (397B)
# ---------------------------------------------------------------------------

TEACHER_TASK_PACK_SYSTEM = (
    "You are a streaming video agent training data constructor.\n"
    "The input is not raw video but a structured timeline of a video, including events, OCR, ASR, entities, actions, causal relations, and candidate support segments.\n"
    "Your task is to generate multiple high-quality streaming tasks at once, with emphasis on recall-required samples.\n\n"
    "Hard requirements:\n"
    "- Only use evidence from the input timeline. No fabrication.\n"
    "- Questions must be suitable for online streaming video scenarios.\n"
    "- For need_recall=true tasks, the gold support must be outside the recent window at ask_time.\n"
    "- Prefer verifiable answers: yes/no, slot, number, entity, ordered_step.\n"
    "- Output 3 query candidates per task for downstream automatic retrieval verification.\n"
    "- Output JSON only, no explanations."
)

TEACHER_TASK_PACK_USER_TEMPLATE = (
    "recent_window_sec = {recent_window_sec}\n"
    "video_type = {video_type}\n"
    "target_task_count = {target_task_count}\n\n"
    "timeline_json =\n{timeline_json}\n\n"
    "archive_summary =\n{archive_summary}\n\n"
    "Output:\n"
    '{{\n'
    '  "video_id": "...",\n'
    '  "tasks": [\n'
    '    {{\n'
    '      "task_id": "...",\n'
    '      "task_type": "retrospective_detail|procedural_state|compare_past_present|long_causal|ocr_history|multi_turn_followup|current_perception|short_temporal|delayed_trigger|continuous_narration",\n'
    '      "question": "...",\n'
    '      "ask_time_ms": 0,\n'
    '      "earliest_response_time_ms": 0,\n'
    '      "support_event_ids": ["..."],\n'
    '      "support_segment_ids": ["..."],\n'
    '      "need_recall": true,\n'
    '      "recall_reason": "...",\n'
    '      "canonical_answer": {{\n'
    '        "answer_type": "slot|entity|yesno|number|ordered_steps|span",\n'
    '        "value": {{}}\n'
    '      }},\n'
    '      "natural_response": "...",\n'
    '      "query_candidates": [\n'
    '        {{"query": "...", "time_bias": "past_recent|past_far|any", "target": "entity|action|event|ocr|dialogue|procedure|cause"}},\n'
    '        {{"query": "...", "time_bias": "...", "target": "..."}},\n'
    '        {{"query": "...", "time_bias": "...", "target": "..."}}\n'
    '      ],\n'
    '      "sparse_think_milestones": [\n'
    '        {{"time_ms": 0, "text": "..."}},\n'
    '        {{"time_ms": 0, "text": "..."}}\n'
    '      ]\n'
    '    }}\n'
    '  ]\n'
    '}}'
)

TEACHER_QUERY_REPAIR_SYSTEM = (
    "You are a recall query compressor.\n"
    "Your goal is to compress \"current question + visible context + gold support summary\" into a short retrieval query.\n\n"
    "Rules:\n"
    "- The query must not be a question.\n"
    "- The query must not use pronouns.\n"
    "- The query must contain entity or object anchors.\n"
    "- Keep the query short but retain discriminative actions/attributes/OCR/numbers.\n"
    "- Generate 3 candidates, from shortest to most robust.\n"
    "- Output JSON only, no explanations."
)

TEACHER_RECALL_JUDGE_SYSTEM = (
    "You are a recall necessity judge.\n"
    "Determine whether this streaming question truly requires recall given the recent window.\n\n"
    "Judgment criteria:\n"
    "- If the recent window already has sufficient evidence, need_recall must be false.\n"
    "- If the question can be reliably answered without recall, need_recall must be false.\n"
    "- need_recall is true only when the answer depends on support evidence outside the recent window.\n"
    "- Output strict JSON."
)

TEACHER_REWRITE_SYSTEM = (
    "You are a response rewriter for a streaming video assistant.\n"
    "Rewrite the canonical answer into a natural, concise, conversational response.\n"
    "Requirements:\n"
    "- No more than 2 sentences.\n"
    "- Answer directly, do not beat around the bush.\n"
    "- Do not add information not present in the supporting evidence."
)

# ---------------------------------------------------------------------------
# 13. Dense caption / OCR prompts (for small VL model)
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "Describe what happens in this 4-second video segment in 1-2 sentences.\n"
    "Requirements:\n"
    "- Describe the main action and participating entities\n"
    "- Mention any obvious state changes\n"
    "- Do not speculate about content outside the frame\n"
    "- Keep it concise"
)

OCR_PROMPT = (
    "Extract all visible text content from this image.\n"
    "If there is no visible text, reply \"none\".\n"
    "Output only the text content itself, no explanations."
)

# ---------------------------------------------------------------------------
# 14. Salience & recallability weights
# ---------------------------------------------------------------------------

SALIENCE_WEIGHTS = {
    "motion": 0.30,
    "entity_density": 0.25,
    "has_ocr_or_asr": 0.20,
    "action_density": 0.15,
    "scene_boundary": 0.10,
}

RECALLABILITY_WEIGHTS = {
    "duration": 0.25,
    "entity_recurrence": 0.20,
    "event_chain_depth": 0.20,
    "procedurality": 0.15,
    "ocr_density": 0.10,
    "asr_density": 0.10,
    "cut_rate": -0.20,
}

# ---------------------------------------------------------------------------
# 15. Procedural action keywords (for recallability scoring)
# ---------------------------------------------------------------------------

PROCEDURAL_KEYWORDS_ZH = [
    "加入", "倒入", "切", "搅拌", "打开", "关闭", "移动",
    "拿起", "放下", "安装", "拆卸", "拧", "按", "点击",
    "涂", "刷", "焊", "钉", "连接", "断开",
]

# ---------------------------------------------------------------------------
# 16. Reward spec defaults (RL)
# ---------------------------------------------------------------------------

DEFAULT_REWARD_SPEC = {
    "wrong_action_penalty": 1.0,
    "over_recall_penalty": 0.3,
    "long_think_penalty_after_tokens": 48,
}

# ---------------------------------------------------------------------------
# 17. Recall hard-negative type ratios
# ---------------------------------------------------------------------------

HARD_NEGATIVE_RATIOS = {
    "temporal_near_miss": 2,
    "semantic_confounder": 1,
    "ocr_confounder": 0.5,
    "same_video_random": 0.5,
}

# ---------------------------------------------------------------------------
# 18. Scene detection threshold (PySceneDetect ContentDetector)
# ---------------------------------------------------------------------------

SCENE_DETECT_THRESHOLD = 27.0

# ---------------------------------------------------------------------------
# Helper: ensure all output directories exist
# ---------------------------------------------------------------------------


def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
