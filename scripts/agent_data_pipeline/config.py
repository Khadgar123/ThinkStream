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
    "你是流视频 agent 训练数据构造器。\n"
    "输入不是原始视频，而是一个视频的结构化时间线，包括事件、OCR、ASR、实体、动作、因果关系和候选支持片段。\n"
    "你的任务是一次生成多个高质量 streaming 任务，重点生成需要 recall 的样本。\n\n"
    "硬性要求：\n"
    "- 只能使用输入时间线中的证据，不允许臆造。\n"
    "- 问题必须适合在线流视频场景。\n"
    "- 对于 need_recall=true 的任务，ask_time 时 gold support 必须位于 recent window 之外。\n"
    "- 优先生成可验证答案：yes/no、slot、number、entity、ordered_step。\n"
    "- 每个任务输出 3 个 query candidates，用于后续自动检索验证。\n"
    "- 不要输出长篇解释，只输出 JSON。"
)

TEACHER_TASK_PACK_USER_TEMPLATE = (
    "recent_window_sec = {recent_window_sec}\n"
    "video_type = {video_type}\n"
    "target_task_count = {target_task_count}\n\n"
    "timeline_json =\n{timeline_json}\n\n"
    "archive_summary =\n{archive_summary}\n\n"
    "请输出：\n"
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
    "你是 recall query 压缩器。\n"
    "你的目标是把\"当前问题 + 当前可见上下文 + gold support 摘要\"压缩成短检索 query。\n\n"
    "规则：\n"
    "- query 不是问句。\n"
    "- query 不能用代词。\n"
    "- query 必须包含实体或物体锚点。\n"
    "- query 尽量短，但要保留区分性动作/属性/OCR/数字。\n"
    "- 生成 3 个候选，从最短到最稳。\n"
    "- 输出 JSON，不要解释。"
)

TEACHER_RECALL_JUDGE_SYSTEM = (
    "你是 recall necessity judge。\n"
    "判断该 streaming 问题在给定 recent window 下是否真的需要 recall。\n\n"
    "判定原则：\n"
    "- 如果 recent window 内已经有充分证据，need_recall 必须为 false。\n"
    "- 如果不给 recall 也能稳定答对，need_recall 必须为 false。\n"
    "- 只有答案依赖 recent window 外支持证据时，need_recall 才能为 true。\n"
    "- 输出严格 JSON。"
)

TEACHER_REWRITE_SYSTEM = (
    "你是流视频助手的话术重写器。\n"
    "把 canonical answer 改写成自然、简短、在线口语化的 response。\n"
    "要求：\n"
    "- 不超过 2 句。\n"
    "- 优先直接回答，不绕弯。\n"
    "- 不要加入支持证据里没有的信息。"
)

# ---------------------------------------------------------------------------
# 13. Dense caption / OCR prompts (for small VL model)
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "请用1-2句中文描述这个4秒视频片段中发生了什么。\n"
    "要求：\n"
    "- 描述主要动作和参与实体\n"
    "- 如果有明显的状态变化，请提及\n"
    "- 不要推测画面外的内容\n"
    "- 保持简洁"
)

OCR_PROMPT = (
    "请提取这张图片中所有可见的文字内容。\n"
    "如果没有可见文字，回复\"无\"。\n"
    "只输出文字内容本身，不要添加解释。"
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
