"""
Configuration for Agent Data Pipeline v6.1.

All constants, prompts, and schema definitions.
Matches docs/data_construction_zh.md v6.2.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Directory layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ThinkStream/
DATA_ROOT = PROJECT_ROOT / "data" / "agent_v5"

# Stage outputs
EVIDENCE_DIR = DATA_ROOT / "evidence_graph"
ROLLOUT_DIR = DATA_ROOT / "rollout"
TASKS_DIR = DATA_ROOT / "tasks"
SAMPLES_DIR = DATA_ROOT / "samples"
FINAL_DIR = DATA_ROOT / "final"
AUDIT_DIR = DATA_ROOT / "audits"

ALL_DIRS = [
    DATA_ROOT, EVIDENCE_DIR, ROLLOUT_DIR, TASKS_DIR,
    SAMPLES_DIR, FINAL_DIR, AUDIT_DIR,
]

# ---------------------------------------------------------------------------
# 2. Video & chunk parameters
# ---------------------------------------------------------------------------

AGENT_CHUNK_SEC = 2          # 每个 chunk 2 秒
FPS = 1                      # 1fps
FRAMES_PER_CHUNK = 2         # 每 chunk 2 帧
VISUAL_WINDOW_CHUNKS = 12    # 视觉窗口 = 最近 12 chunks (24s)
VISUAL_WINDOW_FRAMES = VISUAL_WINDOW_CHUNKS * FRAMES_PER_CHUNK  # 24 帧

# ---------------------------------------------------------------------------
# 3. Think & memory parameters
# ---------------------------------------------------------------------------

THINK_TOKENS = (40, 60)             # student think 长度范围
THINK_TOKEN_AVG = 50                # think 平均 token 数（用于估算）

# Token-based compression trigger with hysteresis
RECENT_THINKS_TOKEN_BUDGET = 600    # recent_thinks 总 token 预算
COMPRESS_TRIGGER_RATIO = 0.8        # 达到预算 80% 时系统触发压缩
COMPRESS_TOKEN_THRESHOLD = int(RECENT_THINKS_TOKEN_BUDGET * COMPRESS_TRIGGER_RATIO)  # = 480
COMPRESS_HYSTERESIS_RATIO = 0.55    # 压缩后应降回 55% 以下，否则窗口太短
COMPRESS_HYSTERESIS_THRESHOLD = int(RECENT_THINKS_TOKEN_BUDGET * COMPRESS_HYSTERESIS_RATIO)  # = 330

# Student model tokenizer (用于精确计算 token 数)
# 造数据时加载一次，全局复用
STUDENT_MODEL = "Qwen/Qwen3-VL-8B"  # 学生模型
_tokenizer = None

def get_tokenizer():
    """Lazy-load student model tokenizer for precise token counting."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
        except Exception:
            _tokenizer = "unavailable"
    return _tokenizer if _tokenizer != "unavailable" else None

COMPRESS_RANGE_MIN = 4              # 每次最少压缩 4 条
COMPRESS_RANGE_MAX = 8              # 每次最多压缩 8 条
SUMMARY_TOKENS_MIN = 100            # summary 最短
SUMMARY_TOKENS_MAX = 180            # summary 最长
COMPRESSION_RATIO_MIN = 2.5        # 最小压缩比
RECALL_RETURN_FRAMES = 4           # recall 返回 4 帧 (4s at 1fps)
MAX_COMPRESSED_SEGMENTS = 5        # 最多保留 5 段压缩

# Per-video candidate limits (controls data volume + API cost)
# Set to 0 to disable limiting for that type
MAX_CANDIDATES_PER_VIDEO = {
    "response_from_frames": 8,
    "response_from_memory": 5,
    "recall": 5,
    "compress_recall": 3,
    "compress_response": 3,
    "unanswerable": 5,
    "pending": 3,
    # "compress" is not limited — determined by actual compression events
}

# Per-video FINAL sample cap (applied after Pass4 generation).
# Candidates limit controls API cost; this controls training set balance.
MAX_SAMPLES_PER_VIDEO = 50

# Backward compat aliases (deprecated — use token-based constants above)
OBSERVATION_TOKENS = THINK_TOKENS  # deprecated alias
COMPRESS_THRESHOLD = 10  # deprecated: item-count fallback, prefer COMPRESS_TOKEN_THRESHOLD
COMPRESS_RANGE = COMPRESS_RANGE_MAX  # deprecated alias

# ---------------------------------------------------------------------------
# 4. Token budgets (草算，需用 tokenizer 实测)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TOKENS = 150
COMPRESSED_SEG_TOKENS = 150        # 每段压缩 ~150 tok
OBSERVATION_AVG_TOKENS = 50
VISUAL_TOKENS_PER_CHUNK = 128      # at min_pixels=100352
VISUAL_WINDOW_TOKENS = VISUAL_WINDOW_CHUNKS * VISUAL_TOKENS_PER_CHUNK  # ~1536
RECALL_VISION_TOKENS = 256         # 4 帧 recalled
MAX_SAMPLE_TOKENS = 4096           # 单样本上限 (远在 16K 内)

# ---------------------------------------------------------------------------
# 4b. 397B context / OOM guards
# ---------------------------------------------------------------------------

# Construction-time guards to prevent over-long or too-wide batches.
VLLM_CONTEXT_SAFETY_RATIO = 0.85
VLLM_PREFILL_BATCH_TOKEN_BUDGET = 32_000_000  # KV usage ~2.6% at 64 conc → 1024 conc fits easily

# Per-request token estimates (text + vision + output + thinking).
# Vision passes: 24-28 frames (recall adds 4 recalled frames).
# vLLM: --limit-mm-per-prompt '{"image":28}' to accommodate recall.
# Thinking tokens estimated at ~2K per request (varies).
PASS_CONTEXT_ESTIMATES = {
    "pass1_evidence": {"input": 1_500, "output": 5_000, "thinking": 0},  # 2 frames only
    "pass2_rollout":  {"input": 10_000, "output": 5_000, "thinking": 0},
    # Pass3/4: now include video frames (~2-4 frames per request).
    # max_tokens=16384 bounds thinking time.
    "pass3_tasks":    {"input": 6_000,  "output": 5_000, "thinking": 0},
    "pass4_forks":    {"input": 4_000,  "output": 5_000, "thinking": 0},
}


def estimated_request_tokens(pass_name: str) -> int:
    """Conservative per-request token estimate including output/thinking."""
    est = PASS_CONTEXT_ESTIMATES.get(pass_name, {})
    return int(est.get("input", 0) + est.get("output", 0) + est.get("thinking", 0))


def max_safe_context_tokens() -> int:
    """Safe request-level context ceiling under the configured 397B context."""
    return int(VLLM_MAX_MODEL_LEN * VLLM_CONTEXT_SAFETY_RATIO)


def safe_concurrency_for_pass(pass_name: str) -> int:
    """Clamp configured concurrency by context length and batch-token budget.

    Rule: data quality beats throughput. If the request estimate approaches
    context limit or prefill batch budget, lower concurrency automatically.
    """
    cfg = PASS_CONFIG.get(pass_name, {})
    requested = int(cfg.get("concurrent_videos", cfg.get("concurrent", 1)))
    per_request = max(1, estimated_request_tokens(pass_name))
    if per_request > max_safe_context_tokens():
        return 1
    by_batch = max(1, VLLM_PREFILL_BATCH_TOKEN_BUDGET // per_request)
    return max(1, min(requested, by_batch))


# ---------------------------------------------------------------------------
# 5. Action distribution targets (episode-level)
# ---------------------------------------------------------------------------

# 按 phase 控制不同类型 episode 的比例
PHASE_CONFIG = {
    1: {
        "name": "protocol_alignment",
        "actions": ["silent", "response"],
        "episode_mix": {"silent_only": 0.3, "response": 0.70},
        "lr": 1e-5, "epochs": 3,
    },
    2: {
        "name": "recall",
        "actions": ["silent", "response", "recall"],
        "episode_mix": {"silent_only": 0.1, "response": 0.4, "recall": 0.4,
                        "negative": 0.1},
        "lr": 5e-6, "epochs": 3,
    },
    "C1": {
        "name": "fixed_compress",
        "actions": ["silent", "response", "recall", "compress"],
        "episode_mix": {"response": 0.3, "recall": 0.25, "compress": 0.25,
                        "compress_recall": 0.2},
        "lr": 3e-6, "epochs": 2,
    },
    "C2": {
        "name": "adaptive_compress",
        "actions": ["silent", "response", "recall", "compress"],
        "episode_mix": {"response": 0.25, "recall": 0.2, "compress_adaptive": 0.3,
                        "compress_recall": 0.25},
        "lr": 2e-6, "epochs": 2,
    },
    5: {
        "name": "mixed",
        "actions": ["silent", "response", "recall", "compress"],
        "episode_mix": {"response": 0.35, "recall": 0.25, "compress": 0.15,
                        "compress_recall": 0.15, "negative": 0.1},
        "lr": 1e-6, "epochs": 1,
    },
}

# ---------------------------------------------------------------------------
# 6. Quality thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7          # teacher fact confidence >= this to make task
ENTITY_COVERAGE_THRESHOLD = 0.7     # grounding: obs entities vs caption entities
LEAKAGE_OVERLAP_THRESHOLD = 0.3     # keyword overlap triggering leakage flag
PROACTIVE_RECALL_RATE = 0.05        # ~5% of chunks trigger proactive recall

# ---------------------------------------------------------------------------
# 7. 397B vLLM configuration
# ---------------------------------------------------------------------------

VLLM_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
VLLM_MAX_MODEL_LEN = 65536

PASS_CONFIG = {
    # All passes: thinking enabled, --reasoning-parser qwen3 separates
    # thinking into reasoning_content, content is clean output.
    # max_tokens covers thinking + response total. Set generously
    # to avoid truncation — data quality > token efficiency.
    "pass1_evidence": {
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": True,
        "concurrent_videos": 1024,
    },
    "pass2_rollout": {
        "max_tokens_observation": 16384,
        "max_tokens_compress": 16384,
        "temperature": 0.3,
        "thinking": True,
        "concurrent_videos": 1024,
    },
    "pass3_tasks": {
        "max_tokens": 16384,
        "temperature": 0.7,
        "thinking": True,
        "concurrent": 1024,
    },
    "pass4_forks": {
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": True,
        "concurrent": 1024,
    },
}

# ---------------------------------------------------------------------------
# 8. System prompt (4-action protocol)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3 Prompt Protocol (v8.0)
# Each prompt teaches ONE type of decision. No mixed behavior.
# ---------------------------------------------------------------------------

# Prompt 1: Main loop — observe video, decide silent/response/recall
SYSTEM_PROMPT = (
    "You are a streaming video agent. You observe video chunks and maintain memory.\n\n"
    "Each turn, output exactly ONE of:\n"
    "1) <think>...</think><action>silent</action>\n"
    "2) <think>...</think><action>response</action><response>...</response>\n"
    "3) <think>...</think><action>recall</action>"
    '<query>{"query":"...","time_range":"..."}</query>\n\n'
    "Rules:\n"
    "- think: 40-60 tokens, describe ONLY what is newly visible in the current chunk. "
    "No meta-reasoning, no sound/smell/emotion.\n"
    "- response: answer based on currently visible information only.\n"
    "- recall: only when answer is NOT in visual window, NOT in text memory, "
    "NOT in compressed summaries.\n"
    "- If a query exists in <queries>, respond when you see new relevant information."
)

# Prompt 2: Post-recall — evaluate recall result, decide silent/response
SYSTEM_PROMPT_POST_RECALL = (
    "You are a streaming video agent. You just received recall results.\n\n"
    "Output exactly ONE of:\n"
    "1) <think>...</think><action>silent</action>  (recall result not useful)\n"
    "2) <think>...</think><action>response</action><response>...</response>  (answer the question)\n\n"
    "Rules:\n"
    "- think: 20-40 tokens, analyze whether the recall result answers the question.\n"
    "- response: answer based on the recall result. If insufficient, say so.\n"
    "- silent: recall result is irrelevant or empty, cannot answer."
)

# Prompt 3: Compress — triggered by system between timesteps, only output summary
SYSTEM_PROMPT_COMPRESS = (
    "You are a streaming video agent. The system has triggered memory compression.\n\n"
    "Below are recent observations that need to be compressed into a summary.\n"
    "{compress_context}\n\n"
    "Output a JSON summary:\n"
    '{{"time_range": [start_sec, end_sec], "text": "compressed summary"}}\n\n'
    "Rules:\n"
    "- Keep all entities with their appearance descriptions\n"
    "- Keep ALL OCR content verbatim\n"
    "- Keep state changes as before→after\n"
    "- Target length: {target_length} tokens"
)

# Special tokens required by SFT init_processor (see sft_engineering.md §6.2)
# Approach B: exact-match tags, attributes as JSON inside tags.
SPECIAL_TOKENS_BASE = [
    "<silent>", "<response>", "<think>", "</think>",
    "<action>", "</action>", "<query>", "</query>",
    "</response>", "<recall_result>", "</recall_result>",
]
SPECIAL_TOKENS_PER_TIMESTEP = [
    # Input structure tags
    "<memory>", "</memory>",               # wraps entire memory block
    "<compressed>", "</compressed>",       # memory block: compressed segment
    "<pending>", "</pending>",             # memory block: pending question
    "<visual_window>", "</visual_window>", # visual window header (JSON inside)
    "<recalled_frames>", "</recalled_frames>",  # recalled frames header
    "<user_input>", "</user_input>",       # wraps user input text
    # Output payload tags
    "<summary>", "</summary>",             # compress action payload
    # User input trigger
    "<compress_trigger>", "</compress_trigger>",  # system compress trigger
]

# ---------------------------------------------------------------------------
# 9. Teacher prompts (397B, hidden from student)
# ---------------------------------------------------------------------------

EVIDENCE_GRAPH_PROMPT = """You are annotating a 2-second video clip (t={start}-{end}s).

Based on the frames above, output a STRICT JSON object:
{{
  "time": [{start}, {end}],
  "visible_entities": [
    {{"desc": "appearance description (clothing/color/material/size)", "action": "what doing", "position": "where in frame"}}
  ],
  "atomic_facts": [
    {{
      "fact": "precise observable statement",
      "confidence": 0.0-1.0,
      "target_resolution_visible": true
    }}
  ],
  "ocr": ["exact text if visible"],
  "spatial": "brief spatial layout description"
}}

Rules:
- Only describe what is VISIBLE in these frames (no comparison to other clips)
- Describe entities by appearance, not by ID or assumed identity
- confidence < 0.7 for uncertain observations (small text, fast motion, partial occlusion)
- target_resolution_visible: false if detail would be too small/blurry at training resolution
- Do NOT include sounds, smells, emotions, or inferred intentions

Output JSON only:"""

OBSERVATION_PROMPT = """You are a streaming video agent generating a think (incremental visual memory note).

Compressed memory:
{compressed_memory}

Recent thinks:
{recent_thinks}

Visual window: t={window_start}-{window_end}s (frames above).

Describe what is NEW or CHANGED in the latest 2 seconds (t={start}-{end}s) in 40-60 tokens.

Rules:
- Only observable visual facts
- Describe entities by appearance (clothing, color, material), not by ID
- Focus: entities+attributes, actions, state changes, OCR, spatial
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- If nothing new: brief ongoing state

Output (one paragraph, 40-60 tokens):"""

COMPRESS_PROMPT = """Compress these observations into a structured summary.

Observations to compress:
{observations_text}
{visual_context}
Rules:
- Use coarse time sub-ranges: [X-Y]
- Keep ALL entities with their appearance descriptions
- Keep ALL OCR content verbatim
- Keep state changes as before→after
- Keep user interaction summaries if any
- Target length: {target_length} tokens
- Base the summary primarily on the observation text
- If video frames are provided, use them to verify and refine details (correct entity counts, colors, spatial positions), but do not introduce entirely new events not mentioned in the observations

Output JSON: {{"time_range": [{start}, {end}], "text": "..."}}"""

TASK_QUESTION_PROMPT = """Based on this visual evidence:
Entity: {entity}
Attributes: {attributes}
Fact: {fact}
Time: t={time}s

Generate ONE specific, answerable question about this visual detail.
The full fact is: {answer}

Requirements:
- Natural conversational question
- Answerable from visual observation alone
- Do not include the answer in the question
- "concise_answer" must be a SHORT answer (1-10 words), not the full fact sentence

Output JSON: {{"question": "...", "concise_answer": "...", "answer_type": "factoid|procedural|summary"}}"""

RECALL_QUERY_PROMPT = """Generate a retrieval query for this scenario:
- Question: "{question}"
- Visible memory context: {visible_context}

Based ONLY on the question and the visible memory context, generate 3-5 discriminative
keywords that would help locate the relevant past observation.
NO answer values, NO pronouns, NO articles.
Include entity descriptions + action/attribute anchors from the question and context.

Output JSON (one line): {{"query": "keyword1 keyword2 keyword3", "time_range": "{time_range}"}}"""

POST_RECALL_THINK_PROMPT = """You are a streaming video agent that just received recall results.

Question: "{question}"
Recall result: {recall_result}
Recall source: {recall_source}

Write a brief analysis (20-40 tokens) of the recall result in relation to the question.
- If results are relevant: note what was found and how it relates to the question.
- If results are irrelevant/empty: note the recall failed to find matching evidence.
- NO meta-reasoning ("I think", "I notice"), NO sounds/smells/emotions.
- Focus on factual assessment of the retrieved content.

Output the analysis text only (20-40 tokens):"""

RESPONSE_PROMPT = """Generate a response for this streaming video agent:
- Question: "{question}"
- Available evidence: {evidence}
- Answer type: {answer_type}
- Correct answer: {gold_answer}

Requirements:
- Response length: {length_guide}
- Base answer ONLY on the provided evidence
- If evidence is insufficient, say "I cannot confirm..."
- Do NOT add information beyond what's in the evidence

Output the response text only:"""

# ---------------------------------------------------------------------------
# 10. Helpers
# ---------------------------------------------------------------------------


def ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
