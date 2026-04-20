"""
Configuration for Agent Data Pipeline v5.0.

All constants, prompts, and schema definitions.
Matches docs/data_construction_zh.md v5.0 exactly.
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

ALL_DIRS = [DATA_ROOT, EVIDENCE_DIR, ROLLOUT_DIR, TASKS_DIR, SAMPLES_DIR, FINAL_DIR]

# ---------------------------------------------------------------------------
# 2. Video & chunk parameters
# ---------------------------------------------------------------------------

AGENT_CHUNK_SEC = 2          # 每个 chunk 2 秒
FPS = 1                      # 1fps
FRAMES_PER_CHUNK = 2         # 每 chunk 2 帧
VISUAL_WINDOW_CHUNKS = 12    # 视觉窗口 = 最近 12 chunks (24s)
VISUAL_WINDOW_FRAMES = VISUAL_WINDOW_CHUNKS * FRAMES_PER_CHUNK  # 24 帧

# ---------------------------------------------------------------------------
# 3. Observation & memory parameters
# ---------------------------------------------------------------------------

OBSERVATION_TOKENS = (40, 60)       # student observation 长度
COMPRESS_THRESHOLD = 10             # recent_observations 达到 10 条触发压缩
COMPRESS_RANGE = 10                 # 每次压缩最早的 10 条
SUMMARY_TOKENS_MIN = 100            # summary 最短 (match doc §8)
SUMMARY_TOKENS_MAX = 180            # summary 最长 (match doc §8)
COMPRESSION_RATIO_MIN = 2.5        # 最小压缩比 (doc: >=2:1, target 3.5:1)
RECALL_RETURN_FRAMES = 4           # recall 返回 4 帧 (4s at 1fps)
MAX_COMPRESSED_SEGMENTS = 5        # 最多保留 5 段压缩 (doc §2.4: <=5×150=750 tok)

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
# 5. Action distribution targets (episode-level)
# ---------------------------------------------------------------------------

# 按 phase 控制不同类型 episode 的比例
PHASE_CONFIG = {
    1: {
        "name": "protocol_alignment",
        "actions": ["silent", "response"],
        "episode_mix": {"silent_only": 0.3, "response": 0.65, "uncertain": 0.05},
        "lr": 1e-5, "epochs": 3,
    },
    2: {
        "name": "recall",
        "actions": ["silent", "response", "recall"],
        "episode_mix": {"silent_only": 0.1, "response": 0.3, "recall": 0.4,
                        "uncertain": 0.1, "negative": 0.1},
        "lr": 5e-6, "epochs": 3,
    },
    "C1": {
        "name": "fixed_compress",
        "actions": ["silent", "response", "recall", "compress"],
        "episode_mix": {"response": 0.2, "recall": 0.25, "compress": 0.25,
                        "compress_recall": 0.2, "uncertain": 0.1},
        "lr": 3e-6, "epochs": 2,
    },
    "C2": {
        "name": "adaptive_compress",
        "actions": ["silent", "response", "recall", "compress"],
        "episode_mix": {"response": 0.15, "recall": 0.2, "compress_adaptive": 0.3,
                        "compress_recall": 0.25, "uncertain": 0.1},
        "lr": 2e-6, "epochs": 2,
    },
    5: {
        "name": "mixed",
        "actions": ["silent", "response", "recall", "compress"],
        "episode_mix": {"response": 0.25, "recall": 0.25, "compress": 0.15,
                        "compress_recall": 0.15, "uncertain": 0.1, "negative": 0.1},
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
    "pass1_evidence": {
        "max_tokens": 1024,
        "temperature": 0.3,
        "thinking": True,
        "concurrent_videos": 16,
    },
    "pass2_rollout": {
        "max_tokens_observation": 128,
        "max_tokens_compress": 512,
        "temperature": 0.3,
        "thinking": False,
        "concurrent_videos": 16,
    },
    "pass3_tasks": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "thinking": True,
        "concurrent": 32,
    },
    "pass4_forks": {
        "max_tokens": 512,
        "temperature": 0.3,
        "thinking": False,
        "concurrent": 32,
    },
}

# ---------------------------------------------------------------------------
# 8. System prompt (4-action protocol)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a streaming video agent. You observe video chunks and maintain memory.\n\n"
    "Each turn, output exactly ONE of:\n"
    "1) <observation>...</observation><action>silent</action>\n"
    "2) <observation>...</observation><action>response</action><response>...</response>\n"
    "3) <observation>...</observation><action>recall</action>"
    '<query>{"query":"...","time_range":"..."}</query>\n'
    "4) <observation>...</observation><action>compress</action>"
    '<summary>{"time_range":[s,e],"text":"..."}</summary>\n\n'
    "Rules:\n"
    "- observation: 40-60 tokens, describe ONLY what is newly visible. "
    "No reasoning, no sound/smell/emotion.\n"
    "- response: answer based on currently visible information only.\n"
    "- recall: only when answer is NOT in visual window, NOT in text memory, "
    "NOT in compressed summaries.\n"
    "- compress: when system triggers, compress oldest observations.\n"
    "- If a pending question exists, respond when the answer becomes visible."
)

# ---------------------------------------------------------------------------
# 9. Teacher prompts (397B, hidden from student)
# ---------------------------------------------------------------------------

EVIDENCE_GRAPH_PROMPT = """You are annotating a streaming video chunk by chunk for training data construction.

Previous captions for context:
{previous_captions}

Current chunk: t={start}-{end}s (frames above).

Output a STRICT JSON object with these fields:
{{
  "time": [{start}, {end}],
  "visible_entities": [
    {{"id": "entity_name", "attributes": ["attr1", "attr2"], "action": "what doing", "position": "where"}}
  ],
  "atomic_facts": [
    {{"fact": "precise observable statement", "confidence": 0.0-1.0}}
  ],
  "state_changes": ["what changed from previous chunk"],
  "ocr": ["exact text if visible"],
  "spatial": "brief spatial layout description",
  "not_observable": ["any sounds/smells/emotions mentioned that are NOT visible"]
}}

Rules:
- Only include what is VISIBLE in the frames
- confidence < 0.7 for uncertain observations (small text, fast motion, partial occlusion)
- not_observable: list anything you'd normally say but can't actually SEE
- Maintain consistent entity IDs across chunks (chef_1, pot_1, etc.)

Output JSON only:"""

OBSERVATION_PROMPT = """/no_think

You are a streaming video agent generating a memory observation.

Compressed memory:
{compressed_memory}

Recent observations:
{recent_observations}

Visual window: t={window_start}-{window_end}s (frames above).

Describe what is NEW or CHANGED in the latest 2 seconds (t={start}-{end}s) in 40-60 tokens.

Rules:
- Only observable visual facts
- Maintain entity names from memory (e.g., keep "chef_1" consistent)
- Focus: entities+attributes, actions, state changes, OCR, spatial
- NO reasoning, NO "I notice", NO sounds/smells/emotions
- If nothing new: brief ongoing state

Output (one paragraph, 40-60 tokens):"""

COMPRESS_PROMPT = """/no_think

Compress these observations into a structured summary.

Observations to compress:
{observations_text}

Rules:
- Use coarse time sub-ranges: [X-Y]
- Keep ALL named entities with visual attributes
- Keep ALL OCR content verbatim
- Keep state changes as before→after
- Keep user interaction summaries if any
- Target length: {target_length} tokens
- DO NOT add information not in the observations

Output JSON: {{"time_range": [{start}, {end}], "text": "..."}}"""

TASK_QUESTION_PROMPT = """Based on this visual evidence:
Entity: {entity}
Attributes: {attributes}
Fact: {fact}
Time: t={time}s

Generate ONE specific, answerable question about this visual detail.
The answer must be: {answer}

Requirements:
- Natural conversational question
- Answerable from visual observation alone
- Do not include the answer in the question

Output JSON: {{"question": "...", "answer_type": "factoid|procedural|summary"}}"""

RECALL_QUERY_PROMPT = """/no_think

Generate a retrieval query for this scenario:
- Question: "{question}"
- The answer is about: {answer_topic} (DO NOT include the exact answer value)
- Evidence was observed around t={evidence_time}s

Query must be 3-5 discriminative keywords.
NO answer values, NO pronouns, NO articles.
Include entity names + action/attribute anchors.

Output JSON (one line): {{"query": "keyword1 keyword2 keyword3", "time_range": "{time_range}"}}"""

RESPONSE_PROMPT = """/no_think

Generate a response for this streaming video agent:
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
