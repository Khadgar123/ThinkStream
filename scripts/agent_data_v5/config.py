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
EVIDENCE_1A_DIR = DATA_ROOT / "evidence_1a"     # 1-A raw per-chunk
EVIDENCE_1B_DIR = DATA_ROOT / "evidence_1b"     # 1-B enriched (entity ID hint + state_changes)
ROLLOUT_DIR = DATA_ROOT / "rollout"
TASK_CARDS_DIR = DATA_ROOT / "task_cards"        # 3-A task cards
PLACEMENTS_DIR = DATA_ROOT / "placements"        # 3-B placements + trajectories
SAMPLES_3C_DIR = DATA_ROOT / "samples_3c"        # 3-C trajectory samples
VERIFIED_DIR = DATA_ROOT / "verified"            # 4 verified samples
FINAL_DIR = DATA_ROOT / "final"
AUDIT_DIR = DATA_ROOT / "audits"

ALL_DIRS = [
    DATA_ROOT, EVIDENCE_1A_DIR, EVIDENCE_1B_DIR, ROLLOUT_DIR,
    TASK_CARDS_DIR, PLACEMENTS_DIR, SAMPLES_3C_DIR,
    VERIFIED_DIR, FINAL_DIR, AUDIT_DIR,
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

# Think length: relaxed to match teacher's natural distribution (p50≈83 tok).
# Used for budget estimation (avg) and as a soft target. Pass 4 verification
# applies its own additional margin (see verify_think_token_length).
THINK_TOKENS = (40, 100)            # student think 长度范围 (relaxed from (40,60))
THINK_TOKEN_AVG = 70                # think 平均 token 数（用于估算）

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

COMPRESS_RANGE_MIN = 4              # 每次最少压缩 4 条（保证 summary 值得做）
COMPRESS_RANGE_MAX = 12             # 每次最多压缩 12 条（v11.3: 8 → 12，配合 token-driven 选择）
# v11.3: token-budget-driven range selection. When trigger fires, pick the
# smallest N >= MIN such that the cumulative token count of the oldest N
# thinks reaches this target. Aligns inference (agent_loop / streaming_vllm)
# with pass2's range scoring, which was already implicitly token-driven via
# COMPRESS_HYSTERESIS_THRESHOLD. 350 ≈ 80% gap (480→330) so post-compress
# memory is comfortably below the trigger again.
COMPRESS_REMOVE_TOKENS = 350
SUMMARY_TOKENS_MIN = 100            # summary 最短
# v11.3: 180 → 280. The 180 cap was being hit by 33% of pass2 summaries —
# they're correctly merging 8-12 chunks but the cap forced truncation. 280
# matches the p80 of teacher-generated summary lengths (mean 169, p99 ~330)
# while still keeping memory cost bounded (5 segments × 280 = 1400 tok max).
SUMMARY_TOKENS_MAX = 280
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

# Per-video FINAL sample cap (applied after render).
# v12.0 DENSITY OVERHAUL — research-backed tightening to match streaming
# benchmark targets (OVO 0.6 q/min, StreamingBench 1.0 q/min, MMDuet2 RL
# converges at 3.3 q/video). v11.5 was producing ~12 q/min, ~30 samples/video
# on 2.6-min videos — 12× over the streaming benchmark median, creating an
# "always-respond" prior that hurts silent-decision learning AND inflates
# train→eval distribution shift. New caps target ~1.2 q/min.
MAX_SAMPLES_PER_VIDEO = 15           # was 30 — halve per-video corpus contribution
MAX_TRAJECTORIES_PER_VIDEO = 5       # was 10 — match VideoLLM-online (3 conv/video)
MAX_QUESTIONS_PER_TRAJECTORY = 3     # was 6 — match VideoLLM-online (3 q/conv)
MAX_ACTIVE_QUERIES = 2               # unchanged — realistic user behavior

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
    "pass1a": {"input": 1_500, "output": 5_000, "thinking": 0},  # 2 frames per chunk
    "pass1b": {"input": 3_000, "output": 5_000, "thinking": 0},  # text-only, full video summary
    "pass2_rollout":  {"input": 10_000, "output": 5_000, "thinking": 0},
    # Pass3/4: now include video frames (~2-4 frames per request).
    # max_tokens=16384 bounds thinking time.
    "pass3a": {"input": 700, "output": 400, "thinking": 16_000},  # pure text, max_tokens=16384
    "pass3a_verify": {"input": 800, "output": 200, "thinking": 16_000},  # card verify, thinking enabled
    "pass3b_visibility": {"input": 600, "output": 100, "thinking": 16_000},  # visibility check, thinking enabled
    "pass3c": {"input": 2_000, "output": 5_000, "thinking": 5_000},  # response/query gen, thinking=True
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
# 5. Quality thresholds
# ---------------------------------------------------------------------------
# (v11: PHASE_CONFIG removed — was unused by data construction and
# encoded the deprecated 5-stage SFT curriculum. Production training is
# now 1 SFT (mixed) + 1 GDPO RL; per-category labels live in
# pipeline.assign_phase() for diagnostic file splits only.)

CONFIDENCE_THRESHOLD = 0.7          # teacher fact confidence >= this to make task
ENTITY_COVERAGE_THRESHOLD = 0.7     # grounding: obs entities vs caption entities
LEAKAGE_OVERLAP_THRESHOLD = 0.3     # keyword overlap triggering leakage flag
PROACTIVE_RECALL_RATE = 0.05        # ~5% of chunks trigger proactive recall

# ---------------------------------------------------------------------------
# 6. 397B vLLM configuration
# ---------------------------------------------------------------------------

VLLM_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
VLLM_MAX_MODEL_LEN = 65536

PASS_CONFIG = {
    # All passes: thinking enabled, --reasoning-parser qwen3 separates
    # thinking into reasoning_content, content is clean output.
    # max_tokens covers thinking + response total. Set generously
    # to avoid truncation — data quality > token efficiency.
    #
    # Concurrency rationale (v11, 2026-04-27): each outer pass owns a
    # dedicated VLLMClient with its own semaphore (see pipeline.py).
    # Values below are tuned to avoid the orphan-cascade we hit at 1024
    # on pass3c — same reasoning applies pass-wide. Client timeout is
    # 5400s (90min) per VLLMClient default; do NOT shorten.
    "pass1a": {
        # v9.5: thinking=False (raw POST path) — frees 16K reasoning
        # budget; retry-on-silent + strict-parse keep quality. With no
        # thinking, per-request load drops and we can run wider.
        # max_tokens kept at 16K so a verbose chunk doesn't truncate
        # the JSON before it closes — short chunks early-stop anyway.
        # concurrent=1024: by safe_concurrency_for_pass calc, 32M //
        # (1500+16384) ≈ 1789 fits in vLLM prefill budget. httpx pool
        # uplifted in VLLMClient (limits=2048).
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": False,
        "concurrent": 1024,
    },
    "pass1b": {
        "max_tokens": 60000,  # text-only, needs headroom for thinking explosion
        "temperature": 0.3,
        "thinking": True,
        "concurrent": 64,     # video-level, longest single request — keep low
    },
    "pass2_rollout": {
        "max_tokens_observation": 16384,
        "max_tokens_compress": 16384,
        "temperature": 0.3,
        "thinking": True,
        "concurrent_videos": 256,
    },
    "pass3a": {
        "max_tokens": 16384,
        "temperature": 0.7,
        "thinking": True,
        "concurrent": 256,    # pure text; client_3a also serves verify
    },
    "pass3c": {
        # Keep thinking + 16K — 3C outputs ARE the SFT labels (response,
        # fork_think, recall_query, recall_think). Quality determines what
        # the student learns. Throughput problems are addressed by lowering
        # concurrency (less GPU oversubscription) + extending client timeout,
        # not by crippling teacher reasoning.
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": True,
        "concurrent": 256,    # at 1024 each req waits ~55min (orphan-cascade);
                              # at 256 each gets ~14min, no timeouts.
    },
    # pass3a_verify and pass3b_visibility share their outer pass's client
    # (client_3a and client_3b respectively). The "concurrent" entries
    # below are no longer the binding cap — they exist for documentation
    # only. The actual cap is on the outer client.
    # v11.3: per-call thinking control. The 5 lightweight calls below were
    # downgraded from thinking=True to thinking=False because their tasks
    # (verification / classification / templating / keyword extraction)
    # don't benefit from CoT — empirically the teacher's thinking budget
    # went unused. max_tokens KEPT at 16K so a verbose response never
    # truncates: GPU has the headroom for 16K @ 1024 concurrent and the
    # speedup comes from disabling reasoning, not from cap reduction.
    # Card generation (pass3a) and fork_think (pass3c_fork_think) keep
    # thinking — the former needs multi-family multi-constraint reasoning,
    # the latter needs answer-leakage avoidance.
    "pass3a_verify": {
        "max_tokens": 16384,
        "temperature": 0.1,
        "thinking": False,
        "concurrent": 256,    # bound by client_3a
    },
    "pass3b_visibility": {
        "max_tokens": 16384,
        "temperature": 0.1,
        "thinking": False,
        "concurrent": 512,    # bound by client_3b
    },
    # v11.3: pass3c split into per-call-type sub-configs so thinking can
    # be controlled per call. The umbrella "pass3c" entry above stays as a
    # legacy fallback — new code should read these specific sub-keys.
    "pass3c_response": {
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": False,
    },
    "pass3c_recall_query": {
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": False,
    },
    "pass3c_recall_think": {
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": False,
    },
    "pass3c_fork_think": {
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": True,      # KEEP — answer-leakage avoidance needs CoT
    },
}

# ---------------------------------------------------------------------------
# 7. System prompt (4-action protocol)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3 Prompt Protocol (v8.0)
# Each prompt teaches ONE type of decision. No mixed behavior.
# ---------------------------------------------------------------------------

# NOTE: legacy v11 SYSTEM_PROMPT / SYSTEM_PROMPT_POST_RECALL /
# SYSTEM_PROMPT_COMPRESS were removed when the codebase consolidated on
# the v12 Qwen tool protocol. See thinkstream/data/agent_protocol.py:
# SYSTEM_PROMPT_V12 + TOOLS_SCHEMA for the current single source of truth.

# Special tokens required by SFT init_processor (see sft_engineering.md §6.2)
# Approach B: exact-match tags, attributes as JSON inside tags.
#
# CANONICAL SOURCE: thinkstream/sft/data_processor.py:SPECIAL_TOKENS_AGENT
# (which mirrors thinkstream/data/agent_protocol.py).
# The two lists below are kept here for documentation only — do NOT use them
# to register tokens; the SFT entry point already does that. Any divergence
# from the canonical source is a bug.
SPECIAL_TOKENS_BASE = [
    "<silent>", "<response>", "<think>", "</think>",
    "<action>", "</action>", "<query>", "</query>",
    "</response>", "<recall_result>", "</recall_result>",
]
SPECIAL_TOKENS_PER_TIMESTEP = [
    # Input structure tags
    "<memory>", "</memory>",                    # wraps memory timeline
    "<compressed>", "</compressed>",            # memory timeline: compressed segment (inline)
    "<pending>", "</pending>",                  # memory timeline: pending question
    "<visual_window>", "</visual_window>",      # visual window header
    "<recalled_frames>", "</recalled_frames>",  # recalled frames header
    "<user_input>", "</user_input>",            # wraps user input text
    "<queries>", "</queries>",                  # past Q&A zone
    # Output payload (assistant)
    "<summary>", "</summary>",                  # compress-action summary payload
    # User input trigger
    "<compress_trigger>", "</compress_trigger>",  # system compress trigger
]

# ---------------------------------------------------------------------------
# 8. Teacher prompts (397B, hidden from student)
# ---------------------------------------------------------------------------

EVIDENCE_GRAPH_PROMPT = """You are annotating a 2-second video clip (t={start}-{end}s).

Based on the frames above, output a STRICT JSON object:
{{
  "time": [{start}, {end}],
  "visible_entities": [
    {{"desc": "fine-grained appearance", "action": "verb phrase or 'static'", "position": "left/center/right/top/bottom/foreground/background"}}
  ],
  "atomic_facts": ["precise observable statement", ...],
  "ocr": ["exact text if visible"],
  "spatial": "spatial relations between entities, sentence-form"
}}

CRITICAL — minimum output requirement:
- The frames above almost always contain SOMETHING describable: a person,
  an object being manipulated, a setting, on-screen text, a tool, food, etc.
- visible_entities MUST have ≥1 element AND atomic_facts MUST have ≥1 element,
  even if the scene is dim/blurry/transition. The only exception is a fully
  black or fully white frame.
- Empty arrays mean "I gave up" — not allowed.

visible_entities[].desc — FINE-GRAINED (downstream questions ask 'What style of
  tattoo / what pattern on shorts / what material for the wing'). Include ALL
  observable attributes:
  - For people: clothing color + clothing pattern + hair + skin + accessories
    (e.g., "person with long blonde hair, white sleeveless top with floral
    pattern, blue jeans")
  - For objects: color + material/texture + pattern/style + size + condition
    (e.g., "wooden stick, smooth, light-brown, ~30cm long")
  - For text/graphics: font/style + color + content
  - For animals: species + color/markings + size

visible_entities[].action — MUST be non-empty:
  - moving entities: verb phrase ("picking up wrench", "walking left")
  - static entities: literally "static"

atomic_facts — list of strings (NO confidence/target_resolution fields):
  - ≥1 fact must be ACTION-TYPE ("person opens the box", "the dog runs to
    the door"), not only state descriptions
  - state-type facts also welcome ("box is on the table")
  - include OCR-derived facts when text presents key info ("the price tag
    reads $14.99")

ocr — array of EXACT text strings as they appear, preserving case/punctuation.

spatial — write 1-3 SENTENCES describing inter-entity relations using these
  prepositions: left of / right of / above / below / in front of / behind /
  on / under / inside / holding / near / next to. Example:
  "Person is to the right of the bird cage. The cage is on a wooden table.
  The 'CALL ON ME' note is above the 'I'm here for you' note."
  This sentence-form replaces structured spatial_relations to keep schema simple
  and LLM output stable.

Rules:
- Only describe what is VISIBLE in these frames (no comparison to other clips)
- Describe entities by appearance, not by ID or assumed identity
- Use a CONSISTENT phrase for the same entity across clips when its appearance
  matches (e.g., "the man in the black polo shirt") so downstream entity
  linking can match by string. Do not paraphrase the same entity differently.
- Do NOT include sounds, smells, emotions, or inferred intentions

Output JSON only:"""

OBSERVATION_PROMPT = """You are a streaming video agent generating a think (incremental visual memory note).

Compressed memory:
{compressed_memory}

Recent thinks:
{recent_thinks}

Visual window: t={window_start}-{window_end}s (frames above).

Describe what is NEW or CHANGED in the latest 2 seconds (t={start}-{end}s).
Be concise but complete (target 50-90 tokens, never exceed 120).

Rules:
- Only observable visual facts
- Describe entities by appearance (clothing, color, material). If a similar
  entity already appears in recent thinks, REUSE the same descriptive phrase
  (e.g., "the man in the black polo shirt") so downstream linking can match it
- Focus: entities+attributes, actions, state changes, OCR, spatial
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- If nothing new: one short sentence on ongoing state

Output one paragraph:"""

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
