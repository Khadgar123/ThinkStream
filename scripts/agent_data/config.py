"""
Configuration for Agent Data Pipeline v12.15.

All constants, prompts, and schema definitions.
This module is the source of truth for runtime data-construction constants.
"""

import os
from pathlib import Path
from typing import Dict

from thinkstream.data.schema import (
    DEFAULT_VIDEO_MAX_PIXELS,
    DEFAULT_VIDEO_MIN_PIXELS,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

# ---------------------------------------------------------------------------
# 1. Directory layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ThinkStream/


def _resolve_data_root() -> Path:
    """Resolve the canonical root for one data-construction batch.

    New runs should set exactly one batch root and let every pass write
    underneath it:

        THINKSTREAM_DATA_ROOT=data/agent_v5/batch2

    Backward-compatible shorthands:
      - AGENT_DATA_DIR points at the same batch root.
      - THINKSTREAM_BATCH=batch2 expands to data/agent_v5/batch2.
      - no env keeps the historical data/agent_v5 root.
    """
    explicit = os.environ.get("THINKSTREAM_DATA_ROOT") or os.environ.get("AGENT_DATA_DIR")
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        # Some older scripts used AGENT_DATA_DIR=data/agent_v5/final. Treat
        # that as a final-dir pointer and recover the batch root.
        return p.parent if p.name == "final" else p

    batch = os.environ.get("THINKSTREAM_BATCH", "").strip()
    if batch:
        return PROJECT_ROOT / "data" / "agent_v5" / batch
    return PROJECT_ROOT / "data" / "agent_v5"


DATA_ROOT = _resolve_data_root()
BATCH_ID = DATA_ROOT.name

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

# v12.5 (2026-04-29): 2s/chunk → 1s/chunk (effective FPS 1 → 2 with 2 frames/chunk).
# Rationale: the user noted text-memory < visual-memory inconsistency under
# the old config (compress at 8 thinks ≈ 16s vs visual window 12 chunks × 2s
# = 24s). Halving chunk-sec gives finer temporal labels, doubles per-chunk
# rate of decisions, and lets us bring text-memory horizon back above visual.
# Chunk-based windows downstream (pass3b/pass3c) doubled accordingly so that
# semantic spans (in seconds) are preserved or slightly extended.
AGENT_CHUNK_SEC = 1          # 每个 chunk 1 秒
FPS = 2                      # 2fps (FRAMES_PER_CHUNK / AGENT_CHUNK_SEC)
FRAMES_PER_CHUNK = 2         # 每 chunk 2 帧
# v12.15: 8 chunks × 1s = 8s of visual context (16 frames).
# Other streaming systems for reference: LiveCC ~240s @
# 2fps, VideoLLM-online ~unbounded @ 2fps, MMDuet token-budgeted, Streamo
# 1fps. Current pre-extracted-frame prompts render explicit frame-tag text
# before each image, so timestamps still reflect real 2fps frame indices.
# We're conservative for the 6-min batch1 footprint, but text
# memory now comfortably exceeds visual (see RECENT_THINKS_TOKEN_BUDGET).
VISUAL_WINDOW_CHUNKS = 8     # 视觉窗口 = 最近 8 chunks (8s @ 2fps = 16 帧)
VISUAL_WINDOW_FRAMES = VISUAL_WINDOW_CHUNKS * FRAMES_PER_CHUNK  # 16 帧

# v12.13 (2026-05-02): visual window mode. Read from env so SFT generators
# (pass2/pass5/render_samples) and the RL agent loop pick the same scheme
# without separate plumbing. SFT and RL MUST agree on this value or the
# rollout-time visual context diverges from the training distribution.
#   "sliding"   — window slides 1 chunk per step (legacy default).
#                 Frame token IDs shift left by FRAMES_PER_CHUNK every
#                 chunk → vLLM prefix cache misses on the visual block.
#   "expanding" — window anchored at segment boundary, grows to current
#                 chunk, resets every VISUAL_WINDOW_CHUNKS. ~94% prefix-
#                 cache hit on visual KV. Boundary chunks have less recent
#                 context — re-verify SFT data quality after switching.
VISUAL_WINDOW_MODE = os.environ.get(
    "THINKSTREAM_VISUAL_WINDOW_MODE", "sliding"
).lower()
if VISUAL_WINDOW_MODE not in ("sliding", "expanding"):
    VISUAL_WINDOW_MODE = "sliding"


def compute_visual_window_start(
    chunk_idx: int,
    visual_window_chunks: int = VISUAL_WINDOW_CHUNKS,
    mode: str = None,
) -> int:
    """Single source of truth for window-start computation across pass2,
    pass5/SFT rendering, RL rollout, eval, and deploy inference.

    Returns the inclusive starting chunk_idx of the visual window
    covering chunk ``chunk_idx``.
    """
    eff_mode = (mode or VISUAL_WINDOW_MODE or "sliding").lower()
    if eff_mode == "expanding":
        seg = max(1, int(visual_window_chunks))
        return (int(chunk_idx) // seg) * seg
    return max(0, int(chunk_idx) - int(visual_window_chunks) + 1)

# ---------------------------------------------------------------------------
# 3. Think & memory parameters
# ---------------------------------------------------------------------------

# v12.5 (2026-04-29): 1s/chunk semantics → each think describes only 1s of
# motion (was 2s), so target tighter range. OBSERVATION_PROMPT updated to
# "target 40-80, never exceed 100" (was "50-90, never exceed 120").
# Pass 4 verification applies its own additional margin.
THINK_TOKENS = (40, 80)             # matches new prompt "target 40-80"
THINK_TOKEN_AVG = 60                # was 70; new prompt midpoint
# Pass4 verifier widens THINK_TOKENS by ±15/+30 → effective accept 25-110.

# Token-based compression trigger with hysteresis.
#
# v12.14 (2026-05-12) — compression is keyed to the full visible text
# memory, not raw observations only. This includes raw <m> records
# plus older compact <m> summaries, matching the prompt area the student
# actually sees.
#
# 64K teacher/runtime allocation under 1s/chunk + tool protocol:
#   system + tools schema  ≈   400
#   visual_window (16 fr)  ≈  3760   (8 chunks × 470 tok)
#   recall vision (4 fr)   ≈   256
#   compressed segments    ≈  1400   (5 × 280, counted in text memory)
#   past queries           ≈   300
#   recall result text     ≈   500
#   output budget          ≈  1000   (think + tool call)
#   ─────────────────────────────────
#   subtotal               ≈  5904
#   text_memory budget     ≈  5400   (raw observations + summaries)
#   ─────────────────────────────────
#   total inference window ≈ 11304   (well under 64K, leaves headroom)
TEXT_MEMORY_TOKEN_BUDGET = 5400
# Backward-compatible name used by older code paths. It now means the full
# visible text-memory budget, not raw recent_thinks only.
RECENT_THINKS_TOKEN_BUDGET = TEXT_MEMORY_TOKEN_BUDGET
COMPRESS_TRIGGER_RATIO = 0.8        # 达到预算 80% 时系统触发压缩
COMPRESS_TOKEN_THRESHOLD = int(TEXT_MEMORY_TOKEN_BUDGET * COMPRESS_TRIGGER_RATIO)  # = 4320
COMPRESS_HYSTERESIS_RATIO = 0.55    # 压缩后应降回 55% 以下，否则窗口太短
COMPRESS_HYSTERESIS_THRESHOLD = int(TEXT_MEMORY_TOKEN_BUDGET * COMPRESS_HYSTERESIS_RATIO)  # = 2970

# Student model tokenizer (用于精确计算 token 数)
# 造数据时加载一次，全局复用
STUDENT_MODEL = "/home/tione/notebook/gaozhenkun/model/Qwen3-VL-8B-Instruct"  # 本地 tokenizer 路径
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

# v12.14: wider range so a compress can replace enough visible text to fall
# below hysteresis without becoming too frequent. Range selection may include
# older summaries plus adjacent raw observations, with event-boundary/content
# penalties applied in pass2_rollout.
COMPRESS_RANGE_MIN = 12             # 每次最少压缩 12 条或等量可见记忆项
COMPRESS_RANGE_MAX = 32             # 每次最多压缩 32 条或等量可见记忆项
COMPRESS_REMOVE_TOKENS = 2000
SUMMARY_TOKENS_MIN = 100            # summary 最短
# v11.3: 180 → 280. The 180 cap was being hit by 33% of pass2 summaries —
# they're correctly merging 8-12 chunks but the cap forced truncation. 280
# matches the p80 of teacher-generated summary lengths (mean 169, p99 ~330)
# while still keeping memory cost bounded (5 segments × 280 = 1400 tok max).
SUMMARY_TOKENS_MAX = 280
COMPRESSION_RATIO_MIN = 2.5        # 最小压缩比
RECALL_RETURN_CHUNKS = 4           # recall returns top-4 memory chunks
RECALL_RETURN_FRAMES = RECALL_RETURN_CHUNKS * FRAMES_PER_CHUNK
MAX_COMPRESSED_SEGMENTS = 5        # 最多保留 5 段压缩

# v12.30 (2026-05-12): compact-memory update mode.
#
# Instead of selecting one local range and writing a JSON summary tool call,
# pass2 now asks the teacher to rewrite the whole visible text memory into a
# compact <m t="..."> memory lines. The next trajectory starts from that
# pure-text memory.
# Triggering is deliberately based on text memory only (old compact memory +
# raw think/caption entries) so it is cheap and stable; visual/retrieval/output
# budgets remain reserved outside this counter.
COMPACT_MEMORY_UPDATE_MODE = _env_bool("THINKSTREAM_COMPACT_MEMORY_UPDATE_MODE", True)
COMPACT_MEMORY_TEXT_TOKEN_BUDGET = _env_int("THINKSTREAM_COMPACT_MEMORY_TEXT_TOKEN_BUDGET", 3200)
COMPACT_MEMORY_BALANCE_SEGMENTS = _env_bool("THINKSTREAM_COMPACT_MEMORY_BALANCE_SEGMENTS", True)
COMPACT_MEMORY_MIN_NEW_CHUNKS = _env_int("THINKSTREAM_COMPACT_MEMORY_MIN_NEW_CHUNKS", 25)
COMPACT_MEMORY_TARGET_NEW_CHUNKS = _env_int("THINKSTREAM_COMPACT_MEMORY_TARGET_NEW_CHUNKS", 30)
COMPACT_MEMORY_MAX_NEW_CHUNKS = _env_int("THINKSTREAM_COMPACT_MEMORY_MAX_NEW_CHUNKS", 36)
STREAM_CONTEXT_TARGET_TOKENS = _env_int("THINKSTREAM_STREAM_CONTEXT_TARGET_TOKENS", 12000)
STREAM_VISUAL_TOKEN_RESERVE = _env_int("THINKSTREAM_STREAM_VISUAL_TOKEN_RESERVE", 6000)
STREAM_RECALL_TOKEN_RESERVE = _env_int("THINKSTREAM_STREAM_RECALL_TOKEN_RESERVE", 2500)
STREAM_OUTPUT_TOKEN_RESERVE = _env_int("THINKSTREAM_STREAM_OUTPUT_TOKEN_RESERVE", 1000)

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
MAX_SAMPLES_PER_VIDEO = 0            # v12.9 (2026-04-30): disable cap. pass3c
                                     # now emits ONE sample per chunk (every
                                     # chunk 0..num_chunks-1), so a 150s video
                                     # produces ~150 samples. Train silent rate
                                     # naturally matches runtime ~91%. The
                                     # round-robin cap was hiding the silent
                                     # sample shortage; with full per-chunk
                                     # coverage the cap becomes unnecessary
                                     # (and would actively harm coverage).
                                     # 0 = no cap (sentinel honored by
                                     # pipeline.py:872 round-robin block).
# v12.6 (2026-04-30): 5 → 1, align with VideoLLM-online / MMDuet / VST
# convention of "1 video = 1 trajectory, multiple questions inside".
# Multi-traj per video creates 5× visual + memory_state overlap → effective
# unique-sample count drops to ~30% of nominal → SFT loss collapses fast
# without true generalization. With 1 long traj per video we get true
# cross-video diversity; question density stays roughly constant by bumping
# MAX_QUESTIONS_PER_TRAJECTORY 5 → 8.
MAX_TRAJECTORIES_PER_VIDEO = 1
# v12.15 (2026-05-03): align config with the v2 placement source of truth.
# The actual count is adaptive, roughly one question per 10 chunks, capped
# here; short videos still use the v2 floor of 8 questions. This keeps the
# observed q-interval near the LiveChat/MMDuet 7-15s band while preserving
# enough family/mechanism diversity per trajectory.
# v12.61 (2026-05-10): allow very long videos to carry more questions,
# while pass3B enforces a hard ask-gap floor so short/medium videos do not
# become denser by accident.
MAX_QUESTIONS_PER_TRAJECTORY = 20
MAX_ACTIVE_QUERIES = 1               # one active question; no cross-question interference

# Backward compat aliases (deprecated — use token-based constants above)
OBSERVATION_TOKENS = THINK_TOKENS  # deprecated alias
COMPRESS_THRESHOLD = 10  # deprecated: item-count fallback, prefer COMPRESS_TOKEN_THRESHOLD
COMPRESS_RANGE = COMPRESS_RANGE_MAX  # deprecated alias

# ---------------------------------------------------------------------------
# 4. Token budgets (草算，需用 tokenizer 实测)
# ---------------------------------------------------------------------------

# v12.5: V12 protocol system prompt + tool schema is closer to 400 tokens
# (was ~150 under v11 plain-text protocol). Treat 400 as the reservation.
SYSTEM_PROMPT_TOKENS = 400
COMPRESSED_SEG_TOKENS = 280        # matches SUMMARY_TOKENS_MAX (was 150 stale)
OBSERVATION_AVG_TOKENS = 60        # matches THINK_TOKEN_AVG

# ---------------------------------------------------------------------------
# v12.12 (2026-05-02): Qwen3-VL smart_resize profiles via mm_processor_kwargs
# ---------------------------------------------------------------------------
# vLLM ≥ 0.7.3 forwards top-level `mm_processor_kwargs` to the Qwen3-VL
# processor's smart_resize. Output token count ≈ resized_pixels / 1024
# (Qwen3-VL has patch_size=16, merge_size=2 → 32×32 = 1024 px/token) plus
# ~18 tok of vision_start/vision_end/grid_thw bookkeeping overhead.
#
# Empirical measurement (1920×1080 source frames):
#   no kwargs           → 2,058 tok/frame   (16 frames = 32,928, blows 16K)
#   min=90k  max=130k   →   138 tok/frame   (16 frames =  2,208)
#   min=90k  max=260k   →   249 tok/frame   (16 frames =  3,984)
#   min=56k  max=56k    →    63 tok/frame   (16 frames =  1,008)
#
# Source video resolution distribution (catalog):
#   640×360 38%, 852×480 18%, 1280×720 13%, 640×480 5%, 480×360 4%,
#   360×640 4% (vertical), 480×270 3%, 480×640 2% (vertical), other ~13%.
# smart_resize is aspect-aware → 16:9, 4:3, 9:16, 3:4 all auto-normalize
# to the configured token-count band; no per-aspect special handling.
#
# RUNTIME profile — used by pass2 / pass5 SFT data / agent_loop / streaming
# eval / verl RL recipe / production deploy. ALL these paths MUST use the
# same mm_processor_kwargs because student is trained at this resolution
# and mismatch = OOD vision-token sequence at inference.
#
# HIRES profile — used ONLY by pass1a evidence extraction. Higher visual
# fidelity for OCR / small-entity / state_change detection. Pass1a is single-
# chunk per request (2 frames) so total visual cost is small even with
# higher per-frame tokens. Output (atomic_facts / visible_entities / ocr)
# carries the fine details forward as TEXT, which pass2/student see in
# memory regardless of their lower runtime resolution.
RUNTIME_MM_PROCESSOR_KWARGS: Dict[str, int] = {
    # Same as schema.DEFAULT_VIDEO_{MIN,MAX}_PIXELS,
    # sft.args.video_*_pixels, agent_protocol.build_user_content defaults,
    # stream_data_processor.DEFAULT_INFERENCE_*_PIXELS. SFT/RL/Eval unified.
    "min_pixels": DEFAULT_VIDEO_MIN_PIXELS,
    "max_pixels": DEFAULT_VIDEO_MAX_PIXELS,
}
HIRES_MM_PROCESSOR_KWARGS: Dict[str, int] = {
    "min_pixels": 200_000,    # ~480p area floor; small sources upscale modestly
    "max_pixels": 1_500_000,  # preserves 1280×720 fully; downsamples 1920×1080 only ~28%
}

# Empirical token counts (verify with real vLLM; adjust if measured value differs)
VISUAL_TOKENS_PER_FRAME_RUNTIME       = 235    # conservative image-item estimate at runtime pixels
VISUAL_TOKENS_PER_FRAME_HIRES_TYPICAL = 500    # at min=200k max=1500k, weighted avg by source distribution
VISUAL_TOKENS_PER_FRAME_HIRES_MAX     = 1500   # at min=200k max=1500k, 1280×720 source

# Backward-compat alias (consumed by older code paths). Set to RUNTIME default.
VISUAL_TOKENS_PER_CHUNK = VISUAL_TOKENS_PER_FRAME_RUNTIME * FRAMES_PER_CHUNK  # 235×2 = 470
VISUAL_WINDOW_TOKENS = VISUAL_WINDOW_CHUNKS * VISUAL_TOKENS_PER_CHUNK  # 8 × 470 = 3,760
RECALL_VISION_TOKENS = VISUAL_TOKENS_PER_FRAME_RUNTIME * 4  # 4 frames recalled at runtime res = 940
# v12.5: 4096 → 16384. Single-sample cap raised to match new 16K context
# budget (system+visual+memory+output = ~10K nominal, 16K accommodates
# bursts in compressed-segment count or recall density).
MAX_SAMPLE_TOKENS = 16384

# ---------------------------------------------------------------------------
# 4b. 397B context / OOM guards
# ---------------------------------------------------------------------------

# Construction-time guards to prevent over-long or too-wide batches.
VLLM_CONTEXT_SAFETY_RATIO = _env_float("THINKSTREAM_VLLM_CONTEXT_SAFETY_RATIO", 0.85)
VLLM_PREFILL_BATCH_TOKEN_BUDGET = _env_int(
    "THINKSTREAM_VLLM_PREFILL_BATCH_TOKEN_BUDGET",
    64_000_000,
)  # Allows 1024-way pass3a after the v12.66 long-video prompt audit.

# Per-request token estimates (text + vision + output + thinking).
# v12.12 (2026-05-02): visual budgets reflect mm_processor_kwargs profiles.
# pass1a uses HIRES (~500 tok/frame typical) × 2 frames + template ≈ 2K visual.
# pass2 uses RUNTIME (~235 tok/frame) × 16 frames + template ≈ 3.8K visual.
# vLLM teacher/runtime server: pass1a/pass2/SFT/RL/eval use frame-tag text +
# image/image_url lists for pre-extracted frames. This keeps the latest chunk
# visually explicit even when text memory is stale, and avoids relying on the
# vLLM video_url/pre-sampled-video path to surface temporal anchors inside
# the model. Start vLLM with e.g.
#   --limit-mm-per-prompt '{"image":64,"video":2}'
# image/video limits are per prompt, not concurrency. pass2 uses up to 32
# images per request (16 chunks × 2 fps); 64 leaves room for recalled frames.
# Keep request-level mm_processor_kwargs at RUNTIME_MM_PROCESSOR_KWARGS
# plus do_sample_frames=False.
PASS_CONTEXT_ESTIMATES = {
    # pass1a: 2 hires frames + template + 5K output. ~3K input typical.
    "pass1a": {"input": 3_000, "output": 5_000, "thinking": 0},
    # v12.5 (2026-04-30): input 3_000 → 16_000. Empirical measurement on 87
    # batch1 videos: avg 13,776 tokens, median 13,432, max 30,927.
    # v12.72: pass1b outputs short e<N> ids for merged entity groups instead
    # of copying full desc text, so visible output should stay well under 16K.
    "pass1b": {"input": 16_000, "output": 8_000, "thinking": 0},  # text-only
    # v12.15 (2026-05-12): RUNTIME profile ~235 tok/frame × 16 = 3,760 visual.
    # + system 500 + template 200 + memory ≤4000 + queries 400 + recall ≤1320
    # + pad 300 ≈ 14,240 input worst-case (with recall). Use 13,500 as a
    # representative estimate (most chunks no recall).
    # Output: weighted avg of (obs max_tokens=1024) and (compress max_tokens=4096)
    # at 97/3 frequency = 1116; rounded to 1500.
    "pass2_rollout":  {"input": 13_500, "output": 1_500, "thinking": 0},
    # v12.66: pass3a sees a compact full-video pass1/pass1b timeline. Batch1-8
    # prompt audit after full-coverage compaction: max prompt ≈98k chars,
    # roughly 28k text tokens; keep a 30k estimate plus 8k output headroom.
    "pass3a": {"input": 30_000, "output": 8_000, "thinking": 0},       # text-only card gen
    # pass3c has mostly tiny deterministic calls, but recall hardening can see
    # ~7k chars current memory + ~11k chars historical evidence + rules.
    "pass3c": {"input": 8_000, "output": 4_000, "thinking": 0},        # response/query/hardening gen
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
    env_key = "THINKSTREAM_" + "".join(
        ch if ch.isalnum() else "_" for ch in pass_name.upper()
    ) + "_CONCURRENT"
    requested = _env_int(env_key, requested)
    global_cap = _env_int("THINKSTREAM_VLLM_MAX_CONCURRENT", 0)
    if global_cap > 0:
        requested = min(requested, global_cap)
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

VLLM_MODEL = os.environ.get(
    "THINKSTREAM_VLLM_MODEL",
    "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8",
)
VLLM_MAX_MODEL_LEN = _env_int("THINKSTREAM_VLLM_MAX_MODEL_LEN", 65536)

PASS_CONFIG = {
    # All teacher/data-construction calls run with enable_thinking=False.
    # The `think` field produced by pass1a is an ordinary supervised
    # observation note in the JSON payload, not Qwen reasoning_content.
    # max_tokens covers the visible response only and stays generous enough
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
        # concurrent=1024: by safe_concurrency_for_pass calc, 64M //
        # (1500+16384) ≈ 3578 fits in vLLM prefill budget. httpx pool
        # uplifted in VLLMClient (limits=2048).
        "max_tokens": 16384,
        "temperature": 0.3,
        "thinking": False,
        "concurrent": 1024,
    },
    "pass1b": {
        # v12.72 (2026-05-10): pass1b emits short e<N> refs, not full descs.
        # 24K is still generous but avoids wasting tens of thousands of tokens
        # on malformed over-long entity dumps; _safe_max_tokens further caps it
        # by real tokenizer-estimated remaining context.
        # concurrent=1024: unified cap with pass1a/2 for max throughput.
        "max_tokens": _env_int("THINKSTREAM_PASS1B_MAX_TOKENS", 24576),
        "min_completion_tokens": _env_int("THINKSTREAM_PASS1B_MIN_COMPLETION_TOKENS", 4096),
        "temperature": 0.3,
        "thinking": False,
        "concurrent": 1024,
    },
    "pass2_rollout": {
        # v12.25: pass2 no longer asks the teacher for per-chunk observation
        # notes. It consumes pass1a's independent current-only `think` field
        # and only calls the teacher for text-only compression summaries.
        # Those summaries are template-driven by COMPRESS_PROMPT; no CoT is
        # required.
        # v12.11 hotfix (2026-05-01): 16384 → tight values right-sized to
        # actual output target. KV cache reservation per request is bounded
        # by max_tokens; reducing 16K → 1024/4096 frees ~75% of reserved KV
        # → vLLM batches more concurrently → throughput up 2-3×.
        # User-confirmed safety margins:
        #   observation: think target 40-80 tok → 1024 = 12× margin.
        #   compress:    summary text ≤ SUMMARY_TOKENS_MAX=280 + tags +
        #     optional JSON wrapper ≈ 320 tok worst case. 4096 = 13× margin
        #     (JSON CANNOT be truncated mid-way; keep generous; user
        #     directive 2026-05-01).
        # VISUAL_WINDOW_CHUNKS is still shared by pass2 memory snapshots and
        # student SFT/RL/eval rendering, even though pass2 no longer uses the
        # teacher to generate current observations from that window.
        "max_tokens_observation": 1024,
        "max_tokens_compress": 4096,
        "temperature": 0.3,
        "thinking": False,
        "concurrent_videos": 1024,
    },
    "pass3a": {
        # v12.5 (2026-04-30): thinking True → False per user audit "在pass3
        # 全流程中 enable_think=false". Card generation works on structured
        # FAMILY_PROMPTS with explicit constraints; CoT was a marginal
        # quality lift, not a correctness floor. 16K max_tokens kept as
        # context budget (no truncation risk on dense evidence).
        "max_tokens": _env_int("THINKSTREAM_PASS3A_MAX_TOKENS", 8192),
        "temperature": 0.7,
        "thinking": False,
        "concurrent": 1024,    # pure text; client_3a also serves verify
    },
    "pass3c": {
        # v12.5 (2026-04-30): thinking True → False per user audit. Generation
        # tasks (response / post_recall_think / recall_hardening) are
        # template-driven; CoT marginally improved quality but added latency
        # without floor-shifting correctness.
        "max_tokens": 8192,
        "temperature": 0.3,
        "thinking": False,
        "concurrent": 1024,
    },
    # pass3c split into per-call-type sub-configs so thinking can be controlled
    # per call. Recall itself is rule-based time-range selection, not an LLM
    # query-generation call.
    "pass3c_response": {
        # Descriptive responses are constrained to 1-2 sentences; MC/binary/
        # number/short_exact are deterministic and never call the teacher.
        "max_tokens": 1024,
        "temperature": 0.3,
        "thinking": False,
    },
    "pass3c_post_recall_think": {
        # One short objective observation about returned visual evidence.
        # It is wrapped by build_assistant_content; the teacher must not emit
        # protocol tags or the final answer.
        "max_tokens": 128,
        "temperature": 0.7,
        "thinking": False,
    },
    "pass3c_recall_hardening": {
        # Three compact replacement candidates with MC options when needed.
        "max_tokens": 4096,
        "temperature": 0.4,
        "thinking": False,
    },
}

# ---------------------------------------------------------------------------
# 7. System prompt (streaming/compress protocol)
# ---------------------------------------------------------------------------

# NOTE: legacy v8/v11 SYSTEM_PROMPT / SYSTEM_PROMPT_POST_RECALL /
# SYSTEM_PROMPT_COMPRESS names were removed when the codebase consolidated on
# the v12 Qwen tool protocol. See thinkstream/data/agent_protocol.py for the
# ordinary streaming prompt, the compression-only prompt, and turn-local tools.

# Canonical special-token registration lives in
# thinkstream.data.agent_protocol.AGENT_SPECIAL_TOKENS and is applied by SFT,
# eval, and RL rollout entry points. Keep this mirror documentation-only.
# Do not reintroduce old paired response/action tag protocols.
SPECIAL_TOKENS_BASE = [
    "<think>", "</think>", "</Response>", "</Silence>",
    "<tool_call>", "</tool_call>",
]
SPECIAL_TOKENS_PER_TIMESTEP = []
# Prompt/input structure delimiters such as <active_query> and
# <response_history> are ordinary text. They must not be registered as
# special tokens. Recall tool results use Qwen's tool-response wrapper with a
# short plain status line plus recalled video blocks.

# ---------------------------------------------------------------------------
# 8. Teacher prompts (397B, hidden from student)
# ---------------------------------------------------------------------------

EVIDENCE_GRAPH_PROMPT = """You are annotating a 1-second video clip (t={start}-{end}s, 2 frames).

Based on the timestamp-tagged frames in this message, output a STRICT JSON object:
{{
  "time": [{start}, {end}],
  "visible_entities": [
    {{"desc": "fine-grained appearance", "action": "verb phrase or 'static'", "position": "left/center/right/top/bottom/foreground/background"}}
  ],
  "atomic_facts": ["precise observable statement", ...],
  "ocr": ["exact text if visible"],
  "spatial": "spatial relations between entities, sentence-form",
  "think": "one current-only observation paragraph, 40-80 tokens"
}}

CRITICAL — minimum output requirement:
- The timestamp-tagged frames in this message almost always contain SOMETHING describable: a person,
  an object being manipulated, a setting, on-screen text, a tool, food, etc.
- visible_entities MUST have ≥1 element AND atomic_facts MUST have ≥1 element,
  even if the scene is dim/blurry/transition. The only exception is a fully
  black or fully white frame.
- Empty arrays mean "I gave up" — not allowed.
- think MUST be non-empty. This field is the exact student observation target
  used by pass2/SFT. It must be grounded only in these two current frames.

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

think — write ONE natural paragraph, 40-80 tokens:
  - Re-state the most important current visual facts in fluent prose
  - Use only entities/actions/OCR/spatial relations visible in these two frames
  - Mention current OCR/logos/title cards when visible
  - Do not refer to any previous or future chunk
  - Do not say "continues", "still", "remains", "same", "earlier", or "next"
    unless that continuity is directly visible in the current frames alone
  - Do not mention JSON fields, frame tags, timestamps, protocol, uncertainty,
    or "I notice"

Rules:
- Only describe what is VISIBLE in these frames (no comparison to other clips)
- Describe entities by appearance, not by ID or assumed identity
- Use a CONSISTENT phrase for the same entity across clips when its appearance
  matches (e.g., "the man in the black polo shirt") so downstream entity
  linking can match by string. Do not paraphrase the same entity differently.
- Do NOT include sounds, smells, emotions, or inferred intentions

Output JSON only:"""

OBSERVATION_PROMPT = """You are a streaming video agent generating an observation note for one current 1-second chunk.

CURRENT TASK FIRST: inspect the timestamp-tagged image list for the sliding visual window t={window_start}-{window_end}s. The latest target chunk is ONLY t={start}-{end}s ({current_frame_count} frames) and is the primary evidence.

History ledger below is archival memory for naming only. It may describe older frames and must not be copied if the latest frames differ. It contains compact records like <m t="start-end">visible historical fact</m>; each line is stale history wording, not current evidence.
<history_ledger>
{recent_thinks}
</history_ledger>

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text. Never copy or paraphrase any frame tag in the output.

The timestamp-tagged images are ordered from older context to the latest chunk. Frames labeled t={start}-{end}s are the only evidence for the current think; older timestamps are context only.

Evidence priority:
1. The tagged images at t={start}-{end}s are the only evidence for the current think.
2. Older tagged images are context only.
3. The history ledger is history and entity naming only. Ignore any history record when it conflicts with the latest frames.
4. Never use the history ledger as evidence that a past object/action is still visible.

Rules:
- Ground the note only in observable visual facts from the latest target chunk
- Treat history_ledger lines as machine-readable records, not prose to continue
- Do not copy any XML-like tag, timestamp marker, role marker, metadata line, or history record text into the output
- Mention current OCR, logos, icons, labels, title cards, graphic overlays, and spatial layout when visible
- Reuse a historical entity phrase only when that same entity is visibly present now
- Do not copy a prior sentence or mention any object/action from history unless it is visible in the latest target chunk
- If history says a person/hand is holding, pressing, pouring, cutting, walking, or otherwise manipulating something, write that action only when the actor and contact/motion are visible in the latest target chunk
- If the latest frames show an object at rest, on a stand/table/surface, or as a static screen/card, describe that current state directly instead of repeating an old manipulation
- If the latest frames show a different object/action, title card, branding card, transition card, or static graphic, name it directly
- Avoid "continues", "remains", "persists", "still", "same", and "without change" unless those words are justified by the latest target chunk alone
- Final self-check before answering: if a phrase came from the history ledger rather than the latest two frames, rewrite it
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

OBSERVATION_REPAIR_PROMPT = """You are correcting a streaming video observation note for one current chunk.

Recent history ledger (tagged stale records; use only for naming, never for current evidence):
{recent_thinks}

Previous stale draft to avoid copying:
{stale_text}

The tagged images contain ONLY the current 1 second: t={start}-{end}s
({n_frames} frames at {fps} fps).

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text. Never copy or paraphrase any frame tag in the output.

Task: inspect the current frames first and write the actual visual note for
t={start}-{end}s.

Evidence priority:
1. Current tagged frames at t={start}-{end}s.
2. History/entity names only if the same entity is visibly present.
3. Never use history or the stale draft as evidence for what is visible now.

Rules:
- Describe only observable visual facts in this 1-second chunk
- Treat history lines as stale records, not prose to continue
- Do not copy any XML-like tag, timestamp marker, role marker, metadata line, or history record text into the output
- Keep entity names consistent only when the same entity is visibly present
- Do not say "continues", "remains", "unchanged", or "no new" unless the
  current frames visibly show the same object/action
- If the current frames show a new object/action, name it directly
- 40-80 tokens, one paragraph, no meta-reasoning

Output one paragraph:"""

COMPRESS_PROMPT = """Compress these tagged observation records into a structured summary.

Tagged observations to compress:
{observations_text}

Rules:
- Use coarse time sub-ranges: [X-Y]
- Read each <m t="start-end">...</m> line as one historical observation or older summary
- Keep ALL entities with their appearance descriptions
- Keep ALL OCR content verbatim
- Keep state changes as before→after
- Keep user interaction summaries if any
- Target length: {target_length} tokens
- Base the summary strictly on the observation text — do not introduce entities, counts, colors, or events not present in the observations

Output JSON only: {{"time_range": [{start}, {end}], "text": "<concise factual summary>"}}
Do NOT output literal ellipsis, placeholder text, markdown, or analysis outside the JSON."""

COMPACT_MEMORY_UPDATE_SYSTEM_PROMPT = """You update compact video memory for training data.

Return only the TARGET_RANGES as chronological XML lines and no other text:
<m t="exact-start-exact-end">one concise English event or state.</m>
<m t="exact-start-exact-end">one concise English event or state.</m>
<m t="exact-start-exact-end">one concise English event or state.</m>
<m t="exact-start-exact-end">one concise English event or state.</m>

Requirements:
- Output exactly one <m> line for each TARGET_RANGES line from the user prompt.
- Use the exact t="start-end" values from TARGET_RANGES. Do not split, merge, skip, repeat, or invent ranges.
- Each output line must summarize only OLD_MEMORY or NEW_CAPTIONS whose timestamps overlap that target range.
- The output ranges must be evenly coarse: no per-second caption copying and no giant old-memory prefix followed by tiny recent ranges.
- Every <m ...> line must have its own explicit closing </m> tag.
- A response that omits </m> on any line is invalid; do not leave tags open, including the final line.
- Use only timestamps that appear in OLD_MEMORY or NEW_CAPTIONS.
- Input is video memory only. Ignore and never reproduce questions, answers, active-query tags, or response-history tags if they appear.
- If OLD_MEMORY contains any <m> lines, at least one output <m> must preserve useful historical information from OLD_MEMORY.
- If NEW_CAPTIONS contains any <c> lines, at least one output <m> must cover the latest new caption timestamps.
- Cover BOTH historical state and new events; do not output only old memory or only new captions when both are present.
- Keep important objects, actions, OCR text, counts, colors, and state changes.
- Preserve exact visible names, jersey numbers, team labels, scoreboard values, OCR strings, sponsor/ad text, and distinctive colors when present.
- Do not replace all older memory with a generic event line if OLD_MEMORY contains named players, OCR, or scoreboard values.
- When NEW_CAPTIONS contains names, OCR, or scoreboard text, include the most important ones in the output.
- If one input memory line overlaps several TARGET_RANGES, carry only the relevant high-level facts into each affected output line; do not copy the same sentence into every line.
- It is acceptable to compress repeated generic play-by-play, but not to drop all exact identifiers.
- Merge adjacent repeated captions; split when the main object, action, scene, or state changes.
- Do not answer questions, add analysis, describe future actions, or write text outside the <m> lines."""

COMPACT_MEMORY_UPDATE_PROMPT = """OLD_MEMORY:
{old_memory}

NEW_CAPTIONS:
{new_captions}

Covered latest span: t={start}-{end}
TARGET_RANGES:
{target_ranges}

Coverage check:
- Output exactly the TARGET_RANGES above, in the same order and with the same t values.
- Each TARGET_RANGES line must be filled with a concise factual summary from overlapping OLD_MEMORY and NEW_CAPTIONS.
- If OLD_MEMORY has <m> lines, preserve useful old information in the target ranges it overlaps.
- If NEW_CAPTIONS has <c> lines, cover the latest new caption timestamps in the target ranges they overlap.
- Do not output a single broad old range like <m t="0-60"> when TARGET_RANGES asks for smaller balanced ranges.
- Do not output one line per second or copy raw captions nearly verbatim.
- Every returned line must end with </m>; do not omit closing tags.

Return only the filled XML lines from TARGET_RANGES.
Do not output NEW_MEMORY, markdown, prose, analysis, or any text outside the <m> lines."""

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

POST_RECALL_THINK_PROMPT = """You are a streaming video agent that just received recall results.

Question: "{question}"
Recall metadata: {recall_result}
Recall source: {recall_source}

Write a brief post-recall thought (15-35 tokens) grounded in the recalled visual
frames and the pending question.
- If results are relevant: mention the non-answer visual evidence now available.
- If results are irrelevant/empty: say the recalled frames do not settle it.
- Do NOT include the final answer, option letter, exact count, or answer value.
- NO meta-reasoning ("I think", "I notice"), NO sounds/smells/emotions.

Output the thought text only (15-35 tokens):"""

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
