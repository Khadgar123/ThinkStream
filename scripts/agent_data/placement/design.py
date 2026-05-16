"""Pass3 v2 — model-agnostic design.

Three concepts, three fields:
  question_type    — single_emit | multi_emit
  gold_emits       — list of (chunk, value) pairs; defines gold function
  grounding_frames — necessary evidence frames; defines recall oracle

Card families describe the benchmark skill being asked; they are not tied to
one availability bucket. A single family may produce current/direct,
state-memory direct, forward/wait, and historical-recall placements. Difficulty
is primarily the gap between ask and grounding plus whether the first-turn
prompt still contains enough clear evidence to answer without visual recall.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
import os
import re
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from ..config import MAX_QUESTIONS_PER_TRAJECTORY as CONFIG_MAX_QUESTIONS_PER_TRAJECTORY
from ..config import VISUAL_WINDOW_CHUNKS as CONFIG_VISUAL_WINDOW_CHUNKS
from ..stable_hash import stable_mod


# ---------------------------------------------------------------------------
# Constants (mirrors agent_data/config.py + adds new ones)
# ---------------------------------------------------------------------------

VISUAL_WINDOW_CHUNKS = CONFIG_VISUAL_WINDOW_CHUNKS  # chunks still in visual prompt
RECENT_THINKS_HORIZON = 60       # ~4000 tok / 70 tok-per-think — pre-compress horizon
RECALL_OK_RATE = 0.95            # pure-oracle recall demo
RECALL_NOISY_RATE = 0.05         # oracle ⊕ distractor frames
RECALL_WAIT_PROBE_MAX = int(os.environ.get("THINKSTREAM_RECALL_WAIT_PROBE_MAX", "0"))
RECALL_WAIT_MIN_LEAD = 6         # do not recall immediately for very short waits
RECALL_MULTI_WAIT_PROBE_MAX = int(os.environ.get("THINKSTREAM_RECALL_MULTI_WAIT_PROBE_MAX", "0"))
MEMORY_DIRECT_RECALL_PROBE_RATE = float(
    os.environ.get("THINKSTREAM_MEMORY_DIRECT_RECALL_PROBE_RATE", "0.20")
)
MEMORY_DIRECT_RECALL_FAMILY_RATE = {
    # Exact visual evidence is much better than text memory for these OVO
    # weaknesses: OCR, spatial/temporal relations, object relation/state,
    # action recognition, future/held state, and cross-event reasoning.
    "C1": 0.30,
    "STU1": 0.25,
    "OJR1": 0.25,
    "ACR1": 0.10,
    "CR7": 0.20,
    "CR5": 0.25,
    "CR4": 0.25,
    "CR1": 0.25,
    "CR2": 0.25,
    "M1": 0.20,
    "HLD1": 0.35,
}

# Keep enough non-MCQ questions for active responding generalization. Do not
# turn ordinary perception/backward-tracing families into open-form questions
# just to hit a ratio; non-MCQ pressure should mainly come from F5/REC counting,
# F7/SSR status, and CRR1/CRR status-over-time, with small exploratory support
# from M1/PN1 when feasible.
NON_MCQ_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_NON_MCQ_TARGET_FRACTION", "0.30")
)
NON_MCQ_MIN_QUESTIONS = int(os.environ.get("THINKSTREAM_NON_MCQ_MIN_QUESTIONS", "3"))
BINARY_TARGET_FRACTION = float(os.environ.get("THINKSTREAM_BINARY_TARGET_FRACTION", "0.10"))
BINARY_MAX_FRACTION = float(os.environ.get("THINKSTREAM_BINARY_MAX_FRACTION", "0.16"))
NUMBER_TARGET_FRACTION = float(os.environ.get("THINKSTREAM_NUMBER_TARGET_FRACTION", "0.08"))
NUMBER_MAX_FRACTION = float(os.environ.get("THINKSTREAM_NUMBER_MAX_FRACTION", "0.14"))
SHORT_TEXT_TARGET_FRACTION = float(os.environ.get("THINKSTREAM_SHORT_TEXT_TARGET_FRACTION", "0.10"))
SHORT_TEXT_MAX_FRACTION = float(os.environ.get("THINKSTREAM_SHORT_TEXT_MAX_FRACTION", "0.16"))
ANSWER_FORM_BUCKETS = ("binary", "number", "short_text")

# ── ask placement: STRATIFIED tier ranges (3 difficulty bands per profile) ──
# Each profile picks one band per placement; multi-placement profiles
# (forward, backward) explicitly cover multiple bands to give the model
# difficulty variety.
#
# Forward stratification (silent_then_response lead time):
SE_LEAD_SHORT  = (4, 10)            # quick wait — easier to maintain pending
SE_LEAD_LONG   = (18, 32)           # long wait — hard, tests pending persistence
SE_LEAD_OVO_CRR = ((10, 24), (24, 64), (64, 120))  # OVO CRR clue delays
#
# Real-time / direct fresh stratification (gap from emit):
SE_FRESH_TRIVIAL = (0, 2)            # ask at evidence — trivial
SE_FRESH_EASY    = (3, 8)            # in visual window, recent
SE_FRESH_MEDIUM  = (9, VISUAL_WINDOW_CHUNKS)    # at edge of visual window
#
# Backward stratification (recall_demo gap):
SE_RECALL_NEAR   = (VISUAL_WINDOW_CHUNKS + 1, VISUAL_WINDOW_CHUNKS + 24)         # ~17-40, just out of visual
SE_RECALL_MID    = (VISUAL_WINDOW_CHUNKS + 25, VISUAL_WINDOW_CHUNKS + RECENT_THINKS_HORIZON)  # ~41-76, in recent_thinks
SE_RECALL_DEEP   = (VISUAL_WINDOW_CHUNKS + RECENT_THINKS_HORIZON + 1, 999)        # 77+, fully compressed
#
# multi_emit ask runway before first emit
ME_LEAD_RANGE = (2, 8)
MAX_MULTI_EMIT_ACTIVE_SPAN = int(os.environ.get("THINKSTREAM_MAX_MULTI_EMIT_ACTIVE_SPAN", "48"))
MAX_MULTI_EMIT_RESPONSES = int(os.environ.get("THINKSTREAM_MAX_MULTI_EMIT_RESPONSES", "9"))
MIN_BENCH_MULTI_EMIT_RESPONSES = int(os.environ.get(
    "THINKSTREAM_MIN_BENCH_MULTI_EMIT_RESPONSES",
    "6",
))
MAX_REC_EMIT_RESPONSES = int(os.environ.get("THINKSTREAM_MAX_REC_EMIT_RESPONSES", "9"))
MAX_REC_GLOBAL_EMIT_RESPONSES = int(os.environ.get(
    "THINKSTREAM_MAX_REC_GLOBAL_EMIT_RESPONSES",
    "9",
))
MAX_CRR_EMIT_RESPONSES = int(os.environ.get("THINKSTREAM_MAX_CRR_EMIT_RESPONSES", "5"))
MAX_REC_ACTIVE_SPAN = int(os.environ.get("THINKSTREAM_MAX_REC_ACTIVE_SPAN", "96"))
F5_ASK_LEAD_CHUNKS = int(os.environ.get("THINKSTREAM_F5_ASK_LEAD_CHUNKS", "2"))
F5_GLOBAL_QUERY_PERCENT = int(os.environ.get("THINKSTREAM_F5_GLOBAL_QUERY_PERCENT", "50"))
F5_GLOBAL_QUERY_MAX_FIRST_CHUNK = int(os.environ.get(
    "THINKSTREAM_F5_GLOBAL_QUERY_MAX_FIRST_CHUNK",
    "120",
))
F5_GLOBAL_QUERY_MAX_ACTIVE_SPAN = int(os.environ.get(
    "THINKSTREAM_F5_GLOBAL_QUERY_MAX_ACTIVE_SPAN",
    "72",
))
STATUS_FAR_AFTER_MIN_GAP = int(os.environ.get(
    "THINKSTREAM_STATUS_FAR_AFTER_MIN_GAP",
    str(VISUAL_WINDOW_CHUNKS + 1),
))
STATUS_FAR_AFTER_MAX_GAP = int(os.environ.get(
    "THINKSTREAM_STATUS_FAR_AFTER_MAX_GAP",
    "36",
))
RECALL_TARGET_FRACTION = float(os.environ.get("THINKSTREAM_RECALL_TARGET_FRACTION", "0.36"))
RECALL_MAX_FRACTION = float(os.environ.get("THINKSTREAM_RECALL_MAX_FRACTION", "0.42"))
RECALL_MAX_PER_TRAJECTORY = int(os.environ.get("THINKSTREAM_RECALL_MAX_PER_TRAJECTORY", "0"))
TIMING_QUOTA_DENOMINATOR_FRACTION = float(
    os.environ.get("THINKSTREAM_TIMING_QUOTA_DENOMINATOR_FRACTION", "0.50")
)
DENSITY_FILL_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_DENSITY_FILL_TARGET_FRACTION", "1.00")
)
DENSITY_BLOCKING_SPAN_MAX = int(
    os.environ.get("THINKSTREAM_DENSITY_BLOCKING_SPAN_MAX", "72")
)
DENSITY_BLOCKING_SPAN_FRACTION = float(
    os.environ.get("THINKSTREAM_DENSITY_BLOCKING_SPAN_FRACTION", "0.40")
)
F5_RESPONSE_RECALL_RATE = float(os.environ.get("THINKSTREAM_F5_RESPONSE_RECALL_RATE", "0.0"))
CRR_STATUS_RECALL_RATE = float(os.environ.get("THINKSTREAM_CRR_STATUS_RECALL_RATE", "0.0"))

OURS_TARGET_FRACTION = float(os.environ.get("THINKSTREAM_OURS_TARGET_FRACTION", "0.09"))
OURS_MAX_FRACTION = float(os.environ.get("THINKSTREAM_OURS_MAX_FRACTION", "0.11"))
OURS_MIN_QUESTIONS = int(os.environ.get("THINKSTREAM_OURS_MIN_QUESTIONS", "1"))
OURS_ABS_CAP = int(os.environ.get("THINKSTREAM_OURS_ABS_CAP", "2"))
OURS_TRAJECTORY_PERCENT = int(os.environ.get("THINKSTREAM_OURS_TRAJECTORY_PERCENT", "65"))
OURS_FAMILY_MIX_PERCENT = {
    "CR5": int(os.environ.get("THINKSTREAM_OURS_CR5_PERCENT", "45")),
    "M1": int(os.environ.get("THINKSTREAM_OURS_M1_PERCENT", "35")),
    "PN1": int(os.environ.get("THINKSTREAM_OURS_PN1_PERCENT", "20")),
}
STATE_PROBE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_STATE_PROBE_TARGET_FRACTION", "0.08")
)
STATE_PROBE_MAX_FRACTION = float(os.environ.get("THINKSTREAM_STATE_PROBE_MAX_FRACTION", "0.10"))
STATE_PROBE_FAMILY_RESERVE_PERCENT = {
    "F5": int(os.environ.get("THINKSTREAM_F5_RESERVE_PERCENT", "45")),
    "CRR1": int(os.environ.get("THINKSTREAM_CRR1_RESERVE_PERCENT", "25")),
    "F7": int(os.environ.get("THINKSTREAM_F7_RESERVE_PERCENT", "20")),
}
CURRENT_DIRECT_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_CURRENT_DIRECT_TARGET_FRACTION", "0.50")
)
CURRENT_DIRECT_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_CURRENT_DIRECT_MAX_FRACTION", "0.55")
)
CURRENT_DIRECT_MIN_QUESTIONS = int(os.environ.get("THINKSTREAM_CURRENT_DIRECT_MIN_QUESTIONS", "3"))
PAST_STATE_DIRECT_TARGET_FRACTION = float(
    os.environ.get(
        "THINKSTREAM_PAST_STATE_DIRECT_TARGET_FRACTION",
        os.environ.get("THINKSTREAM_PAST_DIRECT_TARGET_FRACTION", "0.10"),
    )
)
PAST_STATE_DIRECT_MAX_FRACTION = float(
    os.environ.get(
        "THINKSTREAM_PAST_STATE_DIRECT_MAX_FRACTION",
        os.environ.get("THINKSTREAM_PAST_DIRECT_MAX_FRACTION", "0.12"),
    )
)
# Backward-compatible aliases for old audit scripts/env overrides. The bucket
# semantics are now "state memory direct", not historical visual direct.
PAST_DIRECT_TARGET_FRACTION = PAST_STATE_DIRECT_TARGET_FRACTION
PAST_DIRECT_MAX_FRACTION = PAST_STATE_DIRECT_MAX_FRACTION
MULTI_ANSWER_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_MULTI_ANSWER_TARGET_FRACTION", "0.12")
)
MULTI_ANSWER_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MULTI_ANSWER_MAX_FRACTION", "0.16")
)
MULTI_ANSWER_TRAJECTORY_PERCENT = int(
    os.environ.get("THINKSTREAM_MULTI_ANSWER_TRAJECTORY_PERCENT", "100")
)
UNANSWERABLE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_UNANSWERABLE_TARGET_FRACTION", "0.06")
)
UNANSWERABLE_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_UNANSWERABLE_MAX_FRACTION", "0.08")
)
UNANSWERABLE_TRAJECTORY_PERCENT = int(
    os.environ.get("THINKSTREAM_UNANSWERABLE_TRAJECTORY_PERCENT", "80")
)
FUTURE_DELAYED_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_FUTURE_DELAYED_TARGET_FRACTION", "0.10")
)
FUTURE_DELAYED_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_FUTURE_DELAYED_MAX_FRACTION", "0.14")
)
FUTURE_DELAYED_TRAJECTORY_PERCENT = int(
    os.environ.get("THINKSTREAM_FUTURE_DELAYED_TRAJECTORY_PERCENT", "95")
)
FUTURE_CURRENT_CUE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_FUTURE_CURRENT_CUE_TARGET_FRACTION", "0.05")
)
FUTURE_CURRENT_CUE_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_FUTURE_CURRENT_CUE_MAX_FRACTION", "0.06")
)
FUTURE_CURRENT_CUE_TRAJECTORY_PERCENT = int(
    os.environ.get("THINKSTREAM_FUTURE_CURRENT_CUE_TRAJECTORY_PERCENT", "45")
)
MESSAGE_RESPONSE_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_RESPONSE_ROW_MAX_FRACTION", "0.15")
)
MESSAGE_DIRECT_RESPONSE_ROW_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_DIRECT_RESPONSE_ROW_TARGET_FRACTION", "0.525")
)
MESSAGE_DIRECT_RESPONSE_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_DIRECT_RESPONSE_ROW_MAX_FRACTION", "0.55")
)
MESSAGE_RECALL_RESPONSE_ROW_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_RECALL_RESPONSE_ROW_TARGET_FRACTION", "0.275")
)
MESSAGE_RECALL_RESPONSE_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_RECALL_RESPONSE_ROW_MAX_FRACTION", "0.30")
)
MESSAGE_FUTURE_RESPONSE_ROW_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_FUTURE_RESPONSE_ROW_TARGET_FRACTION", "0.10")
)
MESSAGE_FUTURE_RESPONSE_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_FUTURE_RESPONSE_ROW_MAX_FRACTION", "0.12")
)
MESSAGE_MULTI_RESPONSE_ROW_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_MULTI_RESPONSE_ROW_TARGET_FRACTION", "0.10")
)
MESSAGE_MULTI_RESPONSE_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_MULTI_RESPONSE_ROW_MAX_FRACTION", "0.15")
)
MESSAGE_MULTI_RESPONSE_ROW_MIN_CAP = int(
    os.environ.get("THINKSTREAM_MESSAGE_MULTI_RESPONSE_ROW_MIN_CAP", "2")
)
MESSAGE_MULTI_RESPONSE_ROW_ABS_CAP = int(
    os.environ.get("THINKSTREAM_MESSAGE_MULTI_RESPONSE_ROW_ABS_CAP", "0")
)
MESSAGE_MULTI_ACTIVE_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_MULTI_ACTIVE_ROW_MAX_FRACTION", "0.40")
)
MESSAGE_FUTURE_PENDING_ROW_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_MESSAGE_FUTURE_PENDING_ROW_MAX_FRACTION", "0.25")
)
MESSAGE_FUTURE_PENDING_ROW_MIN_CAP = int(
    os.environ.get("THINKSTREAM_MESSAGE_FUTURE_PENDING_ROW_MIN_CAP", "16")
)
RAW_RESPONSE_DENSITY_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_RAW_RESPONSE_DENSITY_TARGET_FRACTION", "0.135")
)
RAW_RESPONSE_DENSITY_MIN_FRACTION = float(
    os.environ.get("THINKSTREAM_RAW_RESPONSE_DENSITY_MIN_FRACTION", "0.12")
)
RAW_RESPONSE_DENSITY_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_RAW_RESPONSE_DENSITY_MAX_FRACTION", "0.15")
)
RAW_RESPONSE_DENSITY_ASK_GAP_CHUNKS = int(
    os.environ.get("THINKSTREAM_RAW_RESPONSE_DENSITY_ASK_GAP_CHUNKS", "1")
)
RAW_RESPONSE_DENSITY_FAMILY_REPEAT_CAP = int(
    os.environ.get("THINKSTREAM_RAW_RESPONSE_DENSITY_FAMILY_REPEAT_CAP", "4")
)
RAW_RESPONSE_DENSITY_QUESTION_CAP = int(
    os.environ.get("THINKSTREAM_RAW_RESPONSE_DENSITY_QUESTION_CAP", "0")
)
RAW_RESPONSE_SOURCE_TARGET_FRACTION = {
    "direct": float(os.environ.get("THINKSTREAM_RAW_DIRECT_RESPONSE_TARGET_FRACTION", "0.525")),
    "recall": float(os.environ.get("THINKSTREAM_RAW_RECALL_RESPONSE_TARGET_FRACTION", "0.275")),
    "future": float(os.environ.get("THINKSTREAM_RAW_FUTURE_RESPONSE_TARGET_FRACTION", "0.10")),
    "multi": float(os.environ.get("THINKSTREAM_RAW_MULTI_RESPONSE_TARGET_FRACTION", "0.10")),
}
RAW_RESPONSE_SOURCE_MAX_FRACTION = {
    "direct": float(os.environ.get("THINKSTREAM_RAW_DIRECT_RESPONSE_MAX_FRACTION", "0.55")),
    "recall": float(os.environ.get("THINKSTREAM_RAW_RECALL_RESPONSE_MAX_FRACTION", "0.30")),
    "future": float(os.environ.get("THINKSTREAM_RAW_FUTURE_RESPONSE_MAX_FRACTION", "0.12")),
    "multi": float(os.environ.get("THINKSTREAM_RAW_MULTI_RESPONSE_MAX_FRACTION", "0.15")),
}
BENCHMARK_VARIANT_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_BENCHMARK_VARIANT_TARGET_FRACTION", "0.22")
)
BENCHMARK_VARIANT_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_BENCHMARK_VARIANT_MAX_FRACTION", "0.25")
)
SUPPORT_BIN_SIZE = int(os.environ.get("THINKSTREAM_SUPPORT_BIN_SIZE", "8"))
SUPPORT_BIN_CAP = int(os.environ.get("THINKSTREAM_SUPPORT_BIN_CAP", "4"))
SUPPORT_BIN_FLOOR_FILL_CAP = int(os.environ.get("THINKSTREAM_SUPPORT_BIN_FLOOR_FILL_CAP", "6"))
RECALL_FAMILY_DIRECT_FILL_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_RECALL_FAMILY_DIRECT_FILL_MAX_FRACTION", "0.12")
)
QUESTION_DENSITY_CHUNKS = int(os.environ.get("THINKSTREAM_QUESTION_DENSITY_CHUNKS", "6"))
POSITION_BIN_COUNT = int(os.environ.get("THINKSTREAM_POSITION_BIN_COUNT", "5"))
POSITION_BIN_SCORE_WEIGHT = float(os.environ.get("THINKSTREAM_POSITION_BIN_SCORE_WEIGHT", "1.1"))
SOURCE_TIME_BIN_SCORE_WEIGHT = float(
    os.environ.get("THINKSTREAM_SOURCE_TIME_BIN_SCORE_WEIGHT", "1.0")
)

ANSWER_MODE_DIRECT = "direct"
ANSWER_MODE_RECALL = "recall"
ANSWER_MODE_FUTURE = "future"
ANSWER_MODE_MULTI = "multi"
ANSWER_MODE_STATE_DIRECT = "state_direct"
ANSWER_MODE_ABSTAIN = "abstain"
ANSWER_MODE_HLD_RECALL = "hld_recall"
ANSWER_MODES = {
    ANSWER_MODE_DIRECT,
    ANSWER_MODE_RECALL,
    ANSWER_MODE_FUTURE,
    ANSWER_MODE_MULTI,
    ANSWER_MODE_STATE_DIRECT,
    ANSWER_MODE_ABSTAIN,
    ANSWER_MODE_HLD_RECALL,
}
TASK_MODE_TARGET_FRACTION = {
    ("C1", ANSWER_MODE_DIRECT): 0.08,
    ("OJR1", ANSWER_MODE_DIRECT): 0.06,
    ("STU1", ANSWER_MODE_DIRECT): 0.06,
    ("ACR1", ANSWER_MODE_DIRECT): 0.03,
    ("CR3", ANSWER_MODE_DIRECT): 0.03,
    ("CR7", ANSWER_MODE_DIRECT): 0.03,
    ("R1", ANSWER_MODE_DIRECT): 0.02,
    ("HLD1", ANSWER_MODE_HLD_RECALL): 0.06,
    ("M1", ANSWER_MODE_STATE_DIRECT): 0.04,
    ("CR1", ANSWER_MODE_RECALL): 0.04,
    ("CR2", ANSWER_MODE_RECALL): 0.03,
    ("CR4", ANSWER_MODE_RECALL): 0.04,
    ("N1", ANSWER_MODE_RECALL): 0.03,
    ("P1", ANSWER_MODE_RECALL): 0.03,
    ("CR5", ANSWER_MODE_RECALL): 0.03,
    ("E2", ANSWER_MODE_FUTURE): 0.07,
    ("CR5", ANSWER_MODE_FUTURE): 0.03,
    ("F5", ANSWER_MODE_MULTI): 0.02,
    ("F7", ANSWER_MODE_DIRECT): 0.02,
    ("CRR1", ANSWER_MODE_MULTI): 0.02,
    ("PN1", ANSWER_MODE_MULTI): 0.01,
}
NORMAL_RECALL_FAMILIES = {"CR1", "CR2", "CR4", "CR5", "N1", "P1"}

QUESTION_STYLE_BENCHMARK_CORE = "benchmark_core"
QUESTION_STYLE_BENCHMARK_VARIANT = "benchmark_variant"
QUESTION_STYLE_OURS_UNIQUE = "ours_unique"
QUESTION_STYLES = {
    QUESTION_STYLE_BENCHMARK_CORE,
    QUESTION_STYLE_BENCHMARK_VARIANT,
    QUESTION_STYLE_OURS_UNIQUE,
}

QUESTION_WAY_OBJECT_ATTRIBUTE = "object_attribute"
QUESTION_WAY_PERSON_IDENTITY = "person_identity_interaction"
QUESTION_WAY_ACTION_RECOGNITION = "action_recognition"
QUESTION_WAY_TEXT_READOUT = "text_readout"
QUESTION_WAY_SPATIAL_RELATION = "spatial_relation"
QUESTION_WAY_TEMPORAL_ORDER = "temporal_order"
QUESTION_WAY_CAUSAL_INTENT = "causal_intent"
QUESTION_WAY_FUTURE_PREDICTION = "future_prediction"
QUESTION_WAY_PROACTIVE_OUTPUT = "proactive_output"
QUESTION_WAY_REPEATED_COUNT = "repeated_count"
QUESTION_WAY_CURRENT_STATUS = "current_status_probe"
QUESTION_WAY_EVIDENCE_SUFFICIENCY = "evidence_sufficiency_probe"
QUESTION_WAY_UNANSWERABLE = "unanswerable_absence"
QUESTION_WAY_SEQUENTIAL_REFERENCE = "sequential_reference"
QUESTION_WAY_EMOTION_CONTEXT = "emotion_context"
QUESTION_WAY_SCENE_SUMMARY = "scene_summary"
QUESTION_WAY_LIVE_NARRATION = "live_narration"
QUESTION_WAY_SOURCE_DISCRIMINATION = "source_discrimination"
QUESTION_WAY_MULTIMODAL_ALIGNMENT = "multimodal_alignment"
QUESTION_WAYS = {
    QUESTION_WAY_OBJECT_ATTRIBUTE,
    QUESTION_WAY_PERSON_IDENTITY,
    QUESTION_WAY_ACTION_RECOGNITION,
    QUESTION_WAY_TEXT_READOUT,
    QUESTION_WAY_SPATIAL_RELATION,
    QUESTION_WAY_TEMPORAL_ORDER,
    QUESTION_WAY_CAUSAL_INTENT,
    QUESTION_WAY_FUTURE_PREDICTION,
    QUESTION_WAY_PROACTIVE_OUTPUT,
    QUESTION_WAY_REPEATED_COUNT,
    QUESTION_WAY_CURRENT_STATUS,
    QUESTION_WAY_EVIDENCE_SUFFICIENCY,
    QUESTION_WAY_UNANSWERABLE,
    QUESTION_WAY_SEQUENTIAL_REFERENCE,
    QUESTION_WAY_EMOTION_CONTEXT,
    QUESTION_WAY_SCENE_SUMMARY,
    QUESTION_WAY_LIVE_NARRATION,
    QUESTION_WAY_SOURCE_DISCRIMINATION,
    QUESTION_WAY_MULTIMODAL_ALIGNMENT,
}

# Fine-grained target mix across OVO-Bench + StreamingBench question forms.
# These are question-level targets, not rendered-row targets: REC/CRR may
# emit multiple response rows per selected question, while SSR/F7 is now an
# immediate single status row.
QUESTION_WAY_TARGET_FRACTION = {
    QUESTION_WAY_OBJECT_ATTRIBUTE: 0.095,
    QUESTION_WAY_PERSON_IDENTITY: 0.02,
    QUESTION_WAY_ACTION_RECOGNITION: 0.075,
    QUESTION_WAY_TEXT_READOUT: 0.085,
    QUESTION_WAY_SPATIAL_RELATION: 0.105,
    QUESTION_WAY_TEMPORAL_ORDER: 0.105,
    QUESTION_WAY_CAUSAL_INTENT: 0.065,
    QUESTION_WAY_FUTURE_PREDICTION: 0.045,
    QUESTION_WAY_PROACTIVE_OUTPUT: 0.055,
    QUESTION_WAY_REPEATED_COUNT: 0.055,
    QUESTION_WAY_CURRENT_STATUS: 0.05,
    QUESTION_WAY_EVIDENCE_SUFFICIENCY: 0.035,
    QUESTION_WAY_UNANSWERABLE: 0.04,
    QUESTION_WAY_SEQUENTIAL_REFERENCE: 0.025,
    QUESTION_WAY_EMOTION_CONTEXT: 0.025,
    QUESTION_WAY_SCENE_SUMMARY: 0.055,
    QUESTION_WAY_LIVE_NARRATION: 0.015,
    QUESTION_WAY_SOURCE_DISCRIMINATION: 0.025,
    QUESTION_WAY_MULTIMODAL_ALIGNMENT: 0.025,
}
QUESTION_WAY_CAP_MULTIPLIER = float(
    os.environ.get("THINKSTREAM_QUESTION_WAY_CAP_MULTIPLIER", "1.6")
)
QUESTION_WAY_MIN_CAP_FRACTION = float(
    os.environ.get("THINKSTREAM_QUESTION_WAY_MIN_CAP_FRACTION", "0.10")
)

EVIDENCE_OBJECT_ATTRIBUTE = "object_attribute_visual"
EVIDENCE_PERSON_RELATION = "person_relation_visual"
EVIDENCE_ACTION_EVENT = "action_event_visual"
EVIDENCE_TEXT_OCR = "text_ocr_visual"
EVIDENCE_SPATIAL_RELATION = "spatial_relation_visual"
EVIDENCE_TEMPORAL_ORDER = "temporal_order_visual"
EVIDENCE_CAUSAL_CONTEXT = "causal_context_visual"
EVIDENCE_FUTURE_CUE = "future_cue_visual"
EVIDENCE_FUTURE_TRIGGER = "future_trigger_visual"
EVIDENCE_REPEATED_EVENT = "repeated_event_stream"
EVIDENCE_STATUS_PROBE = "status_probe_stream"
EVIDENCE_ABSENCE = "absence_unanswerable"
EVIDENCE_GLOBAL_CONTEXT = "global_context_memory"
EVIDENCE_EMOTION_CONTEXT = "emotion_context_visual"
EVIDENCE_LIVE_STATE_CHANGE = "live_state_change"
EVIDENCE_SOURCE_DISCRIMINATION = "source_discrimination_visual"
EVIDENCE_MULTIMODAL_ALIGNMENT = "multimodal_alignment_visual"
EVIDENCE_TYPES = {
    EVIDENCE_OBJECT_ATTRIBUTE,
    EVIDENCE_PERSON_RELATION,
    EVIDENCE_ACTION_EVENT,
    EVIDENCE_TEXT_OCR,
    EVIDENCE_SPATIAL_RELATION,
    EVIDENCE_TEMPORAL_ORDER,
    EVIDENCE_CAUSAL_CONTEXT,
    EVIDENCE_FUTURE_CUE,
    EVIDENCE_FUTURE_TRIGGER,
    EVIDENCE_REPEATED_EVENT,
    EVIDENCE_STATUS_PROBE,
    EVIDENCE_ABSENCE,
    EVIDENCE_GLOBAL_CONTEXT,
    EVIDENCE_EMOTION_CONTEXT,
    EVIDENCE_LIVE_STATE_CHANGE,
    EVIDENCE_SOURCE_DISCRIMINATION,
    EVIDENCE_MULTIMODAL_ALIGNMENT,
}

QUESTION_WAY_DEFAULT_EVIDENCE_TYPE = {
    QUESTION_WAY_OBJECT_ATTRIBUTE: EVIDENCE_OBJECT_ATTRIBUTE,
    QUESTION_WAY_PERSON_IDENTITY: EVIDENCE_PERSON_RELATION,
    QUESTION_WAY_ACTION_RECOGNITION: EVIDENCE_ACTION_EVENT,
    QUESTION_WAY_TEXT_READOUT: EVIDENCE_TEXT_OCR,
    QUESTION_WAY_SPATIAL_RELATION: EVIDENCE_SPATIAL_RELATION,
    QUESTION_WAY_TEMPORAL_ORDER: EVIDENCE_TEMPORAL_ORDER,
    QUESTION_WAY_CAUSAL_INTENT: EVIDENCE_CAUSAL_CONTEXT,
    QUESTION_WAY_FUTURE_PREDICTION: EVIDENCE_FUTURE_CUE,
    QUESTION_WAY_PROACTIVE_OUTPUT: EVIDENCE_FUTURE_TRIGGER,
    QUESTION_WAY_REPEATED_COUNT: EVIDENCE_REPEATED_EVENT,
    QUESTION_WAY_CURRENT_STATUS: EVIDENCE_STATUS_PROBE,
    QUESTION_WAY_EVIDENCE_SUFFICIENCY: EVIDENCE_STATUS_PROBE,
    QUESTION_WAY_UNANSWERABLE: EVIDENCE_ABSENCE,
    QUESTION_WAY_SEQUENTIAL_REFERENCE: EVIDENCE_PERSON_RELATION,
    QUESTION_WAY_EMOTION_CONTEXT: EVIDENCE_EMOTION_CONTEXT,
    QUESTION_WAY_SCENE_SUMMARY: EVIDENCE_GLOBAL_CONTEXT,
    QUESTION_WAY_LIVE_NARRATION: EVIDENCE_LIVE_STATE_CHANGE,
    QUESTION_WAY_SOURCE_DISCRIMINATION: EVIDENCE_SOURCE_DISCRIMINATION,
    QUESTION_WAY_MULTIMODAL_ALIGNMENT: EVIDENCE_MULTIMODAL_ALIGNMENT,
}
EVIDENCE_TYPE_TARGET_FRACTION: Dict[str, float] = {}
for _way, _fraction in QUESTION_WAY_TARGET_FRACTION.items():
    _etype = QUESTION_WAY_DEFAULT_EVIDENCE_TYPE.get(_way)
    if _etype:
        EVIDENCE_TYPE_TARGET_FRACTION[_etype] = (
            EVIDENCE_TYPE_TARGET_FRACTION.get(_etype, 0.0) + _fraction
        )

# Subtype balancing keeps a correct high-level mix from collapsing into a few
# easy variants under pass3B. These are selection-level targets, not pass3A
# generation rates; missing subtypes simply receive no forced slot.
TASK_SUBTYPE_TARGET_FRACTION = {
    "unanswerable_absence": 0.04,
    "contextual_misleading_or_anomaly": 0.04,
    "emotion_context_current": 0.025,
    "scene_understanding_current": 0.04,
    "person_identity_interaction": 0.02,
    "sequential_reference": 0.025,
    "source_discrimination": 0.025,
    "multimodal_alignment": 0.025,
    "text_readout_current": 0.075,
    "delayed_clue_resolution": 0.045,
    "proactive_output": 0.055,
    "current_future_prediction": 0.045,
    "global_prefix_count": 0.03,
    "local_repeated_count": 0.03,
    "current_step_status": 0.03,
    "future_sufficiency_status": 0.035,
}
TASK_SUBTYPE_MAX_FRACTION = {
    "unanswerable_absence": 0.05,
    "contextual_misleading_or_anomaly": 0.06,
    "emotion_context_current": 0.045,
    "sequential_reference": 0.045,
    "source_discrimination": 0.045,
    "multimodal_alignment": 0.045,
    "delayed_clue_resolution": 0.065,
}
TASK_SUBTYPE_DEFAULT_MAX_FRACTION = float(
    os.environ.get("THINKSTREAM_TASK_SUBTYPE_DEFAULT_MAX_FRACTION", "0.18")
)
TASK_SUBTYPE_CAP_MULTIPLIER = float(
    os.environ.get("THINKSTREAM_TASK_SUBTYPE_CAP_MULTIPLIER", "1.7")
)
TASK_SUBTYPE_MIN_CAP_FRACTION = float(
    os.environ.get("THINKSTREAM_TASK_SUBTYPE_MIN_CAP_FRACTION", "0.08")
)

SUPPORT_CURRENT_VISUAL = "current_visual"
SUPPORT_HISTORICAL_VISUAL_RECALL = "historical_visual_recall"
SUPPORT_HISTORICAL_STATE_MEMORY = "historical_state_memory"
SUPPORT_FUTURE_CURRENT_CUE = "future_current_cue"
SUPPORT_PROBE_STATUS = "probe_status"
SUPPORT_POLICIES = {
    SUPPORT_CURRENT_VISUAL,
    SUPPORT_HISTORICAL_VISUAL_RECALL,
    SUPPORT_HISTORICAL_STATE_MEMORY,
    SUPPORT_FUTURE_CURRENT_CUE,
    SUPPORT_PROBE_STATUS,
}

_CURRENT_TEMPORAL_ROLES = {"current_visual", "current_probe"}
_CURRENT_TEMPORAL_BUCKETS = {"current_direct", "current_visual"}
_FUTURE_WAIT_TEMPORAL_ROLES = {"future_event_wait"}
_FUTURE_WAIT_TEMPORAL_BUCKETS = {"future_delayed"}
_FUTURE_CUE_TEMPORAL_ROLES = {"future_current_cue"}
_FUTURE_CUE_TEMPORAL_BUCKETS = {"current_future_prediction", "future_current_cue"}
_MULTI_TEMPORAL_ROLES = {"cumulative_count", "crr_sufficiency_probe", "live_narration"}
_MULTI_TEMPORAL_BUCKETS = {"multi_answer"}

_QUESTION_FUTURE_WAIT_RE = re.compile(
    r"\b(wait|until|once|as soon as|output\b|emit\b|"
    r"when\b[^?]{0,120}\b(output|emit|say|respond))\b",
    re.I,
)
_QUESTION_FUTURE_PRED_RE = re.compile(
    r"\b(about to|next|likely to|will\b|going to|expected to|what will|"
    r"what happens next|must happen|expected result)\b",
    re.I,
)
_QUESTION_EXPLICIT_CURRENT_RE = re.compile(
    r"\b(currently|right now|at this moment|in this (view|frame|scene))\b",
    re.I,
)
_QUESTION_PRESENT_PROGRESSIVE_RE = re.compile(
    r"\b(what|which|who|where|how)\b[^?]{0,100}\b(is|are)\b[^?]{0,50}\bbeing\b",
    re.I,
)
_QUESTION_EXPLICIT_PAST_RE = re.compile(
    r"\b(earlier|previously|before|did\b|was\b|were\b|had\b|initially|"
    r"at the beginning)\b",
    re.I,
)

FAMILY_TARGET_OVO_TASK = {
    "C1": "OCR",
    "ACR1": "ACR",
    "P1": "ATR",
    "N1": "ATR",
    "STU1": "STU",
    "OJR1": "OJR",
    "CR7": "OJR",
    "R1": "OJR",
    "F6": "FPD",
    "F5": "REC",
    "F7": "SSR",
    "CRR1": "CRR",
    "HLD1": "HLD",
    "CR2": "EPM",
    "E2": "EPM",
    "CR1": "ASI",
    "CR3": "ASI",
    "CR4": "ASI",
    "CR5": "EPM",
    "M1": "GLOBAL",
    "PN1": "STREAMING_AGENT",
}

FAMILY_DEFAULT_QUESTION_WAY = {
    "N1": QUESTION_WAY_OBJECT_ATTRIBUTE,
    "P1": QUESTION_WAY_OBJECT_ATTRIBUTE,
    "HLD1": QUESTION_WAY_UNANSWERABLE,
    "CR1": QUESTION_WAY_CAUSAL_INTENT,
    "CR2": QUESTION_WAY_TEMPORAL_ORDER,
    "CR3": QUESTION_WAY_CAUSAL_INTENT,
    "CR4": QUESTION_WAY_TEMPORAL_ORDER,
    "CR5": QUESTION_WAY_TEMPORAL_ORDER,
    "CR7": QUESTION_WAY_SPATIAL_RELATION,
    "CRR1": QUESTION_WAY_EVIDENCE_SUFFICIENCY,
    "M1": QUESTION_WAY_SCENE_SUMMARY,
    "E2": QUESTION_WAY_PROACTIVE_OUTPUT,
    "F6": QUESTION_WAY_FUTURE_PREDICTION,
    "F7": QUESTION_WAY_CURRENT_STATUS,
    "R1": QUESTION_WAY_SPATIAL_RELATION,
    "ACR1": QUESTION_WAY_ACTION_RECOGNITION,
    "STU1": QUESTION_WAY_SPATIAL_RELATION,
    "OJR1": QUESTION_WAY_SPATIAL_RELATION,
    "F5": QUESTION_WAY_REPEATED_COUNT,
    "C1": QUESTION_WAY_TEXT_READOUT,
    "PN1": QUESTION_WAY_LIVE_NARRATION,
}

FAMILY_DEFAULT_EVIDENCE_TYPE = {
    "N1": EVIDENCE_OBJECT_ATTRIBUTE,
    "P1": EVIDENCE_OBJECT_ATTRIBUTE,
    "HLD1": EVIDENCE_ABSENCE,
    "CR1": EVIDENCE_CAUSAL_CONTEXT,
    "CR2": EVIDENCE_TEMPORAL_ORDER,
    "CR3": EVIDENCE_CAUSAL_CONTEXT,
    "CR4": EVIDENCE_TEMPORAL_ORDER,
    "CR5": EVIDENCE_TEMPORAL_ORDER,
    "CR7": EVIDENCE_SPATIAL_RELATION,
    "CRR1": EVIDENCE_STATUS_PROBE,
    "M1": EVIDENCE_GLOBAL_CONTEXT,
    "E2": EVIDENCE_FUTURE_TRIGGER,
    "F6": EVIDENCE_FUTURE_CUE,
    "F7": EVIDENCE_STATUS_PROBE,
    "R1": EVIDENCE_SPATIAL_RELATION,
    "ACR1": EVIDENCE_ACTION_EVENT,
    "STU1": EVIDENCE_SPATIAL_RELATION,
    "OJR1": EVIDENCE_SPATIAL_RELATION,
    "F5": EVIDENCE_REPEATED_EVENT,
    "C1": EVIDENCE_TEXT_OCR,
    "PN1": EVIDENCE_LIVE_STATE_CHANGE,
}

STRICT_OVO_TASKS = {
    "OCR", "ACR", "ATR", "STU", "FPD", "OJR",
    "EPM", "ASI", "HLD", "REC", "SSR", "CRR",
    "GLOBAL", "STREAMING_AGENT",
}

# Not benchmark copies: these teach our streaming-agent behavior on top of the
# OVO skill core. F5/E2 are now treated as OVO-aligned REC/FPD skills, not as
# unique families for split/selection accounting.
OURS_FAMILIES = {"CR5", "PN1", "M1"}
STATE_PROBE_FAMILIES = {"F5", "F7", "CRR1", "PN1"}
STATE_PROBE_POLICIES = {
    SUPPORT_HISTORICAL_STATE_MEMORY,
    SUPPORT_PROBE_STATUS,
}
CURRENT_DIRECT_POLICIES = {
    SUPPORT_CURRENT_VISUAL,
    SUPPORT_FUTURE_CURRENT_CUE,
}

HISTORICAL_QUESTION_RE = re.compile(
    r"\b(before|after|previously|earlier|past|already|had|"
    r"happened|appeared|was|were|did|used to|when i|while i)\b",
    re.I,
)
CURRENT_QUESTION_RE = re.compile(
    r"\b(currently|right now|now|latest|visible|shown|displayed|"
    r"carrying out|doing now)\b",
    re.I,
)
# Families whose answers often require multi-frame temporal/causal reasoning
# or fine visual verification. Exact text memory can still answer some of
# these, but selection should preferentially keep their historical placements
# as recall candidates rather than collapsing them to memory_direct.
HARD_RECALL_FAMILIES = {
    "CR1", "CR2", "CR4", "CR5", "M1",
    "C1", "STU1", "OJR1", "CR7", "ACR1", "HLD1",
}
SIMPLE_MEMORY_FAMILIES = {
    "N1", "P1", "R1", "CR3", "ACR1", "HLD1",
}

# Production trajectory caps (config.py is the source of truth for max cap)
MAX_QUESTIONS_PER_TRAJECTORY = CONFIG_MAX_QUESTIONS_PER_TRAJECTORY
MIN_QUESTIONS_PER_TRAJECTORY = int(os.environ.get("THINKSTREAM_MIN_QUESTIONS_PER_TRAJECTORY", "10"))
FAMILY_REPEAT_FILL_CAP = int(os.environ.get("THINKSTREAM_FAMILY_REPEAT_FILL_CAP", "5"))
MAX_TRAJECTORIES_PER_VIDEO = 1
AGENT_CHUNK_SEC = 1                  # seconds per chunk
MIN_QUESTION_ASK_GAP_CHUNKS = int(os.environ.get("THINKSTREAM_MIN_QUESTION_ASK_GAP_CHUNKS", "4"))
SHORT_VIDEO_ASK_GAP_CHUNKS = int(os.environ.get("THINKSTREAM_SHORT_VIDEO_ASK_GAP_CHUNKS", "3"))
LONG_VIDEO_ASK_GAP_CHUNKS = int(os.environ.get("THINKSTREAM_LONG_VIDEO_ASK_GAP_CHUNKS", "4"))
LONG_VIDEO_ASK_GAP_AT_CHUNKS = int(os.environ.get("THINKSTREAM_LONG_VIDEO_ASK_GAP_AT_CHUNKS", "180"))
DENSITY_FILL_ASK_GAP_CHUNKS = int(
    os.environ.get("THINKSTREAM_DENSITY_FILL_ASK_GAP_CHUNKS", "3")
)
SPREAD_SCORE_DENOM = 8.0
SPREAD_SCORE_CAP = 3.0

def adaptive_q_count(num_chunks: int) -> int:
    """Question count scales with video length, ~1 question per 10s.

    Short videos: 10 questions where feasible.
    Medium/long videos: roughly one question per 6 chunks, capped by config.
    Very long (200+): 20 questions cap           (q-interval varies)
    """
    target = max(MIN_QUESTIONS_PER_TRAJECTORY,
                 min(
                     MAX_QUESTIONS_PER_TRAJECTORY,
                     max(1, num_chunks // max(1, QUESTION_DENSITY_CHUNKS)),
                 ))
    return target


def density_blocking_span_cap(num_chunks: int) -> int:
    """Largest active question span to reserve before short Q/A density fill."""
    return max(
        24,
        min(
            DENSITY_BLOCKING_SPAN_MAX,
            int(max(1, num_chunks) * DENSITY_BLOCKING_SPAN_FRACTION),
        ),
    )


def question_ask_gap_floor(num_chunks: int) -> int:
    """Minimum spacing between independent user questions on one timeline."""
    if num_chunks < 64:
        return SHORT_VIDEO_ASK_GAP_CHUNKS
    if num_chunks >= LONG_VIDEO_ASK_GAP_AT_CHUNKS:
        return LONG_VIDEO_ASK_GAP_CHUNKS
    return MIN_QUESTION_ASK_GAP_CHUNKS

# ── OVOBench-aligned family→profile mapping ─────────────────────────
#   backward  : evidence in past, often compressed → recall_demo dominant
#   forward   : evidence in future → silent_then_response dominant
#   realtime  : evidence right now → direct dominant
PLACEMENT_PROFILE = {
    # backward (memory / recall)
    "CR1": "backward", "CR2": "backward", "CR4": "backward",
    "CR5": "backward", "N1":  "backward", "P1":  "backward",
    "HLD1": "backward", "M1":  "backward",
    # forward (anticipation / wait)
    "E2":  "forward",
    # realtime (immediate)
    "CR3": "realtime", "CR7": "realtime", "R1":  "realtime",
    "ACR1": "realtime", "STU1": "realtime", "OJR1": "realtime",
    "F5":  "realtime", "C1":  "realtime",
    "PN1": "realtime",
    "F6":  "realtime",
    "CRR1": "realtime",
    # v12.13 (P1-7): F7 moved forward → realtime. New F7 is OVO SSR-style
    # multi_emit Yes/No across [change-K, change+K]; not "wait then answer".
    "F7":  "realtime",
}

# Multi_emit families adoption rate per video (each tossed independently).
# Lower → fewer videos carry narration/counting → multi_emit % drops.
MULTI_EMIT_ADOPT_RATE = 0.5
F5_ADOPT_RATE = float(os.environ.get("THINKSTREAM_F5_ADOPT_RATE", "0.95"))
PN1_ADOPT_RATE = float(os.environ.get("THINKSTREAM_PN1_ADOPT_RATE", "0.20"))

# F7/SSR should be present, but not every video should carry a progress-status
# card. Batch3 landed below OVO SSR scale, so use a higher adoption rate and
# let selection/overlap constraints decide whether the card fits each video.
F7_ADOPT_RATE = 1.0

# CRR1 is the OVO-CRR-like repeated probe around a clue/event becoming true.
# Keep it below every-video generation because each selected card can occupy a
# long active span and would otherwise squeeze out regular QA slots.
CRR1_ADOPT_RATE = 1.0

# Rare benchmark-aligned families can lose greedy selection because their
# active span is longer (F7) or because recall slots are already saturated
# (HLD1). Boosting selection, not generation volume, keeps the card pool
# balanced while making selected trajectories carry the intended coverage.
FAMILY_SELECTION_BOOST = {
    "F7": 1.0,     # OVO SSR is an immediate single Yes/No status row
    "F5": 5.0,     # REC-style cumulative counting, capped by the multi budget
    "CRR1": 5.0,   # clue-before/after multi-probe status
    "PN1": 1.0,    # exploratory live narration should stay small
    "M1": 1.0,     # summary/history support, not the main non-MCQ source
    "CR5": 2.5,    # CRR-style clue waits should survive selection
    "E2": 2.5,     # StreamingBench proactive-output wait examples
    "OJR1": 6.0,   # OVO has a large object-joint-reasoning slice
    "STU1": 6.0,   # state transition understanding is under-selected
    "HLD1": 0.0,   # OVO-HLD exists, but StreamingBench has no abstention task
    "C1": 1.5,     # MC OCR should survive selection without dominating current QA
    "ACR1": 4.0,   # action/causal reasoning should stay near OVO scale
    "N1": 1.5,
    "P1": 1.5,
}

# Keep HLD / "Unable to answer" abstention negatives near the previous
# reasonable family share, but do not force them into every feasible video.
HLD_ABSTENTION_RESERVE_PERCENT = 15

# Timeline silence = silent samples for chunks NOT covered by any active placement.
# Multi-turn trajectory SFT/RL needs a dense chunk timeline, so pass3C must
# emit one silent sample for every non-active, non-compress chunk.

# Question type literal
QuestionType = Literal["single_emit", "multi_emit"]

# Action literal (gold output at any chunk)
GoldKind = Literal["silent", "response"]

# Trajectory placement mechanism
PlacementMechanism = Literal[
    "silent_then_response",  # ask < emit (single_emit only)
    "direct",                # gap small, no recall
    "memory_direct",         # historical state/global support, visual recall not needed
    "recall_demo",           # gap large, insert tool_call
    "multi_emit",            # multi-trigger (counting / narration)
]


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GoldEmit:
    chunk: int
    value: str


@dataclass
class Card:
    card_id: str
    family: str                       # taxonomy from pass3a (F5/F7/M1/PN1/...)
    question: str
    answer_form: str                  # binary | multiple_choice | number | short_exact | descriptive
    question_type: QuestionType
    gold_emits: List[GoldEmit]
    grounding_frames: List[int]
    target_ovo_task: str = ""
    temporal_role: str = ""
    support_policy: str = ""
    allowed_support_policies: Tuple[str, ...] = ()
    recall_eligible: bool = False
    state_memory_required: bool = False
    question_style: str = ""          # benchmark_core | benchmark_variant | ours_unique
    question_way: str = ""            # fine-grained benchmark question pattern
    evidence_type: str = ""           # fine-grained visual/state evidence type
    legacy_family_id: str = ""        # old internal id, e.g. N1/F5/CRR1
    task_family: str = ""             # readable broad group used in reports/prompts
    task_subtype: str = ""            # readable fine-grained subtask
    timing_type: str = ""             # readable evidence/answer timing type
    readable_task_name: str = ""      # task_family / task_subtype / timing_type
    slot_group: str = ""              # current_perception | past_memory | temporal_reasoning | future | multi_state | global_context
    slot_subtype: str = ""            # readable Pass3A subtask, e.g. global_prefix_count
    temporal_bucket: str = ""         # current_direct | past_visual_recall_candidate | future_delayed | multi_answer
    benchmark_source: str = ""        # ovo_* | streamingbench_* | ours_*
    benchmark_task: str = ""          # OCR/REC/SSR/... or StreamingBench family
    answer_behavior: str = ""         # single_mcq | multi_number_prefix_count | ...
    question_goal: str = ""           # teacher-facing slot intent
    placement_hint: str = ""          # placement/recall policy hint
    legal_answer_modes: Tuple[str, ...] = ()
    required_answer_mode: str = ""
    forbidden_answer_modes: Tuple[str, ...] = ()
    answer_mode_reason: str = ""
    # OPTIONAL pre-generated by pass3a so pass3c doesn't re-call LLM:
    recall_query: Optional[Dict] = None  # {"start_time": int, "end_time": int}
    # MC-specific:
    options: Optional[List[str]] = None  # ["A) ...", "B) ...", ...]
    correct_option: Optional[str] = None  # "A" | "B" | ... matching options


@dataclass
class Placement:
    card_id: str
    ask_chunk: int
    mechanism: PlacementMechanism
    difficulty_mode: str = ""
    recall_need: str = ""
    support_policy: str = ""
    # Per-chunk gold actions inside this placement's window
    # (chunk -> (kind, value or "")):
    chunk_actions: Dict[int, Tuple[GoldKind, str]] = field(default_factory=dict)
    # Chunks where the assistant should call recall before the final action.
    # recall_demo response chunks use oracle/noisy historical frames.
    # recall_demo response chunks use oracle/noisy historical frames. Forward
    # silent_then_response waits are plain silent turns; answer support is in
    # the future, so a recall call cannot be the minimal action.
    recall_at: Dict[int, str] = field(default_factory=dict)
    # Optional explanation label for recall_at chunks. This is used only to
    # shape the recall tool-call think text; it does not affect gold timing.
    recall_reason_at: Dict[int, str] = field(default_factory=dict)


def _card_get(card, key: str, default=None):
    if isinstance(card, dict):
        return card.get(key, default)
    return getattr(card, key, default)


def _normalize_policy_values(value) -> Tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        raw_values = re.split(r"[,/| ]+", value)
    else:
        raw_values = list(value)
    out: List[str] = []
    for raw in raw_values:
        policy = str(raw or "").strip().lower()
        if policy in SUPPORT_POLICIES and policy not in out:
            out.append(policy)
    return tuple(out)


def _strict_target_ovo_task(card) -> str:
    family = str(_card_get(card, "family", "") or "").strip()
    target = str(_card_get(card, "target_ovo_task", "") or "").strip().upper()
    if target in STRICT_OVO_TASKS:
        return target
    raw_ovo = str(_card_get(card, "ovo_task", "") or "").strip().upper()
    if raw_ovo in STRICT_OVO_TASKS:
        return raw_ovo
    return FAMILY_TARGET_OVO_TASK.get(family, raw_ovo or "")


def infer_card_policy_fields(card) -> Dict:
    """Infer strict OVO/support-policy metadata from a pass3 card.

    The teacher may omit these fields for old batches. This function makes the
    downstream placement logic deterministic: recall is allowed only for
    concrete historical visual facts, not for cumulative state, future cues, or
    probe/status rows.
    """
    family = str(_card_get(card, "family", "") or "").strip()
    question = str(_card_get(card, "question", "") or "")
    answer_form = str(_card_get(card, "answer_form", "") or "")
    target_ovo_task = _strict_target_ovo_task(card)

    existing_policy = str(_card_get(card, "support_policy", "") or "").strip().lower()
    existing_allowed = _normalize_policy_values(_card_get(card, "allowed_support_policies", ()))
    raw_temporal_role = str(_card_get(card, "temporal_role", "") or "").strip()
    raw_temporal_bucket = str(
        _card_get(card, "temporal_bucket", "")
        or _card_get(card, "timing_type", "")
        or ""
    ).strip()

    has_historical_wording = bool(HISTORICAL_QUESTION_RE.search(question))
    has_current_wording = bool(CURRENT_QUESTION_RE.search(question))

    if family == "F5":
        temporal_role = "cumulative_count"
        support_policy = SUPPORT_HISTORICAL_STATE_MEMORY
        allowed = (SUPPORT_HISTORICAL_STATE_MEMORY,)
        recall_eligible = False
        state_memory_required = True
    elif family == "F7":
        temporal_role = "current_step_status"
        support_policy = SUPPORT_CURRENT_VISUAL
        allowed = (SUPPORT_CURRENT_VISUAL,)
        recall_eligible = False
        state_memory_required = False
    elif family == "CRR1":
        temporal_role = "crr_sufficiency_probe"
        support_policy = SUPPORT_PROBE_STATUS
        allowed = (SUPPORT_PROBE_STATUS,)
        recall_eligible = False
        state_memory_required = True
    elif family == "F6":
        temporal_role = "future_current_cue"
        support_policy = SUPPORT_FUTURE_CURRENT_CUE
        allowed = (SUPPORT_FUTURE_CURRENT_CUE,)
        recall_eligible = False
        state_memory_required = False
    elif family == "PN1":
        temporal_role = "live_narration"
        support_policy = SUPPORT_CURRENT_VISUAL
        allowed = (SUPPORT_CURRENT_VISUAL,)
        recall_eligible = False
        state_memory_required = False
    elif family in {"M1"}:
        temporal_role = "global_summary"
        support_policy = SUPPORT_HISTORICAL_STATE_MEMORY
        allowed = (SUPPORT_HISTORICAL_STATE_MEMORY,)
        recall_eligible = False
        state_memory_required = True
    elif family == "HLD1":
        temporal_role = "historical_abstention_check"
        support_policy = SUPPORT_HISTORICAL_VISUAL_RECALL
        allowed = (SUPPORT_CURRENT_VISUAL, SUPPORT_HISTORICAL_VISUAL_RECALL)
        recall_eligible = True
        state_memory_required = False
    elif family == "CR5":
        temporal_role = "delayed_clue_resolution"
        support_policy = SUPPORT_HISTORICAL_VISUAL_RECALL
        allowed = (
            SUPPORT_CURRENT_VISUAL,
            SUPPORT_HISTORICAL_VISUAL_RECALL,
            SUPPORT_FUTURE_CURRENT_CUE,
        )
        recall_eligible = True
        state_memory_required = False
    elif family in {"E2"}:
        temporal_role = "future_event_wait"
        support_policy = SUPPORT_FUTURE_CURRENT_CUE
        allowed = (SUPPORT_FUTURE_CURRENT_CUE,)
        recall_eligible = False
        state_memory_required = False
    elif (
        family in {"CR1", "CR2", "CR4", "CR5", "C1", "ACR1", "STU1", "OJR1", "CR7", "P1", "N1", "R1"}
        and (has_historical_wording or target_ovo_task in {"EPM", "ASI", "ATR", "HLD"})
    ):
        temporal_role = "historical_visual_detail"
        support_policy = SUPPORT_HISTORICAL_VISUAL_RECALL
        allowed = (SUPPORT_CURRENT_VISUAL, SUPPORT_HISTORICAL_VISUAL_RECALL)
        recall_eligible = True
        state_memory_required = False
    else:
        temporal_role = "current_probe" if has_current_wording else "current_visual"
        support_policy = SUPPORT_CURRENT_VISUAL
        allowed = (SUPPORT_CURRENT_VISUAL,)
        recall_eligible = False
        state_memory_required = False

    if existing_policy in SUPPORT_POLICIES:
        support_policy = existing_policy
    if existing_allowed:
        allowed = existing_allowed
        recall_eligible = SUPPORT_HISTORICAL_VISUAL_RECALL in allowed and family not in {"F5", "F6", "F7", "CRR1", "PN1"}
        state_memory_required = SUPPORT_HISTORICAL_STATE_MEMORY in allowed
    if support_policy == SUPPORT_HISTORICAL_VISUAL_RECALL and SUPPORT_HISTORICAL_VISUAL_RECALL not in allowed:
        allowed = tuple(dict.fromkeys((*allowed, SUPPORT_HISTORICAL_VISUAL_RECALL)))
    if support_policy == SUPPORT_CURRENT_VISUAL and SUPPORT_CURRENT_VISUAL not in allowed:
        allowed = tuple(dict.fromkeys((*allowed, SUPPORT_CURRENT_VISUAL)))
    if support_policy == SUPPORT_FUTURE_CURRENT_CUE and SUPPORT_FUTURE_CURRENT_CUE not in allowed:
        allowed = tuple(dict.fromkeys((*allowed, SUPPORT_FUTURE_CURRENT_CUE)))
    if support_policy == SUPPORT_PROBE_STATUS and SUPPORT_PROBE_STATUS not in allowed:
        allowed = tuple(dict.fromkeys((*allowed, SUPPORT_PROBE_STATUS)))

    # Planned-slot timing is a hard contract. Some older cards carried broad
    # allowed_support_policies even when the slot itself was current/future.
    # Narrow them here so pass3b cannot satisfy ratio pressure by moving a
    # current question into recall, or a future-trigger question into direct.
    if family not in {"F5", "F7", "CRR1", "PN1", "E2", "F6", "M1", "HLD1"}:
        if raw_temporal_role in _CURRENT_TEMPORAL_ROLES or raw_temporal_bucket in _CURRENT_TEMPORAL_BUCKETS:
            temporal_role = raw_temporal_role or temporal_role
            support_policy = SUPPORT_CURRENT_VISUAL
            allowed = (SUPPORT_CURRENT_VISUAL,)
            recall_eligible = False
            state_memory_required = False
    if family == "E2" or raw_temporal_role in _FUTURE_WAIT_TEMPORAL_ROLES or raw_temporal_bucket in _FUTURE_WAIT_TEMPORAL_BUCKETS:
        if family in {"E2", "CR5"}:
            temporal_role = raw_temporal_role or temporal_role
            support_policy = SUPPORT_FUTURE_CURRENT_CUE
            allowed = (SUPPORT_FUTURE_CURRENT_CUE,)
            recall_eligible = False
            state_memory_required = False
    if family == "F6" or raw_temporal_role in _FUTURE_CUE_TEMPORAL_ROLES or raw_temporal_bucket in _FUTURE_CUE_TEMPORAL_BUCKETS:
        if family in {"F6", "CR5"}:
            temporal_role = raw_temporal_role or temporal_role
            support_policy = SUPPORT_FUTURE_CURRENT_CUE
            allowed = (SUPPORT_FUTURE_CURRENT_CUE,)
            recall_eligible = False
            state_memory_required = False

    # Binary/number cards can still be state/probe tasks even when teacher
    # wording is loose; keep F5/F7/CRR guarded above and avoid making generic
    # Yes/No current questions recall-eligible solely due to "was" wording.
    if answer_form in {"binary", "number"} and family not in {"HLD1", "F5", "F7", "CRR1"}:
        if support_policy == SUPPORT_HISTORICAL_VISUAL_RECALL and not has_historical_wording:
            support_policy = SUPPORT_CURRENT_VISUAL
            allowed = (SUPPORT_CURRENT_VISUAL,)
            recall_eligible = False

    return {
        "target_ovo_task": target_ovo_task,
        "temporal_role": temporal_role,
        "support_policy": support_policy,
        "allowed_support_policies": allowed,
        "recall_eligible": bool(recall_eligible),
        "state_memory_required": bool(state_memory_required),
    }


def _normalize_answer_modes(value) -> Tuple[str, ...]:
    if not value:
        return ()
    raw_values = [value] if isinstance(value, str) else list(value)
    out: List[str] = []
    for raw in raw_values:
        mode = str(raw or "").strip().lower()
        if mode in ANSWER_MODES and mode not in out:
            out.append(mode)
    return tuple(out)


def infer_card_answer_mode_constraints(card) -> Dict[str, object]:
    """Explicit card-level answer-mode contract.

    The selector can move some cards between current, historical, and future
    placements, but only inside this legal answer-mode set. This prevents
    ratio pressure from turning OCR/current/probe cards into recall examples or
    state/multi cards into visual-recall examples.
    """
    family = str(_card_get(card, "family", "") or "").strip()
    qtype = str(_card_get(card, "question_type", "") or "").strip()
    answer = str(_card_get(card, "canonical_answer", "") or "").strip().lower()
    policy = infer_card_policy_fields(card)
    allowed = set(policy["allowed_support_policies"])

    if family == "HLD1" or answer == "unable to answer":
        return {
            "legal_answer_modes": (ANSWER_MODE_HLD_RECALL,),
            "required_answer_mode": ANSWER_MODE_HLD_RECALL,
            "forbidden_answer_modes": (
                ANSWER_MODE_DIRECT,
                ANSWER_MODE_RECALL,
                ANSWER_MODE_FUTURE,
                ANSWER_MODE_MULTI,
                ANSWER_MODE_STATE_DIRECT,
                ANSWER_MODE_ABSTAIN,
            ),
            "answer_mode_reason": "hld_full_history_absence_check_not_normal_recall",
        }
    if qtype == "multi_emit" or family in {"F5", "CRR1", "PN1"}:
        return {
            "legal_answer_modes": (ANSWER_MODE_MULTI,),
            "required_answer_mode": ANSWER_MODE_MULTI,
            "forbidden_answer_modes": (
                ANSWER_MODE_DIRECT,
                ANSWER_MODE_RECALL,
                ANSWER_MODE_FUTURE,
                ANSWER_MODE_STATE_DIRECT,
                ANSWER_MODE_ABSTAIN,
            ),
            "answer_mode_reason": "stream_state_probe_multi_answer_only",
        }
    if SUPPORT_HISTORICAL_STATE_MEMORY in allowed or family in {"M1"}:
        return {
            "legal_answer_modes": (ANSWER_MODE_STATE_DIRECT,),
            "required_answer_mode": ANSWER_MODE_STATE_DIRECT,
            "forbidden_answer_modes": (
                ANSWER_MODE_RECALL,
                ANSWER_MODE_FUTURE,
                ANSWER_MODE_MULTI,
                ANSWER_MODE_ABSTAIN,
            ),
            "answer_mode_reason": "state_memory_direct_hard_negative_for_recall",
        }
    if family == "F6":
        return {
            "legal_answer_modes": (ANSWER_MODE_DIRECT,),
            "required_answer_mode": ANSWER_MODE_DIRECT,
            "forbidden_answer_modes": (
                ANSWER_MODE_RECALL,
                ANSWER_MODE_FUTURE,
                ANSWER_MODE_MULTI,
                ANSWER_MODE_ABSTAIN,
                ANSWER_MODE_STATE_DIRECT,
            ),
            "answer_mode_reason": "future_current_cue_answer_now",
        }
    if family == "E2":
        return {
            "legal_answer_modes": (ANSWER_MODE_FUTURE,),
            "required_answer_mode": ANSWER_MODE_FUTURE,
            "forbidden_answer_modes": (
                ANSWER_MODE_DIRECT,
                ANSWER_MODE_RECALL,
                ANSWER_MODE_MULTI,
                ANSWER_MODE_ABSTAIN,
                ANSWER_MODE_STATE_DIRECT,
            ),
            "answer_mode_reason": "future_wait_until_trigger",
        }
    if family == "CR5":
        return {
            "legal_answer_modes": (
                ANSWER_MODE_DIRECT,
                ANSWER_MODE_RECALL,
                ANSWER_MODE_FUTURE,
            ),
            "required_answer_mode": "",
            "forbidden_answer_modes": (
                ANSWER_MODE_MULTI,
                ANSWER_MODE_ABSTAIN,
                ANSWER_MODE_STATE_DIRECT,
            ),
            "answer_mode_reason": "delayed_clue_can_be_current_recall_or_wait",
        }
    if SUPPORT_HISTORICAL_VISUAL_RECALL in allowed and policy["recall_eligible"]:
        return {
            "legal_answer_modes": (ANSWER_MODE_DIRECT, ANSWER_MODE_RECALL),
            "required_answer_mode": "",
            "forbidden_answer_modes": (
                ANSWER_MODE_FUTURE,
                ANSWER_MODE_MULTI,
                ANSWER_MODE_STATE_DIRECT,
                ANSWER_MODE_ABSTAIN,
            ),
            "answer_mode_reason": "historical_visual_card_direct_or_recall_by_gap",
        }
    return {
        "legal_answer_modes": (ANSWER_MODE_DIRECT,),
        "required_answer_mode": ANSWER_MODE_DIRECT,
        "forbidden_answer_modes": (
            ANSWER_MODE_RECALL,
            ANSWER_MODE_FUTURE,
            ANSWER_MODE_MULTI,
            ANSWER_MODE_STATE_DIRECT,
            ANSWER_MODE_ABSTAIN,
        ),
        "answer_mode_reason": "current_visual_direct_only",
    }


def infer_card_semantic_fields(card) -> Dict[str, str]:
    """Infer fine-grained benchmark question/evidence buckets.

    These fields are separate from family. A family says what skill generated
    the card; question_way/evidence_type say which benchmark-style surface form
    and visual evidence type the selector should balance.
    """
    family = str(_card_get(card, "family", "") or "").strip()
    q = str(_card_get(card, "question", "") or "").lower()
    raw_way = str(_card_get(card, "question_way", "") or "").strip()
    raw_evidence = str(_card_get(card, "evidence_type", "") or "").strip()

    way = raw_way if raw_way in QUESTION_WAYS else FAMILY_DEFAULT_QUESTION_WAY.get(
        family, QUESTION_WAY_OBJECT_ATTRIBUTE
    )
    evidence = raw_evidence if raw_evidence in EVIDENCE_TYPES else FAMILY_DEFAULT_EVIDENCE_TYPE.get(
        family, EVIDENCE_OBJECT_ATTRIBUTE
    )

    if "previous question" in q or "mentioned in the previous" in q or "first question" in q:
        way = QUESTION_WAY_SEQUENTIAL_REFERENCE
        evidence = EVIDENCE_PERSON_RELATION
    elif re.search(r"\bwhen\b.*\boutput\b", q):
        way = QUESTION_WAY_PROACTIVE_OUTPUT
        evidence = EVIDENCE_FUTURE_TRIGGER
    elif re.search(r"\bhow many\b|\bcount\b|\bso far\b", q):
        way = QUESTION_WAY_REPEATED_COUNT if family == "F5" else way
        evidence = EVIDENCE_REPEATED_EVENT if family == "F5" else evidence
    elif re.search(r"\bwhat (?:text|word|logo|number|sign|label)\b|\bshown on\b|\bdisplayed\b", q):
        way = QUESTION_WAY_TEXT_READOUT if family == "C1" else way
        evidence = EVIDENCE_TEXT_OCR if family == "C1" else evidence
    elif re.search(r"\bwhere\b|\brelative position\b|\bleft of\b|\bright of\b|\bin relation to\b", q):
        if family in {"STU1", "OJR1", "CR7", "R1"}:
            way = QUESTION_WAY_SPATIAL_RELATION
            evidence = EVIDENCE_SPATIAL_RELATION
    elif re.search(r"\bwhy\b|\bcause\b|\bpurpose\b|\bintend\b|\btrying to\b", q):
        way = QUESTION_WAY_CAUSAL_INTENT
        evidence = EVIDENCE_CAUSAL_CONTEXT
    elif re.search(r"\bemotion\b|\bmood\b|\bfeeling\b|\bflustered\b|\bexcited\b", q):
        way = QUESTION_WAY_EMOTION_CONTEXT
        evidence = EVIDENCE_EMOTION_CONTEXT
    elif re.search(r"\bwhich (?:source|cue|evidence)\b|\bvisual source\b|\bfrom the (?:audio|image|video|text|caption)\b", q):
        way = QUESTION_WAY_SOURCE_DISCRIMINATION
        evidence = EVIDENCE_SOURCE_DISCRIMINATION
    elif re.search(r"\balign(?:ed|ment)?\b|\bmatch(?:es|ed)?\b|\bcontradict(?:s|ed|ion)?\b|\bconsistent with\b", q):
        way = QUESTION_WAY_MULTIMODAL_ALIGNMENT
        evidence = EVIDENCE_MULTIMODAL_ALIGNMENT
    elif re.search(r"\bnext\b|\bwill\b|\blikely\b|\babout to\b", q) and family == "F6":
        way = QUESTION_WAY_FUTURE_PREDICTION
        evidence = EVIDENCE_FUTURE_CUE

    return {"question_way": way, "evidence_type": evidence}


# ---------------------------------------------------------------------------
# Gold function — the ONE source of truth for benchmark and SFT label
# ---------------------------------------------------------------------------


def gold_action_at(
    c: int, ask_chunk: int, emits: List[GoldEmit], question_type: QuestionType,
) -> Tuple[GoldKind, str]:
    """Pure function. Returns (kind, value).

    single_emit rule:
      response_chunk = max(ask_chunk, emits[0].chunk)
      c == response_chunk → response(value); else silent
      → covers (a) ask after evidence: respond at ask
              (b) ask before evidence: respond at emit
              (c) deep history (recall demo): respond at ask, mechanism inserts recall

    multi_emit rule:
      c == any emit.chunk AND c >= ask → response(emit.value)
      else silent
    """
    if c < ask_chunk:
        return ("silent", "")
    if question_type == "single_emit":
        if not emits:
            return ("silent", "")
        response_chunk = max(ask_chunk, emits[0].chunk)
        if c == response_chunk:
            return ("response", emits[0].value)
        return ("silent", "")
    # multi_emit
    for e in emits:
        if e.chunk == c:
            return ("response", e.value)
    return ("silent", "")


def gold_window_for_card(card: Card, ask_chunk: int, num_chunks: int) -> Dict[int, Tuple[GoldKind, str]]:
    """Materialize gold_action over the chunks where this card is 'active'.

    single_emit: [ask, response_chunk + 1] where response_chunk = max(ask, emit)
    multi_emit:  [ask, last_emit + 1]
    The +1 gives one trailing silent for post-response supervision.
    """
    if not card.gold_emits:
        return {}
    if card.question_type == "single_emit":
        response_chunk = max(ask_chunk, card.gold_emits[0].chunk)
        end = min(num_chunks - 1, response_chunk + 1)
    else:
        last_emit = max(e.chunk for e in card.gold_emits)
        end = min(num_chunks - 1, last_emit + 1)
    actions: Dict[int, Tuple[GoldKind, str]] = {}
    for c in range(ask_chunk, end + 1):
        actions[c] = gold_action_at(c, ask_chunk, card.gold_emits, card.question_type)
    return actions


# ---------------------------------------------------------------------------
# Placement — model-agnostic; pure function of (card, num_chunks, rng)
# ---------------------------------------------------------------------------


def _make_placement(
    card: Card,
    ask: int,
    num_chunks: int,
    mech: PlacementMechanism,
    *,
    difficulty_mode: str = "",
    recall_need: str = "",
    support_policy: str = "",
) -> Placement:
    return Placement(
        card_id=card.card_id,
        ask_chunk=ask,
        mechanism=mech,
        difficulty_mode=difficulty_mode,
        recall_need=recall_need,
        support_policy=support_policy or infer_card_policy_fields(card)["support_policy"],
        chunk_actions=gold_window_for_card(card, ask, num_chunks),
    )


def _randint_safe(rng: random.Random, lo: int, hi: int) -> int:
    """Inclusive randint with safe handling when lo >= hi."""
    if hi <= lo:
        return lo
    return rng.randint(lo, hi)


def _ask_from_band(emit: int, band: Tuple[int, int], num_chunks: int,
                   rng: random.Random, sign: int) -> Optional[int]:
    """Pick an ask chunk inside the given (lo, hi) gap band.

    sign = +1 → ask = emit + gap (placement after emit)
    sign = -1 → ask = emit - lead (placement before emit)
    Returns None if band has no feasible gap.
    """
    lo, hi = band
    if sign > 0:
        max_feasible = num_chunks - 1 - emit
        hi_eff = min(hi, max_feasible)
        if hi_eff < lo:
            return None
        gap = _randint_safe(rng, lo, hi_eff)
        return emit + gap
    else:
        max_feasible = emit
        hi_eff = min(hi, max_feasible)
        if hi_eff < lo:
            return None
        lead = _randint_safe(rng, lo, hi_eff)
        return emit - lead


def _dedupe_placements(placements: List[Placement]) -> List[Placement]:
    out: List[Placement] = []
    seen: set = set()
    for p in placements:
        key = (p.card_id, p.ask_chunk, p.mechanism)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def place_single_emit(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    """Generate availability variants for one single-emit card.

    Support policy decides which availability variants are legal. OVO-like
    current perception stays direct; historical visual detail may become recall
    only when asked after the visual window; state/count/probe tasks do not
    create visual recall supervision.
    """
    if not card.gold_emits:
        return []
    emit = card.gold_emits[0].chunk
    placements: List[Placement] = []
    policy = infer_card_policy_fields(card)
    allowed = set(policy["allowed_support_policies"])

    # Fresh/current direct placement. This is legal for normal current visual
    # tasks and also for future-current-cue prompts such as FPD, where the
    # current cue is the evidence.
    if SUPPORT_CURRENT_VISUAL in allowed or SUPPORT_FUTURE_CURRENT_CUE in allowed:
        placements.append(_make_placement(
            card, emit, num_chunks, "direct",
            difficulty_mode=(
                "future_current_cue"
                if SUPPORT_FUTURE_CURRENT_CUE in allowed and SUPPORT_CURRENT_VISUAL not in allowed
                else "current_at_support"
            ),
            recall_need="current_window",
            support_policy=(
                SUPPORT_FUTURE_CURRENT_CUE
                if SUPPORT_FUTURE_CURRENT_CUE in allowed and SUPPORT_CURRENT_VISUAL not in allowed
                else SUPPORT_CURRENT_VISUAL
            ),
        ))
        band_choice = (SE_FRESH_TRIVIAL, SE_FRESH_EASY, SE_FRESH_MEDIUM)[
            stable_mod(card.card_id, "direct", modulo=3)
        ]
        ask = _ask_from_band(emit, band_choice, num_chunks, rng, sign=+1)
        if ask is None:
            ask = _ask_from_band(emit, SE_FRESH_TRIVIAL, num_chunks, rng, sign=+1)
        if ask is not None:
            placements.append(_make_placement(
                card, ask, num_chunks, "direct",
                difficulty_mode=(
                    "future_current_cue"
                    if SUPPORT_FUTURE_CURRENT_CUE in allowed and SUPPORT_CURRENT_VISUAL not in allowed
                    else "current_direct"
                ),
                recall_need="current_window",
                support_policy=(
                    SUPPORT_FUTURE_CURRENT_CUE
                    if SUPPORT_FUTURE_CURRENT_CUE in allowed and SUPPORT_CURRENT_VISUAL not in allowed
                    else SUPPORT_CURRENT_VISUAL
                ),
            ))

    # Historical state/memory tasks are answered from trajectory state, not
    # from injected visual recall frames.
    if SUPPORT_HISTORICAL_STATE_MEMORY in allowed:
        ask_mem = _ask_from_band(emit, SE_RECALL_NEAR, num_chunks, rng, sign=+1)
        if ask_mem is not None:
            placements.append(_make_placement(
                card, ask_mem, num_chunks, "memory_direct",
                difficulty_mode="state_memory_direct",
                recall_need="state_memory_enough",
                support_policy=SUPPORT_HISTORICAL_STATE_MEMORY,
            ))

    # Historical visual-recall variants only for concrete visual facts. Near
    # recall is kept because OVO often asks just beyond the active window.
    if SUPPORT_HISTORICAL_VISUAL_RECALL in allowed and policy["recall_eligible"]:
        ask_near = _ask_from_band(emit, SE_RECALL_NEAR, num_chunks, rng, sign=+1)
        if ask_near is not None:
            placements.append(_make_placement(
                card, ask_near, num_chunks, "recall_demo",
                difficulty_mode="recall_near",
                recall_need=(
                    "abstention_visual_check"
                    if card.family == "HLD1"
                    else "fine_visual_verification"
                ),
                support_policy=SUPPORT_HISTORICAL_VISUAL_RECALL,
            ))
        ask_mid = _ask_from_band(emit, SE_RECALL_MID, num_chunks, rng, sign=+1)
        if ask_mid is not None:
            placements.append(_make_placement(
                card, ask_mid, num_chunks, "recall_demo",
                difficulty_mode="recall_mid",
                recall_need="historical_visual",
                support_policy=SUPPORT_HISTORICAL_VISUAL_RECALL,
            ))
        ask_deep = _ask_from_band(emit, SE_RECALL_DEEP, num_chunks, rng, sign=+1)
        if ask_deep is not None:
            placements.append(_make_placement(
                card, ask_deep, num_chunks, "recall_demo",
                difficulty_mode="recall_deep",
                recall_need="compressed_history",
                support_policy=SUPPORT_HISTORICAL_VISUAL_RECALL,
            ))

    profile = PLACEMENT_PROFILE.get(card.family, "realtime")
    if (
        SUPPORT_FUTURE_CURRENT_CUE in allowed
        and (profile == "forward" or card.family in {"CR2", "CR5", "E2"})
        and card.family != "F6"
    ):
        # Keep explicit wait/silent supervision for naturally temporal
        # families without making every static attribute question a future
        # prediction prompt.
        lead_bands = [SE_LEAD_SHORT, SE_LEAD_LONG]
        if card.family == "CR5":
            lead_bands.extend(SE_LEAD_OVO_CRR)
        for band in lead_bands:
            ask_forward = _ask_from_band(emit, band, num_chunks, rng, sign=-1)
            if ask_forward is not None:
                placements.append(_make_placement(
                    card, ask_forward, num_chunks, "silent_then_response",
                    difficulty_mode=(
                        "ovo_crr_wait" if card.family == "CR5" and band in SE_LEAD_OVO_CRR
                        else "future_wait"
                    ),
                    recall_need="future_not_available",
                    support_policy=SUPPORT_FUTURE_CURRENT_CUE,
                ))
        if not any(p.mechanism == "silent_then_response" for p in placements):
            ask_forward = _ask_from_band(emit, (4, 8), num_chunks, rng, sign=-1)
            if ask_forward is not None:
                placements.append(_make_placement(
                    card, ask_forward, num_chunks, "silent_then_response",
                    difficulty_mode="future_wait",
                    recall_need="future_not_available",
                    support_policy=SUPPORT_FUTURE_CURRENT_CUE,
                ))

    if not placements:
        # very short video — fallback to direct
        gap = _randint_safe(rng, 0, max(0, num_chunks - 1 - emit))
        placements.append(_make_placement(
            card, emit + gap, num_chunks, "direct",
            difficulty_mode="current_direct",
            recall_need="fallback",
            support_policy=SUPPORT_CURRENT_VISUAL,
        ))

    return _dedupe_placements(placements)


def _select_multi_emit_subset(card: Card) -> List[GoldEmit]:
    """Pick a compact local subset for one active multi-answer episode."""
    emits = sorted(card.gold_emits, key=lambda e: e.chunk)
    if card.family == "F5":
        best: List[GoldEmit] = []
        best_key: Tuple[int, int, int, int] = (-1, -1, -10**9, -10**9)
        min_responses = min(MIN_BENCH_MULTI_EMIT_RESPONSES, MAX_REC_EMIT_RESPONSES)
        for i, start in enumerate(emits):
            cur = [
                e for e in emits[i:]
                if e.chunk - start.chunk <= MAX_REC_ACTIVE_SPAN
            ][:MAX_REC_EMIT_RESPONSES]
            if len(cur) < 2:
                continue
            span = cur[-1].chunk - cur[0].chunk
            # Prefer more cumulative probes, then a compact span, then earlier
            # placement. The answer values remain the original cumulative
            # counts, so this still trains state memory without holding a query
            # open from frame 0 across the whole video.
            key = (
                1 if len(cur) >= min_responses else 0,
                len(cur),
                -span,
                -cur[0].chunk,
            )
            if key > best_key:
                best_key = key
                best = cur
        if best:
            return best
        if len(emits) >= 2:
            pair = min(
                zip(emits, emits[1:]),
                key=lambda ab: (ab[1].chunk - ab[0].chunk, ab[0].chunk),
            )
            return [pair[0], pair[1]]
        return emits

    if card.family == "F7":
        first_yes_idx = next(
            (i for i, e in enumerate(emits)
             if str(e.value).strip().lower() == "yes"),
            None,
        )
        if first_yes_idx is not None:
            first_yes = emits[first_yes_idx]
            no_before = [
                e for e in emits[:first_yes_idx]
                if str(e.value).strip().lower() == "no"
                and first_yes.chunk - e.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ][-2:]
            no_after = [
                e for e in emits[first_yes_idx + 1:]
                if str(e.value).strip().lower() == "no"
                and e.chunk - first_yes.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ][:2]
            yes_cluster = [
                e for e in emits[first_yes_idx:]
                if str(e.value).strip().lower() == "yes"
                and e.chunk - first_yes.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ][:5]
            subset = sorted(
                {int(e.chunk): e for e in (no_before + yes_cluster + no_after)}.values(),
                key=lambda e: e.chunk,
            )[:MAX_MULTI_EMIT_RESPONSES]
            vals = {str(e.value).strip().lower() for e in subset}
            if (
                {"no", "yes"}.issubset(vals)
                and subset[-1].chunk - subset[0].chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ):
                return subset

    if card.family == "CRR1":
        first_yes_idx = next(
            (i for i, e in enumerate(emits)
             if str(e.value).strip().lower() == "yes"),
            None,
        )
        if first_yes_idx is not None and first_yes_idx > 0:
            first_yes = emits[first_yes_idx]
            no_before = [
                e for e in emits[:first_yes_idx]
                if str(e.value).strip().lower() == "no"
                and first_yes.chunk - e.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ][-2:]
            yes_after = [
                e for e in emits[first_yes_idx + 1:]
                if str(e.value).strip().lower() == "yes"
                and e.chunk - first_yes.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ]
            far_yes = next(
                (
                    e for e in yes_after
                    if STATUS_FAR_AFTER_MIN_GAP <= e.chunk - first_yes.chunk <= STATUS_FAR_AFTER_MAX_GAP
                ),
                None,
            )
            if far_yes is not None:
                middle_yes = next(
                    (
                        e for e in yes_after
                        if e.chunk < far_yes.chunk
                        and e.chunk - first_yes.chunk <= VISUAL_WINDOW_CHUNKS
                    ),
                    None,
                )
                subset = no_before + [first_yes]
                if middle_yes is not None:
                    subset.append(middle_yes)
                subset.append(far_yes)
                subset = sorted(
                    {int(e.chunk): e for e in subset}.values(),
                    key=lambda e: e.chunk,
                )
            else:
                subset = sorted(
                    {int(e.chunk): e for e in (no_before + [first_yes] + yes_after)}.values(),
                    key=lambda e: e.chunk,
                )
            if len(subset) < MAX_CRR_EMIT_RESPONSES:
                present = {int(e.chunk) for e in subset}
                fill = [
                    e for e in emits
                    if int(e.chunk) not in present
                    and abs(int(e.chunk) - int(first_yes.chunk)) <= MAX_MULTI_EMIT_ACTIVE_SPAN
                ]
                subset = sorted(
                    {int(e.chunk): e for e in (subset + fill)}.values(),
                    key=lambda e: e.chunk,
                )
            subset = subset[:MAX_CRR_EMIT_RESPONSES]
            vals = {str(e.value).strip().lower() for e in subset}
            if (
                {"no", "yes"}.issubset(vals)
                and subset[-1].chunk - subset[0].chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
            ):
                return subset

    if len(emits) <= MAX_MULTI_EMIT_RESPONSES:
        if emits[-1].chunk - emits[0].chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN:
            return emits

    best: List[GoldEmit] = []
    best_key: Tuple[int, int, int] = (-1, 10**9, 10**9)
    for i, start in enumerate(emits):
        cur = [
            e for e in emits[i:]
            if e.chunk - start.chunk <= MAX_MULTI_EMIT_ACTIVE_SPAN
        ][:MAX_MULTI_EMIT_RESPONSES]
        if len(cur) < 2:
            continue
        span = cur[-1].chunk - cur[0].chunk
        key = (len(cur), -span, -cur[0].chunk)
        if key > best_key:
            best_key = key
            best = cur
    return best


def _select_f5_global_query_subset(card: Card) -> List[GoldEmit]:
    """Prefix-style REC subset for OVO's chunk-0 cumulative-count protocol."""
    emits = sorted(
        {int(e.chunk): e for e in card.gold_emits}.values(),
        key=lambda e: e.chunk,
    )
    if len(emits) < 2:
        return []
    if stable_mod(card.card_id, "f5_global_query", modulo=100) >= max(0, F5_GLOBAL_QUERY_PERCENT):
        return []
    first = int(emits[0].chunk)
    if first > max(0, F5_GLOBAL_QUERY_MAX_FIRST_CHUNK):
        return []
    subset = [
        e for e in emits
        if int(e.chunk) <= max(0, F5_GLOBAL_QUERY_MAX_ACTIVE_SPAN)
    ][:MAX_REC_GLOBAL_EMIT_RESPONSES]
    return subset if len(subset) >= 2 else []


def place_multi_emit(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    """Generate ONE compact multi-answer placement.

    Most multi-answer questions remain local episodes. F5/REC additionally gets
    a bounded chunk-0 prefix variant because OVO REC asks once at the beginning
    and scores cumulative counts later in the same video.
    """
    if not card.gold_emits:
        return []
    f5_global = False
    if card.family == "F5":
        forced_global = (
            str(getattr(card, "slot_subtype", "") or "") == "global_prefix_count"
            or str(getattr(card, "answer_behavior", "") or "") == "multi_number_prefix_count"
        )
        if forced_global:
            emits = sorted(
                {int(e.chunk): e for e in card.gold_emits}.values(),
                key=lambda e: e.chunk,
            )[:MAX_REC_GLOBAL_EMIT_RESPONSES]
        else:
            emits = _select_f5_global_query_subset(card)
        f5_global = bool(emits)
        if not emits:
            emits = _select_multi_emit_subset(card)
    else:
        emits = _select_multi_emit_subset(card)
    emits = sorted(
        {int(e.chunk): e for e in emits}.values(),
        key=lambda e: int(e.chunk),
    )
    if len(emits) < 2:
        return []
    first = min(e.chunk for e in emits)
    last = max(e.chunk for e in emits)
    difficulty_mode = "multi_emit"
    if card.family == "F5":
        if f5_global:
            ask = 0
            difficulty_mode = "ovo_rec_cumulative_from_start"
        else:
            # Keep most REC/counting as a local probe episode. The response
            # values stay cumulative over the video so the model must use state
            # memory, but the query does not monopolize every trajectory.
            ask = max(0, first - min(F5_ASK_LEAD_CHUNKS, first))
            difficulty_mode = "ovo_rec_cumulative"
    elif card.family in {"F7", "CRR1"}:
        # Status probes are immediate Yes/No checks at probe time, not a
        # persistent wait-before-first-answer question.
        ask = first
        difficulty_mode = "status_probe"
    else:
        lead = _randint_safe(rng, ME_LEAD_RANGE[0], min(ME_LEAD_RANGE[1], first))
        ask = max(0, first - lead)
    emit_by_chunk = {e.chunk: e.value for e in emits}
    end = min(num_chunks - 1, last + 1)
    actions: Dict[int, Tuple[GoldKind, str]] = {}
    for c in range(ask, end + 1):
        if c in emit_by_chunk:
            actions[c] = ("response", emit_by_chunk[c])
        else:
            actions[c] = ("silent", "")
    return [Placement(
        card_id=card.card_id,
        ask_chunk=ask,
        mechanism="multi_emit",
        difficulty_mode=difficulty_mode,
        support_policy=infer_card_policy_fields(card)["support_policy"],
        chunk_actions=actions,
    )]


def place_card(card: Card, num_chunks: int, rng: random.Random) -> List[Placement]:
    if card.question_type == "single_emit":
        return place_single_emit(card, num_chunks, rng)
    return place_multi_emit(card, num_chunks, rng)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "best", "by", "chunk",
    "did", "does", "for", "from", "has", "in", "is", "it", "of", "on",
    "or", "the", "this", "to", "was", "what", "which", "with",
}


def _strip_mc_label(text: str) -> str:
    return re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][\).:])\s*", "", str(text or "")).strip()


def _card_answer_text(card: Card) -> str:
    if card.answer_form == "multiple_choice" and card.options and card.correct_option:
        idx = ord(str(card.correct_option).strip().upper()[:1] or "A") - ord("A")
        if 0 <= idx < len(card.options):
            return _strip_mc_label(card.options[idx])
    if card.gold_emits:
        return str(card.gold_emits[-1].value or "")
    return ""


def _tokens(text: str) -> List[str]:
    toks = _TOKEN_RE.findall(str(text or "").lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _evidence_text_for_chunks(evidence_by_chunk: Dict[int, Dict], chunks: Iterable[int]) -> str:
    parts: List[str] = []
    for c in chunks:
        cap = evidence_by_chunk.get(int(c)) or {}
        parts.append(str(cap.get("think", "")))
        for ent in cap.get("visible_entities") or []:
            if isinstance(ent, dict):
                parts.append(str(ent.get("desc", "")))
                parts.append(str(ent.get("action", "")))
            else:
                parts.append(str(ent))
        for fact in cap.get("atomic_facts") or []:
            parts.append(str(fact.get("fact", "")) if isinstance(fact, dict) else str(fact))
        for ocr in cap.get("ocr") or []:
            parts.append(str(ocr.get("text", "")) if isinstance(ocr, dict) else str(ocr))
        parts.append(str(cap.get("spatial", "")))
    return " ".join(parts).lower()


def _answer_terms_present(card: Card, text: str) -> bool:
    answer = _card_answer_text(card)
    answer_norm = " ".join(_tokens(answer))
    if not answer_norm:
        return False
    text_l = str(text or "").lower()
    answer_tokens = _tokens(answer)
    if card.answer_form in {"short_exact", "number", "binary"}:
        return answer_norm in " ".join(_tokens(text_l))
    if card.answer_form == "multiple_choice":
        # Need at least two meaningful option tokens unless the answer is an
        # exact OCR-like string. This avoids relabeling on common words.
        if len(answer_tokens) == 1:
            return answer_tokens[0] in _tokens(text_l) and len(answer_tokens[0]) >= 4
        hit = sum(1 for t in set(answer_tokens) if t in _tokens(text_l))
        return hit >= min(2, len(set(answer_tokens)))
    hit = sum(1 for t in set(answer_tokens) if t in _tokens(text_l))
    return hit >= max(2, min(4, len(set(answer_tokens))))


def _correct_option_text(card: Card) -> str:
    correct = str(card.correct_option or "").strip().upper()
    options = list(card.options or [])
    if len(correct) != 1 or correct < "A" or correct > "Z" or not options:
        return ""
    idx = ord(correct) - ord("A")
    if idx < 0 or idx >= len(options):
        return ""
    return _strip_mc_label(str(options[idx]))


def _is_unanswerable_card(card: Card) -> bool:
    if card.family == "HLD1":
        return True
    return _correct_option_text(card).strip().lower() == "unable to answer"


def _support_span(card: Card) -> int:
    support = [int(x) for x in (card.grounding_frames or [])]
    if not support:
        return 0
    return max(support) - min(support) + 1


def refine_placements_with_evidence(
    card: Card,
    placements: List[Placement],
    evidence: Optional[List[Dict]] = None,
) -> List[Placement]:
    """Low-cost first-turn availability refinement.

    This is deliberately heuristic and local. It prevents obvious bad recall
    labels when the answer is already visible in the current window or when a
    simple factual answer is explicitly present in recent text memory, while
    preserving recall candidates for temporal/causal/fine-grained families.
    The expensive semantic judge can be added later as an optional parallel
    pass over these already-filtered candidates.
    """
    if not evidence or card.question_type != "single_emit":
        return _dedupe_placements(placements)

    evidence_by_chunk = {
        int(cap.get("chunk_idx", -1)): cap
        for cap in evidence
        if cap.get("chunk_idx") is not None
    }
    refined: List[Placement] = []
    for p in placements:
        if p.mechanism != "recall_demo":
            refined.append(p)
            continue

        current_lo = max(0, int(p.ask_chunk) - VISUAL_WINDOW_CHUNKS + 1)
        current_text = _evidence_text_for_chunks(
            evidence_by_chunk, range(current_lo, int(p.ask_chunk) + 1)
        )
        if _answer_terms_present(card, current_text):
            # This historical recall candidate is low value: the answer text
            # already appears in the current model-visible context. Do not
            # relabel it as memory_direct, because memory_direct is reserved
            # for explicit state-memory policies (REC/global state). The card's
            # normal current/direct placement remains available.
            continue

        memory_hi = max(0, int(p.ask_chunk) - VISUAL_WINDOW_CHUNKS)
        memory_text = _evidence_text_for_chunks(evidence_by_chunk, range(0, memory_hi))
        simple_memory_case = (
            card.family in SIMPLE_MEMORY_FAMILIES
            and _support_span(card) <= 4
            and _answer_terms_present(card, memory_text)
        )
        if simple_memory_case:
            # In multi-turn training, compact text memory is only a lossy
            # state prior; old visual detail may no longer be recoverable from
            # KV. Keep the recall candidate when evidence is outside the
            # current visual window instead of demoting it solely because the
            # answer words appear in historical text.
            p.recall_need = p.recall_need or "text_memory_but_outside_kv"
            refined.append(p)
            continue

        # Keep as recall. Hard families and multi-support questions are the
        # main source of temporal/order/causal/fine-grained recall difficulty.
        if card.family in HARD_RECALL_FAMILIES or _support_span(card) > 4:
            p.recall_need = p.recall_need or "hard_historical_visual"
        refined.append(p)
    return _dedupe_placements(refined)


def _question_temporal_surface_flags(question: str) -> set:
    q = str(question or "")
    flags = set()
    if _QUESTION_FUTURE_WAIT_RE.search(q):
        flags.add("future_wait")
    if _QUESTION_FUTURE_PRED_RE.search(q):
        flags.add("future_prediction")
    if _QUESTION_EXPLICIT_CURRENT_RE.search(q):
        flags.add("explicit_current")
    if _QUESTION_PRESENT_PROGRESSIVE_RE.search(q):
        flags.add("present_progressive")
    if _QUESTION_EXPLICIT_PAST_RE.search(q):
        flags.add("explicit_past")
    return flags


def _structured_temporal_intent(card: Card) -> str:
    family = str(getattr(card, "family", "") or "")
    qtype = str(getattr(card, "question_type", "") or "")
    role = str(getattr(card, "temporal_role", "") or "")
    bucket = str(getattr(card, "temporal_bucket", "") or getattr(card, "timing_type", "") or "")
    policy = str(getattr(card, "support_policy", "") or "")
    if not (role or bucket or policy):
        inferred = infer_card_policy_fields(card)
        role = str(inferred.get("temporal_role", "") or "")
        policy = str(inferred.get("support_policy", "") or "")
    if family == "E2" or role in _FUTURE_WAIT_TEMPORAL_ROLES or bucket in _FUTURE_WAIT_TEMPORAL_BUCKETS:
        return "future_wait"
    if family == "F6" or role in _FUTURE_CUE_TEMPORAL_ROLES or bucket in _FUTURE_CUE_TEMPORAL_BUCKETS:
        return "future_prediction"
    if family in {"F5", "CRR1", "PN1"} or qtype == "multi_emit" or role in _MULTI_TEMPORAL_ROLES or bucket in _MULTI_TEMPORAL_BUCKETS:
        return "multi"
    if (
        role in _CURRENT_TEMPORAL_ROLES
        or bucket in _CURRENT_TEMPORAL_BUCKETS
        or (policy == SUPPORT_CURRENT_VISUAL and family in {"C1", "ACR1", "STU1", "OJR1", "R1", "CR3", "CR7"})
    ):
        return "current"
    if policy == SUPPORT_HISTORICAL_STATE_MEMORY or role == "global_summary":
        return "past_state"
    if policy == SUPPORT_HISTORICAL_VISUAL_RECALL or "historical" in role or "past" in bucket:
        return "past_visual"
    return ""


def _placement_temporal_group(card: Card, placement: Placement) -> str:
    if placement.mechanism == "multi_emit":
        return "multi"
    if placement.mechanism == "silent_then_response":
        return "future_wait"
    if placement.mechanism == "recall_demo":
        return "past_visual"
    if placement.mechanism == "memory_direct":
        return "past_state"
    policy = str(getattr(placement, "support_policy", "") or getattr(card, "support_policy", "") or "")
    if policy == SUPPORT_FUTURE_CURRENT_CUE:
        return "future_prediction"
    return "current"


def _placement_temporal_surface_verdict(card: Card, placement: Placement) -> Tuple[bool, str]:
    """Reject high-confidence wording/type/placement timing contradictions.

    The teacher owns wording; this guard only prevents pass3b from placing a
    generated card into an incompatible availability bucket.
    """
    family = str(getattr(card, "family", "") or "")
    intent = _structured_temporal_intent(card)
    group = _placement_temporal_group(card, placement)
    flags = _question_temporal_surface_flags(getattr(card, "question", ""))

    if intent == "future_wait" and group != "future_wait":
        return False, "temporal_intent_future_wait_not_future"
    if intent == "future_prediction" and group not in {"future_prediction", "current"}:
        return False, "temporal_intent_future_prediction_not_current_cue"
    if intent == "multi" and group != "multi":
        return False, "temporal_intent_multi_not_multi"
    if intent == "current" and group != "current":
        return False, "temporal_intent_current_not_current"
    if intent == "past_visual" and group == "future_wait" and not (family == "CR5" and "future_wait" in flags):
        return False, "temporal_intent_past_not_future"
    if intent == "future_prediction" and "future_prediction" not in flags:
        return False, "question_future_prediction_missing_prospective_surface"
    if intent == "future_wait" and "future_wait" not in flags:
        return False, "question_future_wait_missing_trigger_surface"
    if intent == "past_visual" and group == "past_visual" and "explicit_past" not in flags:
        return False, "question_past_visual_missing_past_surface"

    if "future_wait" in flags and group != "future_wait" and family != "F6":
        return False, "question_future_wait_surface_not_future"
    if "future_prediction" in flags and group not in {"future_prediction", "current"}:
        return False, "question_future_prediction_surface_not_current_cue"
    if (
        {"explicit_current", "present_progressive"} & flags
        and group in {"past_visual", "past_state", "future_wait"}
        and family not in {"E2", "F5", "CRR1", "M1"}
    ):
        return False, "question_current_surface_not_current"
    if (
        "explicit_past" in flags
        and not ({"future_wait", "future_prediction"} & flags)
        and group in {"future_wait", "future_prediction"}
        and family not in {"CR5", "F6"}
    ):
        return False, "question_past_surface_not_past"
    return True, "pass"


def placement_timing_verdict(card: Card, placement: Placement) -> Tuple[bool, str]:
    """Validate ask/answer timing against support availability.

    This is intentionally stricter than schema validation: pass3a decides
    what evidence supports the answer; pass3b must ensure the question is
    asked only at a time when the chosen mechanism is causally valid.
    """
    if not card.gold_emits:
        return False, "no_gold_emits"
    if not placement.chunk_actions:
        return False, "no_chunk_actions"

    surface_ok, surface_reason = _placement_temporal_surface_verdict(card, placement)
    if not surface_ok:
        return False, surface_reason

    response_chunks = [
        int(c) for c, (kind, _value) in placement.chunk_actions.items()
        if kind == "response"
    ]
    if not response_chunks:
        return False, "no_response_chunk"

    if card.question_type == "multi_emit":
        # Multi-emit placements may intentionally select a compact local
        # subset from a longer card. Validate against the selected placement's
        # first response chunk, not the card's global first emit.
        first_emit = min(response_chunks)
        if placement.ask_chunk > first_emit:
            return False, "multi_ask_after_first_emit"
        return True, "pass"

    emit = int(card.gold_emits[0].chunk)
    support = [int(g) for g in (card.grounding_frames or [emit])]
    max_support = max(support)
    first_response = min(response_chunks)
    if emit < max_support:
        return False, "emit_before_latest_support"
    if first_response < max_support:
        return False, "response_before_latest_support"

    policy = placement.support_policy or infer_card_policy_fields(card)["support_policy"]

    if placement.mechanism == "silent_then_response":
        if policy != SUPPORT_FUTURE_CURRENT_CUE:
            return False, "forward_not_future_policy"
        if placement.ask_chunk >= emit:
            return False, "forward_ask_not_before_emit"
        return True, "pass"

    if placement.mechanism == "recall_demo":
        if policy != SUPPORT_HISTORICAL_VISUAL_RECALL:
            return False, "recall_not_historical_visual_policy"
        if placement.ask_chunk <= max_support:
            return False, "recall_ask_before_support"
        if placement.ask_chunk - max_support <= VISUAL_WINDOW_CHUNKS:
            return False, "recall_support_still_visual"
        if first_response != placement.ask_chunk:
            return False, "recall_response_not_at_ask"
        return True, "pass"

    if placement.mechanism == "memory_direct":
        if policy != SUPPORT_HISTORICAL_STATE_MEMORY:
            return False, "memory_not_state_policy"
        if placement.ask_chunk <= max_support:
            return False, "memory_ask_before_support"
        if placement.ask_chunk - max_support <= VISUAL_WINDOW_CHUNKS:
            return False, "memory_support_still_visual"
        if first_response != placement.ask_chunk:
            return False, "memory_response_not_at_ask"
        return True, "pass"

    if placement.mechanism == "direct":
        if policy not in {SUPPORT_CURRENT_VISUAL, SUPPORT_FUTURE_CURRENT_CUE}:
            return False, "direct_not_current_policy"
        if placement.ask_chunk < max_support:
            return False, "direct_ask_before_support"
        if placement.ask_chunk - max_support > VISUAL_WINDOW_CHUNKS:
            return False, "direct_gap_requires_recall"
        return True, "pass"

    return True, "pass"


# ---------------------------------------------------------------------------
# Trajectory selection — pick MAX_QUESTIONS_PER_TRAJECTORY placements per video
# ---------------------------------------------------------------------------


def _placement_chunks(p: Placement) -> set:
    return {int(c) for c in p.chunk_actions.keys()}


def _placement_span_len(p: Placement) -> int:
    chunks = _placement_chunks(p)
    if not chunks:
        return 0
    return max(chunks) - min(chunks) + 1


@dataclass(frozen=True)
class PlacementMessageCost:
    active_rows: int = 0
    silent_rows: int = 0
    response_rows: int = 0
    direct_response_rows: int = 0
    recall_response_rows: int = 0
    hld_recall_response_rows: int = 0
    future_response_rows: int = 0
    multi_response_rows: int = 0
    future_pending_silent_rows: int = 0
    multi_pending_silent_rows: int = 0
    multi_active_rows: int = 0

    def as_counter(self) -> Counter:
        return Counter({
            "active_rows": self.active_rows,
            "silent_rows": self.silent_rows,
            "response_rows": self.response_rows,
            "direct_response_rows": self.direct_response_rows,
            "recall_response_rows": self.recall_response_rows,
            "hld_recall_response_rows": self.hld_recall_response_rows,
            "future_response_rows": self.future_response_rows,
            "multi_response_rows": self.multi_response_rows,
            "future_pending_silent_rows": self.future_pending_silent_rows,
            "multi_pending_silent_rows": self.multi_pending_silent_rows,
            "multi_active_rows": self.multi_active_rows,
        })


def _placement_response_source(p: Placement, card: Optional[Card] = None) -> str:
    """Action-level answer source for boundary/cost balancing.

    This deliberately differs from benchmark task labels. For SFT, the model is
    learning whether the next action is direct answer, recall, keep waiting for
    a future trigger, or keep a multi-answer state probe open.
    """
    family = str(getattr(card, "family", "") or "") if card is not None else ""
    canonical = ""
    if card is not None and getattr(card, "gold_emits", None):
        canonical = str(card.gold_emits[-1].value or "").strip().lower()
    if (family == "HLD1" or canonical == "unable to answer") and p.mechanism == "recall_demo":
        return ANSWER_MODE_HLD_RECALL
    bucket = _placement_timing_bucket(p, card)
    if bucket == "multi_answer" or p.mechanism == "multi_emit":
        return "multi"
    if bucket == "future_delayed" or p.mechanism == "silent_then_response":
        return "future"
    if p.mechanism == "recall_demo":
        return "recall"
    return "direct"


def _placement_answer_mode(p: Placement, card: Optional[Card] = None) -> str:
    family = str(getattr(card, "family", "") or "")
    canonical = ""
    if card is not None and getattr(card, "gold_emits", None):
        canonical = str(card.gold_emits[-1].value or "").strip().lower()
    if family == "HLD1" or canonical == "unable to answer":
        if p.mechanism == "recall_demo":
            return ANSWER_MODE_HLD_RECALL
        return ANSWER_MODE_ABSTAIN
    bucket = _placement_timing_bucket(p, card)
    if bucket == "multi_answer" or p.mechanism == "multi_emit":
        return ANSWER_MODE_MULTI
    if bucket == "future_delayed" or p.mechanism == "silent_then_response":
        return ANSWER_MODE_FUTURE
    if p.mechanism == "recall_demo":
        return ANSWER_MODE_RECALL
    if bucket == "past_state_direct" or p.mechanism == "memory_direct":
        return ANSWER_MODE_STATE_DIRECT
    return ANSWER_MODE_DIRECT


def _placement_mode_allowed_by_card(p: Placement, card: Card) -> bool:
    mode = _placement_answer_mode(p, card)
    constraints = infer_card_answer_mode_constraints(card)
    legal = _normalize_answer_modes(
        getattr(card, "legal_answer_modes", None)
        or constraints.get("legal_answer_modes")
    )
    forbidden = set(_normalize_answer_modes(
        getattr(card, "forbidden_answer_modes", None)
        or constraints.get("forbidden_answer_modes")
    ))
    required = str(
        getattr(card, "required_answer_mode", "")
        or constraints.get("required_answer_mode", "")
        or ""
    ).strip().lower()
    if required and mode != required:
        return False
    if mode in forbidden:
        return False
    return not legal or mode in legal


def _placement_message_cost(p: Placement, card: Optional[Card] = None) -> PlacementMessageCost:
    response_chunks = sorted(
        int(c) for c, (kind, _value) in p.chunk_actions.items()
        if kind == "response"
    )
    response_rows = len(response_chunks)
    active_rows = len(p.chunk_actions)
    silent_rows = max(0, active_rows - response_rows)
    source = _placement_response_source(p, card)
    first_response = min(response_chunks) if response_chunks else None
    pre_answer_silent = 0
    if first_response is not None:
        pre_answer_silent = sum(
            1
            for c, (kind, _value) in p.chunk_actions.items()
            if kind == "silent" and int(c) < first_response
        )
    return PlacementMessageCost(
        active_rows=active_rows,
        silent_rows=silent_rows,
        response_rows=response_rows,
        direct_response_rows=response_rows if source == "direct" else 0,
        recall_response_rows=response_rows if source == "recall" else 0,
        hld_recall_response_rows=response_rows if source == ANSWER_MODE_HLD_RECALL else 0,
        future_response_rows=response_rows if source == "future" else 0,
        multi_response_rows=response_rows if source == "multi" else 0,
        future_pending_silent_rows=pre_answer_silent if source == "future" else 0,
        multi_pending_silent_rows=silent_rows if source == "multi" else 0,
        multi_active_rows=active_rows if source == "multi" else 0,
    )


def _recall_floor(max_q: int) -> int:
    if max_q <= 0:
        return 0
    return max(1, int(max_q * RECALL_TARGET_FRACTION + 0.999))


def _ceil_fraction(max_q: int, fraction: float) -> int:
    if max_q <= 0 or fraction <= 0:
        return 0
    return int(max_q * fraction + 0.999)


def _floor_fraction(max_q: int, fraction: float) -> int:
    if max_q <= 0 or fraction <= 0:
        return 0
    return int(max_q * fraction)


def _bucket_target(max_q: int, fraction: float, minimum: int = 0) -> int:
    if max_q <= 0:
        return 0
    return min(max_q, max(minimum, _ceil_fraction(max_q, fraction)))


def _weighted_bucket_choice(key: str, weights: Dict[str, int]) -> str:
    total = sum(max(0, int(v)) for v in weights.values())
    if total <= 0:
        return ""
    bucket = stable_mod(key, "weighted_bucket", modulo=total)
    acc = 0
    for name, weight in weights.items():
        acc += max(0, int(weight))
        if bucket < acc:
            return name
    return next(iter(weights), "")


def _is_ours_card(card: Card) -> bool:
    style = str(getattr(card, "question_style", "") or "").strip()
    return style == QUESTION_STYLE_OURS_UNIQUE or str(card.family or "") in OURS_FAMILIES


def _card_question_style(card: Card) -> str:
    style = str(getattr(card, "question_style", "") or "").strip()
    if style in QUESTION_STYLES:
        return style
    if _is_ours_card(card):
        return QUESTION_STYLE_OURS_UNIQUE
    return QUESTION_STYLE_BENCHMARK_CORE


def _is_benchmark_variant_card(card: Card) -> bool:
    return (
        _card_question_style(card) == QUESTION_STYLE_BENCHMARK_VARIANT
        and not _is_ours_card(card)
    )


def _card_question_way(card: Card) -> str:
    way = str(getattr(card, "question_way", "") or "").strip()
    if way in QUESTION_WAYS:
        return way
    return infer_card_semantic_fields(card)["question_way"]


def _card_evidence_type(card: Card) -> str:
    evidence_type = str(getattr(card, "evidence_type", "") or "").strip()
    if evidence_type in EVIDENCE_TYPES:
        return evidence_type
    return infer_card_semantic_fields(card)["evidence_type"]


def _card_task_subtype(card: Card) -> str:
    subtype = str(getattr(card, "task_subtype", "") or "").strip()
    if subtype:
        return subtype
    subtype = str(getattr(card, "slot_subtype", "") or "").strip()
    if subtype:
        return subtype
    return str(getattr(card, "family", "") or "unknown").strip() or "unknown"


def _answer_form_bucket(card: Optional[Card]) -> str:
    answer_form = str(getattr(card, "answer_form", "") or "").strip()
    if answer_form in {"short_exact", "descriptive"}:
        return "short_text"
    if answer_form in {"binary", "number", "multiple_choice"}:
        return answer_form
    return answer_form or "unknown"


def _is_state_probe_placement(p: Placement, card: Optional[Card] = None) -> bool:
    family = str(getattr(card, "family", "") or "")
    policy = str(getattr(p, "support_policy", "") or "")
    return family in STATE_PROBE_FAMILIES or policy in STATE_PROBE_POLICIES


def _is_current_direct_placement(p: Placement) -> bool:
    return (
        p.mechanism == "direct"
        and str(getattr(p, "support_policy", "") or "") in CURRENT_DIRECT_POLICIES
    )


def _placement_timing_bucket(p: Placement, card: Optional[Card] = None) -> str:
    """Question-timing bucket used for selection quotas.

    Benchmark labels do not include recall. We derive this from ask/answer
    timing and support policy:
      - current_direct: current/recent evidence or current cue, answer now
      - past_state_direct: historical state/global answer without visual recall
      - past_recall: historical visual evidence older than the current window
      - future_delayed: ask now, answer when future trigger appears
      - multi_answer: persistent REC/CRR/narration style probes
    """
    if p.mechanism == "multi_emit":
        return "multi_answer"
    if p.mechanism == "silent_then_response":
        return "future_delayed"
    if p.mechanism == "recall_demo":
        return "past_recall"
    policy = str(getattr(p, "support_policy", "") or "")
    card_policy = ""
    card_timing = ""
    if card is not None:
        card_policy = str(getattr(card, "support_policy", "") or "")
        card_timing = str(getattr(card, "timing_type", "") or getattr(card, "temporal_bucket", "") or "")
    if (
        p.mechanism == "memory_direct"
        or policy == SUPPORT_HISTORICAL_STATE_MEMORY
        or card_policy == SUPPORT_HISTORICAL_STATE_MEMORY
        or card_timing == "global_state_memory"
    ):
        return "past_state_direct"
    if policy == SUPPORT_FUTURE_CURRENT_CUE or card_policy == SUPPORT_FUTURE_CURRENT_CUE:
        return "future_current_cue"
    support: List[int] = []
    if card is not None:
        for raw in getattr(card, "grounding_frames", []) or []:
            try:
                support.append(int(raw))
            except (TypeError, ValueError):
                continue
    if _is_current_direct_placement(p):
        if not support:
            return "current_direct"
        latest_support = max(support)
        support_gap = int(p.ask_chunk) - latest_support
        if 0 <= support_gap <= VISUAL_WINDOW_CHUNKS:
            return "current_direct"
    if support:
        latest_support = max(support)
        if latest_support < int(p.ask_chunk):
            return "past_state_direct"
    if _is_current_direct_placement(p):
        return "current_direct"
    return "past_state_direct"


def _primary_support_bin(card: Optional[Card]) -> Optional[int]:
    if card is None:
        return None
    support: List[int] = []
    for raw in getattr(card, "grounding_frames", []) or []:
        try:
            support.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not support or SUPPORT_BIN_SIZE <= 0:
        return None
    return max(support) // SUPPORT_BIN_SIZE


def _chunk_position_bin(chunk: int, num_chunks: int) -> int:
    bins = max(1, POSITION_BIN_COUNT)
    if num_chunks <= 1:
        return 0
    frac = max(0.0, min(0.999999, float(chunk) / float(max(1, num_chunks))))
    return min(bins - 1, int(frac * bins))


def _placement_response_chunks(p: Placement) -> List[int]:
    return sorted(
        int(c) for c, (kind, _value) in p.chunk_actions.items()
        if kind == "response"
    )


def _placement_response_bin(p: Placement, num_chunks: int) -> Optional[int]:
    chunks = _placement_response_chunks(p)
    if not chunks:
        return None
    return _chunk_position_bin(chunks[-1], num_chunks)


def select_trajectory(
    cards: List[Card],
    placements_by_card: Dict[str, List[Placement]],
    num_chunks: int,
    rng: random.Random,
    max_q: int = MAX_QUESTIONS_PER_TRAJECTORY,
) -> List[Placement]:
    """Select up to max_q placements for ONE trajectory per video.

    Greedy diversity scoring:
      +3 unseen family
      +2 unseen mechanism
      +1 unseen answer_form
      +1 unseen card (one placement per card max)
      +spread: distance to nearest already-picked ask_chunk / 8 (cap 3.0)

    Strict constraints:
      - at most ONE placement per card_id
      - at most ONE placement per family per trajectory
      - independent ask chunks must satisfy the video-length ask-gap floor
      - no overlapping placement chunks. This enforces a single active
        question at a time; multi-answer supervision is allowed only inside
        one multi_emit question, never as competing questions on the same
        answer chunk.
    """
    cards_by_id = {c.card_id: c for c in cards}
    pool: List[Tuple[Placement, Card]] = []
    for cid, plcs in placements_by_card.items():
        card = cards_by_id.get(cid)
        if not card:
            continue
        for p in plcs:
            if not _placement_mode_allowed_by_card(p, card):
                continue
            pool.append((p, card))
    if not pool:
        return []

    if max_q <= 3:
        # Tiny diagnostic trajectories validate structural constraints rather
        # than production distribution quotas. Use a minimal no-overlap selector
        # so semantic/support caps cannot collapse these toy cases to one row.
        selected_small: List[Placement] = []
        used_small: set = set()
        for p, _card in sorted(pool, key=lambda pc: (pc[0].ask_chunk, pc[0].card_id)):
            chunks = _placement_chunks(p)
            if chunks & used_small:
                continue
            selected_small.append(p)
            used_small.update(chunks)
            if len(selected_small) >= max_q:
                break
        return selected_small

    selection_key = min((str(c.card_id) for c in cards if c.card_id), default="trajectory")
    selected: List[Placement] = []
    seen_families: set = set()
    family_counts: Dict[str, int] = {}
    seen_mechs: set = set()
    seen_aforms: set = set()
    seen_question_ways: set = set()
    seen_evidence_types: set = set()
    seen_task_subtypes: set = set()
    seen_cards: set = set()
    used_chunks: set = set()
    used_ask: List[int] = []
    support_bin_counts: Dict[int, int] = {}
    selected_message_cost: Counter = Counter()
    ask_gap_floor = question_ask_gap_floor(num_chunks)
    span_cap = density_blocking_span_cap(num_chunks)
    raw_density_min_rows = _ceil_fraction(
        max(1, num_chunks),
        RAW_RESPONSE_DENSITY_MIN_FRACTION,
    )
    raw_density_target_rows = _ceil_fraction(
        max(1, num_chunks),
        RAW_RESPONSE_DENSITY_TARGET_FRACTION,
    )
    raw_density_cap_rows = _ceil_fraction(
        max(1, num_chunks),
        RAW_RESPONSE_DENSITY_MAX_FRACTION,
    )
    density_selection_q = max(1, min(max(1, num_chunks), raw_density_cap_rows))
    if RAW_RESPONSE_DENSITY_QUESTION_CAP > 0:
        density_selection_q = min(
            density_selection_q,
            max(raw_density_target_rows, RAW_RESPONSE_DENSITY_QUESTION_CAP),
        )
    # `max_q` is now only the first-pass diversity budget. The final raw
    # trajectory is capped by response rows, so short videos cannot be forced
    # above 15% just because the historical adaptive question floor was 10.
    timing_quota_q = density_selection_q
    density_floor_q = min(
        density_selection_q,
        max(
            min(max_q, MIN_QUESTIONS_PER_TRAJECTORY),
            _ceil_fraction(density_selection_q, DENSITY_FILL_TARGET_FRACTION),
        ),
    )
    recall_target_base = _floor_fraction(timing_quota_q, RECALL_TARGET_FRACTION)
    if RECALL_TARGET_FRACTION > 0 and timing_quota_q > 0 and recall_target_base == 0:
        recall_target_base = 1
    recall_cap_base = _floor_fraction(timing_quota_q, RECALL_MAX_FRACTION)
    if RECALL_MAX_FRACTION > 0 and timing_quota_q > 0 and recall_cap_base == 0:
        recall_cap_base = 1
    recall_cap = min(density_selection_q, max(recall_target_base, recall_cap_base))
    if RECALL_MAX_PER_TRAJECTORY > 0:
        recall_cap = min(recall_cap, RECALL_MAX_PER_TRAJECTORY)
    own_target = min(
        max_q,
        max(
            min(OURS_MIN_QUESTIONS, max_q) if OURS_TARGET_FRACTION > 0 else 0,
            _ceil_fraction(timing_quota_q, OURS_TARGET_FRACTION),
        ),
    )
    own_cap = min(
        max_q,
        max(own_target, _floor_fraction(timing_quota_q, OURS_MAX_FRACTION)),
    )
    if OURS_ABS_CAP > 0:
        own_target = min(own_target, OURS_ABS_CAP)
        own_cap = min(own_cap, OURS_ABS_CAP)
    if stable_mod(selection_key, "ours_trajectory", modulo=100) >= max(0, OURS_TRAJECTORY_PERCENT):
        own_target = 0
        own_cap = 0
    state_probe_target = min(
        _ceil_fraction(timing_quota_q, STATE_PROBE_MAX_FRACTION),
        _bucket_target(timing_quota_q, STATE_PROBE_TARGET_FRACTION, 1),
    )
    state_probe_cap = max(
        state_probe_target,
        _ceil_fraction(timing_quota_q, STATE_PROBE_MAX_FRACTION),
    )
    future_delayed_target = _bucket_target(
        timing_quota_q,
        FUTURE_DELAYED_TARGET_FRACTION,
        1 if FUTURE_DELAYED_TARGET_FRACTION > 0 else 0,
    )
    future_delayed_cap = max(
        future_delayed_target,
        _ceil_fraction(timing_quota_q, FUTURE_DELAYED_MAX_FRACTION),
    )
    if stable_mod(selection_key, "future_delayed_trajectory", modulo=100) >= max(
        0,
        FUTURE_DELAYED_TRAJECTORY_PERCENT,
    ):
        future_delayed_target = 0
        future_delayed_cap = 0
    future_current_cue_target = _bucket_target(
        timing_quota_q,
        FUTURE_CURRENT_CUE_TARGET_FRACTION,
        1 if FUTURE_CURRENT_CUE_TARGET_FRACTION > 0 else 0,
    )
    future_current_cue_cap = max(
        future_current_cue_target,
        _ceil_fraction(timing_quota_q, FUTURE_CURRENT_CUE_MAX_FRACTION),
    )
    if stable_mod(selection_key, "future_current_cue_trajectory", modulo=100) >= max(
        0,
        FUTURE_CURRENT_CUE_TRAJECTORY_PERCENT,
    ):
        future_current_cue_target = 0
        future_current_cue_cap = 0
    multi_answer_target = _bucket_target(
        timing_quota_q,
        MULTI_ANSWER_TARGET_FRACTION,
        1 if MULTI_ANSWER_TARGET_FRACTION > 0 else 0,
    )
    multi_answer_cap = max(
        multi_answer_target,
        _ceil_fraction(timing_quota_q, MULTI_ANSWER_MAX_FRACTION),
    )
    if stable_mod(selection_key, "multi_answer_trajectory", modulo=100) >= max(
        0,
        MULTI_ANSWER_TRAJECTORY_PERCENT,
    ):
        multi_answer_target = 0
        multi_answer_cap = 0
    unanswerable_target = _bucket_target(
        max_q,
        UNANSWERABLE_TARGET_FRACTION,
        1 if UNANSWERABLE_TARGET_FRACTION > 0 else 0,
    )
    unanswerable_cap = max(
        unanswerable_target,
        _ceil_fraction(max_q, UNANSWERABLE_MAX_FRACTION),
    )
    if stable_mod(selection_key, "unanswerable_trajectory", modulo=100) >= max(
        0,
        UNANSWERABLE_TRAJECTORY_PERCENT,
    ):
        unanswerable_target = 0
        unanswerable_cap = 0
    variant_target = _bucket_target(
        timing_quota_q,
        BENCHMARK_VARIANT_TARGET_FRACTION,
        1 if BENCHMARK_VARIANT_TARGET_FRACTION > 0 else 0,
    )
    variant_cap = max(
        variant_target,
        _ceil_fraction(timing_quota_q, BENCHMARK_VARIANT_MAX_FRACTION),
    )
    current_direct_target = _bucket_target(
        timing_quota_q,
        CURRENT_DIRECT_TARGET_FRACTION,
        min(CURRENT_DIRECT_MIN_QUESTIONS, timing_quota_q),
    )
    current_direct_cap = max(
        current_direct_target,
        _ceil_fraction(timing_quota_q, CURRENT_DIRECT_MAX_FRACTION),
    )
    past_state_direct_target = _bucket_target(
        timing_quota_q,
        PAST_STATE_DIRECT_TARGET_FRACTION,
        1 if PAST_STATE_DIRECT_TARGET_FRACTION > 0 else 0,
    )
    past_state_direct_cap = max(
        past_state_direct_target,
        _ceil_fraction(timing_quota_q, PAST_STATE_DIRECT_MAX_FRACTION),
    )
    if max_q <= 3:
        # Unit tests and tiny diagnostic trajectories often use max_q=3 to
        # validate structural constraints such as no-overlap. Do not let the
        # production distribution caps collapse those toy selections to one
        # question.
        ask_gap_floor = 0
        future_delayed_cap = max_q
        future_current_cue_cap = max_q
        multi_answer_cap = max_q
        current_direct_cap = max_q
        past_state_direct_cap = max_q
    recall_family_direct_cap = _bucket_target(
        timing_quota_q,
        RECALL_FAMILY_DIRECT_FILL_MAX_FRACTION,
        1 if RECALL_FAMILY_DIRECT_FILL_MAX_FRACTION > 0 else 0,
    )
    if max_q <= 3:
        recall_family_direct_cap = max_q
    question_way_targets = {
        way: _bucket_target(density_selection_q, frac, 1 if frac >= 0.05 else 0)
        for way, frac in QUESTION_WAY_TARGET_FRACTION.items()
    }
    question_way_caps = {
        way: max(
            1,
            _ceil_fraction(
                density_selection_q,
                max(
                    frac * QUESTION_WAY_CAP_MULTIPLIER,
                    QUESTION_WAY_MIN_CAP_FRACTION,
                ),
            ),
        )
        for way, frac in QUESTION_WAY_TARGET_FRACTION.items()
    }
    evidence_type_targets = {
        evidence_type: _bucket_target(density_selection_q, frac, 1 if frac >= 0.05 else 0)
        for evidence_type, frac in EVIDENCE_TYPE_TARGET_FRACTION.items()
    }
    evidence_type_caps = {
        evidence_type: max(
            1,
            _ceil_fraction(
                density_selection_q,
                max(
                    frac * QUESTION_WAY_CAP_MULTIPLIER,
                    QUESTION_WAY_MIN_CAP_FRACTION,
                ),
            ),
        )
        for evidence_type, frac in EVIDENCE_TYPE_TARGET_FRACTION.items()
    }
    task_subtype_targets = {
        subtype: _bucket_target(density_selection_q, frac, 1 if frac >= 0.025 else 0)
        for subtype, frac in TASK_SUBTYPE_TARGET_FRACTION.items()
    }
    task_subtype_caps = {
        subtype: max(
            1,
            _ceil_fraction(
                density_selection_q,
                TASK_SUBTYPE_MAX_FRACTION.get(
                    subtype,
                    max(
                        frac * TASK_SUBTYPE_CAP_MULTIPLIER,
                        TASK_SUBTYPE_MIN_CAP_FRACTION,
                    ),
                ),
            ),
        )
        for subtype, frac in TASK_SUBTYPE_TARGET_FRACTION.items()
    }
    default_task_subtype_cap = max(
        1,
        _ceil_fraction(density_selection_q, TASK_SUBTYPE_DEFAULT_MAX_FRACTION),
    )
    task_mode_targets = {
        pair: _bucket_target(density_selection_q, frac, 1 if frac >= 0.025 else 0)
        for pair, frac in TASK_MODE_TARGET_FRACTION.items()
    }
    position_bin_target = max(
        1,
        _floor_fraction(max(1, density_floor_q), 1.0 / max(1, POSITION_BIN_COUNT)),
    )
    answer_form_targets = {
        "binary": _bucket_target(density_selection_q, BINARY_TARGET_FRACTION, 1 if BINARY_TARGET_FRACTION > 0 else 0),
        "number": _bucket_target(density_selection_q, NUMBER_TARGET_FRACTION, 1 if NUMBER_TARGET_FRACTION > 0 else 0),
        "short_text": _bucket_target(density_selection_q, SHORT_TEXT_TARGET_FRACTION, 1 if SHORT_TEXT_TARGET_FRACTION > 0 else 0),
    }
    answer_form_caps = {
        "binary": max(answer_form_targets["binary"], _ceil_fraction(density_selection_q, BINARY_MAX_FRACTION)),
        "number": max(answer_form_targets["number"], _ceil_fraction(density_selection_q, NUMBER_MAX_FRACTION)),
        "short_text": max(answer_form_targets["short_text"], _ceil_fraction(density_selection_q, SHORT_TEXT_MAX_FRACTION)),
    }
    # Message/action budgets. The learner sees one decision per row, so a
    # multi-answer placement is much more expensive than a one-shot question.
    # These are hard caps; semantic/timing quotas below are only soft balance
    # pressure once a candidate fits the row-level budget.
    answer_row_target = min(raw_density_target_rows, raw_density_cap_rows)
    answer_row_cap = max(1, raw_density_cap_rows)

    def planned_source_response_rows(total_rows: int) -> Dict[str, int]:
        total_rows = max(0, int(total_rows))
        if total_rows <= 0:
            return {"direct": 0, "recall": 0, "future": 0, "multi": 0}
        rows = {
            "direct": max(
                1,
                int(total_rows * MESSAGE_DIRECT_RESPONSE_ROW_TARGET_FRACTION + 0.5),
            ),
            "recall": max(
                1 if total_rows >= 4 else 0,
                int(total_rows * MESSAGE_RECALL_RESPONSE_ROW_TARGET_FRACTION + 0.5),
            ),
            "future": max(
                1 if total_rows >= 6 else 0,
                int(total_rows * MESSAGE_FUTURE_RESPONSE_ROW_TARGET_FRACTION + 0.5),
            ),
        }
        rows["multi"] = max(
            1 if total_rows >= 10 else 0,
            total_rows - sum(rows.values()),
        )
        if total_rows < 10:
            rows["direct"] += rows["multi"]
            rows["multi"] = 0
        floors = {
            "direct": 1 if total_rows > 0 else 0,
            "recall": 1 if total_rows >= 4 else 0,
            "future": 1 if total_rows >= 6 else 0,
            "multi": 1 if total_rows >= 10 else 0,
        }
        while sum(rows.values()) > total_rows:
            for source in ("direct", "recall", "multi", "future"):
                if rows.get(source, 0) > floors.get(source, 0):
                    rows[source] -= 1
                    break
            else:
                break
        while sum(rows.values()) < total_rows:
            eligible_sources = ["direct"]
            if total_rows >= 4:
                eligible_sources.append("recall")
            if total_rows >= 6:
                eligible_sources.append("future")
            if total_rows >= 10:
                eligible_sources.append("multi")
            source = min(
                eligible_sources,
                key=lambda name: (
                    rows.get(name, 0)
                    / max(
                        0.001,
                        {
                            "direct": MESSAGE_DIRECT_RESPONSE_ROW_TARGET_FRACTION,
                            "recall": MESSAGE_RECALL_RESPONSE_ROW_TARGET_FRACTION,
                            "future": MESSAGE_FUTURE_RESPONSE_ROW_TARGET_FRACTION,
                            "multi": MESSAGE_MULTI_RESPONSE_ROW_TARGET_FRACTION,
                        }.get(name, 0.001),
                    ),
                    name,
                ),
            )
            rows[source] += 1
        return rows

    planned_source_rows = planned_source_response_rows(answer_row_target)
    direct_response_row_target = min(answer_row_cap, planned_source_rows["direct"])
    direct_response_row_cap = min(
        answer_row_cap,
        max(
            direct_response_row_target,
            _floor_fraction(answer_row_cap, MESSAGE_DIRECT_RESPONSE_ROW_MAX_FRACTION),
        ),
    )
    recall_response_row_target = min(answer_row_cap, planned_source_rows["recall"])
    recall_response_row_cap = max(
        recall_response_row_target,
        _ceil_fraction(answer_row_cap, MESSAGE_RECALL_RESPONSE_ROW_MAX_FRACTION),
    )
    if RECALL_MAX_PER_TRAJECTORY > 0:
        recall_response_row_cap = min(recall_response_row_cap, RECALL_MAX_PER_TRAJECTORY)
    future_response_row_target = min(answer_row_cap, planned_source_rows["future"])
    future_response_row_cap = max(
        future_response_row_target,
        _ceil_fraction(answer_row_cap, MESSAGE_FUTURE_RESPONSE_ROW_MAX_FRACTION),
    )
    multi_response_row_target = min(answer_row_cap, planned_source_rows["multi"])
    multi_response_row_cap = max(
        MESSAGE_MULTI_RESPONSE_ROW_MIN_CAP,
        multi_response_row_target,
        _ceil_fraction(answer_row_cap, MESSAGE_MULTI_RESPONSE_ROW_MAX_FRACTION),
    )
    if MESSAGE_MULTI_RESPONSE_ROW_ABS_CAP > 0:
        multi_response_row_cap = min(multi_response_row_cap, MESSAGE_MULTI_RESPONSE_ROW_ABS_CAP)
    multi_active_row_cap = span_cap
    future_pending_row_cap = min(
        span_cap,
        max(
        MESSAGE_FUTURE_PENDING_ROW_MIN_CAP,
        _ceil_fraction(max(1, num_chunks), MESSAGE_FUTURE_PENDING_ROW_MAX_FRACTION),
        ),
    )
    if max_q <= 3:
        direct_response_row_cap = answer_row_cap

    def non_mcq_selected() -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and cards_by_id[p.card_id].answer_form != "multiple_choice"
        )

    def non_mcq_floor() -> int:
        if max_q <= 0:
            return 0
        by_fraction = int(max_q * NON_MCQ_TARGET_FRACTION + 0.999)
        return min(max_q, max(NON_MCQ_MIN_QUESTIONS, by_fraction))

    def answer_form_bucket_selected(bucket: str) -> int:
        return sum(
            1
            for p in selected
            if _answer_form_bucket(cards_by_id.get(p.card_id)) == bucket
        )

    def own_selected() -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id) and _is_ours_card(cards_by_id[p.card_id])
        )

    def state_probe_selected() -> int:
        return sum(
            1
            for p in selected
            if _is_state_probe_placement(p, cards_by_id.get(p.card_id))
        )

    def current_direct_selected() -> int:
        return sum(
            1
            for p in selected
            if _placement_timing_bucket(p, cards_by_id.get(p.card_id)) == "current_direct"
        )

    def recall_family_direct_selected() -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and cards_by_id[p.card_id].family in NORMAL_RECALL_FAMILIES
            and _placement_answer_mode(p, cards_by_id[p.card_id]) == ANSWER_MODE_DIRECT
        )

    def future_delayed_selected() -> int:
        return sum(
            1
            for p in selected
            if _placement_timing_bucket(p, cards_by_id.get(p.card_id)) == "future_delayed"
        )

    def future_current_cue_selected() -> int:
        return sum(
            1
            for p in selected
            if _placement_timing_bucket(p, cards_by_id.get(p.card_id)) == "future_current_cue"
        )

    def past_state_direct_selected() -> int:
        return sum(
            1
            for p in selected
            if _placement_timing_bucket(p, cards_by_id.get(p.card_id)) == "past_state_direct"
        )

    def multi_answer_selected() -> int:
        return sum(
            1
            for p in selected
            if _placement_timing_bucket(p, cards_by_id.get(p.card_id)) == "multi_answer"
        )

    def unanswerable_selected() -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and _is_unanswerable_card(cards_by_id[p.card_id])
        )

    def variant_selected() -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and _is_benchmark_variant_card(cards_by_id[p.card_id])
        )

    def question_way_selected(way: str) -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and _card_question_way(cards_by_id[p.card_id]) == way
        )

    def evidence_type_selected(evidence_type: str) -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and _card_evidence_type(cards_by_id[p.card_id]) == evidence_type
        )

    def task_subtype_selected(subtype: str) -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and _card_task_subtype(cards_by_id[p.card_id]) == subtype
        )

    def task_mode_selected(subtype: str, mode: str) -> int:
        return sum(
            1
            for p in selected
            if cards_by_id.get(p.card_id)
            and _card_task_subtype(cards_by_id[p.card_id]) == subtype
            and _placement_answer_mode(p, cards_by_id[p.card_id]) == mode
        )

    def ask_bin_selected(bin_idx: int) -> int:
        return sum(
            1
            for p in selected
            if _chunk_position_bin(int(p.ask_chunk), num_chunks) == bin_idx
        )

    def response_bin_selected(bin_idx: int) -> int:
        return sum(
            1
            for p in selected
            if _placement_response_bin(p, num_chunks) == bin_idx
        )

    def family_selected(family: str) -> bool:
        return any(
            cards_by_id.get(p.card_id) and cards_by_id[p.card_id].family == family
            for p in selected
        )

    def family_reserve_enabled(family: str, percent: int) -> bool:
        return stable_mod(selection_key, f"{family}_reserve", modulo=100) < max(0, int(percent))

    def is_normal_recall_placement(p: Placement, card: Optional[Card]) -> bool:
        return (
            p.mechanism == "recall_demo"
            and card is not None
            and not _is_unanswerable_card(card)
        )

    def normal_recall_selected() -> int:
        return sum(
            1
            for p in selected
            if is_normal_recall_placement(p, cards_by_id.get(p.card_id))
        )

    def normalized_response_source(p: Placement, card: Optional[Card]) -> str:
        source = _placement_response_source(p, card)
        return "recall" if source == ANSWER_MODE_HLD_RECALL else source

    def selected_source_response_rows(source: str) -> int:
        if source == "direct":
            return int(selected_message_cost["direct_response_rows"])
        if source == "recall":
            return int(
                selected_message_cost["recall_response_rows"]
                + selected_message_cost["hld_recall_response_rows"]
            )
        if source == "future":
            return int(selected_message_cost["future_response_rows"])
        if source == "multi":
            return int(selected_message_cost["multi_response_rows"])
        return 0

    source_response_row_targets = {
        "direct": direct_response_row_target,
        "recall": recall_response_row_target,
        "future": future_response_row_target,
        "multi": multi_response_row_target,
    }
    target_recall = min(recall_cap, recall_response_row_target)

    def source_response_bin_selected(source: str, bin_idx: int) -> int:
        rows = 0
        for p in selected:
            card = cards_by_id.get(p.card_id)
            if not card or normalized_response_source(p, card) != source:
                continue
            for chunk in _placement_response_chunks(p):
                if _chunk_position_bin(chunk, num_chunks) == bin_idx:
                    rows += 1
        return rows

    def placement_response_bins(p: Placement) -> set:
        return {
            _chunk_position_bin(chunk, num_chunks)
            for chunk in _placement_response_chunks(p)
        }

    def source_budget_predicate(source: str):
        if source == "multi":
            return lambda p, card: (
                _placement_timing_bucket(p, card) == "multi_answer"
                and _placement_span_len(p) <= span_cap
            )
        if source == "future":
            return lambda p, card: (
                _placement_timing_bucket(p, card) == "future_delayed"
                and _placement_span_len(p) <= span_cap
            )
        if source == "recall":
            return lambda p, card: is_normal_recall_placement(p, card)
        return lambda p, card: (
            normalized_response_source(p, card) == "direct"
            and _placement_span_len(p) <= span_cap
        )

    def take_source_row(source: str, *, allow_family_repeat: bool = False) -> bool:
        target = int(source_response_row_targets.get(source, 0))
        if len(selected) >= density_selection_q:
            return False
        if selected_source_response_rows(source) >= target:
            return False
        predicate = source_budget_predicate(source)
        desired_bins = max(1, min(POSITION_BIN_COUNT, target))
        bin_counts = [
            (bin_idx, source_response_bin_selected(source, bin_idx))
            for bin_idx in range(max(1, POSITION_BIN_COUNT))
        ]
        covered_bins = sum(1 for _bin_idx, count in bin_counts if count > 0)
        took = False
        if covered_bins < desired_bins:
            min_count = min(count for _bin_idx, count in bin_counts)
            under_bins = {
                bin_idx
                for bin_idx, count in bin_counts
                if count == min_count
            }
            took = take_best(
                lambda p, card: (
                    predicate(p, card)
                    and bool(placement_response_bins(p) & under_bins)
                ),
                allow_family_repeat=allow_family_repeat,
            )
        if not took:
            took = take_best(
                predicate,
                allow_family_repeat=allow_family_repeat,
            )
        return took

    def reserve_source_rows(source: str, *, allow_family_repeat: bool = False) -> None:
        while take_source_row(source, allow_family_repeat=allow_family_repeat):
            pass

    def feasible(
        p: Placement,
        card: Card,
        *,
        allow_family_repeat: bool = False,
        relax_semantic_caps: bool = False,
        relax_timing_caps: bool = False,
        relax_ask_gap: bool = False,
        relax_message_caps: bool = False,
        ask_gap_override: Optional[int] = None,
        family_repeat_cap: Optional[int] = None,
    ) -> bool:
        relax_caps = relax_semantic_caps or max_q <= 3
        relax_timing = relax_timing_caps or max_q <= 3
        if card.card_id in seen_cards:
            return False
        if card.family in seen_families and not allow_family_repeat:
            return False
        effective_family_repeat_cap = (
            FAMILY_REPEAT_FILL_CAP
            if family_repeat_cap is None
            else max(1, int(family_repeat_cap))
        )
        if (
            allow_family_repeat
            and family_counts.get(card.family, 0) >= effective_family_repeat_cap
        ):
            return False
        cost = _placement_message_cost(p, card)
        if (
            not relax_message_caps
            and selected_message_cost["response_rows"] + cost.response_rows > answer_row_cap
        ):
            return False
        if (
            not relax_message_caps
            and
            selected_message_cost["direct_response_rows"] + cost.direct_response_rows
            > direct_response_row_cap
        ):
            return False
        if (
            not relax_message_caps
            and
            selected_message_cost["recall_response_rows"]
            + selected_message_cost["hld_recall_response_rows"]
            + cost.recall_response_rows
            + cost.hld_recall_response_rows
            > recall_response_row_cap
        ):
            return False
        if (
            not relax_message_caps
            and
            selected_message_cost["future_response_rows"] + cost.future_response_rows
            > future_response_row_cap
        ):
            return False
        if (
            not relax_message_caps
            and
            selected_message_cost["multi_response_rows"] + cost.multi_response_rows
            > multi_response_row_cap
        ):
            return False
        if (
            selected_message_cost["multi_active_rows"] + cost.multi_active_rows
            > multi_active_row_cap
        ):
            return False
        if (
            selected_message_cost["future_pending_silent_rows"] + cost.future_pending_silent_rows
            > future_pending_row_cap
        ):
            return False
        bucket = _placement_timing_bucket(p, card)
        if bucket == "future_delayed" and future_delayed_cap <= 0:
            return False
        if bucket == "future_current_cue" and future_current_cue_cap <= 0:
            return False
        if bucket == "multi_answer" and multi_answer_cap <= 0:
            return False
        if (
            is_normal_recall_placement(p, card)
            and normal_recall_selected() >= recall_cap
        ):
            return False
        if (
            card.family in NORMAL_RECALL_FAMILIES
            and _placement_answer_mode(p, card) == ANSWER_MODE_DIRECT
            and recall_family_direct_selected() >= recall_family_direct_cap
        ):
            return False
        if _is_ours_card(card) and own_selected() >= own_cap:
            return False
        if (
            not relax_timing
            and _is_state_probe_placement(p, card)
            and state_probe_selected() >= state_probe_cap
        ):
            return False
        if (
            not relax_timing
            and
            _placement_timing_bucket(p, card) == "future_delayed"
            and future_delayed_selected() >= future_delayed_cap
        ):
            return False
        if (
            not relax_timing
            and
            _placement_timing_bucket(p, card) == "future_current_cue"
            and future_current_cue_selected() >= future_current_cue_cap
        ):
            return False
        if (
            not relax_timing
            and
            _placement_timing_bucket(p, card) == "multi_answer"
            and multi_answer_selected() >= multi_answer_cap
        ):
            return False
        if _is_unanswerable_card(card) and unanswerable_selected() >= unanswerable_cap:
            return False
        if _is_benchmark_variant_card(card) and variant_selected() >= variant_cap:
            return False
        way = _card_question_way(card)
        if (
            not relax_caps
            and question_way_caps.get(way, max_q) >= 0
            and question_way_selected(way) >= question_way_caps.get(way, max_q)
        ):
            return False
        evidence_type = _card_evidence_type(card)
        if (
            not relax_caps
            and evidence_type_caps.get(evidence_type, max_q) >= 0
            and evidence_type_selected(evidence_type) >= evidence_type_caps.get(evidence_type, max_q)
        ):
            return False
        subtype = _card_task_subtype(card)
        subtype_cap = task_subtype_caps.get(subtype, default_task_subtype_cap)
        if (
            not relax_caps
            and subtype
            and subtype_cap >= 0
            and task_subtype_selected(subtype) >= subtype_cap
        ):
            return False
        answer_bucket = _answer_form_bucket(card)
        if (
            not relax_caps
            and answer_bucket in answer_form_caps
            and answer_form_bucket_selected(answer_bucket) >= answer_form_caps[answer_bucket]
        ):
            return False
        bucket = _placement_timing_bucket(p, card)
        if (
            not relax_timing
            and bucket == "current_direct"
            and current_direct_selected() >= current_direct_cap
        ):
            return False
        if (
            not relax_timing
            and bucket == "past_state_direct"
            and past_state_direct_selected() >= past_state_direct_cap
        ):
            return False
        if used_ask:
            effective_gap_floor = ask_gap_floor
            if relax_ask_gap:
                effective_gap_floor = max(
                    0,
                    min(ask_gap_floor, DENSITY_FILL_ASK_GAP_CHUNKS),
                )
            if ask_gap_override is not None:
                effective_gap_floor = max(0, int(ask_gap_override))
            if min(abs(int(p.ask_chunk) - x) for x in used_ask) < effective_gap_floor:
                return False
        support_bin = _primary_support_bin(card)
        support_bin_cap = SUPPORT_BIN_FLOOR_FILL_CAP if allow_family_repeat else SUPPORT_BIN_CAP
        if (
            not relax_caps
            and
            support_bin is not None
            and support_bin_cap > 0
            and support_bin_counts.get(support_bin, 0) >= support_bin_cap
        ):
            return False
        return not (_placement_chunks(p) & used_chunks)

    def score(p: Placement, card: Card, *, allow_family_repeat: bool = False) -> float:
        s = 0.0
        cost = _placement_message_cost(p, card)
        source = normalized_response_source(p, card)
        if card.family not in seen_families:
            s += 3.0
        elif allow_family_repeat:
            # Final density fill may reuse a family when the one-family rule
            # blocks the 8-question floor, but repeats should lose to genuinely
            # new independent skills whenever those remain feasible.
            s -= 3.0 + 1.5 * family_counts.get(card.family, 0)
        if p.mechanism not in seen_mechs:
            s += 2.0
        if card.answer_form not in seen_aforms:
            s += 1.0
        answer_bucket = _answer_form_bucket(card)
        if answer_form_bucket_selected(answer_bucket) < answer_form_targets.get(answer_bucket, 0):
            s += 1.2
        way = _card_question_way(card)
        evidence_type = _card_evidence_type(card)
        task_subtype = _card_task_subtype(card)
        answer_mode = _placement_answer_mode(p, card)
        if way not in seen_question_ways:
            s += 1.0
        if evidence_type not in seen_evidence_types:
            s += 0.8
        if task_subtype not in seen_task_subtypes:
            s += 0.8
        if task_mode_selected(task_subtype, answer_mode) < task_mode_targets.get(
            (task_subtype, answer_mode),
            0,
        ):
            s += 2.0
        if question_way_selected(way) < question_way_targets.get(way, 0):
            s += 1.4
        if evidence_type_selected(evidence_type) < evidence_type_targets.get(evidence_type, 0):
            s += 0.8
        if task_subtype_selected(task_subtype) < task_subtype_targets.get(task_subtype, 0):
            s += 1.2
        s += 1.0  # base for new card
        if p.mechanism == "recall_demo":
            if selected_message_cost["recall_response_rows"] < recall_response_row_target:
                s += 3.4
            if p.difficulty_mode == "recall_deep":
                s += 1.0
            elif p.difficulty_mode == "recall_mid":
                s += 0.6
            elif p.difficulty_mode == "recall_near":
                s += 0.2
        elif p.mechanism == "memory_direct":
            # Useful hard negatives against overusing recall, but do not let
            # them crowd out true recall slots.
            s += 0.15
        if source == "future" and selected_message_cost["future_response_rows"] < future_response_row_target:
            s += 1.4
        if source == "multi":
            if selected_message_cost["multi_response_rows"] < multi_response_row_target:
                s += 2.8
            # Multi-answer tasks are valuable boundary examples, but each
            # extra answer row competes with many direct/recall decisions.
            s -= max(0, cost.response_rows - 1) * 0.30
            s -= min(cost.multi_active_rows / 36.0, 2.0)
        elif source == "direct":
            if selected_message_cost["direct_response_rows"] < direct_response_row_target:
                s += 0.35
            else:
                s -= 0.8
            if (
                selected_message_cost["recall_response_rows"] < recall_response_row_target
                or selected_message_cost["future_response_rows"] < future_response_row_target
                or selected_message_cost["multi_response_rows"] < multi_response_row_target
            ):
                s -= 0.35
        if used_ask:
            min_dist = min(abs(p.ask_chunk - x) for x in used_ask)
            s += min(min_dist / SPREAD_SCORE_DENOM, SPREAD_SCORE_CAP)
        else:
            s += SPREAD_SCORE_CAP
        ask_bin = _chunk_position_bin(int(p.ask_chunk), num_chunks)
        if ask_bin_selected(ask_bin) < position_bin_target:
            s += POSITION_BIN_SCORE_WEIGHT
        response_bin = _placement_response_bin(p, num_chunks)
        if response_bin is not None and response_bin_selected(response_bin) < position_bin_target:
            s += POSITION_BIN_SCORE_WEIGHT * 0.75
        if (
            response_bin is not None
            and source in source_response_row_targets
            and source_response_bin_selected(source, response_bin) == 0
        ):
            s += SOURCE_TIME_BIN_SCORE_WEIGHT
        # Long active spans are legitimate for one-question multi-answer
        # tasks, but they reduce the number of independent Q/A episodes
        # in a trajectory. Penalize them rather than banning them.
        s -= min(_placement_span_len(p) / 12.0, 6.0)
        s += FAMILY_SELECTION_BOOST.get(card.family, 0.0)
        return s

    def take_best(
        predicate,
        *,
        allow_family_repeat: bool = False,
        relax_semantic_caps: bool = False,
        relax_timing_caps: bool = False,
        relax_ask_gap: bool = False,
        relax_message_caps: bool = False,
        ask_gap_override: Optional[int] = None,
        family_repeat_cap: Optional[int] = None,
    ) -> bool:
        best_score = -1e9
        best_idx = -1
        for i, (p, card) in enumerate(pool):
            if (
                not feasible(
                    p,
                    card,
                    allow_family_repeat=allow_family_repeat,
                    relax_semantic_caps=relax_semantic_caps,
                    relax_timing_caps=relax_timing_caps,
                    relax_ask_gap=relax_ask_gap,
                    relax_message_caps=relax_message_caps,
                    ask_gap_override=ask_gap_override,
                    family_repeat_cap=family_repeat_cap,
                )
                or not predicate(p, card)
            ):
                continue
            s = score(p, card, allow_family_repeat=allow_family_repeat)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            return False
        p, card = pool.pop(best_idx)
        selected.append(p)
        seen_families.add(card.family)
        family_counts[card.family] = family_counts.get(card.family, 0) + 1
        seen_mechs.add(p.mechanism)
        seen_aforms.add(card.answer_form)
        seen_question_ways.add(_card_question_way(card))
        seen_evidence_types.add(_card_evidence_type(card))
        seen_task_subtypes.add(_card_task_subtype(card))
        seen_cards.add(card.card_id)
        used_chunks.update(_placement_chunks(p))
        used_ask.append(p.ask_chunk)
        selected_message_cost.update(_placement_message_cost(p, card).as_counter())
        support_bin = _primary_support_bin(card)
        if support_bin is not None:
            support_bin_counts[support_bin] = support_bin_counts.get(support_bin, 0) + 1
        return True

    unanswerable_reserve_key = min(
        (
            card.card_id
            for p, card in pool
            if p.mechanism == "recall_demo" and _is_unanswerable_card(card)
        ),
        default="",
    )
    reserve_unanswerable = bool(unanswerable_reserve_key) and (
        stable_mod(unanswerable_reserve_key, "abstention_reserve", modulo=100)
        < HLD_ABSTENTION_RESERVE_PERCENT
    )

    # One clean source-row reserve phase. It protects final response-row
    # proportions before broad semantic filling can consume timeline windows.
    # Keep the first reserve family-diverse; bounded repeat density fill runs
    # later only if the trajectory is still below the row target.
    for source in ("multi", "future", "recall", "direct"):
        reserve_source_rows(source, allow_family_repeat=(source == "future"))

    # Keep OVO-like benchmark skills as the majority, while reserving one
    # controlled slot for our streaming-agent variants. Pick the preferred
    # unique family by a stable per-video bucket so CR5 does not monopolize the
    # slice merely because it has the highest selection boost.
    while (
        len(selected) < max_q
        and own_selected() < own_target
        and take_best(
            lambda _p, card: card.family == _weighted_bucket_choice(
                selection_key, OURS_FAMILY_MIX_PERCENT
            )
        )
    ):
        pass
    while (
        len(selected) < max_q
        and own_selected() < own_target
        and take_best(lambda _p, card: _is_ours_card(card))
    ):
        pass

    # Keep a controlled slice of benchmark-compatible wording variants:
    # same task/answer form as OVO/StreamingBench, but different surface
    # question phrasing. These improve robustness without changing the answer
    # space or recall boundary.
    while (
        len(selected) < max_q
        and variant_selected() < variant_target
        and take_best(lambda _p, card: _is_benchmark_variant_card(card))
    ):
        pass

    for (subtype, mode), _target in sorted(
        task_mode_targets.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    ):
        while (
            len(selected) < max_q
            and task_mode_selected(subtype, mode) < task_mode_targets.get((subtype, mode), 0)
            and take_best(
                lambda p, card, subtype=subtype, mode=mode: (
                    _card_task_subtype(card) == subtype
                    and _placement_answer_mode(p, card) == mode
                )
            )
        ):
            pass

    for subtype, _target in sorted(
        task_subtype_targets.items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        while (
            len(selected) < max_q
            and task_subtype_selected(subtype) < task_subtype_targets.get(subtype, 0)
            and take_best(
                lambda _p, card, subtype=subtype: _card_task_subtype(card) == subtype
            )
        ):
            pass

    while (
        len(selected) < max_q
        and future_delayed_selected() < future_delayed_target
        and take_best(
            lambda p, card: (
                _placement_timing_bucket(p, card) == "future_delayed"
                and _placement_span_len(p) <= span_cap
            )
        )
    ):
        pass

    while (
        len(selected) < max_q
        and future_current_cue_selected() < future_current_cue_target
        and take_best(
            lambda p, card: _placement_timing_bucket(p, card) == "future_current_cue"
        )
    ):
        pass

    while (
        len(selected) < max_q
        and current_direct_selected() < current_direct_target
        and take_best(lambda p, _card: _is_current_direct_placement(p))
    ):
        pass

    while (
        len(selected) < max_q
        and past_state_direct_selected() < past_state_direct_target
        and take_best(
            lambda p, card: _placement_timing_bucket(p, card) == "past_state_direct"
        )
    ):
        pass

    # Multi-answer state probes are useful boundary data, but they expand into
    # several response rows and many pending silent rows. Reserve them only
    # after the one-shot direct/recall/future boundaries have claimed space.
    while (
        len(selected) < max_q
        and multi_answer_selected() < multi_answer_target
        and take_best(
            lambda p, card: (
                _placement_timing_bucket(p, card) == "multi_answer"
                and _placement_span_len(p) <= span_cap
            ),
        )
    ):
        pass

    # OVO REC/SSR/CRR are answer-space-sensitive tasks. Select them through
    # separate soft reserves after direct placement so one state-probe family
    # cannot consume the whole trajectory before current/retrieval examples.
    for family in ("F5", "CRR1", "F7"):
        percent = STATE_PROBE_FAMILY_RESERVE_PERCENT.get(family, 0)
        if not family_reserve_enabled(family, percent):
            continue
        while (
            len(selected) < max_q
            and not family_selected(family)
            and take_best(lambda _p, card, family=family: card.family == family)
        ):
            pass

    while (
        len(selected) < max_q
        and state_probe_selected() < state_probe_target
        and take_best(lambda p, card: _is_state_probe_placement(p, card))
    ):
        pass

    for answer_bucket in ANSWER_FORM_BUCKETS:
        while (
            len(selected) < max_q
            and answer_form_bucket_selected(answer_bucket) < answer_form_targets.get(answer_bucket, 0)
            and take_best(
                lambda _p, card, answer_bucket=answer_bucket: _answer_form_bucket(card) == answer_bucket
            )
        ):
            pass

    target_non_mcq = non_mcq_floor()
    while (
        len(selected) < max_q
        and non_mcq_selected() < target_non_mcq
        and take_best(lambda _p, card: card.answer_form != "multiple_choice")
    ):
        pass

    if len(selected) < max_q:
        take_best(
            lambda p, card: (
                _is_ours_card(card)
                and p.mechanism == "silent_then_response"
                and _placement_span_len(p) >= 12
            )
        )

    if len(selected) < max_q:
        take_best(lambda p, card: p.mechanism == "recall_demo" and _is_unanswerable_card(card))

    while (
        len(selected) < max_q
        and normal_recall_selected() < target_recall
        and take_best(lambda p, card: is_normal_recall_placement(p, card))
    ):
        pass

    if reserve_unanswerable and len(selected) < max_q:
        take_best(lambda p, card: p.mechanism == "recall_demo" and _is_unanswerable_card(card))

    while len(selected) < max_q and pool:
        if not take_best(lambda _p, _card: True):
            break

    # The first pass keeps one-family-per-trajectory for benchmark diversity,
    # but long multi_emit windows and no-overlap can otherwise leave many
    # videos below the intended 8-question floor. Fill the remaining floor with
    # short, single-emit placements first; this raises independent Q/A density
    # without inflating REC/SSR/CRR active spans or recall frequency.
    floor_q = min(max_q, MIN_QUESTIONS_PER_TRAJECTORY)
    while (
        len(selected) < floor_q
        and take_best(
            lambda p, card: card.question_type == "single_emit"
            and p.mechanism in {"direct", "memory_direct", "silent_then_response"},
            allow_family_repeat=True,
        )
    ):
        pass
    while (
        len(selected) < floor_q
        and take_best(
            lambda _p, card: card.question_type == "single_emit",
            allow_family_repeat=True,
        )
    ):
        pass

    # Floor fill above only guarantees short videos do not collapse below the
    # minimum. For medium/long videos, add a final density fill to a
    # conservative fraction of the adaptive target. It still preserves one
    # active question, ask spacing, no duplicate card, recall/source caps,
    # bounded family repeats, and the semantic/timing bucket caps.
    target_density_recall = target_recall
    while (
        len(selected) < density_floor_q
        and normal_recall_selected() < target_density_recall
        and take_best(
            lambda p, card: is_normal_recall_placement(p, card),
            allow_family_repeat=True,
            relax_ask_gap=True,
        )
    ):
        pass
    while (
        len(selected) < density_floor_q
        and selected_message_cost["multi_response_rows"] < multi_response_row_target
        and take_best(
            lambda p, card: (
                _placement_timing_bucket(p, card) == "multi_answer"
                and _placement_span_len(p) <= span_cap
            ),
            allow_family_repeat=True,
            relax_ask_gap=True,
        )
    ):
        pass
    while (
        len(selected) < density_floor_q
        and take_best(
            lambda p, card: (
                card.question_type == "single_emit"
                and p.mechanism in {"direct", "memory_direct", "silent_then_response"}
                and (
                    p.mechanism != "silent_then_response"
                    or _placement_span_len(p) <= span_cap
                )
            ),
            allow_family_repeat=True,
            relax_ask_gap=True,
        )
    ):
        pass
    while (
        len(selected) < density_floor_q
        and take_best(
            lambda p, _card: _placement_span_len(p) <= span_cap,
            allow_family_repeat=True,
            relax_ask_gap=True,
        )
    ):
        pass

    # Raw trajectories keep every silent row, so long videos can still end up
    # response-sparse even after the semantic quotas are satisfied. This final
    # fill is deliberately row-aware: it may exceed the adaptive question cap,
    # but only with non-overlapping legal placements and only while source
    # ratios stay inside the direct/recall/future/multi envelope.
    raw_density_question_cap = density_selection_q

    def source_response_rows(source: str) -> int:
        if source == "direct":
            return int(selected_message_cost["direct_response_rows"])
        if source == "recall":
            return int(
                selected_message_cost["recall_response_rows"]
                + selected_message_cost["hld_recall_response_rows"]
            )
        if source == "future":
            return int(selected_message_cost["future_response_rows"])
        if source == "multi":
            return int(selected_message_cost["multi_response_rows"])
        return 0

    def source_cost_rows(cost: PlacementMessageCost, source: str) -> int:
        if source == "direct":
            return int(cost.direct_response_rows)
        if source == "recall":
            return int(cost.recall_response_rows + cost.hld_recall_response_rows)
        if source == "future":
            return int(cost.future_response_rows)
        if source == "multi":
            return int(cost.multi_response_rows)
        return 0

    def raw_quota_source(p: Placement, card: Card) -> str:
        source = _placement_response_source(p, card)
        if source == ANSWER_MODE_HLD_RECALL:
            return "recall"
        return source

    def raw_source_target(source: str) -> int:
        return max(
            0,
            _floor_fraction(
                raw_density_target_rows,
                RAW_RESPONSE_SOURCE_TARGET_FRACTION.get(source, 0.0),
            ),
        )

    def raw_source_cap(source: str) -> int:
        return max(
            raw_source_target(source),
            _ceil_fraction(
                raw_density_cap_rows,
                RAW_RESPONSE_SOURCE_MAX_FRACTION.get(source, 0.0),
            ),
        )

    def take_best_raw_density(
        *,
        require_under_target: bool,
        relax_semantic_caps: bool = False,
    ) -> bool:
        if len(selected) >= raw_density_question_cap:
            return False
        if selected_message_cost["response_rows"] >= max(raw_density_min_rows, raw_density_target_rows):
            return False
        best_score = -1e9
        best_idx = -1
        for i, (p, card) in enumerate(pool):
            source = raw_quota_source(p, card)
            if source not in RAW_RESPONSE_SOURCE_TARGET_FRACTION:
                continue
            cost = _placement_message_cost(p, card)
            source_cost = source_cost_rows(cost, source)
            if source_cost <= 0:
                continue
            if (
                selected_message_cost["response_rows"] + cost.response_rows
                > raw_density_cap_rows
            ):
                continue
            if source_response_rows(source) + source_cost > raw_source_cap(source):
                continue
            source_need = raw_source_target(source) - source_response_rows(source)
            if require_under_target and source_need <= 0:
                continue
            if source == "multi" and _placement_span_len(p) > span_cap:
                continue
            if not feasible(
                p,
                card,
                allow_family_repeat=True,
                relax_semantic_caps=relax_semantic_caps,
                relax_ask_gap=True,
                relax_message_caps=True,
                ask_gap_override=RAW_RESPONSE_DENSITY_ASK_GAP_CHUNKS,
                family_repeat_cap=RAW_RESPONSE_DENSITY_FAMILY_REPEAT_CAP,
            ):
                continue
            s = score(p, card, allow_family_repeat=True)
            s += min(max(source_need, 0) / 2.0, 6.0)
            if source in {"recall", "future"}:
                s += 1.6
            elif source == "multi":
                s += 0.8
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx < 0:
            return False
        p, card = pool.pop(best_idx)
        selected.append(p)
        seen_families.add(card.family)
        family_counts[card.family] = family_counts.get(card.family, 0) + 1
        seen_mechs.add(p.mechanism)
        seen_aforms.add(card.answer_form)
        seen_question_ways.add(_card_question_way(card))
        seen_evidence_types.add(_card_evidence_type(card))
        seen_task_subtypes.add(_card_task_subtype(card))
        seen_cards.add(card.card_id)
        used_chunks.update(_placement_chunks(p))
        used_ask.append(p.ask_chunk)
        selected_message_cost.update(_placement_message_cost(p, card).as_counter())
        support_bin = _primary_support_bin(card)
        if support_bin is not None:
            support_bin_counts[support_bin] = support_bin_counts.get(support_bin, 0) + 1
        return True

    while take_best_raw_density(require_under_target=True):
        pass
    while take_best_raw_density(require_under_target=False):
        pass
    while (
        selected_message_cost["response_rows"] < raw_density_min_rows
        and take_best_raw_density(
            require_under_target=False,
            relax_semantic_caps=True,
        )
    ):
        pass

    return selected


# ---------------------------------------------------------------------------
# Recall augmentation — independent of rollout
# ---------------------------------------------------------------------------


def assign_recall_noise(
    placements: List[Placement],
    rng: random.Random,
    cards_by_id: Optional[Dict[str, Card]] = None,
) -> None:
    """Assign recall demonstrations without creating no-answer questions.

    Mutates placement.recall_at in place.

    - recall_demo: recall at the response chunk with oracle/noisy evidence.
    - silent_then_response/future questions never recall: their evidence is in
      the future or the current trigger, so recall is not the minimal action.
    - F5/CRR1/PN1 multi-answer tasks and F7 immediate SSR rows never recall in
      production; they train state/probe tracking, not visual retrieval.
    - memory_direct never gets upgraded to recall here. If a question needs
      visual recall, it must be selected as recall_demo up front.

    Every selected question still has a grounded answer in the same
    trajectory. If failure-mode data is needed later, it should live in a
    separate diagnostic dataset, not in SFT/RL/eval training trajectories.
    """
    def mark_response_recall(p: Placement, c: int, reason: str) -> None:
        r = rng.random()
        p.recall_at[int(c)] = "oracle" if r < RECALL_OK_RATE else "noisy"
        p.recall_reason_at[int(c)] = reason

    def schedule_wait_recalls(p: Placement) -> None:
        response_chunks = sorted(
            int(c) for c, (kind, _value) in p.chunk_actions.items()
            if kind == "response"
        )
        if not response_chunks:
            return
        first_response = response_chunks[0]
        lead = first_response - int(p.ask_chunk)
        if lead < RECALL_WAIT_MIN_LEAD:
            return
        exact_candidates: List[Tuple[int, str]] = []
        if lead >= RECENT_THINKS_HORIZON + RECALL_WAIT_MIN_LEAD:
            exact_candidates.append((
                int(p.ask_chunk) + RECENT_THINKS_HORIZON + 1,
                "long_wait_history_check",
            ))
        if lead >= 36:
            fractions = (0.25, 0.50, 0.75)
        elif lead >= 16:
            fractions = (0.33, 0.67)
        else:
            fractions = (0.50,)
        reasons = ("memory_unclear", "related_history_check", "pre_answer_check")
        chosen: List[int] = []
        for c, reason in exact_candidates:
            c = min(first_response - 1, max(int(p.ask_chunk) + 1, int(c)))
            if c <= 0 or c in chosen:
                continue
            if p.chunk_actions.get(c, ("", ""))[0] != "silent":
                continue
            p.recall_at[c] = "not_yet"
            p.recall_reason_at[c] = reason
            chosen.append(c)
        for i, frac in enumerate(fractions[:RECALL_WAIT_PROBE_MAX]):
            c = int(p.ask_chunk) + max(2, round(lead * frac))
            c = min(first_response - 1, max(int(p.ask_chunk) + 1, c))
            if c <= 0 or c in chosen:
                continue
            if p.chunk_actions.get(c, ("", ""))[0] != "silent":
                continue
            p.recall_at[c] = "not_yet"
            p.recall_reason_at[c] = reasons[min(i, len(reasons) - 1)]
            chosen.append(c)

    def response_needs_historical_recall(
        card: Optional[Card],
        c: int,
    ) -> bool:
        if card is None:
            return False
        support = []
        for raw in card.grounding_frames or []:
            try:
                support.append(int(raw))
            except (TypeError, ValueError):
                continue
        if not support:
            return False
        return any(int(c) - s > VISUAL_WINDOW_CHUNKS for s in support)

    for p in placements:
        if p.mechanism == "recall_demo":
            card = cards_by_id.get(p.card_id) if cards_by_id else None
            policy = p.support_policy or (
                infer_card_policy_fields(card)["support_policy"] if card else ""
            )
            if policy != SUPPORT_HISTORICAL_VISUAL_RECALL:
                continue
            for c, (kind, _) in p.chunk_actions.items():
                if kind != "response":
                    continue
                mark_response_recall(p, int(c), "historical_answer")
        elif p.mechanism in {"silent_then_response", "multi_emit", "memory_direct"}:
            continue


# ---------------------------------------------------------------------------
# Trajectory rendering — produces SFT samples
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    chunk_idx: int
    sample_kind: str               # silent | response | recall+response
    placement_id: str
    card_id: str
    ask_chunk: int
    mechanism: PlacementMechanism
    response_text: str = ""        # empty for silent
    recall_query: Optional[Dict] = None
    recall_result_kind: Optional[str] = None  # oracle/noisy
    extra: Dict = field(default_factory=dict)


def render_placement(
    card: Card,
    placement: Placement,
) -> List[Sample]:
    """Walk the placement's gold window and emit one Sample per active chunk."""
    samples: List[Sample] = []
    pid = f"{card.card_id}@{placement.ask_chunk}"
    for c in sorted(placement.chunk_actions.keys()):
        kind, value = placement.chunk_actions[c]
        if kind == "silent":
            if placement.recall_at.get(c) == "not_yet":
                samples.append(Sample(
                    chunk_idx=c,
                    sample_kind="recall+silent",
                    placement_id=pid,
                    card_id=card.card_id,
                    ask_chunk=placement.ask_chunk,
                    mechanism=placement.mechanism,
                    recall_query=card.recall_query,
                    recall_result_kind="not_yet",
                    extra={"recall_reason": placement.recall_reason_at.get(c, "")},
                ))
                continue
            samples.append(Sample(
                chunk_idx=c,
                sample_kind="silent",
                placement_id=pid,
                card_id=card.card_id,
                ask_chunk=placement.ask_chunk,
                mechanism=placement.mechanism,
            ))
            continue
        # response chunk
        if c in placement.recall_at:
            rkind = placement.recall_at[c]
            if rkind == "failure":
                raise ValueError(
                    f"recall failure is disabled in production trajectories: "
                    f"card={card.card_id} chunk={c}"
                )
            samples.append(Sample(
                chunk_idx=c,
                sample_kind="recall+response",
                placement_id=pid,
                card_id=card.card_id,
                ask_chunk=placement.ask_chunk,
                mechanism=placement.mechanism,
                response_text=value,
                recall_query=card.recall_query,
                recall_result_kind=rkind,
                extra={"recall_reason": placement.recall_reason_at.get(c, "")},
            ))
        else:
            samples.append(Sample(
                chunk_idx=c,
                sample_kind="response",
                placement_id=pid,
                card_id=card.card_id,
                ask_chunk=placement.ask_chunk,
                mechanism=placement.mechanism,
                response_text=value,
            ))
    return samples


def render_video_samples(
    cards: List[Card],
    placements_by_card: Dict[str, List[Placement]],
    num_chunks: int,
    evidence: Optional[List[Dict]] = None,
    rng: Optional[random.Random] = None,
    compression_event_chunks: Optional[List[int]] = None,
) -> List[Sample]:
    """Render placements + dense timeline-silent samples.

    Multi-placement priority: response > recall+response > recall+silent > silent.
    """
    if rng is None:
        rng = random.Random(0)
    per_chunk: Dict[int, List[Tuple[str, str, Placement, Card, Dict]]] = {}
    cards_by_id = {c.card_id: c for c in cards}
    for cid, plcs in placements_by_card.items():
        card = cards_by_id[cid]
        for p in plcs:
            for s in render_placement(card, p):
                per_chunk.setdefault(s.chunk_idx, []).append(
                    (s.sample_kind, s.response_text, p, card, dict(s.extra or {}))
                )

    PRIORITY = {
        "response": 5,
        "recall+response": 4,
        "recall+silent": 3,
        "silent": 2,
        "compress_silent": 1,    # NEW silent situation type
        "patrol": 0,              # legacy cached rows only; new data emits silent
    }

    compress_set = set(compression_event_chunks or [])

    all_samples: List[Sample] = []
    for c in range(num_chunks):
        candidates = per_chunk.get(c, [])
        if c in compress_set:
            all_samples.append(Sample(
                chunk_idx=c,
                sample_kind="compress_silent",
                placement_id="",
                card_id="",
                ask_chunk=-1,
                mechanism="multi_emit",
                extra={"role": "compress_event"},
            ))
        if not candidates:
            if c in compress_set:
                continue
            # Dense timeline: every non-active, non-compress chunk must be an
            # ordinary silent action. Do not emit a separate "patrol" label;
            # downstream SFT/RL should only see the agent protocol actions.
            all_samples.append(Sample(
                chunk_idx=c,
                sample_kind="silent",
                placement_id="",
                card_id="",
                ask_chunk=-1,
                mechanism="multi_emit",  # placeholder
            ))
            continue
        if len(candidates) > 1:
            owners = [
                f"{card.card_id}@{p.ask_chunk}:{kind}"
                for kind, _value, p, card, _extra in candidates
            ]
            raise ValueError(
                f"overlapping question placements at chunk {c}: "
                + ", ".join(owners)
            )
        candidates.sort(key=lambda t: -PRIORITY[t[0]])
        kind, value, p, card, extra = candidates[0]
        all_samples.append(Sample(
            chunk_idx=c,
            sample_kind=kind,
            placement_id=f"{card.card_id}@{p.ask_chunk}",
            card_id=card.card_id,
            ask_chunk=p.ask_chunk,
            mechanism=p.mechanism,
            response_text=value,
            recall_query=card.recall_query if "recall" in kind else None,
            recall_result_kind=p.recall_at.get(c) if "recall" in kind else None,
            extra=extra,
        ))
    return all_samples


# ---------------------------------------------------------------------------
# Sanity helpers (used by simulator)
# ---------------------------------------------------------------------------


def is_response_kind(k: str) -> bool:
    return k in ("response", "recall+response")


def is_silent_kind(k: str) -> bool:
    return k in ("silent", "patrol", "recall+silent", "compress_silent")
