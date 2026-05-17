"""Deterministic Pass3 slot planning.

Pass3A should not independently choose every evidence position. This planner
first assigns per-video card slots with fixed support/probe/answer chunks, then
the LLM only turns those slots into natural questions and options.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .stable_hash import stable_mod
from .placement.design import (
    EVIDENCE_ABSENCE,
    EVIDENCE_ACTION_EVENT,
    EVIDENCE_CAUSAL_CONTEXT,
    EVIDENCE_EMOTION_CONTEXT,
    EVIDENCE_FUTURE_CUE,
    EVIDENCE_FUTURE_TRIGGER,
    EVIDENCE_GLOBAL_CONTEXT,
    EVIDENCE_LIVE_STATE_CHANGE,
    EVIDENCE_MULTIMODAL_ALIGNMENT,
    EVIDENCE_OBJECT_ATTRIBUTE,
    EVIDENCE_PERSON_RELATION,
    EVIDENCE_REPEATED_EVENT,
    EVIDENCE_SOURCE_DISCRIMINATION,
    EVIDENCE_SPATIAL_RELATION,
    EVIDENCE_STATUS_PROBE,
    EVIDENCE_TEMPORAL_ORDER,
    EVIDENCE_TEXT_OCR,
    QUESTION_STYLE_BENCHMARK_CORE,
    QUESTION_STYLE_BENCHMARK_VARIANT,
    QUESTION_STYLE_OURS_UNIQUE,
    QUESTION_WAY_ACTION_RECOGNITION,
    QUESTION_WAY_CAUSAL_INTENT,
    QUESTION_WAY_CURRENT_STATUS,
    QUESTION_WAY_EMOTION_CONTEXT,
    QUESTION_WAY_EVIDENCE_SUFFICIENCY,
    QUESTION_WAY_FUTURE_PREDICTION,
    QUESTION_WAY_LIVE_NARRATION,
    QUESTION_WAY_MULTIMODAL_ALIGNMENT,
    QUESTION_WAY_OBJECT_ATTRIBUTE,
    QUESTION_WAY_PERSON_IDENTITY,
    QUESTION_WAY_PROACTIVE_OUTPUT,
    QUESTION_WAY_REPEATED_COUNT,
    QUESTION_WAY_SCENE_SUMMARY,
    QUESTION_WAY_SEQUENTIAL_REFERENCE,
    QUESTION_WAY_SOURCE_DISCRIMINATION,
    QUESTION_WAY_SPATIAL_RELATION,
    QUESTION_WAY_TEMPORAL_ORDER,
    QUESTION_WAY_TEXT_READOUT,
    QUESTION_WAY_UNANSWERABLE,
    SUPPORT_CURRENT_VISUAL,
    SUPPORT_FUTURE_CURRENT_CUE,
    SUPPORT_HISTORICAL_STATE_MEMORY,
    SUPPORT_HISTORICAL_VISUAL_RECALL,
    SUPPORT_PROBE_STATUS,
    SUPPORT_BIN_SIZE,
    VISUAL_WINDOW_CHUNKS,
)
from .placement.llm_prompts import FAMILY_RULES, QUESTION_TYPE_BY_FAMILY, family_taxonomy


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "and", "are",
    "around", "before", "being", "between", "chunk", "clear", "clearly",
    "could", "does", "during", "from", "have", "into", "near", "only",
    "person", "people", "scene", "shown", "shows", "that", "their", "there",
    "this", "through", "under", "video", "visible", "what", "when", "where",
    "while", "with", "would",
}
_SPATIAL_FAMILIES = {"CR7", "R1", "STU1", "OJR1"}
_PAST_VISUAL_FAMILIES = {"N1", "P1", "HLD1"}
_CROSS_EVENT_FAMILIES = {"CR1", "CR2", "CR4", "CR5"}
_CURRENT_ACTION_FAMILIES = {"ACR1", "CR3", "F6", "F7", "E2"}
_MULTI_EMIT_MIN_CHUNKS = {"F5": 2, "CRR1": 3, "PN1": 2}
_MAX_EXACT_ANSWER_CHUNK_USES = 1
_MAX_CHUNK0_ANSWER_USES = 1
PASS3A_RESPONSE_ROW_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_RESPONSE_ROW_TARGET_FRACTION", "0.135")
)
PASS3A_DIRECT_SOURCE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_DIRECT_SOURCE_TARGET_FRACTION", "0.525")
)
PASS3A_RECALL_SOURCE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_RECALL_SOURCE_TARGET_FRACTION", "0.275")
)
PASS3A_FUTURE_SOURCE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_FUTURE_SOURCE_TARGET_FRACTION", "0.10")
)
PASS3A_MULTI_SOURCE_TARGET_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_MULTI_SOURCE_TARGET_FRACTION", "0.10")
)
PASS3A_SLOT_RETRY_HEADROOM_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_SLOT_RETRY_HEADROOM_FRACTION", "0.25")
)
PASS3A_RECALL_SLOT_RETRY_HEADROOM_FRACTION = float(
    os.environ.get("THINKSTREAM_PASS3A_RECALL_SLOT_RETRY_HEADROOM_FRACTION", "0.50")
)
PASS3A_DIRECT_MIN_RETRY_SLOTS = int(
    os.environ.get("THINKSTREAM_PASS3A_DIRECT_MIN_RETRY_SLOTS", "1")
)
PASS3A_RECALL_MIN_RETRY_SLOTS = int(
    os.environ.get("THINKSTREAM_PASS3A_RECALL_MIN_RETRY_SLOTS", "1")
)
PASS3A_FUTURE_MIN_RETRY_SLOTS = int(
    os.environ.get("THINKSTREAM_PASS3A_FUTURE_MIN_RETRY_SLOTS", "2")
)
PASS3A_MULTI_MIN_RETRY_SLOTS = int(
    os.environ.get("THINKSTREAM_PASS3A_MULTI_MIN_RETRY_SLOTS", "1")
)
PASS3A_MULTI_EXPECTED_RESPONSE_ROWS_PER_SLOT = float(
    os.environ.get("THINKSTREAM_PASS3A_MULTI_EXPECTED_RESPONSE_ROWS_PER_SLOT", "3.0")
)
PASS3A_F5_GLOBAL_MAX_ACTIVE_SPAN = int(
    os.environ.get("THINKSTREAM_PASS3A_F5_GLOBAL_MAX_ACTIVE_SPAN", "64")
)
PASS3A_SUPPORT_BIN_BASE_CAP = int(
    os.environ.get("THINKSTREAM_PASS3A_SUPPORT_BIN_BASE_CAP", "4")
)
PASS3A_SUPPORT_BIN_LONG_STEP = int(
    os.environ.get("THINKSTREAM_PASS3A_SUPPORT_BIN_LONG_STEP", "128")
)
PASS3A_SUPPORT_BIN_MAX_CAP = int(
    os.environ.get("THINKSTREAM_PASS3A_SUPPORT_BIN_MAX_CAP", "8")
)

_DIRECT_SUPPLY_FAMILIES = ("C1", "ACR1", "STU1", "OJR1", "R1", "CR3", "CR7", "F6", "F7")
_RECALL_SUPPLY_FAMILIES = ("N1", "P1", "HLD1", "CR1", "CR2", "CR4", "CR5")
_FUTURE_SUPPLY_FAMILIES = ("E2", "CR5")
_MULTI_SUPPLY_FAMILIES = ("F5", "CRR1", "PN1")

SLOT_METADATA_KEYS = (
    "legacy_family_id",
    "task_family",
    "task_subtype",
    "timing_type",
    "readable_task_name",
    "slot_group",
    "slot_subtype",
    "temporal_bucket",
    "benchmark_source",
    "benchmark_task",
    "answer_behavior",
    "question_goal",
    "placement_hint",
)


FAMILY_SLOT_DEFAULTS: Dict[str, Dict[str, str]] = {
    "N1": {
        "question_way": QUESTION_WAY_OBJECT_ATTRIBUTE,
        "evidence_type": EVIDENCE_OBJECT_ATTRIBUTE,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "historical_visual_detail",
        "lifecycle": "past_visual_candidate",
    },
    "P1": {
        "question_way": QUESTION_WAY_OBJECT_ATTRIBUTE,
        "evidence_type": EVIDENCE_OBJECT_ATTRIBUTE,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "historical_visual_detail",
        "lifecycle": "past_visual_candidate",
    },
    "HLD1": {
        "question_way": QUESTION_WAY_UNANSWERABLE,
        "evidence_type": EVIDENCE_ABSENCE,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "historical_abstention_check",
        "lifecycle": "small_unanswerable_negative",
    },
    "CR1": {
        "question_way": QUESTION_WAY_CAUSAL_INTENT,
        "evidence_type": EVIDENCE_CAUSAL_CONTEXT,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "historical_visual_detail",
        "lifecycle": "past_visual_candidate",
    },
    "CR2": {
        "question_way": QUESTION_WAY_TEMPORAL_ORDER,
        "evidence_type": EVIDENCE_TEMPORAL_ORDER,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "historical_visual_detail",
        "lifecycle": "past_visual_candidate",
    },
    "CR4": {
        "question_way": QUESTION_WAY_TEMPORAL_ORDER,
        "evidence_type": EVIDENCE_TEMPORAL_ORDER,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "historical_visual_detail",
        "lifecycle": "past_visual_candidate",
    },
    "CR5": {
        "question_way": QUESTION_WAY_TEMPORAL_ORDER,
        "evidence_type": EVIDENCE_TEMPORAL_ORDER,
        "support_policy": SUPPORT_HISTORICAL_VISUAL_RECALL,
        "temporal_role": "delayed_clue_resolution",
        "lifecycle": "past_or_delayed_clue_candidate",
    },
    "M1": {
        "question_way": QUESTION_WAY_SCENE_SUMMARY,
        "evidence_type": EVIDENCE_GLOBAL_CONTEXT,
        "support_policy": SUPPORT_HISTORICAL_STATE_MEMORY,
        "temporal_role": "global_summary",
        "lifecycle": "state_memory_direct",
    },
    "E2": {
        "question_way": QUESTION_WAY_PROACTIVE_OUTPUT,
        "evidence_type": EVIDENCE_FUTURE_TRIGGER,
        "support_policy": SUPPORT_FUTURE_CURRENT_CUE,
        "temporal_role": "future_event_wait",
        "lifecycle": "future_delayed",
    },
    "F6": {
        "question_way": QUESTION_WAY_FUTURE_PREDICTION,
        "evidence_type": EVIDENCE_FUTURE_CUE,
        "support_policy": SUPPORT_FUTURE_CURRENT_CUE,
        "temporal_role": "future_current_cue",
        "lifecycle": "current_future_cue",
    },
    "CR3": {
        "question_way": QUESTION_WAY_CAUSAL_INTENT,
        "evidence_type": EVIDENCE_CAUSAL_CONTEXT,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "CR7": {
        "question_way": QUESTION_WAY_SPATIAL_RELATION,
        "evidence_type": EVIDENCE_SPATIAL_RELATION,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "R1": {
        "question_way": QUESTION_WAY_SPATIAL_RELATION,
        "evidence_type": EVIDENCE_SPATIAL_RELATION,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "ACR1": {
        "question_way": QUESTION_WAY_ACTION_RECOGNITION,
        "evidence_type": EVIDENCE_ACTION_EVENT,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "STU1": {
        "question_way": QUESTION_WAY_SPATIAL_RELATION,
        "evidence_type": EVIDENCE_SPATIAL_RELATION,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "OJR1": {
        "question_way": QUESTION_WAY_SPATIAL_RELATION,
        "evidence_type": EVIDENCE_SPATIAL_RELATION,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "C1": {
        "question_way": QUESTION_WAY_TEXT_READOUT,
        "evidence_type": EVIDENCE_TEXT_OCR,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_visual",
        "lifecycle": "current_direct",
    },
    "F5": {
        "question_way": QUESTION_WAY_REPEATED_COUNT,
        "evidence_type": EVIDENCE_REPEATED_EVENT,
        "support_policy": SUPPORT_HISTORICAL_STATE_MEMORY,
        "temporal_role": "cumulative_count",
        "lifecycle": "multi_local_or_global_count",
    },
    "F7": {
        "question_way": QUESTION_WAY_CURRENT_STATUS,
        "evidence_type": EVIDENCE_STATUS_PROBE,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "current_step_status",
        "lifecycle": "current_direct",
    },
    "CRR1": {
        "question_way": QUESTION_WAY_EVIDENCE_SUFFICIENCY,
        "evidence_type": EVIDENCE_STATUS_PROBE,
        "support_policy": SUPPORT_PROBE_STATUS,
        "temporal_role": "crr_sufficiency_probe",
        "lifecycle": "multi_local_sufficiency",
    },
    "PN1": {
        "question_way": QUESTION_WAY_LIVE_NARRATION,
        "evidence_type": EVIDENCE_LIVE_STATE_CHANGE,
        "support_policy": SUPPORT_CURRENT_VISUAL,
        "temporal_role": "live_narration",
        "lifecycle": "multi_local_live_narration",
    },
}


FAMILY_SLOT_META: Dict[str, Dict[str, str]] = {
    "N1": {
        "slot_group": "past_memory",
        "slot_subtype": "appearance_entity_memory",
        "temporal_bucket": "past_visual_recall_candidate",
        "benchmark_source": "ovo_atr",
        "benchmark_task": "ATR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask an OVO-style past appearance/entity question with plausible MC options.",
        "placement_hint": "Current-window direct only if asked while visual support is still active; otherwise recall after the visual window.",
    },
    "P1": {
        "slot_group": "past_memory",
        "slot_subtype": "attribute_memory",
        "temporal_bucket": "past_visual_recall_candidate",
        "benchmark_source": "ovo_atr",
        "benchmark_task": "ATR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask for a visible object/entity attribute such as color, material, state, or appearance.",
        "placement_hint": "Avoid OCR/brand reading; that is handled by C1.",
    },
    "HLD1": {
        "slot_group": "past_memory",
        "slot_subtype": "unanswerable_absence",
        "temporal_bucket": "past_unanswerable",
        "benchmark_source": "ovo_hld",
        "benchmark_task": "HLD",
        "answer_behavior": "single_mcq_unable",
        "question_goal": "Ask a bounded historical question whose correct MC option is Unable to answer.",
        "placement_hint": "Use this sparingly; StreamingBench has no dedicated unanswerable task.",
    },
    "CR1": {
        "slot_group": "past_memory",
        "slot_subtype": "causal_context_history",
        "temporal_bucket": "past_visual_recall_candidate",
        "benchmark_source": "ovo_asi_streaming_causal",
        "benchmark_task": "ASI",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask why/what caused a visible action or outcome using historical context.",
        "placement_hint": "Evidence can be a historical visual detail that may need recall.",
    },
    "CR2": {
        "slot_group": "past_memory",
        "slot_subtype": "temporal_order_history",
        "temporal_bucket": "past_visual_recall_candidate",
        "benchmark_source": "ovo_epm",
        "benchmark_task": "EPM",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask before/after/event-order questions over observed events.",
        "placement_hint": "Grounding should include only events up to the planned answer chunk.",
    },
    "CR4": {
        "slot_group": "past_memory",
        "slot_subtype": "cross_event_reasoning",
        "temporal_bucket": "past_visual_recall_candidate",
        "benchmark_source": "ovo_asi_streaming_context",
        "benchmark_task": "ASI",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask a multi-evidence reasoning question over separated observations.",
        "placement_hint": "Used for StreamingBench contextual/anomaly-style variants when slot_subtype overrides it.",
    },
    "CR5": {
        "slot_group": "temporal_reasoning",
        "slot_subtype": "delayed_clue_resolution",
        "temporal_bucket": "past_or_future_clue",
        "benchmark_source": "ours_variant_of_epm_crr",
        "benchmark_task": "EPM",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask an early ambiguous clue question resolved by later visual evidence.",
        "placement_hint": "Selected as either delayed wait, current-window direct, or recall depending on placement.",
    },
    "M1": {
        "slot_group": "global_context",
        "slot_subtype": "global_scene_summary",
        "temporal_bucket": "global_state_memory",
        "benchmark_source": "streamingbench_scene_summary_ours",
        "benchmark_task": "GLOBAL",
        "answer_behavior": "single_descriptive",
        "question_goal": "Ask about the overall activity, broad scene trajectory, or repeated pattern.",
        "placement_hint": "State memory task, not visual recall.",
    },
    "E2": {
        "slot_group": "future",
        "slot_subtype": "proactive_output",
        "temporal_bucket": "future_delayed",
        "benchmark_source": "streamingbench_proactive_output",
        "benchmark_task": "EPM",
        "answer_behavior": "single_wait_short_exact",
        "question_goal": "Ask now, stay silent, then output a short phrase/number when a future trigger appears.",
        "placement_hint": "No recall; evidence is unavailable until the future trigger.",
    },
    "F6": {
        "slot_group": "future",
        "slot_subtype": "current_future_prediction",
        "temporal_bucket": "current_future_prediction",
        "benchmark_source": "ovo_fpd_streaming_prospective",
        "benchmark_task": "FPD",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask what is likely/about to happen from current visible cues.",
        "placement_hint": "Answer immediately from current cue; do not wait for later verification.",
    },
    "CR3": {
        "slot_group": "current_perception",
        "slot_subtype": "current_intent",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_asi_streaming_causal",
        "benchmark_task": "ASI",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask current visible intent/cause when it is supported by the current scene.",
        "placement_hint": "No recall.",
    },
    "CR7": {
        "slot_group": "current_perception",
        "slot_subtype": "object_persistence_relation",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_ojr",
        "benchmark_task": "OJR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask a current object tracking/persistence/relation question.",
        "placement_hint": "No recall.",
    },
    "R1": {
        "slot_group": "current_perception",
        "slot_subtype": "visible_reasoning",
        "temporal_bucket": "current_direct",
        "benchmark_source": "streamingbench_scene_understanding",
        "benchmark_task": "OJR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask a current visible scene reasoning question.",
        "placement_hint": "No recall.",
    },
    "ACR1": {
        "slot_group": "current_perception",
        "slot_subtype": "current_action_recognition",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_acr_streaming_action",
        "benchmark_task": "ACR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask what action is currently visible.",
        "placement_hint": "No recall.",
    },
    "STU1": {
        "slot_group": "current_perception",
        "slot_subtype": "spatial_temporal_current",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_stu_streaming_spatial",
        "benchmark_task": "STU",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask current spatial relation, visible count, direction, or location.",
        "placement_hint": "No recall.",
    },
    "OJR1": {
        "slot_group": "current_perception",
        "slot_subtype": "object_relation_current",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_ojr",
        "benchmark_task": "OJR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask current relation/state between visible objects.",
        "placement_hint": "No recall.",
    },
    "C1": {
        "slot_group": "current_perception",
        "slot_subtype": "text_readout_current",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_ocr_streaming_text",
        "benchmark_task": "OCR",
        "answer_behavior": "single_mcq",
        "question_goal": "Ask an exact visible text/logo/number question with MC options.",
        "placement_hint": "No recall by default; text is current visual evidence.",
    },
    "F5": {
        "slot_group": "multi_state",
        "slot_subtype": "local_repeated_count",
        "temporal_bucket": "multi_answer",
        "benchmark_source": "ovo_rec",
        "benchmark_task": "REC",
        "answer_behavior": "multi_number_cumulative",
        "question_goal": "Ask cumulative repeated-action counting with digit responses at multiple planned chunks.",
        "placement_hint": "State-memory task; no recall.",
    },
    "F7": {
        "slot_group": "current_perception",
        "slot_subtype": "current_step_status",
        "temporal_bucket": "current_direct",
        "benchmark_source": "ovo_ssr",
        "benchmark_task": "SSR",
        "answer_behavior": "single_binary_current_status",
        "question_goal": "Ask whether a concrete step/action is currently being carried out at the current probe.",
        "placement_hint": "Immediate Yes/No; no recall and no multi-answer state.",
    },
    "CRR1": {
        "slot_group": "multi_state",
        "slot_subtype": "future_sufficiency_status",
        "temporal_bucket": "future_delayed_sufficiency",
        "benchmark_source": "ovo_crr",
        "benchmark_task": "CRR",
        "answer_behavior": "multi_binary_sufficiency",
        "question_goal": "Ask whether current/latest visual evidence is now sufficient to answer the original visual question.",
        "placement_hint": "No recall by default; this trains wait/respond sufficiency boundaries.",
    },
    "PN1": {
        "slot_group": "multi_state",
        "slot_subtype": "live_narration",
        "temporal_bucket": "multi_answer",
        "benchmark_source": "ours_streaming_agent",
        "benchmark_task": "STREAMING_AGENT",
        "answer_behavior": "multi_descriptive_sparse_events",
        "question_goal": "Emit concise narration only at sparse state-change chunks.",
        "placement_hint": "No recall.",
    },
}


SLOT_VARIANT_OVERRIDES: Dict[str, Dict[int, Dict[str, str]]] = {
    "F5": {
        0: {
            "slot_subtype": "global_prefix_count",
            "temporal_bucket": "multi_from_start_count",
            "answer_behavior": "multi_number_prefix_count",
            "question_goal": (
                "OVO REC-style: ask from the beginning how many times a repeated "
                "action has happened so far; emit cumulative digit counts at 6-9 "
                "planned chunks when evidence supports them."
            ),
            "placement_hint": "Prefer ask_chunk=0 in Pass3B; no recall.",
        },
    },
    "CR3": {
        1: {
            "slot_subtype": "emotion_context_current",
            "question_way": QUESTION_WAY_EMOTION_CONTEXT,
            "evidence_type": EVIDENCE_EMOTION_CONTEXT,
            "benchmark_source": "streamingbench_emotion",
            "question_goal": "Ask a current emotion/mood question grounded in visible expression/context.",
        },
    },
    "R1": {
        1: {
            "slot_subtype": "scene_understanding_current",
            "question_way": QUESTION_WAY_SCENE_SUMMARY,
            "evidence_type": EVIDENCE_GLOBAL_CONTEXT,
            "benchmark_source": "streamingbench_scene_understanding",
            "question_goal": "Ask a current scene understanding or clip-summary style MCQ.",
        },
        2: {
            "slot_subtype": "multimodal_alignment",
            "question_way": QUESTION_WAY_MULTIMODAL_ALIGNMENT,
            "evidence_type": EVIDENCE_MULTIMODAL_ALIGNMENT,
            "benchmark_source": "streamingbench_multimodal_alignment",
            "question_goal": (
                "Ask whether visible text/object cues, action context, or scene "
                "description are mutually consistent; ground the answer in visual evidence."
            ),
        },
    },
    "N1": {
        1: {
            "slot_subtype": "epm_event_entity_memory",
            "question_way": QUESTION_WAY_PERSON_IDENTITY,
            "evidence_type": EVIDENCE_PERSON_RELATION,
            "benchmark_source": "ovo_epm_event_entity_memory",
            "benchmark_task": "EPM",
            "answer_behavior": "single_mcq_event_memory",
            "question_goal": (
                "Ask an OVO EPM-style past event-property question: what/who "
                "was involved in a completed action such as put, pick, remove, "
                "use, hand, carry, or place. The answer must be the event "
                "participant from the support chunks, not the currently visible object."
            ),
        },
        2: {
            "slot_subtype": "sequential_reference",
            "question_way": QUESTION_WAY_SEQUENTIAL_REFERENCE,
            "evidence_type": EVIDENCE_PERSON_RELATION,
            "benchmark_source": "streamingbench_sequential_qa",
            "question_goal": (
                "Ask a compact follow-up-style question that resolves an earlier "
                "visual referent; keep it self-contained enough for independent rendering."
            ),
        },
    },
    "P1": {
        1: {
            "slot_subtype": "epm_event_location_memory",
            "question_way": QUESTION_WAY_SPATIAL_RELATION,
            "evidence_type": EVIDENCE_SPATIAL_RELATION,
            "benchmark_source": "ovo_epm_event_location_memory",
            "benchmark_task": "EPM",
            "answer_behavior": "single_mcq_event_memory",
            "question_goal": (
                "Ask an OVO EPM-style where/location question about a completed "
                "event, e.g. where an object was picked from, placed, removed "
                "from, or left. The answer must come from the past support "
                "chunks and should not be a current-frame location guess."
            ),
        },
    },
    "CR4": {
        1: {
            "slot_subtype": "epm_event_count_memory",
            "question_way": QUESTION_WAY_REPEATED_COUNT,
            "evidence_type": EVIDENCE_REPEATED_EVENT,
            "benchmark_source": "ovo_epm_event_count_memory",
            "benchmark_task": "EPM",
            "answer_behavior": "single_mcq_event_count",
            "question_goal": (
                "Ask an OVO EPM-style count/quantity question about a completed "
                "bounded event, such as how many items were put on, removed "
                "from, picked up, or moved. The correct MC option is the final "
                "event count, including none/nothing only when the support "
                "chunks establish that no item was involved."
            ),
        },
        2: {
            "slot_subtype": "source_discrimination",
            "question_way": QUESTION_WAY_SOURCE_DISCRIMINATION,
            "evidence_type": EVIDENCE_SOURCE_DISCRIMINATION,
            "benchmark_source": "streamingbench_source_discrimination",
            "question_goal": (
                "Ask which visual source, cue, or observed evidence supports the "
                "answer, distinguishing visible evidence from inferred background context."
            ),
        },
        3: {
            "slot_subtype": "contextual_misleading_or_anomaly",
            "question_way": QUESTION_WAY_CAUSAL_INTENT,
            "evidence_type": EVIDENCE_CAUSAL_CONTEXT,
            "benchmark_source": "streamingbench_contextual_understanding",
            "question_goal": "Ask a contextual/anomaly-style MCQ that requires comparing the apparent context with actual visual evidence.",
        },
    },
    "CR2": {
        1: {
            "slot_subtype": "asi_adjacent_action_after",
            "question_way": QUESTION_WAY_TEMPORAL_ORDER,
            "evidence_type": EVIDENCE_TEMPORAL_ORDER,
            "benchmark_source": "ovo_asi_adjacent_step_after",
            "benchmark_task": "ASI",
            "answer_behavior": "single_mcq_adjacent_action",
            "question_goal": (
                "Ask what the person does immediately after a named anchor "
                "step. The correct option must be the adjacent next action, "
                "not the anchor action itself and not the current visible step."
            ),
        },
        2: {
            "slot_subtype": "asi_adjacent_action_before",
            "question_way": QUESTION_WAY_TEMPORAL_ORDER,
            "evidence_type": EVIDENCE_TEMPORAL_ORDER,
            "benchmark_source": "ovo_asi_adjacent_step_before",
            "benchmark_task": "ASI",
            "answer_behavior": "single_mcq_adjacent_action",
            "question_goal": (
                "Ask what the person does immediately before a named anchor "
                "step. The correct option must be the adjacent previous action, "
                "not the anchor action itself and not a later/current step."
            ),
        },
    },
    "CR5": {
        1: {
            "slot_subtype": "future_wait_delayed_clue",
            "temporal_bucket": "future_delayed",
            "timing_type": "future_delayed",
            "support_policy": SUPPORT_FUTURE_CURRENT_CUE,
            "temporal_role": "future_event_wait",
            "lifecycle": "future_delayed",
            "question_way": QUESTION_WAY_PROACTIVE_OUTPUT,
            "evidence_type": EVIDENCE_FUTURE_TRIGGER,
            "answer_behavior": "single_wait_mcq",
            "question_goal": (
                "Ask an early delayed-clue question that must wait for a later "
                "visual trigger before answering."
            ),
            "placement_hint": "Future wait only; do not convert this slot to direct or recall.",
        },
        3: {
            "slot_subtype": "future_wait_delayed_clue",
            "temporal_bucket": "future_delayed",
            "timing_type": "future_delayed",
            "support_policy": SUPPORT_FUTURE_CURRENT_CUE,
            "temporal_role": "future_event_wait",
            "lifecycle": "future_delayed",
            "question_way": QUESTION_WAY_PROACTIVE_OUTPUT,
            "evidence_type": EVIDENCE_FUTURE_TRIGGER,
            "answer_behavior": "single_wait_mcq",
            "question_goal": (
                "Ask an early delayed-clue question that must wait for a later "
                "visual trigger before answering."
            ),
            "placement_hint": "Future wait only; do not convert this slot to direct or recall.",
        },
    },
}


BALANCED_BASE_RATES = {
    "HLD1": 100,
    "PN1": 45,
    "M1": 75,
    "N1": 85,
    "P1": 85,
    "CR1": 80,
    "CR2": 80,
    "CR4": 80,
    "CR5": 80,
    "E2": 90,
    "F6": 95,
    "F5": 95,
    "F7": 80,
    "CRR1": 70,
}

BALANCED_SECOND_SLOT_RATES = {
    "HLD1": 45,
    "C1": 90,
    "ACR1": 75,
    "STU1": 55,
    "OJR1": 60,
    "R1": 55,
    "CR3": 65,
    "CR7": 45,
    "F5": 70,
    "F7": 35,
    "CRR1": 25,
    "CR5": 45,
    "E2": 45,
    "F6": 35,
    "N1": 70,
    "P1": 35,
    "CR1": 35,
    "CR2": 35,
    "CR4": 60,
}

BALANCED_THIRD_SLOT_RATES = {
    "C1": 55,
    "OJR1": 35,
    "STU1": 35,
    "ACR1": 35,
    "E2": 30,
    "F6": 25,
    "CR5": 30,
    "CR1": 25,
    "CR2": 25,
    "N1": 35,
    "P1": 25,
    "CR4": 35,
    "R1": 35,
    "F5": 25,
    "CRR1": 20,
}

BALANCED_FOURTH_SLOT_RATES = {
    "C1": 25,
    "N1": 20,
    "CR4": 20,
    "OJR1": 20,
    "STU1": 15,
    "E2": 15,
    "CR5": 15,
    "CR1": 15,
    "CR2": 15,
    "P1": 15,
}


def balanced_family_targets(video_id: str, mode: str = "balanced") -> Dict[str, int]:
    """Return per-family candidate targets from the shared Pass3 policy.

    This is candidate generation, not final trajectory selection. Final ratios
    are enforced again in pass3B, but Pass3A must generate enough current,
    future, multi-answer, past, and ours-unique cards for selection to work.
    """
    if mode != "balanced":
        from .pass3a_cards import PASS3A_TARGETS_BY_FAMILY
        return {
            family: int(PASS3A_TARGETS_BY_FAMILY.get(family, 1))
            for family in FAMILY_RULES
        }

    targets: Dict[str, int] = {}
    for family in FAMILY_RULES:
        rate = int(BALANCED_BASE_RATES.get(family, 100))
        if stable_mod(video_id, f"slot_base_{family}", modulo=100) < max(0, rate):
            targets[family] = 1
    for family, rate in BALANCED_SECOND_SLOT_RATES.items():
        if targets.get(family, 0) > 0 and stable_mod(video_id, f"slot_second_{family}", modulo=100) < max(0, int(rate)):
            targets[family] += 1
    for family, rate in BALANCED_THIRD_SLOT_RATES.items():
        if targets.get(family, 0) >= 2 and stable_mod(video_id, f"slot_third_{family}", modulo=100) < max(0, int(rate)):
            targets[family] += 1
    for family, rate in BALANCED_FOURTH_SLOT_RATES.items():
        if targets.get(family, 0) >= 3 and stable_mod(video_id, f"slot_fourth_{family}", modulo=100) < max(0, int(rate)):
            targets[family] += 1
    return {k: v for k, v in targets.items() if v > 0}


def _chunk_idx(cap: Dict, fallback: int) -> int:
    try:
        return int(cap.get("chunk_idx", fallback))
    except (AttributeError, TypeError, ValueError):
        return int(fallback)


def _text_for_cap(cap: Dict) -> str:
    parts: List[str] = [str(cap.get("think", ""))]
    for key in ("spatial", "summary"):
        parts.append(str(cap.get(key, "")))
    for ent in cap.get("visible_entities") or []:
        if isinstance(ent, dict):
            parts.append(str(ent.get("desc", "")))
            parts.append(str(ent.get("action", "")))
        else:
            parts.append(str(ent))
    for fact in cap.get("atomic_facts") or []:
        parts.append(str(fact.get("fact", "")) if isinstance(fact, dict) else str(fact))
    for item in cap.get("ocr") or []:
        parts.append(str(item.get("text", "")) if isinstance(item, dict) else str(item))
    for item in cap.get("state_changes") or []:
        parts.append(str(item))
    return " ".join(parts)


def _meaningful_tokens(text: str, *, limit: int = 8) -> List[str]:
    tokens: List[str] = []
    seen: Set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        token = _canonical_action_token(raw)
        if len(token) <= 2 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens


def _caps_by_chunk(evidence: List[Dict]) -> Dict[int, Dict]:
    return {
        _chunk_idx(cap, pos): cap
        for pos, cap in enumerate(evidence)
        if isinstance(cap, dict)
    }


def _caps_for_chunks(by_chunk: Mapping[int, Dict], chunks: Iterable[int]) -> List[Dict]:
    caps: List[Dict] = []
    for raw in chunks:
        try:
            c = int(raw)
        except (TypeError, ValueError):
            continue
        cap = by_chunk.get(c)
        if isinstance(cap, dict):
            caps.append(cap)
    return caps


def _has_visual_signal(cap: Dict) -> bool:
    return bool(
        cap.get("visible_entities")
        or cap.get("atomic_facts")
        or cap.get("spatial")
        or cap.get("ocr")
        or cap.get("state_changes")
    )


def _has_spatial_signal(cap: Dict) -> bool:
    if cap.get("spatial"):
        return True
    entities = cap.get("visible_entities") or []
    return isinstance(entities, list) and len(entities) >= 2


def _has_ocr_signal(cap: Dict) -> bool:
    return any(
        str(item.get("text", "") if isinstance(item, dict) else item).strip()
        for item in (cap.get("ocr") or [])
    )


def _has_action_signal(cap: Dict) -> bool:
    return bool(_action_tokens_for_cap(cap))


def _has_state_change_signal(cap: Dict) -> bool:
    return bool(cap.get("state_changes"))


def _slot_semantic_group(family: str, fields: Mapping[str, str]) -> str:
    question_way = str(fields.get("question_way", "") or "")
    evidence_type = str(fields.get("evidence_type", "") or "")
    if family == "C1" or evidence_type == EVIDENCE_TEXT_OCR:
        return "ocr_text"
    if family in _SPATIAL_FAMILIES or evidence_type == EVIDENCE_SPATIAL_RELATION:
        return "current_spatial_relation"
    if family == "F7":
        return "current_step_status"
    if family in {"F5", "CRR1", "PN1"}:
        return f"multi_{family.lower()}"
    if family in _CURRENT_ACTION_FAMILIES or question_way in {
        QUESTION_WAY_ACTION_RECOGNITION,
        QUESTION_WAY_CAUSAL_INTENT,
        QUESTION_WAY_FUTURE_PREDICTION,
        QUESTION_WAY_PROACTIVE_OUTPUT,
    }:
        return "current_action_or_intent"
    if family in _CROSS_EVENT_FAMILIES:
        return "cross_event"
    if family in _PAST_VISUAL_FAMILIES:
        return "past_visual_fact"
    if family == "M1":
        return "global_summary"
    return str(fields.get("slot_subtype") or family or "generic")


def _signature_terms(group: str, caps: List[Dict]) -> List[str]:
    if group == "ocr_text":
        terms: List[str] = []
        for cap in caps:
            for item in cap.get("ocr") or []:
                text = str(item.get("text", "") if isinstance(item, dict) else item)
                terms.extend(_meaningful_tokens(text, limit=5))
        return list(dict.fromkeys(terms))[:8]
    if "spatial" in group:
        text = " ".join(
            " ".join(
                str(ent.get(k, ""))
                for k in ("id", "desc", "action")
                if isinstance(ent, dict)
            )
            for cap in caps
            for ent in (cap.get("visible_entities") or [])
        )
        text = " ".join([text] + [str(cap.get("spatial", "")) for cap in caps])
        return _meaningful_tokens(text, limit=8)
    if "action" in group or group.startswith("multi_") or group == "cross_event":
        actions: List[str] = []
        for cap in caps:
            actions.extend(sorted(_action_tokens_for_cap(cap)))
        if actions:
            return list(dict.fromkeys(actions))[:8]
    text = " ".join(_text_for_cap(cap) for cap in caps)
    return _meaningful_tokens(text, limit=8)


def _slot_signature(
    family: str,
    fields: Mapping[str, str],
    support_chunks: Iterable[int],
    answer_chunks: Iterable[int],
    evidence_by_chunk: Mapping[int, Dict],
) -> str:
    answers = sorted({int(c) for c in answer_chunks})
    support = sorted({int(c) for c in support_chunks})
    caps = _caps_for_chunks(evidence_by_chunk, support or answers)
    group = _slot_semantic_group(family, fields)
    terms = _signature_terms(group, caps)
    if not terms:
        return ""
    bin_id = (min(answers) if answers else min(support or [0])) // max(1, SUPPORT_BIN_SIZE)
    return f"{group}:b{bin_id}:{','.join(terms)}"


def _slot_signal_reject_reason(
    family: str,
    fields: Mapping[str, str],
    support_chunks: Iterable[int],
    answer_chunks: Iterable[int],
    evidence_by_chunk: Mapping[int, Dict],
) -> str:
    answers = sorted({int(c) for c in answer_chunks})
    support = sorted({int(c) for c in support_chunks})
    if not answers:
        return "no_answer_chunks"
    if not support:
        return "no_support_chunks"
    if any(c not in evidence_by_chunk for c in answers):
        return "answer_chunk_missing_evidence"
    if any(c not in evidence_by_chunk for c in support):
        return "support_chunk_missing_evidence"

    caps = _caps_for_chunks(evidence_by_chunk, sorted(set(support) | set(answers)))
    if not caps:
        return "no_local_evidence"

    qtype = QUESTION_TYPE_BY_FAMILY.get(family, "single_emit")
    if qtype == "multi_emit":
        min_chunks = _MULTI_EMIT_MIN_CHUNKS.get(family, 2)
        if len(answers) < min_chunks:
            return f"{family.lower()}_too_few_preplanned_probes"

    if family == "C1" and not any(_has_ocr_signal(cap) for cap in caps):
        return "ocr_missing"
    if family in _SPATIAL_FAMILIES and not any(_has_spatial_signal(cap) for cap in caps):
        return "spatial_relation_missing"
    if family in {"ACR1", "CR3"} and not any(_has_action_signal(cap) or _has_state_change_signal(cap) for cap in caps):
        return "action_or_state_missing"
    if family in {"F6", "E2"} and not any(_has_visual_signal(cap) for cap in caps):
        return "future_cue_missing"
    if family in {"N1", "P1", "HLD1"} and not any(_has_visual_signal(cap) for cap in caps):
        return "visual_fact_missing"
    if family in {"CR1", "CR2", "CR4", "CR5"}:
        signal_chunks = {
            _chunk_idx(cap, 0)
            for cap in caps
            if _has_visual_signal(cap) or _has_action_signal(cap)
        }
        if len(signal_chunks) < 2:
            return "cross_event_too_thin"
    if family == "M1" and sum(1 for cap in caps if _has_visual_signal(cap)) < 2:
        return "summary_too_thin"
    if family == "F5" and len(answers) < 2:
        return "count_no_repeated_action"
    if family == "F7" and not any(_has_action_signal(cap) or _has_state_change_signal(cap) for cap in caps):
        return "status_probe_no_action"
    if family == "CRR1" and not any(
        _has_action_signal(cap) or _has_state_change_signal(cap) or _has_ocr_signal(cap) or cap.get("spatial")
        for cap in caps
    ):
        return "sufficiency_probe_no_clue"
    if family == "PN1" and not any(_has_state_change_signal(cap) for cap in caps):
        return "narration_no_state_change"
    return ""


def _slot_candidate_reject_reason(
    family: str,
    fields: Mapping[str, str],
    support_chunks: Iterable[int],
    answer_chunks: Iterable[int],
    evidence_by_chunk: Mapping[int, Dict],
    used_signatures: Set[str],
    exact_answer_uses: Counter,
    used_bins: Optional[Counter] = None,
    num_chunks: int = 0,
) -> str:
    reason = _slot_signal_reject_reason(
        family,
        fields,
        support_chunks,
        answer_chunks,
        evidence_by_chunk,
    )
    if reason:
        return reason
    answers = sorted({int(c) for c in answer_chunks})
    support = sorted({int(c) for c in support_chunks})
    for c in answers:
        cap = _MAX_CHUNK0_ANSWER_USES if c == 0 else _MAX_EXACT_ANSWER_CHUNK_USES
        if exact_answer_uses.get(c, 0) >= cap:
            return "answer_chunk_capacity_full"
    if used_bins is not None:
        support_bin_cap = _pass3a_support_bin_cap(num_chunks)
        bins = {
            int(c) // max(1, SUPPORT_BIN_SIZE)
            for c in (support or answers)
        }
        for bin_id in bins:
            if used_bins.get(bin_id, 0) >= support_bin_cap:
                return "support_bin_capacity_full"
    signature = _slot_signature(
        family,
        fields,
        support_chunks,
        answer_chunks,
        evidence_by_chunk,
    )
    if signature and signature in used_signatures:
        return "semantic_duplicate"
    return ""


def filter_pass3_slot_plan(evidence: List[Dict], slots: Iterable[Dict]) -> List[Dict]:
    """Apply the same pre-LLM slot checks to externally supplied slot plans."""
    ordered = sorted(
        [cap for cap in evidence if isinstance(cap, dict)],
        key=lambda cap: _chunk_idx(cap, 0),
    )
    evidence_by_chunk = _caps_by_chunk(ordered)
    num_chunks = max((_chunk_idx(cap, i) for i, cap in enumerate(ordered)), default=0) + 1
    used_signatures: Set[str] = set()
    exact_answer_uses: Counter = Counter()
    used_bins: Counter = Counter()
    out: List[Dict] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        family = str(slot.get("family", "") or "")
        if family not in FAMILY_SLOT_DEFAULTS:
            continue
        try:
            slot_index = int(slot.get("slot_index", 0) or 0)
        except (TypeError, ValueError):
            slot_index = 0
        fields = _default_slot_fields(family, slot_index)
        fields.update({
            key: str(slot.get(key, fields.get(key, "")) or "")
            for key in ("question_way", "evidence_type", "support_policy")
        })
        try:
            support_chunks = sorted({int(c) for c in (slot.get("support_chunks") or [])})
            answer_chunks = sorted({int(c) for c in (slot.get("answer_chunks") or [])})
        except (TypeError, ValueError):
            continue
        if _slot_candidate_reject_reason(
            family,
            fields,
            support_chunks,
            answer_chunks,
            evidence_by_chunk,
            used_signatures,
            exact_answer_uses,
            used_bins,
            num_chunks,
        ):
            continue
        signature = _slot_signature(family, fields, support_chunks, answer_chunks, evidence_by_chunk)
        if signature:
            used_signatures.add(signature)
        exact_answer_uses.update(answer_chunks)
        for c in support_chunks:
            used_bins[int(c) // max(1, SUPPORT_BIN_SIZE)] += 1
        out.append(dict(slot))
    return out


def _with_readable_slot_names(family: str, fields: Dict[str, str]) -> Dict[str, str]:
    """Add readable public names while keeping old slot_* aliases for compatibility."""
    out = dict(fields)
    if not out.get("slot_group") and out.get("task_family"):
        out["slot_group"] = str(out["task_family"])
    if not out.get("slot_subtype") and out.get("task_subtype"):
        out["slot_subtype"] = str(out["task_subtype"])
    if not out.get("temporal_bucket") and out.get("timing_type"):
        out["temporal_bucket"] = str(out["timing_type"])

    out["legacy_family_id"] = str(out.get("legacy_family_id") or family)
    out["task_family"] = str(out.get("task_family") or out.get("slot_group") or "")
    out["task_subtype"] = str(out.get("task_subtype") or out.get("slot_subtype") or "")
    out["timing_type"] = str(out.get("timing_type") or out.get("temporal_bucket") or "")
    if not out.get("readable_task_name"):
        pieces = [
            out.get("task_family", ""),
            out.get("task_subtype", ""),
            out.get("timing_type", ""),
        ]
        out["readable_task_name"] = " / ".join(p for p in pieces if p)
    return out


def _default_slot_fields(family: str, slot_index: int = 0) -> Dict[str, str]:
    fields = dict(FAMILY_SLOT_DEFAULTS.get(family, {}))
    fields.update(FAMILY_SLOT_META.get(family, {}))
    overrides = SLOT_VARIANT_OVERRIDES.get(family, {})
    if slot_index in overrides:
        fields.update(overrides[slot_index])
    return _with_readable_slot_names(family, fields)


def default_slot_metadata_for_family(family: str, slot_index: int = 0) -> Dict[str, str]:
    fields = _default_slot_fields(family, slot_index)
    return {k: v for k, v in fields.items() if k in SLOT_METADATA_KEYS}


def apply_slot_metadata_to_card(card: Dict, planned_slots: List[Dict]) -> None:
    """Stamp deterministic slot metadata onto a generated card in-place."""
    if not card or not planned_slots:
        return
    slot_id = str(card.get("slot_id", "") or "")
    slot = next((s for s in planned_slots if str(s.get("slot_id", "")) == slot_id), None)
    if not slot:
        return
    for key in (
        "question_style",
        "question_way",
        "evidence_type",
        "support_policy",
        "temporal_role",
        "target_ovo_task",
        *SLOT_METADATA_KEYS,
    ):
        if slot.get(key) not in (None, ""):
            card[key] = slot[key]


def _score_chunk_for_family(family: str, cap: Dict) -> float:
    text = _text_for_cap(cap).lower()
    score = 1.0
    if cap.get("visible_entities"):
        score += 1.2
    if cap.get("atomic_facts"):
        score += 1.2
    if cap.get("state_changes"):
        score += 1.5
    if cap.get("ocr"):
        score += 2.5 if family == "C1" else 0.5
    if cap.get("spatial"):
        score += 1.5 if family in {"STU1", "OJR1", "CR7", "R1"} else 0.3
    tokens = set(_TOKEN_RE.findall(text))
    if family == "ACR1" and tokens & {"open", "close", "pick", "put", "move", "walk", "run", "hold", "use"}:
        score += 1.5
    if family in {"CR1", "CR3", "CR4", "CR5"} and tokens & {"because", "so", "therefore", "trying", "prepare"}:
        score += 1.0
    return score


_ACTION_CANON = {
    "holding": "hold",
    "held": "hold",
    "holds": "hold",
    "showing": "show",
    "shows": "show",
    "displaying": "show",
    "applying": "apply",
    "applies": "apply",
    "applied": "apply",
    "pouring": "pour",
    "pours": "pour",
    "stirring": "stir",
    "stirs": "stir",
    "mixing": "mix",
    "mixes": "mix",
    "opening": "open",
    "opens": "open",
    "closing": "close",
    "closes": "close",
    "picking": "pick",
    "picked": "pick",
    "placing": "place",
    "places": "place",
    "moving": "move",
    "moves": "move",
    "walking": "walk",
    "walks": "walk",
    "speaking": "speak",
    "talking": "speak",
    "gesturing": "gesture",
    "gestures": "gesture",
    "looking": "look",
    "turning": "turn",
    "turns": "turn",
    "cutting": "cut",
    "chopping": "chop",
    "writing": "write",
    "using": "use",
    "dipping": "dip",
    "eating": "eat",
    "drinking": "drink",
}
_ACTION_TERMS = {
    "hold", "show", "apply", "pour", "stir", "mix", "open", "close",
    "pick", "place", "move", "walk", "speak", "gesture", "look", "turn",
    "cut", "chop", "write", "use", "dip", "eat", "drink", "put", "take",
}


def _canonical_action_token(token: str) -> str:
    token = str(token or "").lower()
    if token in _ACTION_CANON:
        return _ACTION_CANON[token]
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 4:
        token = token[:-1]
    return _ACTION_CANON.get(token, token)


def _action_tokens_for_cap(cap: Dict) -> set:
    text = _text_for_cap(cap).lower()
    toks = {_canonical_action_token(t) for t in _TOKEN_RE.findall(text)}
    return {t for t in toks if t in _ACTION_TERMS}


def _action_group_candidates(
    evidence: List[Dict],
    *,
    desired: int,
    global_prefix: bool,
    used_bins: Counter,
    min_chunks: int = 1,
) -> List[Tuple[str, List[int]]]:
    groups: Dict[str, List[int]] = {}
    for fallback, cap in enumerate(evidence):
        c = _chunk_idx(cap, fallback)
        for token in _action_tokens_for_cap(cap):
            groups.setdefault(token, []).append(c)
    candidates = []
    for token, raw_chunks in groups.items():
        chunks = sorted(dict.fromkeys(raw_chunks))
        if len(chunks) < max(1, min_chunks):
            continue
        first = chunks[0]
        span = chunks[-1] - first
        if global_prefix and first > 120:
            continue
        bin_penalty = sum(used_bins.get(c // max(1, SUPPORT_BIN_SIZE), 0) for c in chunks)
        key = (
            1 if len(chunks) >= max(3, desired // 2) else 0,
            min(len(chunks), desired),
            span if global_prefix else -span,
            -first if global_prefix else first,
            -bin_penalty,
            stable_mod(token, "repeated_action_token", modulo=1000),
        )
        candidates.append((key, token, chunks))
    candidates.sort(reverse=True)
    return [(token, chunks) for _key, token, chunks in candidates]


def _evenly_sample(chunks: List[int], desired: int) -> List[int]:
    chunks = sorted(dict.fromkeys(int(c) for c in chunks))
    desired = max(1, int(desired))
    if len(chunks) <= desired:
        return chunks
    last = len(chunks) - 1
    return sorted(dict.fromkeys(chunks[round(i * last / (desired - 1))] for i in range(desired)))


def _repeated_action_chunks(
    evidence: List[Dict],
    *,
    desired: int,
    global_prefix: bool,
    used_bins: Counter,
) -> List[int]:
    candidates = _action_group_candidates(
        evidence,
        desired=desired,
        global_prefix=global_prefix,
        used_bins=used_bins,
        min_chunks=2,
    )
    if not candidates:
        return []
    chunks = candidates[0][1]
    if global_prefix:
        chunks = [c for c in chunks if c <= chunks[0] + max(1, PASS3A_F5_GLOBAL_MAX_ACTIVE_SPAN)]
    return _evenly_sample(chunks, desired)


def _repeated_action_candidate_chunks(
    evidence: List[Dict],
    *,
    desired: int,
    global_prefix: bool,
    used_bins: Counter,
) -> List[List[int]]:
    out: List[List[int]] = []
    for _token, chunks in _action_group_candidates(
        evidence,
        desired=desired,
        global_prefix=global_prefix,
        used_bins=used_bins,
        min_chunks=2,
    ):
        if global_prefix:
            chunks = [c for c in chunks if c <= chunks[0] + max(1, PASS3A_F5_GLOBAL_MAX_ACTIVE_SPAN)]
        sampled = _evenly_sample(chunks, desired)
        if len(sampled) >= 2:
            out.append(sampled)
    return out


def _nearest_before_after(existing: List[int], start: int, end: int, max_span: int) -> Tuple[List[int], List[int]]:
    before = [c for c in existing if start - max_span <= c < start]
    after = [c for c in existing if end < c <= end + max_span]
    return before, after


def _status_probe_candidate_chunks(
    family: str,
    evidence: List[Dict],
    *,
    used_bins: Counter,
    desired: int,
    max_span: int,
) -> List[List[int]]:
    existing = sorted(_chunk_idx(cap, i) for i, cap in enumerate(evidence))
    out: List[List[int]] = []
    for _token, chunks in _action_group_candidates(
        evidence,
        desired=desired,
        global_prefix=False,
        used_bins=used_bins,
        min_chunks=1,
    ):
        if not chunks:
            continue
        start, end = chunks[0], chunks[-1]
        before, after = _nearest_before_after(existing, start, end, max_span)
        if family == "CRR1":
            if not before:
                continue
            no_probe = before[-1:]
            yes_probes = _evenly_sample([c for c in existing if start <= c <= min(existing[-1], end + max_span)], desired - len(no_probe))
            candidate = sorted(dict.fromkeys(no_probe + yes_probes))
        else:
            no_probes = before[-1:] + after[:1]
            if not no_probes:
                continue
            yes_budget = max(1, desired - len(no_probes))
            yes_probes = _evenly_sample(chunks, yes_budget)
            candidate = sorted(dict.fromkeys(no_probes + yes_probes))
        if len(candidate) >= _MULTI_EMIT_MIN_CHUNKS.get(family, 3):
            out.append(candidate[:desired])
    if out:
        return out
    anchors = [
        _chunk_idx(cap, i)
        for i, cap in enumerate(evidence)
        if _has_state_change_signal(cap)
    ]
    if family == "CRR1" and not anchors:
        anchors = [
            _chunk_idx(cap, i)
            for i, cap in enumerate(evidence)
            if _has_ocr_signal(cap) or cap.get("spatial")
        ]
    for anchor in sorted(dict.fromkeys(anchors))[:6]:
        before, after = _nearest_before_after(existing, anchor, anchor, max_span)
        if not before:
            continue
        if family == "CRR1":
            yes_tail = [anchor] + after[: max(0, desired - 2)]
            candidate = sorted(dict.fromkeys(before[-1:] + yes_tail))
        else:
            candidate = sorted(dict.fromkeys(before[-1:] + [anchor] + after[:1]))
        if len(candidate) >= _MULTI_EMIT_MIN_CHUNKS.get(family, 3):
            out.append(candidate[:desired])
    return out


def _state_change_candidate_chunks(
    evidence: List[Dict],
    *,
    desired: int,
    used_bins: Counter,
    max_span: int,
) -> List[List[int]]:
    chunks = [
        _chunk_idx(cap, i)
        for i, cap in enumerate(evidence)
        if _has_state_change_signal(cap)
    ]
    chunks = sorted(dict.fromkeys(chunks))
    if len(chunks) < 2:
        return []
    scored = sorted(
        chunks,
        key=lambda c: (used_bins.get(c // max(1, SUPPORT_BIN_SIZE), 0), c),
    )
    out: List[List[int]] = []
    for anchor in scored[: min(len(scored), 6)]:
        local = [c for c in chunks if abs(c - anchor) <= max_span]
        sampled = _evenly_sample(local, desired)
        if len(sampled) >= 2:
            out.append(sampled)
    if not out:
        out.append(_evenly_sample(chunks, desired))
    return out


def _nearest_existing_chunks(evidence: List[Dict], candidates: Iterable[int]) -> List[int]:
    existing = sorted(_chunk_idx(cap, i) for i, cap in enumerate(evidence))
    if not existing:
        return []
    out: List[int] = []
    for cand in candidates:
        best = min(existing, key=lambda c: (abs(c - int(cand)), c))
        if best not in out:
            out.append(best)
    return out


def _support_window(center: int, num_chunks: int) -> List[int]:
    # For single_emit cards, the answer chunk must be the latest support
    # chunk. Do not include future support after the answer position, or the
    # teacher will correctly move gold_emits to that later chunk.
    lo = max(0, int(center) - 2)
    hi = min(num_chunks - 1, int(center))
    return list(range(lo, hi + 1))


def _dense_existing_chunks(evidence: List[Dict], start: int, end: int) -> List[int]:
    """Return existing chunk indices in [start, end]."""
    lo = int(min(start, end))
    hi = int(max(start, end))
    chunks = sorted(_chunk_idx(cap, i) for i, cap in enumerate(evidence))
    return [c for c in chunks if lo <= c <= hi]


def _single_support_chunks(
    family: str,
    center: int,
    evidence: List[Dict],
    num_chunks: int,
) -> List[int]:
    center = int(center)
    if family == "M1":
        # Global summary questions need broad evidence, with the answer emitted
        # only after the latest support chunk.
        return _nearest_existing_chunks(
            evidence,
            [0, num_chunks // 3, (num_chunks * 2) // 3, num_chunks - 1],
        )
    if family in {"CR1", "CR2", "CR4", "CR5"}:
        # Temporal/causal questions often need earlier context plus the
        # resolving/latest support chunk. Keep all support <= answer chunk.
        # Do not include every chunk between anchors: that creates huge
        # slot-local prompts and makes the evidence window too diffuse for
        # pass3B placement. A few anchored local windows are enough to force
        # cross-event evidence without turning one card into a half-video span.
        candidates = [
            max(0, center - 48),
            max(0, center - 24),
            max(0, center - 8),
            center,
        ]
        anchors = [c for c in _nearest_existing_chunks(evidence, candidates) if c <= center] or [center]
        # Keep cross-event evidence compact. Two or three anchored chunks are
        # enough for temporal/causal/source-discrimination prompts, and avoid
        # one slot consuming several support bins before later variants run.
        selected: set[int] = {anchors[0], center}
        if len(anchors) > 2:
            selected.add(anchors[-2])
        elif len(anchors) > 1:
            selected.add(anchors[-1])
        return sorted(selected) or anchors
    return _support_window(center, num_chunks)


def _ranked_chunks_for_family(
    family: str,
    evidence: List[Dict],
    used_bins: Counter,
) -> List[int]:
    ranked = []
    for fallback, cap in enumerate(evidence):
        if family == "C1" and not cap.get("ocr"):
            continue
        c = _chunk_idx(cap, fallback)
        score = _score_chunk_for_family(family, cap)
        bin_id = c // max(1, SUPPORT_BIN_SIZE)
        score -= used_bins.get(bin_id, 0) * 2.0
        score += stable_mod(f"{family}:{c}", "slot_chunk_tiebreak", modulo=1000) / 100000.0
        ranked.append((score, c))
    ranked.sort(reverse=True)
    return [c for _score, c in ranked]


def _pick_chunk(
    family: str,
    evidence: List[Dict],
    used_bins: Counter,
    used_chunks: set,
) -> int:
    for c in _ranked_chunks_for_family(family, evidence, used_bins):
        if c in used_chunks:
            continue
        return c
    return -1


def _multi_probe_chunks(
    family: str,
    evidence: List[Dict],
    used_bins: Counter,
    desired: int,
    max_span: int,
) -> List[int]:
    if not evidence:
        return [0]
    ranked = _ranked_chunks_for_family(family, evidence, used_bins)
    anchor = ranked[0] if ranked else _chunk_idx(evidence[0], 0)
    half = max(1, max_span // 2)
    candidates = [
        _chunk_idx(cap, i)
        for i, cap in enumerate(evidence)
        if abs(_chunk_idx(cap, i) - anchor) <= half
    ]
    if len(candidates) < desired:
        candidates = [
            _chunk_idx(cap, i)
            for i, cap in enumerate(evidence)
            if abs(_chunk_idx(cap, i) - anchor) <= max_span
        ]
    candidates = sorted(dict.fromkeys(candidates))
    if len(candidates) <= desired:
        return candidates
    last = len(candidates) - 1
    return sorted(dict.fromkeys(candidates[round(i * last / (desired - 1))] for i in range(desired)))


def _style_for_slot(family: str, slot_index: int) -> str:
    if family_taxonomy(family).get("ours_unique"):
        return QUESTION_STYLE_OURS_UNIQUE
    return QUESTION_STYLE_BENCHMARK_VARIANT if slot_index % 2 else QUESTION_STYLE_BENCHMARK_CORE


def _ceil_count(value: float) -> int:
    if value <= 0:
        return 0
    return int(value + 0.999)


def _round_count(value: float) -> int:
    if value <= 0:
        return 0
    return int(value + 0.5)


def _floor_count(value: float) -> int:
    if value <= 0:
        return 0
    return int(value)


def pass3a_response_row_budget(num_chunks: int) -> int:
    return _ceil_count(max(1, int(num_chunks)) * PASS3A_RESPONSE_ROW_TARGET_FRACTION)


def pass3a_source_row_targets(num_chunks: int) -> Dict[str, int]:
    """Final trajectory response-row targets that Pass3B should be able to hit."""
    target_rows = pass3a_response_row_budget(num_chunks)
    direct_rows = _round_count(target_rows * PASS3A_DIRECT_SOURCE_TARGET_FRACTION)
    recall_rows = _round_count(target_rows * PASS3A_RECALL_SOURCE_TARGET_FRACTION)
    future_rows = (
        _round_count(target_rows * PASS3A_FUTURE_SOURCE_TARGET_FRACTION)
        if target_rows >= 10 else 0
    )
    if target_rows >= 10 and future_rows == 0 and PASS3A_FUTURE_SOURCE_TARGET_FRACTION > 0:
        future_rows = 1

    rows = {
        "direct": max(1 if target_rows > 0 else 0, direct_rows),
        "recall": max(1 if target_rows >= 4 else 0, recall_rows),
        "future": max(1 if target_rows >= 10 else 0, future_rows),
    }
    residual_rows = max(0, target_rows - sum(rows.values()))
    rows["multi"] = residual_rows if target_rows >= 10 else 0
    if target_rows < 10:
        rows["direct"] += residual_rows
    if target_rows >= 10 and rows["multi"] == 0 and PASS3A_MULTI_SOURCE_TARGET_FRACTION > 0:
        rows["multi"] = 1

    # Keep the rounded buckets on the exact row budget. Prefer preserving
    # recall/future/multi minima and adjust direct, which is easiest to supply.
    while sum(rows.values()) > target_rows:
        for source in ("direct", "recall", "multi", "future"):
            floor = 1 if source in {"direct", "recall"} and target_rows >= 4 else 0
            if source == "future":
                floor = 1 if target_rows >= 6 else 0
            if source == "multi":
                floor = 1 if target_rows >= 10 else 0
            if rows.get(source, 0) > floor:
                rows[source] -= 1
                break
        else:
            break
    while sum(rows.values()) < target_rows:
        source = "direct"
        if rows["direct"] / max(1, target_rows) >= PASS3A_DIRECT_SOURCE_TARGET_FRACTION:
            source = "recall"
        if rows[source] / max(1, target_rows) >= (
            PASS3A_RECALL_SOURCE_TARGET_FRACTION if source == "recall" else 1.0
        ):
            source = "multi" if target_rows >= 10 else "direct"
        rows[source] += 1
    return rows


def _pass3a_source_row_bounds(target_rows: int, source: str) -> Tuple[int, int]:
    target_rows = max(0, int(target_rows))
    if target_rows <= 0:
        return 0, 0
    if source == "direct":
        return (
            max(1, _floor_count(target_rows * 0.50)),
            max(1, _ceil_count(target_rows * 0.55)),
        )
    if source == "recall":
        return (
            max(1 if target_rows >= 4 else 0, _floor_count(target_rows * 0.25)),
            max(1 if target_rows >= 4 else 0, _ceil_count(target_rows * 0.30)),
        )
    if source == "future":
        if target_rows < 10:
            return 0, 0
        return (
            max(1, _floor_count(target_rows * 0.08)),
            max(1, _ceil_count(target_rows * 0.12)),
        )
    if source == "multi":
        if target_rows < 10:
            return 0, 0
        return (
            max(1, _floor_count(target_rows * 0.10)),
            max(1, _ceil_count(target_rows * 0.15)),
        )
    return 0, target_rows


def pass3a_batch_source_row_targets(video_num_chunks: Mapping[str, int]) -> Dict[str, Dict[str, int]]:
    """Allocate source row targets across a batch while preserving per-video balance.

    Per-video row totals stay proportional to video length. Batch-level source
    totals are then corrected by transferring rows between direct and the
    under/over source inside each video's source bounds. Direct is the exchange
    bucket because it is easiest to supply and least timing-constrained.
    """
    row_budget_by_video = {
        str(video_id): pass3a_response_row_budget(int(num_chunks))
        for video_id, num_chunks in video_num_chunks.items()
    }
    out = {
        video_id: pass3a_source_row_targets(int(video_num_chunks[video_id]))
        for video_id in row_budget_by_video
    }
    total_rows = sum(row_budget_by_video.values())
    if total_rows <= 0:
        return out

    global_targets = {
        "direct": _round_count(total_rows * PASS3A_DIRECT_SOURCE_TARGET_FRACTION),
        "recall": _round_count(total_rows * PASS3A_RECALL_SOURCE_TARGET_FRACTION),
        "future": _round_count(total_rows * PASS3A_FUTURE_SOURCE_TARGET_FRACTION),
    }
    global_targets["multi"] = max(0, total_rows - sum(global_targets.values()))

    def global_rows(source: str) -> int:
        return sum(rows.get(source, 0) for rows in out.values())

    def transfer(video_id: str, src: str, dst: str) -> bool:
        rows = out[video_id]
        target_rows = row_budget_by_video[video_id]
        src_min, _src_max = _pass3a_source_row_bounds(target_rows, src)
        _dst_min, dst_max = _pass3a_source_row_bounds(target_rows, dst)
        if rows.get(src, 0) <= src_min or rows.get(dst, 0) >= dst_max:
            return False
        rows[src] -= 1
        rows[dst] = rows.get(dst, 0) + 1
        return True

    for source in ("future", "multi", "recall"):
        while global_rows(source) < global_targets.get(source, 0):
            candidates = [
                video_id
                for video_id in row_budget_by_video
                if out[video_id].get("direct", 0) > _pass3a_source_row_bounds(row_budget_by_video[video_id], "direct")[0]
                and out[video_id].get(source, 0) < _pass3a_source_row_bounds(row_budget_by_video[video_id], source)[1]
            ]
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda video_id: (
                    out[video_id].get(source, 0) / max(1, row_budget_by_video[video_id]),
                    stable_mod(video_id, f"batch_source_under_{source}", modulo=10_000),
                ),
            )
            if not transfer(chosen, "direct", source):
                break
        while global_rows(source) > global_targets.get(source, 0):
            candidates = [
                video_id
                for video_id in row_budget_by_video
                if out[video_id].get(source, 0) > _pass3a_source_row_bounds(row_budget_by_video[video_id], source)[0]
                and out[video_id].get("direct", 0) < _pass3a_source_row_bounds(row_budget_by_video[video_id], "direct")[1]
            ]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda video_id: (
                    out[video_id].get(source, 0) / max(1, row_budget_by_video[video_id]),
                    -stable_mod(video_id, f"batch_source_over_{source}", modulo=10_000),
                ),
            )
            if not transfer(chosen, source, "direct"):
                break
    return out


def _slots_with_retry_headroom(
    final_slots: int,
    min_retry_slots: int,
    *,
    headroom_fraction: Optional[float] = None,
) -> int:
    if final_slots <= 0:
        return 0
    fraction = (
        PASS3A_SLOT_RETRY_HEADROOM_FRACTION
        if headroom_fraction is None
        else max(0.0, float(headroom_fraction))
    )
    retry = max(
        0,
        int(min_retry_slots),
        _ceil_count(final_slots * fraction),
    )
    return final_slots + retry


def pass3a_source_slot_requirements_from_rows(row_targets: Mapping[str, int]) -> Dict[str, int]:
    multi_rows_per_slot = max(1.0, PASS3A_MULTI_EXPECTED_RESPONSE_ROWS_PER_SLOT)
    final_multi_slots = _ceil_count(int(row_targets.get("multi", 0)) / multi_rows_per_slot)
    return {
        "direct": _slots_with_retry_headroom(
            int(row_targets.get("direct", 0)),
            PASS3A_DIRECT_MIN_RETRY_SLOTS,
        ),
        "recall": _slots_with_retry_headroom(
            int(row_targets.get("recall", 0)),
            PASS3A_RECALL_MIN_RETRY_SLOTS,
            headroom_fraction=PASS3A_RECALL_SLOT_RETRY_HEADROOM_FRACTION,
        ),
        "future": _slots_with_retry_headroom(
            int(row_targets.get("future", 0)),
            PASS3A_FUTURE_MIN_RETRY_SLOTS,
        ),
        "multi": _slots_with_retry_headroom(
            final_multi_slots,
            PASS3A_MULTI_MIN_RETRY_SLOTS,
        ),
    }


def pass3a_source_slot_requirements(num_chunks: int) -> Dict[str, int]:
    """Candidate slot supply needed before Pass3B row selection.

    Pass3B selects final response rows. Pass3A only needs enough candidates for
    those rows plus a bounded retry/headroom pool for teacher rejection,
    placement conflicts, and source/subtype quotas. This keeps generation close
    to the final trajectory budget instead of blindly producing 2x questions.
    """
    return pass3a_source_slot_requirements_from_rows(pass3a_source_row_targets(num_chunks))


def _slot_supply_source(family: str, slot_index: int) -> str:
    fields = _default_slot_fields(family, slot_index)
    qtype = QUESTION_TYPE_BY_FAMILY.get(family, "single_emit")
    temporal_bucket = str(fields.get("temporal_bucket") or fields.get("timing_type") or "")
    support_policy = str(fields.get("support_policy") or "")
    temporal_role = str(fields.get("temporal_role") or "")
    if qtype == "multi_emit" or temporal_bucket.startswith("multi_") or family in _MULTI_SUPPLY_FAMILIES:
        return "multi"
    if (
        temporal_bucket == "future_delayed"
        or temporal_role == "future_event_wait"
        or (family in _FUTURE_SUPPLY_FAMILIES and support_policy == SUPPORT_FUTURE_CURRENT_CUE)
    ):
        return "future"
    if (
        support_policy == SUPPORT_HISTORICAL_VISUAL_RECALL
        or temporal_bucket.startswith("past_")
        or family in _RECALL_SUPPLY_FAMILIES
    ):
        return "recall"
    return "direct"


def _count_source_slots(targets: Mapping[str, int]) -> Counter:
    counts: Counter = Counter()
    for family, raw_target in targets.items():
        if family not in FAMILY_SLOT_DEFAULTS:
            continue
        for slot_index in range(max(0, int(raw_target or 0))):
            counts[_slot_supply_source(family, slot_index)] += 1
    return counts


def _source_family_pool(source: str) -> Tuple[str, ...]:
    if source == "direct":
        return _DIRECT_SUPPLY_FAMILIES
    if source == "recall":
        return _RECALL_SUPPLY_FAMILIES
    if source == "future":
        return _FUTURE_SUPPLY_FAMILIES
    if source == "multi":
        return _MULTI_SUPPLY_FAMILIES
    return ()


def _bump_targets_for_source(
    targets: Dict[str, int],
    *,
    source: str,
    required: int,
    video_id: str,
) -> None:
    pool = [family for family in _source_family_pool(source) if family in FAMILY_SLOT_DEFAULTS]
    if not pool:
        return
    while _count_source_slots(targets).get(source, 0) < max(0, int(required)):
        step = sum(targets.values())
        family = min(
            pool,
            key=lambda fam: (
                targets.get(fam, 0),
                stable_mod(video_id, f"slot_supply_{source}_{fam}_{step}", modulo=10_000),
            ),
        )
        targets[family] = int(targets.get(family, 0)) + 1


def _augment_targets_for_response_budget(
    family_targets: Mapping[str, int],
    *,
    num_chunks: int,
    video_id: str,
    source_slot_requirements: Optional[Mapping[str, int]] = None,
) -> Dict[str, int]:
    candidate_caps: Dict[str, int] = {}
    for family, raw_target in family_targets.items():
        if family not in FAMILY_SLOT_DEFAULTS:
            continue
        try:
            target = max(0, int(raw_target or 0))
        except (TypeError, ValueError):
            continue
        if target > 0:
            candidate_caps[family] = target
    requirements = dict(source_slot_requirements or pass3a_source_slot_requirements(num_chunks))

    # Earlier versions treated the balanced family targets as a floor and then
    # only bumped sources that were short. That made pass3a ask the teacher for
    # nearly every family on nearly every video, while pass3b kept only a small
    # row budget. Here the response-row budget is the hard supply target:
    # balanced_family_targets defines the eligible family mix, and requirements
    # define how many candidate slots are worth paying for.
    targets: Dict[str, int] = {}
    served: Counter = Counter()

    def _next_source(family: str) -> str:
        return _slot_supply_source(family, int(targets.get(family, 0)))

    def _add_family_slot(family: str) -> bool:
        source = _next_source(family)
        if not source:
            return False
        targets[family] = int(targets.get(family, 0)) + 1
        served[source] += 1
        return True

    def _candidate_families(source: str, *, allow_bump: bool) -> List[str]:
        out: List[str] = []
        for family in _source_family_pool(source):
            if family not in FAMILY_SLOT_DEFAULTS:
                continue
            current = int(targets.get(family, 0))
            cap = int(candidate_caps.get(family, 0))
            if not allow_bump and current >= cap:
                continue
            if current >= 4:
                continue
            if _next_source(family) == source:
                out.append(family)
        return out

    def _choose_family(source: str, *, allow_bump: bool) -> Optional[str]:
        candidates = _candidate_families(source, allow_bump=allow_bump)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda family: (
                int(targets.get(family, 0)),
                0 if int(targets.get(family, 0)) < int(candidate_caps.get(family, 0)) else 1,
                stable_mod(
                    video_id,
                    f"slot_budget_{source}_{family}_{int(targets.get(family, 0))}",
                    modulo=10_000,
                ),
            ),
        )

    for source in ("future", "multi", "recall", "direct"):
        required = max(0, int(requirements.get(source, 0) or 0))
        while served[source] < required:
            family = _choose_family(source, allow_bump=False)
            if family is None:
                family = _choose_family(source, allow_bump=True)
            if family is None or not _add_family_slot(family):
                break

    for source in ("direct", "recall", "future", "multi"):
        _bump_targets_for_source(
            targets,
            source=source,
            required=requirements.get(source, 0),
            video_id=video_id,
        )
    return targets


def _ordered_slot_jobs(
    slot_jobs: Iterable[Tuple[str, int, int, str]],
    requirements: Mapping[str, int],
) -> List[Tuple[int, str]]:
    """Interleave slot construction by source budget."""
    protected_variants = {
        ("N1", 1),
        ("N1", 2),
        ("P1", 1),
        ("CR2", 1),
        ("CR2", 2),
        ("CR4", 1),
        ("CR4", 2),
        ("R1", 2),
        ("CR5", 1),
    }

    def _variant_order(family: str, slot_index: int) -> int:
        if slot_index == 0:
            return 0
        if (family, slot_index) in protected_variants:
            return 1
        return 2 + int(slot_index)

    by_source: Dict[str, List[Tuple[int, int, str]]] = {}
    for source, slot_index, tie, family in slot_jobs:
        by_source.setdefault(source, []).append((slot_index, tie, family))
    for jobs in by_source.values():
        jobs.sort(key=lambda item: (_variant_order(item[2], item[0]), item[0], item[1], item[2]))

    source_order = ("direct", "recall", "future", "multi")
    served: Counter = Counter()
    ordered: List[Tuple[int, str]] = []
    while any(by_source.get(source) for source in source_order):
        best_source = ""
        best_key = (float("inf"), 99)
        for rank, source in enumerate(source_order):
            if not by_source.get(source):
                continue
            denominator = max(1, int(requirements.get(source, 0) or 0))
            key = (served[source] / denominator, rank)
            if key < best_key:
                best_key = key
                best_source = source
        if not best_source:
            break
        slot_index, _tie, family = by_source[best_source].pop(0)
        served[best_source] += 1
        ordered.append((slot_index, family))
    return ordered


def _pass3a_support_bin_cap(num_chunks: int) -> int:
    step = max(1, PASS3A_SUPPORT_BIN_LONG_STEP)
    cap = PASS3A_SUPPORT_BIN_BASE_CAP + max(0, int(num_chunks) // step)
    if PASS3A_SUPPORT_BIN_MAX_CAP > 0:
        cap = min(cap, PASS3A_SUPPORT_BIN_MAX_CAP)
    return max(1, cap)


def build_pass3_slot_plan(
    evidence: List[Dict],
    video_id: str,
    family_targets: Mapping[str, int],
    *,
    audit: Optional[Counter] = None,
    source_row_targets: Optional[Mapping[str, int]] = None,
    seed: int = 42,
) -> List[Dict]:
    """Return planned card slots for one video.

    The returned slots are intentionally model-independent JSON-compatible
    dicts so they can be injected into prompts, audits, or future batch-level
    quota planners.
    """
    if not evidence:
        return []
    ordered = sorted(
        [cap for cap in evidence if isinstance(cap, dict)],
        key=lambda cap: _chunk_idx(cap, 0),
    )
    if not ordered:
        return []
    num_chunks = max(_chunk_idx(cap, i) for i, cap in enumerate(ordered)) + 1
    evidence_by_chunk = _caps_by_chunk(ordered)
    used_bins: Counter = Counter()
    exact_answer_uses: Counter = Counter()
    used_signatures: Set[str] = set()
    slots: List[Dict] = []
    source_slot_requirements = (
        pass3a_source_slot_requirements_from_rows(source_row_targets)
        if source_row_targets
        else None
    )
    planned_targets = _augment_targets_for_response_budget(
        family_targets,
        num_chunks=num_chunks,
        video_id=video_id,
        source_slot_requirements=source_slot_requirements,
    )

    def _append_slot(
        *,
        family: str,
        slot_index: int,
        defaults: Dict[str, str],
        qtype: str,
        support_chunks: List[int],
        answer_chunks: List[int],
        lifecycle: str,
    ) -> bool:
        support_chunks = sorted({int(c) for c in support_chunks})
        answer_chunks = sorted({int(c) for c in answer_chunks})
        reason = _slot_candidate_reject_reason(
            family,
            defaults,
            support_chunks,
            answer_chunks,
            evidence_by_chunk,
            used_signatures,
            exact_answer_uses,
            used_bins,
            num_chunks,
        )
        if reason:
            if audit is not None:
                audit[f"drop:{reason}"] += 1
            return False
        signature = _slot_signature(
            family,
            defaults,
            support_chunks,
            answer_chunks,
            evidence_by_chunk,
        )
        if signature:
            used_signatures.add(signature)
        exact_answer_uses.update(answer_chunks)
        for c in support_chunks:
            used_bins[c // max(1, SUPPORT_BIN_SIZE)] += 1
        source = _slot_supply_source(family, slot_index)
        slots.append({
            "slot_id": f"{video_id}_{family}_slot{slot_index}",
            "family": family,
            "slot_index": slot_index,
            "slot_source": source,
            "slot_keep_reason": "kept",
            "question_style": _style_for_slot(family, slot_index),
            "question_way": defaults["question_way"],
            "evidence_type": defaults["evidence_type"],
            "answer_form": str(FAMILY_RULES.get(family, {}).get("answer_form", "")),
            "question_type": qtype,
            "support_policy": defaults["support_policy"],
            "temporal_role": defaults["temporal_role"],
            "lifecycle": lifecycle,
            "target_ovo_task": defaults.get("benchmark_task", ""),
            "support_chunks": support_chunks,
            "answer_chunks": answer_chunks,
            "max_active_span": (
                max(answer_chunks) - min(answer_chunks) + 1
                if answer_chunks else 1
            ),
            "visual_window_chunks": VISUAL_WINDOW_CHUNKS,
            "seed": int(seed),
            **{
                key: defaults[key]
                for key in SLOT_METADATA_KEYS
                if key in defaults
            },
        })
        if audit is not None:
            audit["keep:kept"] += 1
            audit[f"keep_source:{source}"] += 1
        return True

    slot_jobs: List[Tuple[str, int, int, str]] = []
    for family, raw_target in planned_targets.items():
        if family not in FAMILY_SLOT_DEFAULTS:
            continue
        target = max(0, int(raw_target or 0))
        if family == "M1":
            target = min(target, 1)
        for slot_index in range(target):
            source = _slot_supply_source(family, slot_index)
            slot_jobs.append((
                source,
                slot_index,
                stable_mod(video_id, f"slot_job_{source}_{family}_{slot_index}", modulo=10_000),
                family,
            ))

    requirements = dict(source_slot_requirements or pass3a_source_slot_requirements(num_chunks))
    for slot_index, family in _ordered_slot_jobs(slot_jobs, requirements):
        defaults = _default_slot_fields(family, slot_index)
        qtype = QUESTION_TYPE_BY_FAMILY.get(family, "single_emit")
        if qtype == "multi_emit":
            candidate_answer_sets: List[List[int]] = []
            if family == "F5":
                global_prefix = defaults.get("slot_subtype") == "global_prefix_count"
                desired = 6 if global_prefix else 7
                candidate_answer_sets = _repeated_action_candidate_chunks(
                    ordered,
                    desired=desired,
                    global_prefix=global_prefix,
                    used_bins=used_bins,
                )
                max_span = 72 if global_prefix else 96
            elif family in {"F7", "CRR1"}:
                desired = 5 if family == "CRR1" else 7
                max_span = 36 if family == "CRR1" else 48
                candidate_answer_sets = _status_probe_candidate_chunks(
                    family,
                    ordered,
                    used_bins=used_bins,
                    desired=desired,
                    max_span=max_span,
                )
            elif family == "PN1":
                desired = 4
                max_span = 32
                candidate_answer_sets = _state_change_candidate_chunks(
                    ordered,
                    desired=desired,
                    used_bins=used_bins,
                    max_span=max_span,
                )
            else:
                desired = 7
                max_span = 48
                candidate = _multi_probe_chunks(family, ordered, used_bins, desired, max_span)
                candidate_answer_sets = [candidate] if candidate else []
            for answer_chunks in candidate_answer_sets:
                if not answer_chunks:
                    continue
                support_chunks = sorted(set(answer_chunks))
                if _append_slot(
                    family=family,
                    slot_index=slot_index,
                    defaults=defaults,
                    qtype=qtype,
                    support_chunks=support_chunks,
                    answer_chunks=answer_chunks,
                    lifecycle=defaults["lifecycle"],
                ):
                    break
        else:
            for chunk in _ranked_chunks_for_family(family, ordered, used_bins):
                if chunk < 0:
                    continue
                if family == "M1":
                    support_chunks = _single_support_chunks(family, chunk, ordered, num_chunks)
                    answer_chunks = [max(support_chunks)]
                else:
                    support_chunks = _single_support_chunks(family, chunk, ordered, num_chunks)
                    answer_chunks = [chunk]
                if _append_slot(
                    family=family,
                    slot_index=slot_index,
                    defaults=defaults,
                    qtype=qtype,
                    support_chunks=support_chunks,
                    answer_chunks=answer_chunks,
                    lifecycle=defaults["lifecycle"],
                ):
                    break
    return slots


def group_slots_by_family(slots: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for slot in slots:
        family = str(slot.get("family", ""))
        if family:
            grouped.setdefault(family, []).append(dict(slot))
    return grouped


def card_matches_planned_slot(card: Dict, planned_slots: List[Dict]) -> bool:
    """Validate a generated card against its planned slot contract."""
    if not planned_slots:
        return True
    slot_id = str(card.get("slot_id", "") or "")
    by_id = {str(slot.get("slot_id", "")): slot for slot in planned_slots}
    slot = by_id.get(slot_id)
    if not slot:
        return False
    if str(card.get("family", "")) != str(slot.get("family", "")):
        return False
    if str(card.get("question_way", "")) and str(card.get("question_way")) != str(slot.get("question_way")):
        return False
    if str(card.get("evidence_type", "")) and str(card.get("evidence_type")) != str(slot.get("evidence_type")):
        return False
    if str(card.get("support_policy", "")) and str(card.get("support_policy")) != str(slot.get("support_policy")):
        return False
    for key in (
        "legacy_family_id",
        "task_family",
        "task_subtype",
        "timing_type",
        "slot_group",
        "slot_subtype",
        "temporal_bucket",
        "answer_behavior",
    ):
        if str(card.get(key, "") or "") and str(card.get(key)) != str(slot.get(key, "")):
            return False
    try:
        expected_answers = sorted(int(c) for c in slot.get("answer_chunks") or [])
        got_answers = sorted(
            int(e.get("chunk"))
            for e in (card.get("gold_emits") or [])
            if isinstance(e, dict) and e.get("chunk") is not None
        )
    except (TypeError, ValueError):
        return False
    if expected_answers != got_answers:
        return False
    try:
        allowed_support = set(int(c) for c in slot.get("support_chunks") or [])
    except (TypeError, ValueError):
        return False
    allowed_support.update(expected_answers)
    try:
        got_support = {
            int(c)
            for c in (card.get("grounding_frames") or [])
            if c is not None
        }
    except (TypeError, ValueError):
        return False
    if str(card.get("family", "")) == "M1":
        return bool(got_support) and all(c <= max(expected_answers or [c]) for c in got_support)
    return bool(got_support) and got_support.issubset(allowed_support)


def slot_plan_summary(slots: Iterable[Dict]) -> Dict:
    slots = [dict(s) for s in slots]
    return {
        "slots": len(slots),
        "family": dict(Counter(str(s.get("family", "")) for s in slots)),
        "answer_form": dict(Counter(str(s.get("answer_form", "")) for s in slots)),
        "question_way": dict(Counter(str(s.get("question_way", "")) for s in slots)),
        "lifecycle": dict(Counter(str(s.get("lifecycle", "")) for s in slots)),
        "slot_source": dict(Counter(str(s.get("slot_source", "")) for s in slots)),
        "slot_keep_reason": dict(Counter(str(s.get("slot_keep_reason", "")) for s in slots)),
        "task_family": dict(Counter(str(s.get("task_family", "")) for s in slots)),
        "task_subtype": dict(Counter(str(s.get("task_subtype", "")) for s in slots)),
        "timing_type": dict(Counter(str(s.get("timing_type", "")) for s in slots)),
        "slot_group": dict(Counter(str(s.get("slot_group", "")) for s in slots)),
        "slot_subtype": dict(Counter(str(s.get("slot_subtype", "")) for s in slots)),
        "temporal_bucket": dict(Counter(str(s.get("temporal_bucket", "")) for s in slots)),
        "answer_behavior": dict(Counter(str(s.get("answer_behavior", "")) for s in slots)),
        "support_bin": dict(Counter(
            min(int(c) for c in (s.get("answer_chunks") or [0])) // max(1, SUPPORT_BIN_SIZE)
            for s in slots
        )),
    }
