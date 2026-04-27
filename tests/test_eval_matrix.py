"""Tests for the new eval framework (base + agent × test + OVO).

Covers the GPU-free helpers in:
  - scripts/eval/ovo/base.py  — frame sampling, aggregation, dispatch routing
  - scripts/eval/test_set_agent.py — gold extraction, scoring, question pull
  - scripts/eval/run_matrix.sh — bash syntax + matrix coverage

We do NOT instantiate models, the StreamingAgentLoop, or call .generate() —
those need GPUs and are exercised by the actual eval runs. These tests
catch the logic bugs that you'd only otherwise discover after a multi-hour
eval finishes with garbage numbers.

Test isolation: we read the eval scripts' source and exec selected def
blocks into a private namespace via `_load_helpers_from_source`. We do NOT
import the eval scripts as modules (which would import torch / transformers
and either fail or pollute sys.modules for other test files). The blocks
we test are pure-Python — string parsing, regex, frame-path math — none of
them call torch at runtime, so this approach works without any stubs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import subprocess
from typing import Dict

import pytest


# ─── 1. test_set_agent helpers (pure-Python, no torch needed) ────────────────


def _load_helpers_from_source(path: str, names):
    """Re-exec selected def blocks from a script. Faster + more isolated than
    importing the full module — even with stubs, importing pulls module-level
    side effects we don't need. Pre-seeds common deps (re, Path) since most
    helpers use them.
    """
    src = Path(path).read_text()
    out: Dict = {
        "re": re,
        "Path": Path,
        "defaultdict": __import__("collections").defaultdict,
    }
    for name in names:
        m = re.search(rf"^def {name}\(.*?(?=^def |\Z)", src,
                      re.MULTILINE | re.DOTALL)
        if not m:
            raise AssertionError(f"def {name} not found in {path}")
        exec(m.group(0), out)
    return out


# Module-level regex constant referenced by test_set_agent.extract_gold.
# Pre-load into the namespace whenever we exec that def.
_GOLD_RE_PATTERN = r"<response>(.*?)</response>"


def test_test_set_agent_gold_kind_extraction():
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "test_set_agent.py"),
        ["extract_gold", "gold_kind"],
    )
    h["GOLD_RE"] = re.compile(_GOLD_RE_PATTERN, re.DOTALL)
    extract_gold = h["extract_gold"]
    gold_kind = h["gold_kind"]

    # Assemble synthetic samples
    s_yes = {"output": "<think>obs</think><action>response</action><response>Yes</response>"}
    s_no = {"output": "<response>No</response>"}
    s_int = {"output": "<response>3</response>"}
    s_letter = {"output": "<response>A</response>"}
    s_desc = {"output": "<response>The chef adds salt to the pot</response>"}
    s_none = {"output": "<think>obs</think><action>silent</action>"}

    assert extract_gold(s_yes) == "Yes" and gold_kind("Yes") == "yes_no"
    assert extract_gold(s_no) == "No" and gold_kind("No") == "yes_no"
    assert extract_gold(s_int) == "3" and gold_kind("3") == "int"
    assert extract_gold(s_letter) == "A" and gold_kind("A") == "letter"
    assert extract_gold(s_desc) == "The chef adds salt to the pot"
    assert gold_kind("The chef adds salt to the pot") == "descriptive"
    assert extract_gold(s_none) is None
    assert gold_kind(None) is None


def test_test_set_agent_question_extraction_priority():
    """user_input > most-recent-unanswered query > most-recent query."""
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "test_set_agent.py"),
        ["extract_question"],
    )
    extract_question = h["extract_question"]

    # 1. user_input wins
    s = {"input": {"user_input": "What color?", "queries": [
        {"question": "old", "answers": [{"text": "red"}]},
    ]}}
    assert extract_question(s) == "What color?"

    # 2. No user_input → most-recent UNANSWERED
    s = {"input": {"user_input": "", "queries": [
        {"question": "Q1", "answers": []},
        {"question": "Q2", "answers": [{"text": "ans"}]},
        {"question": "Q3", "answers": []},
    ]}}
    assert extract_question(s) == "Q3"

    # 3. All answered → most-recent of any kind
    s = {"input": {"user_input": "", "queries": [
        {"question": "QX", "answers": [{"text": "a"}]},
        {"question": "QY", "answers": [{"text": "b"}]},
    ]}}
    assert extract_question(s) == "QY"

    # 4. No queries → None
    assert extract_question({"input": {}}) is None


def test_test_set_agent_score_yes_no_int_letter():
    """score() must mirror OVO scoring: case-insensitive Yes/No, first int,
    first letter."""
    # We need is_yes/is_no/extract_int/extract_letter from eval_full —
    # also from source to avoid torch import.
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "ovo" / "eval_full.py"),
        ["is_yes", "is_no", "extract_int", "extract_letter"],
    )
    # Inject module-level regex constants the helpers reference at call time.
    h["_LETTER_RE"] = re.compile(r"\b([A-Da-d])\b")
    h["_INT_RE"] = re.compile(r"\d+")
    # Compose score inline with the same logic test_set_agent.score uses.
    def score(pred_text, gold, kind):
        if not pred_text or not gold:
            return False
        pred = pred_text.strip()
        if kind == "yes_no":
            if h["is_yes"](pred):
                return gold == "Yes"
            if h["is_no"](pred):
                return gold == "No"
            return False
        if kind == "int":
            v = h["extract_int"](pred)
            return v is not None and str(v) == gold
        if kind == "letter":
            let = h["extract_letter"](pred)
            return let is not None and let.upper() == gold.upper()
        return False

    # Yes/No
    assert score("Yes", "Yes", "yes_no")
    assert score("yes, the chef did", "Yes", "yes_no")
    assert score("No", "No", "yes_no")
    assert score("no, never", "No", "yes_no")
    assert not score("Yes", "No", "yes_no")
    assert not score("maybe", "Yes", "yes_no")

    # Int (extract first integer)
    assert score("3", "3", "int")
    assert score("There are 3 tomatoes", "3", "int")
    assert not score("There are 5 tomatoes", "3", "int")
    assert not score("three", "3", "int")  # word not digit

    # Letter (first A-D)
    assert score("A", "A", "letter")
    assert score("A. eggplant", "A", "letter")
    assert score("a", "A", "letter")
    assert not score("B", "A", "letter")


# ─── 2. ovo/base.py frame sampling ───────────────────────────────────────────


def test_ovo_base_frame_sampling_in_range(tmp_path):
    """sample_frame_paths picks frames from [t_start, t_end] only and caps at n."""
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "ovo" / "base.py"),
        ["sample_frame_paths"],
    )
    sample_frame_paths = h["sample_frame_paths"]

    # Build fake frame dir: video "demo" with frames 0-9 (10 frames at 1 fps).
    frames_root = tmp_path / "frames"
    video_dir = frames_root / "demo"
    video_dir.mkdir(parents=True)
    for i in range(10):
        (video_dir / f"{i:06d}.jpg").write_bytes(b"fake")

    # Full range, n=10 → all 10
    out = sample_frame_paths(frames_root, "demo", 0, 9, 10)
    assert len(out) == 10
    assert all(p.endswith(".jpg") for p in out)

    # Partial range [3, 7] → 5 frames in range
    out = sample_frame_paths(frames_root, "demo", 3, 7, 5)
    assert len(out) == 5
    assert all("00000" in p for p in out)  # all single-digit indices padded

    # Cap n=3 from 10 candidates → uniformly sampled
    out = sample_frame_paths(frames_root, "demo", 0, 9, 3)
    assert len(out) == 3
    # First, middle-ish, last
    indices = [int(Path(p).stem) for p in out]
    assert indices[0] == 0 and indices[-1] == 9

    # Out-of-range t_start/t_end → empty/None
    out = sample_frame_paths(frames_root, "demo", 100, 200, 5)
    assert out is None


def test_ovo_base_frame_sampling_missing_dir(tmp_path):
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "ovo" / "base.py"),
        ["sample_frame_paths"],
    )
    out = h["sample_frame_paths"](tmp_path, "nonexistent", 0, 10, 5)
    assert out is None


# ─── 3. ovo/base.py aggregation ──────────────────────────────────────────────


def test_ovo_base_aggregate_categories():
    """aggregate() must produce per-task acc + RT/BT/FT category averages +
    overall = mean of category averages (matches OVO paper)."""
    # Mirror RT/BT/FT taxonomy from eval_full.py — copy to avoid the import.
    RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
    BT_TASKS = {"EPM", "ASI", "HLD"}
    FT_TASKS = {"REC", "SSR", "CRR"}

    src = (ROOT / "scripts" / "eval" / "ovo" / "base.py").read_text()
    ns: Dict = {
        "defaultdict": __import__("collections").defaultdict,
        "RT_TASKS": RT_TASKS, "BT_TASKS": BT_TASKS, "FT_TASKS": FT_TASKS,
    }
    m = re.search(r"^def aggregate\(.*?(?=^def |\Z)", src,
                  re.MULTILINE | re.DOTALL)
    exec(m.group(0), ns)
    aggregate = ns["aggregate"]

    results = [
        {"task": "OCR", "probes": [{"correct": True}, {"correct": False}]},
        {"task": "EPM", "probes": [{"correct": True}]},
        {"task": "REC", "probes": [{"correct": True}, {"correct": True}]},
    ]
    agg = aggregate(results)
    assert agg["per_task"]["OCR"]["acc"] == 0.5
    assert agg["per_task"]["EPM"]["acc"] == 1.0
    assert agg["per_task"]["REC"]["acc"] == 1.0
    # RT contains OCR (just 1 task in this fixture)
    assert "RT" in agg["category"]
    # Overall is mean of (RT_avg, BT_avg, FT_avg)
    assert 0.0 <= agg["overall"] <= 1.0


# ─── 4. run_matrix.sh sanity ─────────────────────────────────────────────────


def test_run_matrix_bash_syntax():
    """`bash -n` ensures the matrix script parses."""
    res = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "eval" / "run_matrix.sh")],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"


def test_run_matrix_covers_full_grid():
    """Inspect the script source: it must invoke all 4 entries and cover
    {test,ovo} × all retrievers × both compress modes."""
    src = (ROOT / "scripts" / "eval" / "run_matrix.sh").read_text()
    # Entry points
    for entry in (
        "scripts/eval/test_set_base.py",
        "scripts/eval/ovo/base.py",
        "scripts/eval/test_set_agent.py",
        "scripts/eval/ovo/eval_full.py",
    ):
        assert entry in src, f"matrix script missing {entry}"
    # Frame budgets
    for n in ("64", "128", "256", "512", "1024"):
        assert n in src, f"matrix script missing budget {n}"
    # Retrievers and compress modes
    assert "bm25" in src and "hybrid" in src
    assert "system" in src and "self" in src
    # The 3 base ckpts
    assert "QWEN3VL_2B" in src
    assert "QWEN3VL_4B" in src
    assert "QWEN3VL_8B" in src


def test_run_matrix_help_modes_exist():
    """`./run_matrix.sh foo` must reject; `base|agent|all` should be valid."""
    res = subprocess.run(
        ["bash", str(ROOT / "scripts" / "eval" / "run_matrix.sh"), "bogus"],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "Usage:" in res.stdout or "Usage:" in res.stderr


# ─── 5. dispatch routing for OVO base ────────────────────────────────────────


def test_test_set_base_score_lenient_vs_strict():
    """test_set_base.score(): lenient = first-token-wins, strict = exact match."""
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "test_set_base.py"),
        ["score"],
    )
    score = h["score"]

    # Lenient (default-equivalent) — first token wins
    assert score("Yes", "Yes", "yes_no", "lenient")
    assert score("yes, definitely", "Yes", "yes_no", "lenient")
    assert score("A. eggplant", "A", "letter", "lenient")
    assert score("There are 3 tomatoes", "3", "int", "lenient")

    # Strict — must equal gold exactly
    assert score("Yes", "Yes", "yes_no", "strict")
    assert not score("yes", "Yes", "yes_no", "strict")          # case
    assert not score("Yes, definitely", "Yes", "yes_no", "strict")  # extra text
    assert score("A", "A", "letter", "strict")
    assert not score("A.", "A", "letter", "strict")             # extra period
    assert not score("a", "A", "letter", "strict")              # case
    assert score("3", "3", "int", "strict")
    assert not score("3 times", "3", "int", "strict")           # extra words


def test_ovo_base_strict_format_helpers():
    """The strict-letter / strict-yes-no / strict-int helpers reject anything
    that's not the exact expected token."""
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "eval" / "ovo" / "base.py"),
        ["_strict_letter", "_strict_yes_no", "_strict_int"],
    )
    sl, sy, si = h["_strict_letter"], h["_strict_yes_no"], h["_strict_int"]
    # Strict letter
    assert sl("A") == "A" and sl("D") == "D"
    assert sl("a") is None       # case-sensitive
    assert sl("E") is None       # out of A-D
    assert sl("A. eggplant") is None
    assert sl("") is None
    # Strict yes/no
    assert sy("Yes") == "Yes" and sy("No") == "No"
    assert sy("yes") is None and sy("Yes, definitely") is None
    # Strict int
    assert si("3") == "3" and si("12") == "12"
    assert si("3 times") is None and si("three") is None


def test_eval_full_lenient_constant_set():
    """eval_full.py defines LENIENT_MAX_EXTRA_CHUNKS — verify it's a
    sensibly-large value (must allow the agent multiple chunks past the
    question to actually produce a late response, but not so large it
    walks past video end on every sample)."""
    src = (ROOT / "scripts" / "eval" / "ovo" / "eval_full.py").read_text()
    m = re.search(r"LENIENT_MAX_EXTRA_CHUNKS\s*=\s*(\d+)", src)
    assert m, "LENIENT_MAX_EXTRA_CHUNKS not found"
    val = int(m.group(1))
    assert 30 <= val <= 200, f"unreasonable LENIENT cap: {val}"


def test_test_set_agent_walk_modes_in_signature():
    """walk_and_score signature must accept scoring + num_chunks_video."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    m = re.search(r"def walk_and_score\(([^)]*)\)", src, re.DOTALL)
    assert m, "walk_and_score not found"
    sig = m.group(1)
    assert "scoring" in sig
    assert "num_chunks_video" in sig


def test_eval_pixels_match_sft():
    """v9.4.2 — eval scripts must use SFT-aligned pixel budget. Larger
    values silently double per-frame visual tokens and overflow
    model_max_length. We assert SFT defaults appear; we strip comments
    before checking that the legacy 100352*2 / *4 expressions are gone
    (the comments referencing 'was 100352*2' should remain as audit trail).
    """
    for f in ("scripts/eval/ovo/eval_full.py",
              "scripts/eval/test_set_agent.py"):
        src = (ROOT / f).read_text()
        # Strip line comments before checking for legacy literals
        stripped = re.sub(r"#.*", "", src)
        assert "min_pixels=100352" in stripped or "min_pixels = 100352" in stripped, \
            f"{f}: min_pixels must equal SFT 100352 (was 200704)"
        assert "max_pixels=150528" in stripped or "max_pixels = 150528" in stripped, \
            f"{f}: max_pixels must equal SFT 150528 (was 401408)"
        # Legacy expressions must be gone from CODE (comments OK)
        assert "100352*2" not in stripped, f"{f}: legacy min_pixels=200704 still present in code"
        assert "100352*4" not in stripped, f"{f}: legacy max_pixels=401408 still present in code"


def test_queries_cap_default_safe():
    """Default QUERIES_HISTORY_CAP must equal SFT-aligned safe value (8).
    The 32k profile bumps this at runtime via apply_profile()."""
    src = (ROOT / "thinkstream" / "data" / "agent_protocol.py").read_text()
    m = re.search(r"^QUERIES_HISTORY_CAP\s*=\s*(\d+)", src, re.MULTILINE)
    assert m, "QUERIES_HISTORY_CAP not set"
    assert int(m.group(1)) == 8, \
        f"default queries cap must be 8 (SFT-aligned); got {m.group(1)}"
    # Pending-vs-answered policy still documented in comment
    assert "pending" in src.lower() and "keep" in src.lower()


def test_recall_text_cap_default_safe():
    """Default RECALL_TEXT_MAX_CHARS must be the SFT-aligned safe value (800)."""
    src = (ROOT / "thinkstream" / "data" / "agent_protocol.py").read_text()
    m = re.search(r"^RECALL_TEXT_MAX_CHARS\s*=\s*(\d+)", src, re.MULTILINE)
    assert m, "RECALL_TEXT_MAX_CHARS not set"
    assert int(m.group(1)) == 800, \
        f"default recall char-cap must be 800; got {m.group(1)}"
    # And the build_user_content path uses the constant (not a hardcoded literal)
    block = src[src.find("Zone C continued: Recall result"):
                src.find("Zone D: User input")]
    assert "RECALL_TEXT_MAX_CHARS" in block, \
        "recall block must reference RECALL_TEXT_MAX_CHARS, not a literal"


def test_eval_profiles_present():
    """eval_profiles.py exposes both 16k and 32k profiles + apply_profile()."""
    from scripts.eval.eval_profiles import (
        EVAL_PROFILES, apply_profile, describe_profile,
    )
    assert "16k" in EVAL_PROFILES and "32k" in EVAL_PROFILES
    p16 = EVAL_PROFILES["16k"]
    p32 = EVAL_PROFILES["32k"]
    # Required keys
    for key in ("model_max_length", "max_new_tokens_default",
                "queries_history_cap", "recall_text_max_chars",
                "subtotal_tokens_estimate", "headroom_tokens_estimate"):
        assert key in p16 and key in p32, f"profile missing key {key}"
    # 32k must be strictly larger across the four eval-side caps
    assert p32["model_max_length"] > p16["model_max_length"]
    assert p32["queries_history_cap"] > p16["queries_history_cap"]
    assert p32["recall_text_max_chars"] > p16["recall_text_max_chars"]
    assert p32["max_new_tokens_default"] >= p16["max_new_tokens_default"]
    # Subtotals add up to less than max_length
    for name in ("16k", "32k"):
        cfg = EVAL_PROFILES[name]
        assert cfg["subtotal_tokens_estimate"] < cfg["model_max_length"], \
            f"{name}: subtotal exceeds max_length"
        assert cfg["headroom_tokens_estimate"] > 0


def test_apply_profile_mutates_globals():
    """apply_profile('32k') must update agent_protocol module globals,
    then apply_profile('16k') must restore them."""
    from scripts.eval.eval_profiles import apply_profile, EVAL_PROFILES
    from thinkstream.data import agent_protocol
    # Save originals
    orig_q = agent_protocol.QUERIES_HISTORY_CAP
    orig_r = agent_protocol.RECALL_TEXT_MAX_CHARS
    try:
        apply_profile("32k")
        assert agent_protocol.QUERIES_HISTORY_CAP == EVAL_PROFILES["32k"]["queries_history_cap"]
        assert agent_protocol.RECALL_TEXT_MAX_CHARS == EVAL_PROFILES["32k"]["recall_text_max_chars"]
        apply_profile("16k")
        assert agent_protocol.QUERIES_HISTORY_CAP == EVAL_PROFILES["16k"]["queries_history_cap"]
        assert agent_protocol.RECALL_TEXT_MAX_CHARS == EVAL_PROFILES["16k"]["recall_text_max_chars"]
    finally:
        agent_protocol.QUERIES_HISTORY_CAP = orig_q
        agent_protocol.RECALL_TEXT_MAX_CHARS = orig_r


def test_eval_scripts_have_profile_flag():
    """Both agent eval scripts must expose --profile {16k,32k}, plumb the
    chosen cfg's model_max_length to the tokenizer, and stamp the profile
    name into the output JSON."""
    for f in ("scripts/eval/test_set_agent.py", "scripts/eval/ovo/eval_full.py"):
        src = (ROOT / f).read_text()
        assert '"--profile"' in src, f"{f}: missing --profile flag"
        assert 'apply_profile' in src, f"{f}: missing apply_profile() call"
        assert 'profile_cfg["model_max_length"]' in src, \
            f"{f}: tokenizer not using profile_cfg model_max_length"
        # Output JSON must record which profile ran
        assert '"profile"' in src and '"profile_cfg"' in src, \
            f"{f}: output JSON missing profile metadata"


def test_walk_and_score_signature_includes_support_chunks():
    """walk_and_score must accept support_chunks for recall hit-rate calc."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    m = re.search(r"def walk_and_score\(([^)]*)\)", src, re.DOTALL)
    assert m, "walk_and_score signature not found"
    sig = m.group(1)
    assert "support_chunks" in sig
    assert "scoring" in sig and "num_chunks_video" in sig


def test_walk_and_score_telemetry_fields():
    """walk_and_score return dict must surface the v9.4.2 telemetry fields."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    # Extract the return-statement block
    fn_block = src[src.find("def walk_and_score("):
                   src.find("# ─── Main")]
    for field in (
        "n_compress_events", "compress_thinks_at_trigger",
        "compress_chunks_per_event",
        "n_recall", "recall_events",
        "response_offset_chunks",
        "n_premature_responses", "n_late_responses",
    ):
        assert f'"{field}"' in fn_block, \
            f"walk_and_score must return field {field!r}"


def test_walk_and_score_premature_bug_fixed():
    """Premature responses (chunk < ask_chunk) must NOT be captured as the
    accepted answer — they only increment n_premature_responses telemetry."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    fn_block = src[src.find("def walk_and_score("):
                   src.find("# ─── Main")]
    # Both response paths (direct + after recall) must gate on chunk_idx
    # >= ask_chunk before assigning response_text.
    assert "if chunk_idx < ask_chunk:" in fn_block
    assert "n_premature_responses += 1" in fn_block


def test_agent_loop_emits_compress_telemetry():
    """agent_loop.step() must populate parsed['compress_telemetry'] when
    compression fires (else None) and parsed['recall_returned_chunks']."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    assert "compress_telemetry" in src
    assert '"thinks_count_at_trigger"' in src
    assert '"compressed_chunks"' in src
    # parsed dict carries them
    assert 'parsed["compress_telemetry"]' in src
    assert 'parsed["recall_returned_chunks"]' in src


def test_agent_loop_emits_extra_telemetry():
    """v9.4.2 — agent_loop.step() must populate 4 extra per-step metrics:
    prompt_text_token_count, think_token_count, format_ok,
    compress_succeeded (None when no trigger fired)."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    for field in ("prompt_text_token_count", "think_token_count",
                  "format_ok", "compress_succeeded"):
        assert f'parsed["{field}"]' in src, f"missing parsed[{field!r}]"
    # self.tokenizer must be set in __init__ (used by token-counting telemetry)
    assert "self.tokenizer = tokenizer" in src


def test_walk_and_score_aggregates_extra_telemetry():
    """walk_and_score returns prompt-tokens / think-tokens / format-violations
    / compress-attempts vs successes."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    fn_block = src[src.find("def walk_and_score("):
                   src.find("# ─── Main")]
    for field in ("prompt_text_tokens_per_step", "think_tokens_per_step",
                  "n_format_violations", "n_steps",
                  "n_compress_attempts", "n_compress_succeeded"):
        assert f'"{field}"' in fn_block, \
            f"walk_and_score must return field {field!r}"


def test_test_set_agent_report_has_two_blocks():
    """Report must show two blocks: ACC/CMP/RECALL/TIMING and PROMPT/THINK/FORMAT."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    assert "ACCURACY / COMPRESS / RECALL / TIMING" in src
    assert "PROMPT LENGTH / THINK CHATTINESS / FORMAT" in src
    # Specific column headers
    assert "cmp_ok" in src, "compress success rate column missing"
    assert "fmt_err%" in src, "format violation column missing"
    assert "pt_avg" in src and "pt_max" in src, "prompt-token columns missing"


def test_test_set_agent_resets_visual_index_per_sample():
    """v9.4.2 BUG FIX: walk_and_score must call reset_visual_index(retriever)
    per sample, else hybrid retriever's siglip embeddings leak across
    sample boundaries → false recalls."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    assert "reset_visual_index" in src, \
        "test_set_agent.py must import + call reset_visual_index"
    # Specifically inside walk_and_score, after loop.reset()
    fn_block = src[src.find("def walk_and_score("):
                   src.find("# ─── Main")]
    assert "reset_visual_index(loop.retriever)" in fn_block, \
        "reset_visual_index must be called inside walk_and_score"


def test_extract_question_rejects_whitespace():
    """v9.4.2: '   ' is truthy but not a real question. extract_question
    must require .strip() to be non-empty."""
    for f in ("scripts/eval/test_set_agent.py", "scripts/eval/test_set_base.py"):
        src = (ROOT / f).read_text()
        eq_block = src[src.find("def extract_question("):
                       src.find("def extract_question(") + 800]
        assert ".strip()" in eq_block, \
            f"{f}: extract_question must reject whitespace-only inputs"


def test_base_evals_have_profile_flag():
    """Both base eval scripts now expose --profile for parity with agent
    eval. Effect is metadata-only (base VLM doesn't use queries/recall caps),
    but stamping the profile in output JSON enables fair comparison."""
    for f in ("scripts/eval/test_set_base.py", "scripts/eval/ovo/base.py"):
        src = (ROOT / f).read_text()
        assert '"--profile"' in src, f"{f}: missing --profile CLI"
        assert '"profile"' in src and "args.profile" in src, \
            f"{f}: profile not threaded into output metadata"


def test_walk_and_score_tolerates_eof_errors():
    """In lenient mode, walk past video end may raise on step(). Tolerate
    up to 3 errors AFTER ask_chunk; one error before ask_chunk still bails."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    fn_block = src[src.find("def walk_and_score("):
                   src.find("# ─── Main")]
    assert "n_step_errors" in fn_block
    assert "if chunk_idx < ask_chunk:" in fn_block and "break" in fn_block
    assert "n_step_errors >= 3" in fn_block


def test_eval_full_telemetry_extras():
    """eval_full.py:run_agent must capture prompt/think/format/compress-success
    extras into the telemetry dict; eval_mcq surfaces them."""
    src = (ROOT / "scripts" / "eval" / "ovo" / "eval_full.py").read_text()
    # run_agent collects per-step
    run_block = src[src.find("def run_agent("):src.find("def make_loop(")]
    for field in ("prompt_tokens_per_step", "think_tokens_per_step",
                  "n_format_violations", "total_steps"):
        assert field in run_block, f"run_agent missing {field}"
    assert '"succeeded"' in run_block, \
        "compress event must record success/fail flag"
    # eval_mcq surfaces them
    mcq_block = src[src.find("def eval_mcq("):src.find("def eval_rec(")]
    assert "n_compress_succeeded" in mcq_block
    assert "prompt_tokens_per_step" in mcq_block


def test_eval_full_run_agent_takes_telemetry():
    """eval_full.py:run_agent must accept a telemetry dict for OVO MCQ stats."""
    src = (ROOT / "scripts" / "eval" / "ovo" / "eval_full.py").read_text()
    m = re.search(r"def run_agent\(([^)]*)\)", src, re.DOTALL)
    assert m
    assert "telemetry" in m.group(1)
    # eval_mcq must return a 'telemetry' field on the result
    mcq_block = src[src.find("def eval_mcq("):src.find("def eval_rec(")]
    assert '"telemetry"' in mcq_block
    assert 'n_compress_events' in mcq_block
    assert 'n_premature_responses' in mcq_block


def test_run_matrix_runs_both_profiles():
    src = (ROOT / "scripts" / "eval" / "run_matrix.sh").read_text()
    assert "PROFILES=(16k 32k)" in src or "PROFILES=(32k 16k)" in src
    assert 'for prof in "${PROFILES[@]}"' in src


def test_should_compress_emergency_trigger():
    """v9.4.2: emergency compress trigger when individual thinks are
    pathologically long (1.5× normal threshold + len≥2). Without this,
    a verbose model emitting 1000-tok thinks would carry 2000+ tok in
    recent_thinks before COMPRESS_RANGE_MIN=4 fires."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    sc_block = src[src.find("def should_compress("):
                   src.find("def compress(")]
    # Standard trigger: tokens>=480 AND len>=4
    assert "COMPRESS_TOKEN_THRESHOLD" in sc_block
    assert "COMPRESS_RANGE_MIN" in sc_block
    # Emergency trigger: tokens >= 1.5× threshold AND len >= 2
    assert "1.5" in sc_block, "emergency trigger should use 1.5× threshold"
    assert "n_thinks >= 2" in sc_block or "len(self.recent_thinks) >= 2" in sc_block, \
        "emergency trigger needs minimum-2 condition"


def test_summary_capped_at_storage_time():
    """v9.4.2: incoming <summary> text must be capped at 200 tok at compress()
    time, NOT only when merging. Verbose model summaries used to bloat the
    compressed_segments zone to ~2000 tok before merge fired."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    # Find compress() body
    cmp_block = src[src.find("def compress("):
                    src.find("def add_query(") if "def add_query(" in src
                    else src.find("# --- Queries tracking")]
    # Must encode and truncate BEFORE append
    assert "Cap summary text BEFORE storing" in cmp_block, \
        "compress() must cap summary text before append"
    assert "len(ids) > 200" in cmp_block
    assert "_tokenizer.decode(ids[:200])" in cmp_block
    # Truncation must happen before the .append(summary) line
    cap_pos = cmp_block.find("len(ids) > 200")
    append_pos = cmp_block.find("self.compressed_segments.append(summary)")
    assert cap_pos < append_pos, \
        "cap must be applied BEFORE append, else 5 segments can stack uncapped"


def test_compressed_segment_merge_cap_aligned_with_sft():
    """Merged compressed segments capped at 200 tokens — matches the SFT
    data construction value (config.py); going below would OOD the model
    relative to training distribution."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    merge_block = src[src.find("Merge oldest two"):
                      src.find("self.compressed_segments.insert(0, merged)") + 60]
    m = re.search(r"len\(ids\)\s*>\s*(\d+)", merge_block)
    assert m, "merge cap not found"
    cap = int(m.group(1))
    # SFT was constructed at 200; eval cap should match (or be tighter,
    # which still works because SFT saw segments ≤200 already).
    assert cap == 200, f"merge cap {cap} should equal SFT-baked value 200"


def test_no_dangling_model_max_length_flag():
    """v9.4.2 transitional: --model_max_length flag was replaced by
    --profile. Make sure neither script still parses the old flag (would
    cause CLI confusion / silent-ignore)."""
    for f in ("scripts/eval/test_set_agent.py", "scripts/eval/ovo/eval_full.py"):
        src = (ROOT / f).read_text()
        assert '"--model_max_length"' not in src, \
            f"{f}: --model_max_length CLI flag should be removed (use --profile)"


def test_run_matrix_runs_both_scorings():
    """Driver must loop over both strict and lenient — that's the whole point."""
    src = (ROOT / "scripts" / "eval" / "run_matrix.sh").read_text()
    assert "SCORINGS=(strict lenient)" in src or \
        "SCORINGS=(lenient strict)" in src, \
        "Driver must enumerate both scoring modes"
    # And actually loop over them
    assert 'for sc in "${SCORINGS[@]}"' in src


def test_ovo_base_dispatch_routes_by_task():
    """dispatch() returns the right callable per task family. We only check
    the routing branches via mock; actual evaluators need GPU."""
    src = (ROOT / "scripts" / "eval" / "ovo" / "base.py").read_text()
    m = re.search(r"^def dispatch\(.*?(?=^def |\Z)", src,
                  re.MULTILINE | re.DOTALL)
    # Build minimal namespace with stubs for the 4 evaluators
    calls = {}
    def _stub(name):
        def _f(sample, **kw):
            calls["last"] = (name, sample.get("task"))
            return {"task": sample.get("task"), "probes": []}
        return _f
    # Mirror RT/BT taxonomy without importing eval_full
    RT_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
    BT_TASKS = {"EPM", "ASI", "HLD"}
    ns: Dict = {
        "RT_TASKS": RT_TASKS, "BT_TASKS": BT_TASKS,
        "eval_mcq_base":  _stub("mcq"),
        "eval_rec_base":  _stub("rec"),
        "eval_ssr_base":  _stub("ssr"),
        "eval_crr_base":  _stub("crr"),
    }
    exec(m.group(0), ns)
    dispatch = ns["dispatch"]

    dispatch({"task": "OCR"});  assert calls["last"] == ("mcq", "OCR")
    dispatch({"task": "EPM"});  assert calls["last"] == ("mcq", "EPM")
    dispatch({"task": "REC"});  assert calls["last"] == ("rec", "REC")
    dispatch({"task": "SSR"});  assert calls["last"] == ("ssr", "SSR")
    dispatch({"task": "CRR"});  assert calls["last"] == ("crr", "CRR")
    # Unknown task → None
    assert dispatch({"task": "ZZZ"}) is None
