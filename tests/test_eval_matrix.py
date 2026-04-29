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


def _load_helpers_from_source(path: str, names, extra_globals=None):
    """Re-exec selected def blocks from a script. Faster + more isolated than
    importing the full module — even with stubs, importing pulls module-level
    side effects we don't need. Pre-seeds common deps (re, Path, typing) since
    most helpers use them.

    extra_globals: optional dict of extra names to seed (constants like
    AGENT_CHUNK_SEC, module-level config) before exec.
    """
    import typing
    src = Path(path).read_text()
    out: Dict = {
        "re": re,
        "Path": Path,
        "defaultdict": __import__("collections").defaultdict,
        # Type-annotation names commonly used in signatures
        "List": typing.List,
        "Dict": typing.Dict,
        "Optional": typing.Optional,
        "Any": typing.Any,
        "Tuple": typing.Tuple,
        "Callable": typing.Callable,
    }
    if extra_globals:
        out.update(extra_globals)
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


def test_recall_text_cap_aligned_to_sft():
    """RECALL_TEXT_MAX_CHARS=1600 ≈ 4 thinks × THINK_TOKENS.max(100) × ~4
    char/tok — the upper bound of SFT's recall_result text_content
    distribution. Below this is over-tightening; above is wasted budget."""
    src = (ROOT / "thinkstream" / "data" / "agent_protocol.py").read_text()
    m = re.search(r"^RECALL_TEXT_MAX_CHARS\s*=\s*(\d+)", src, re.MULTILINE)
    assert m, "RECALL_TEXT_MAX_CHARS not set"
    assert int(m.group(1)) == 1600, \
        f"default recall char-cap must be 1600 (SFT upper bound); got {m.group(1)}"
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


def test_should_compress_matches_sft():
    """should_compress is single-condition (>=480 tok AND >=4 thinks),
    matching SFT pass2_rollout. No emergency / 1.5× / len>=2 branches —
    those would be OOD vs training."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    sc_block = src[src.find("def should_compress("):
                   src.find("def compress(")]
    assert "COMPRESS_TOKEN_THRESHOLD" in sc_block
    assert "COMPRESS_RANGE_MIN" in sc_block
    assert "1.5" not in sc_block, "no 1.5× emergency trigger (OOD vs SFT)"
    assert "len(self.recent_thinks) >= 2" not in sc_block


def test_summary_cap_aligned_with_sft():
    """Both incoming summary and merged segments cap at SUMMARY_TOKENS_MAX
    (=180), matching SFT config.py. Going above 180 is OOD; going below is
    over-tightening (model never trained on shorter)."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    assert "SUMMARY_TOKENS_MAX = 180" in src, "module-level cap constant"
    cmp_block = src[src.find("def compress("):
                    src.find("# --- Queries tracking")]
    assert "len(ids) > SUMMARY_TOKENS_MAX" in cmp_block
    assert "_tokenizer.decode(ids[:SUMMARY_TOKENS_MAX])" in cmp_block
    # Incoming cap must run BEFORE append
    cap_pos = cmp_block.find("len(ids) > SUMMARY_TOKENS_MAX")
    append_pos = cmp_block.find("self.compressed_segments.append(summary)")
    assert cap_pos < append_pos, "incoming cap must precede append"


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


def test_render_sample_injects_recalled_frames():
    """SFT recall_response samples must carry `recalled_frames` with
    frame_paths so the model trains on visual recall, not text-only.
    Without this, eval feeding `<recalled_frames>` + frames is OOD vs SFT."""
    from scripts.agent_data_v5.render_samples import _build_recalled_frames
    # historical_frames recall with returned_chunks → produces frame_paths
    rr = {
        "source": "historical_frames",
        "text_content": "[10s] foo",
        "returned_chunks": [5, 6, 7, 8],
    }
    # Per-video frame list: 1fps, 2 frames per chunk → indices [10..15] map
    # to chunks 5..7. Provide enough to cover chunk 8.
    all_frames = [f"frame_{i:06d}.jpg" for i in range(1, 21)]
    rf = _build_recalled_frames(rr, all_frames)
    assert rf is not None
    assert rf["source"] == "historical_frames"
    assert rf["time_range"] == [10, 18]   # min(5)*2 to (max(8)+1)*2
    assert rf["n_frames"] == 8            # 4 chunks × 2 frames
    assert "frame_paths" in rf and len(rf["frame_paths"]) == 8

    # failure / no chunks → None (text-only recall, legacy behaviour)
    assert _build_recalled_frames({"source": "failure",
                                   "returned_chunks": []}, all_frames) is None
    assert _build_recalled_frames({"source": "historical_frames",
                                   "returned_chunks": []}, all_frames) is None
    assert _build_recalled_frames(None, all_frames) is None

    # Without all_frame_paths → header still emitted, but no frame_paths
    rf_no_paths = _build_recalled_frames(rr, None)
    assert rf_no_paths is not None
    assert "frame_paths" not in rf_no_paths


def test_render_sample_threads_frame_paths():
    """render_sample / render_trajectory / render_video_samples must accept
    and forward all_frame_paths so the pipeline can plumb it from
    extract_frames output."""
    import inspect
    from scripts.agent_data_v5 import render_samples
    for fn_name in ("render_sample", "render_trajectory", "render_video_samples"):
        sig = inspect.signature(getattr(render_samples, fn_name))
        assert "all_frame_paths" in sig.parameters, \
            f"{fn_name} must accept all_frame_paths"


def test_pipeline_passes_video_frames_to_render():
    """pipeline.py's RENDER step must pass the per-video extracted frame
    list to render_video_samples (otherwise recalled_frames.frame_paths
    stays empty and SFT trains on text-only recall)."""
    src = (ROOT / "scripts" / "agent_data_v5" / "pipeline.py").read_text()
    block = src[src.find("RENDER: Building SFT-ready samples"):
                src.find("PASS 4: Verify + Filter")]
    assert "all_frame_paths=" in block, \
        "render call must thread all_frame_paths"
    assert "video_frames.get(vid" in block, \
        "must source from per-video extracted frame list"


def test_parse_time_range_handles_all_formats():
    """parse_time_range accepts 'a-b', [a,b], (a,b), and rejects junk."""
    h = _load_helpers_from_source(
        str(ROOT / "thinkstream" / "model" / "agent_loop.py"),
        ["parse_time_range"],
    )
    parse_time_range = h["parse_time_range"]
    # Stub Optional (referenced in signature)
    assert parse_time_range("10-30") == (10.0, 30.0)
    assert parse_time_range("10.5-30.5") == (10.5, 30.5)
    assert parse_time_range([5, 15]) == (5.0, 15.0)
    assert parse_time_range((5, 15)) == (5.0, 15.0)
    # Falsy / malformed → None (caller falls back to full archive)
    assert parse_time_range(None) is None
    assert parse_time_range("") is None
    assert parse_time_range("abc") is None
    assert parse_time_range("10") is None       # no dash
    assert parse_time_range([1, 2, 3]) is None  # wrong arity


def test_filter_archive_by_time_range_keeps_overlapping_chunks():
    """An item is kept iff its chunk window [c*2, c*2+2] overlaps
    [t_start, t_end]."""
    h = _load_helpers_from_source(
        str(ROOT / "thinkstream" / "model" / "agent_loop.py"),
        ["parse_time_range", "filter_archive_by_time_range"],
        extra_globals={"AGENT_CHUNK_SEC": 2.0},
    )
    filter_archive_by_time_range = h["filter_archive_by_time_range"]
    archive = [
        {"chunk": 0, "text": "a"},   # window 0-2
        {"chunk": 1, "text": "b"},   # window 2-4
        {"chunk": 2, "text": "c"},   # window 4-6
        {"chunk": 3, "text": "d"},   # window 6-8
        {"chunk": 5, "text": "e"},   # window 10-12
    ]
    out = filter_archive_by_time_range(archive, "3-7", chunk_sec=2.0)
    assert [a["chunk"] for a in out] == [1, 2, 3]
    assert filter_archive_by_time_range(archive, None, chunk_sec=2.0) == archive
    out2 = filter_archive_by_time_range(archive, "7-3", chunk_sec=2.0)
    assert [a["chunk"] for a in out2] == [1, 2, 3]


def test_bm25_retrieve_calls_time_range_filter():
    """bm25_retrieve must invoke filter_archive_by_time_range before
    scoring (source-level check — env can't import the module)."""
    src = (ROOT / "thinkstream" / "model" / "agent_loop.py").read_text()
    fn = src[src.find("def bm25_retrieve("):
             src.find("# Backward compat alias")]
    assert "filter_archive_by_time_range(archive, query.get(\"time_range\"))" in fn, \
        "bm25_retrieve must filter archive by query.time_range"
    # And HybridRetriever must do the same
    hsrc = (ROOT / "thinkstream" / "model" / "retrieval.py").read_text()
    h_block = hsrc[hsrc.find("class HybridRetriever"):
                   hsrc.find("@staticmethod")]
    assert "filter_archive_by_time_range" in h_block, \
        "HybridRetriever must filter archive by time_range too"


def test_recall_query_two_schemas_present():
    """pass3c_samples must define BOTH RECALL_QUERY_PROMPT_WITH_RANGE and
    RECALL_QUERY_PROMPT_KEYWORD_ONLY, and the keyword-only schema must
    NOT include time_range in its JSON output template."""
    src = (ROOT / "scripts" / "agent_data_v5" / "pass3c_samples.py").read_text()
    assert "RECALL_QUERY_PROMPT_WITH_RANGE" in src
    assert "RECALL_QUERY_PROMPT_KEYWORD_ONLY" in src
    # Keyword-only block must not template a time_range field
    ko = src[src.find("RECALL_QUERY_PROMPT_KEYWORD_ONLY"):
             src.find("RECALL_TIME_RANGE_FRACTION")]
    assert '"time_range"' not in ko, \
        "keyword_only schema must NOT contain time_range field"
    # Mix ratio is exposed
    assert "RECALL_TIME_RANGE_FRACTION" in src


def test_recall_time_range_uses_support_chunks():
    """_compute_recall_time_range must derive the window from card's
    support_chunks (with slack), not the legacy 0-max-history."""
    h = _load_helpers_from_source(
        str(ROOT / "scripts" / "agent_data_v5" / "pass3c_samples.py"),
        ["_compute_recall_time_range"],
        extra_globals={"AGENT_CHUNK_SEC": 2.0},
    )
    fn = h["_compute_recall_time_range"]
    # support_chunks = [5, 6, 7] → t in [10, 16] → with 4s slack → "6-20"
    card = {"support_chunks": [5, 6, 7]}
    snapshot = {"compressed_segments": [], "recent_thinks": []}
    assert fn(card, snapshot, slack_sec=4) == "6-20"
    # No support → fallback to visible-history bound
    snap2 = {"compressed_segments": [], "recent_thinks": [{"time": "0-2", "text": "x"}]}
    assert fn({}, snap2) == "0-2"


@pytest.mark.skip(reason="v11 helper removed in v12.5 cleanup")
def test_recall_hit_rate_helper_correctness():
    """_compute_recall_hit_rate returns |union ∩ gold|/|gold|, None when
    either input is empty/missing."""
    h = _load_helpers_from_source(
        str(ROOT / "thinkstream" / "trainer" / "grpo.py"),
        ["_compute_recall_hit_rate"],
    )
    fn = h["_compute_recall_hit_rate"]
    # Perfect hit
    assert fn([[5, 6, 7]], [5, 6, 7]) == 1.0
    # Partial hit (2/3)
    assert abs(fn([[5, 6]], [5, 6, 7]) - (2 / 3)) < 1e-6
    # Multiple recall events: union counts
    assert fn([[5], [6, 7]], [5, 6, 7]) == 1.0
    # Miss
    assert fn([[10, 11]], [5, 6, 7]) == 0.0
    # No support → None (no signal)
    assert fn([[5]], []) is None
    # Never recalled → None (legacy: query_quality alone)
    assert fn([[], [], []], [5, 6]) is None


def test_rollout_records_recall_returned_chunks():
    """Rollout must persist `recall_returned_chunks` per chunk per gen so
    the reward function can compute hit-rate."""
    src = (ROOT / "thinkstream" / "trainer" / "grpo.py").read_text()
    # Per-gen capture
    assert '"recall_returned_chunks": list(' in src
    # Per-chunk merged capture
    merge_block = src[src.find("max_chunks_seen = max(len(g)"):
                      src.find("merged_chunk_results.append(merged)") + 50]
    assert '"recall_returned_chunks": []' in merge_block
    assert 'merged["recall_returned_chunks"]' in merge_block


def test_eval_default_retriever_is_hybrid():
    """eval scripts default to hybrid (BM25 + visual embedding) — pure
    BM25 is kept as a baseline via --retriever bm25."""
    for f in ("scripts/eval/ovo/eval_full.py",
              "scripts/eval/test_set_agent.py"):
        src = (ROOT / f).read_text()
        m = re.search(
            r'add_argument\(\s*"--retriever",\s*default="(\w+)"',
            src,
        )
        assert m, f"{f}: --retriever flag with default not found"
        assert m.group(1) == "hybrid", \
            f"{f}: default retriever should be 'hybrid', got {m.group(1)!r}"


def test_reward_keys_split_recall_signal():
    """REWARD_DICT_KEYS must list recall_quality, recall_hit_rate,
    range_tightness as three separate columns. Weights must sum to ~1.0
    and put correctness as the largest single weight."""
    src = (ROOT / "thinkstream" / "trainer" / "gdpo_advantage.py").read_text()
    keys_block = src[src.find("REWARD_DICT_KEYS"):
                     src.find("# Per-reward weights")]
    for k in ("recall_quality", "recall_hit_rate", "range_tightness"):
        assert f'"{k}"' in keys_block, f"REWARD_DICT_KEYS missing {k}"
    # Weights map exists for every key
    weights_block = src[src.find("DEFAULT_REWARD_WEIGHTS:"):
                        src.find("def per_reward_group_norm(")]
    for k in ("correctness", "recall_quality", "recall_hit_rate",
              "range_tightness", "format", "timing", "silent_quality",
              "overflow_pen"):
        assert f'"{k}":' in weights_block, f"weights missing {k}"


@pytest.mark.skip(reason="v11 helper removed in v12.5 cleanup")
def test_recall_quality_reward_no_longer_mixes_hit_rate():
    """After v11.3 split, _compute_recall_quality_reward must NOT mix
    hit_rate (that lives in its own column now)."""
    src = (ROOT / "thinkstream" / "trainer" / "grpo.py").read_text()
    block = src[src.find("def _compute_recall_quality_reward("):
                src.find("def _parse_query_time_range(")]
    assert "0.5 * query_quality + 0.5 * hit_rate" not in block, \
        "recall_quality must not mix hit_rate after the v11.3 column split"
    # And the helper for range_tightness must exist
    assert "def _compute_range_tightness_reward(" in src
    assert "def _parse_query_time_range(" in src


@pytest.mark.skip(reason="v11 helper removed in v12.5 cleanup")
def test_range_tightness_reward_correctness():
    """_compute_range_tightness_reward = (1 - range_width/duration) × coverage,
    None when no query has time_range / no support / no coverage."""
    h = _load_helpers_from_source(
        str(ROOT / "thinkstream" / "trainer" / "grpo.py"),
        ["_parse_query_time_range", "_compute_range_tightness_reward"],
        extra_globals={"json": __import__("json")},
    )
    fn = h["_compute_range_tightness_reward"]
    parse_q = h["_parse_query_time_range"]
    # Tight range covering all support → high score
    chunk_texts = ['<query>{"query": "x", "time_range": "10-20"}</query>']
    # support [5,6,7] in seconds → [10-12, 12-14, 14-16] all in [10,20]
    score = fn(chunk_texts, [5, 6, 7], video_duration=100.0, chunk_sec=2.0)
    assert score is not None
    # range_width=10, duration=100 → tightness=0.9, coverage=1.0 → 0.9
    assert abs(score - 0.9) < 1e-6
    # No support → None
    assert fn(chunk_texts, None, 100.0) is None
    # No time_range in query → None
    chunk_texts_no_tr = ['<query>{"query": "x"}</query>']
    assert fn(chunk_texts_no_tr, [5, 6], 100.0) is None
    # Range that misses support entirely → None (don't reward width-only)
    chunk_texts_miss = ['<query>{"query": "x", "time_range": "50-60"}</query>']
    assert fn(chunk_texts_miss, [5, 6, 7], 100.0) is None


def test_streaming_eval_records_query_schema_and_compress_quality():
    """walk_and_score must record query schema (with_time_range vs
    keyword_only), partial-compress count, and chunk-revisit count so
    the SFT/RL eval reports can show schema-split hit rates."""
    src = (ROOT / "scripts" / "eval" / "test_set_agent.py").read_text()
    # Per-recall schema
    assert '"schema": schema' in src
    assert '"hit_fraction": hit_frac' in src
    # Partial compress detection
    assert "n_partial_compress" in src
    assert "COMPRESS_RANGE_MIN" in src
    # Chunk revisit
    assert "n_chunks_revisited" in src
    assert "compress_chunk_count" in src
    # Block 3 in the report
    assert "RECALL SCHEMA / COMPRESS QUALITY" in src


def test_render_metadata_includes_gold_compress_chunks():
    """Compress samples must carry metadata.gold_compress_chunks (the
    teacher's chosen chunks for that compression event) so RL/eval can
    score the model's summary time_range vs teacher's choice."""
    src = (ROOT / "scripts" / "agent_data_v5" / "render_samples.py").read_text()
    # The metadata dict must list gold_compress_chunks
    meta_block = src[src.find("metadata = {"):src.find("# Build complete SFT sample")]
    assert '"gold_compress_chunks":' in meta_block, \
        "render_sample metadata must include gold_compress_chunks"
    # And it must be populated from the compression_events lookup
    assert "gold_compress_chunks: List[int]" in src or \
           "gold_compress_chunks = sorted" in src, \
        "gold_compress_chunks must be derived from compression_events"


def test_make_retriever_accepts_agent_model():
    """make_retriever's signature exposes agent_model + agent_processor
    so the hybrid retriever can reuse Qwen3-VL's vision tower instead of
    loading SigLIP."""
    src = (ROOT / "thinkstream" / "model" / "retrieval.py").read_text()
    sig_block = src[src.find("def make_retriever("):
                    src.find("    if kind == \"bm25\"")]
    assert "agent_model" in sig_block
    assert "agent_processor" in sig_block
    # And the helper that uses them
    assert "def _make_qwen3vl_encoders(" in src
    # Eval scripts expose --use_agent_vision
    for f in ("scripts/eval/ovo/eval_full.py",
              "scripts/eval/test_set_agent.py"):
        s = (ROOT / f).read_text()
        assert '"--use_agent_vision"' in s, f"{f}: --use_agent_vision flag missing"
