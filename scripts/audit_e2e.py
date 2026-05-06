"""End-to-end consistency audit — Data → SFT → RL → Eval.

Programmatically verifies that every stage uses the canonical v12.5
constants (1s/chunk, 16-chunk window, 4000-token memory, etc.) and the
v12 protocol (Qwen3-VL official tool calls). No GPU required.

Stages checked:
  [1] Canonical config (scripts/agent_data_v5/config.py)
  [2] Agent protocol (thinkstream/data/agent_protocol.py)
  [3] Data construction (pass1a → pass5)
  [4] SFT (thinkstream/sft/)
  [5] RL reward helpers (thinkstream/trainer/)
  [6] RL verl recipe (verl/recipe_thinkstream/)
  [7] Eval (thinkstream/eval/)
  [8] Cross-stage prompt + system message identity

Each check is self-contained and prints PASS/FAIL with a one-line reason.
Exits non-zero if any check fails.
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Audit:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.failures = []

    def check(self, label: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✓ {label}" + (f" — {detail}" if detail else ""))
        else:
            self.failed += 1
            self.failures.append((label, detail))
            print(f"  ✗ {label}" + (f" — {detail}" if detail else ""))

    def warn(self, label: str, detail: str = ""):
        self.warnings += 1
        print(f"  ⚠ {label}" + (f" — {detail}" if detail else ""))

    def section(self, name: str):
        print(f"\n[{name}]")


# ===========================================================================
# Stage 1: canonical config
# ===========================================================================

def check_config(a: Audit):
    a.section("1. Canonical config (scripts/agent_data_v5/config.py)")
    from scripts.agent_data_v5 import config as c

    a.check("AGENT_CHUNK_SEC == 1",          c.AGENT_CHUNK_SEC == 1, str(c.AGENT_CHUNK_SEC))
    a.check("FRAMES_PER_CHUNK == 2",          c.FRAMES_PER_CHUNK == 2, str(c.FRAMES_PER_CHUNK))
    a.check("FPS == 2",                        c.FPS == 2, str(c.FPS))
    a.check("VISUAL_WINDOW_CHUNKS == 16",     c.VISUAL_WINDOW_CHUNKS == 16, str(c.VISUAL_WINDOW_CHUNKS))
    a.check("VISUAL_WINDOW_FRAMES == 32",     c.VISUAL_WINDOW_FRAMES == 32)
    a.check("RECENT_THINKS_TOKEN_BUDGET == 4000",
                                              c.RECENT_THINKS_TOKEN_BUDGET == 4000, str(c.RECENT_THINKS_TOKEN_BUDGET))
    a.check("COMPRESS_TOKEN_THRESHOLD == 3200",
                                              c.COMPRESS_TOKEN_THRESHOLD == 3200, str(c.COMPRESS_TOKEN_THRESHOLD))
    a.check("COMPRESS_HYSTERESIS_THRESHOLD == 2200",
                                              c.COMPRESS_HYSTERESIS_THRESHOLD == 2200)
    a.check("COMPRESS_RANGE_MIN/MAX == 8/24", c.COMPRESS_RANGE_MIN == 8 and c.COMPRESS_RANGE_MAX == 24)
    a.check("MAX_COMPRESSED_SEGMENTS == 5",   c.MAX_COMPRESSED_SEGMENTS == 5)
    a.check("SUMMARY_TOKENS_MAX == 280",      c.SUMMARY_TOKENS_MAX == 280)
    a.check("THINK_TOKENS == (40, 80)",       c.THINK_TOKENS == (40, 80))
    a.check("MAX_SAMPLE_TOKENS == 16384",     c.MAX_SAMPLE_TOKENS == 16384)
    a.check("VISUAL_TOKENS_PER_CHUNK == 470", c.VISUAL_TOKENS_PER_CHUNK == 470)

    # Prompts
    a.check("OBSERVATION_PROMPT mentions '1-second'",
            "1-second" in c.OBSERVATION_PROMPT, "")
    a.check("OBSERVATION_PROMPT does NOT mention '2 second'",
            "2 second" not in c.OBSERVATION_PROMPT)
    a.check("OBSERVATION_PROMPT think target 40-80",
            "40-80" in c.OBSERVATION_PROMPT)


# ===========================================================================
# Stage 2: agent protocol
# ===========================================================================

def check_protocol(a: Audit):
    a.section("2. Agent protocol (thinkstream/data/agent_protocol.py)")
    from thinkstream.data import agent_protocol as ap

    a.check("AGENT_CHUNK_SEC import resolves to 1", ap.AGENT_CHUNK_SEC == 1)
    a.check("VISUAL_WINDOW_CHUNKS == 16",            ap.VISUAL_WINDOW_CHUNKS == 16)
    a.check("FRAMES_PER_CHUNK == 2",                  ap.FRAMES_PER_CHUNK == 2)

    # SYSTEM_PROMPT_V12 is the ordinary streaming prompt; compression has a
    # separate system prompt selected via system_prompt_for_frame_protocol.
    sp = ap.SYSTEM_PROMPT_V12
    cp = ap.system_prompt_for_frame_protocol(inter_chunk=True)
    a.check("SYSTEM_PROMPT mentions '1-second video chunks'",
            "1-second video chunks" in sp)
    a.check("SYSTEM_PROMPT mentions '16s window'", "16s window" in sp)
    a.check("COMPRESS prompt is separated",
            "Directly compress memory now" in cp and "Do not call recall" in cp)
    a.check("SYSTEM_PROMPT does NOT mention '2-second'", "2-second" not in sp)
    a.check("SYSTEM_PROMPT does NOT mention '24s'", "24s window" not in sp)
    mem_text = ap.format_memory_block({
        "compressed": [{"time_range": [0, 8], "text": "setup"}],
        "recent_thinks": [{"time": "8-9", "text": "new object appears"}],
    })
    a.check("memory renders tagged records",
            "<compressed>{" in mem_text and "<memory_think>{" in mem_text)

    # Tool schema
    tools = ap.TOOLS_SCHEMA
    a.check("TOOLS_SCHEMA has 2 tools",                len(tools) == 2)
    tool_names = sorted(t["function"]["name"] for t in tools)
    a.check("TOOLS_SCHEMA has [compress, recall]",
            tool_names == ["compress", "recall"])

    # Output construction
    msg = ap.build_assistant_content_v12(
        think="x", kind="answer", answer_text="yes",
    )
    a.check("answer assistant content well-formed",
            "<think>x</think>" in msg and "<answer>yes</answer>" in msg)

    msg = ap.build_assistant_content_v12(
        think="x", kind="recall",
        recall_query={"query": "kw", "time_range": "10-20"},
    )
    a.check("recall assistant content has tool_call",
            '<tool_call>' in msg and '"name": "recall"' in msg)

    msg = ap.build_assistant_content_v12(
        think="x", kind="compress",
        compress_summary={"time_range": [0, 10], "text": "sum"},
    )
    a.check("compress assistant content has tool_call",
            '<tool_call>' in msg and '"name": "compress"' in msg)

    # Parser round-trips
    parsed = ap.parse_agent_output_v12(msg)
    a.check("compress output parses to kind='compress'",
            parsed.get("kind") == "compress")


# ===========================================================================
# Stage 3: data construction
# ===========================================================================

def check_data_pipeline(a: Audit):
    a.section("3. Data construction (pass1a → pass5)")

    # pass2 MemoryState
    from scripts.agent_data_v5.pass2_rollout import MemoryState
    ms = MemoryState()
    ms.add_think(0, "A red bowl is on the table.")
    a.check("MemoryState has timeline + retrieval_archive",
            hasattr(ms, "timeline") and hasattr(ms, "_retrieval_archive"))
    a.check("pass2 teacher memory uses student memory tags",
            "<memory_think>{" in ms.format_for_observation_prompt())

    # add_think uses AGENT_CHUNK_SEC
    src = inspect.getsource(MemoryState.add_think)
    a.check("MemoryState.add_think uses AGENT_CHUNK_SEC",
            "AGENT_CHUNK_SEC" in src)

    # should_compress uses COMPRESS_TOKEN_THRESHOLD
    src = inspect.getsource(MemoryState.should_compress)
    a.check("MemoryState.should_compress uses COMPRESS_TOKEN_THRESHOLD",
            "COMPRESS_TOKEN_THRESHOLD" in src)

    # pass3c renders with v12
    from scripts.agent_data_v5 import pass3c_samples
    src = inspect.getsource(pass3c_samples)
    a.check("pass3c builds compress as inter_chunk",
            "v12_inter_chunk" in src)
    a.check("pass3c uses build_assistant_content_v12",
            "build_assistant_content_v12" in src)
    # v12.12 (2026-05-02): trigger no longer carries range; model must
    # derive it from <memory>. Range stays in the gold tool_call output.
    a.check("pass3c injects bare <compress_trigger/> (no range)",
            "<compress_trigger/>" in src and
            "<compress_trigger range='" not in src)

    # pass5 messages converter
    from scripts.agent_data_v5 import pass5_messages
    src = inspect.getsource(pass5_messages)
    a.check("pass5_messages uses protocol system prompt selector",
            "from thinkstream.data.agent_protocol import" in src and
            "system_prompt_for_frame_protocol" in src)
    a.check("pass5_messages reads chunk_sec from config",
            "AGENT_CHUNK_SEC" in src)

    # Output messages format compliance (LLaMA-Factory ShareGPT)
    final = ROOT / "data/agent_v5/final"
    if (final / "train_sft_messages.jsonl").exists():
        import json
        with open(final / "train_sft_messages.jsonl") as f:
            d = json.loads(f.readline())
        a.check("train_sft_messages first record has 'messages' key",
                "messages" in d)
        a.check("first message role == 'system'",
                d["messages"][0]["role"] == "system")
    else:
        a.warn("train_sft_messages.jsonl not yet generated (run pass5_messages.py)")


# ===========================================================================
# Stage 4: SFT
# ===========================================================================

def check_sft(a: Audit):
    a.section("4. SFT (thinkstream/sft/)")

    arg_path = ROOT / "thinkstream/sft/argument.py"
    arg_src = arg_path.read_text() if arg_path.exists() else ""
    try:
        from thinkstream.sft.argument import ModelArguments, DataArguments, TrainingArguments
        ma, da = ModelArguments(), DataArguments()
        a.check("Default model = Qwen3-VL-8B-Instruct",
                "Qwen3-VL" in ma.model_name_or_path,
                ma.model_name_or_path)
        a.check("Default agent_chunk_sec == 1.0", da.agent_chunk_sec == 1.0)
        a.check("Default visual_window_chunks == 16", da.visual_window_chunks == 16)
        a.check("Default video_max_pixels == 220_000", da.video_max_pixels == 220_000)
        a.check("Default video_min_pixels == 130_000", da.video_min_pixels == 130_000)
        a.check("Default video_fps == 2.0", da.video_fps == 2.0)
        a.check("max_sample_tokens == 12000 (legacy filter)",
                da.max_sample_tokens == 12000)
    except ImportError as e:
        a.warn(f"SFT argument import skipped due local env: {e}")
        a.check("argument.py default model mentions Qwen3-VL",
                "Qwen3-VL-8B-Instruct" in arg_src)
        a.check("argument.py video_min_pixels default 130000",
                "video_min_pixels" in arg_src and "130_000" in arg_src)
        a.check("argument.py video_max_pixels default 220000",
                "video_max_pixels" in arg_src and "220_000" in arg_src)

    # data_processor strict messages format
    import importlib.util
    spec = importlib.util.find_spec("thinkstream.sft.data_processor")
    if spec and spec.origin:
        src = Path(spec.origin).read_text()
        a.check("preprocess_per_timestep requires messages key",
                "missing 'messages' key" in src)
        a.check("data_processor reads chunk_sec from config",
                "from scripts.agent_data_v5.config import AGENT_CHUNK_SEC" in src)
        a.check("data_processor passes tools=TOOLS_SCHEMA",
                "tools=TOOLS_SCHEMA" in src)
        a.check("v11 register_special_tokens removed",
                "register_special_tokens" not in src)
        a.check("v11 SPAN_WEIGHTS removed",
                "SPAN_WEIGHTS = " not in src)


# ===========================================================================
# Stage 5: RL reward helpers
# ===========================================================================

def check_rl_reward_helpers(a: Audit):
    a.section("5. RL reward helpers (thinkstream/trainer/)")

    # Reward keys (drop compress_quality / recall_quality)
    from thinkstream.trainer.gdpo_advantage import (
        V12_REWARD_DICT_KEYS, V12_DEFAULT_REWARD_WEIGHTS, V12_ADVANTAGE_MIX_ALPHA,
    )
    expected_keys = {"outcome", "timing", "format", "spam", "silent_quality"}
    actual_keys = set(V12_REWARD_DICT_KEYS)
    a.check("Reward keys = {outcome, timing, format, spam, silent_quality}",
            actual_keys == expected_keys, str(actual_keys))
    a.check("compress_quality NOT in reward keys",
            "compress_quality" not in actual_keys)
    a.check("recall_quality NOT in reward keys",
            "recall_quality" not in actual_keys)
    a.check("V12_ADVANTAGE_MIX_ALPHA == 0.7", V12_ADVANTAGE_MIX_ALPHA == 0.7)

    # Reward functions exist + removed ones gone
    from thinkstream.trainer import v12_rewards as vr
    has_outcome = hasattr(vr, "compute_outcome_reward_v12")
    has_compress_q = hasattr(vr, "compute_compress_quality_v12")
    has_recall_q = hasattr(vr, "compute_recall_quality_v12")
    a.check("compute_outcome_reward_v12 present", has_outcome)
    a.check("compute_compress_quality_v12 REMOVED", not has_compress_q)
    a.check("compute_recall_quality_v12 REMOVED", not has_recall_q)

    # ChunkLevelRolloutConfig defaults
    from thinkstream.trainer.v12_rollout import ChunkLevelRolloutConfig
    cfg = ChunkLevelRolloutConfig()
    a.check("ChunkLevelRolloutConfig.max_chunks_per_video >= 300",
            cfg.max_chunks_per_video >= 300, str(cfg.max_chunks_per_video))
    a.check("max_prompt_length == 16384",        cfg.max_prompt_length == 16384)
    a.check("chunk_visual_tokens == 4096",        cfg.chunk_visual_tokens == 4096)
    a.check("compress_trigger_every == 0 (token-budget)",
            cfg.compress_trigger_every == 0)


# ===========================================================================
# Stage 6: RL verl recipe
# ===========================================================================

def check_rl_verl(a: Audit):
    a.section("6. RL verl recipe (verl/recipe_thinkstream/)")

    recipe = ROOT / "verl/recipe_thinkstream"
    cfg = recipe / "configs/thinkstream_grpo.yaml"
    loop = recipe / "streaming_agent_loop.py"
    entry = recipe / "thinkstream.py"
    run = recipe / "run_thinkstream_grpo.sh"

    a.check("verl recipe config exists", cfg.exists())
    a.check("verl streaming agent loop exists", loop.exists())
    a.check("verl reward/dataset entry exists", entry.exists())
    if cfg.exists():
        txt = cfg.read_text()
        a.check("verl uses custom ThinkStream agent loop",
                "thinkstream_streaming_agent" in txt)
        a.check("verl sets vLLM image limit for timestamped frames",
                "limit_images: 64" in txt)
        a.check("verl reward path = recipe_thinkstream/thinkstream.py",
                'path: "recipe_thinkstream/thinkstream.py"' in txt)
    if loop.exists():
        txt = loop.read_text()
        a.check("verl rollout appends protocol visual frames",
                "append_visual_frames" in txt)
    if run.exists():
        txt = run.read_text()
        a.check("verl launcher resolves THINKSTREAM_DATA_ROOT",
                "THINKSTREAM_DATA_ROOT" in txt)


# ===========================================================================
# Stage 7: Eval
# ===========================================================================

def check_eval(a: Audit):
    a.section("7. Eval (thinkstream/eval/) — file-level scan (avoids flash_attn dep)")

    # File-level inspection: avoids importing thinkstream.model.inference
    # which has a hard flash_attn dep (CUDA-only). Audit verifies the
    # source text contains the right canonical references.
    eval_files = [
        ROOT / "thinkstream/eval/streaming_vllm.py",
        ROOT / "thinkstream/eval/eval_common.py",
        ROOT / "thinkstream/model/agent_loop.py",
    ]
    for p in eval_files:
        if not p.exists():
            a.check(f"{p.relative_to(ROOT)} exists", False, "missing file")
            continue
        src = p.read_text()
        rel = str(p.relative_to(ROOT))
        # AGENT_CHUNK_SEC must be sourced from canonical: directly from
        # agent_protocol, OR transitively from agent_loop (which itself
        # imports from agent_protocol). The audit accepts either route as
        # long as no hardcoded `AGENT_CHUNK_SEC = 2` (or similar) appears.
        canonical_source = (
            ("AGENT_CHUNK_SEC" in src and "agent_protocol" in src)
            or ("AGENT_CHUNK_SEC" in src and "agent_loop" in src)
        )
        a.check(f"{rel}: uses canonical AGENT_CHUNK_SEC",
                canonical_source,
                "must import from agent_protocol or agent_loop (no hardcoded copy)")
        a.check(f"{rel}: imports FRAMES_PER_CHUNK",
                "FRAMES_PER_CHUNK" in src)
        if "streaming_vllm" in rel or "agent_loop" in rel:
            a.check(f"{rel}: uses VISUAL_WINDOW_CHUNKS",
                    "VISUAL_WINDOW_CHUNKS" in src)
            a.check(f"{rel}: handles recalled_frames / returned_chunks",
                    "recalled_frames" in src or "returned_chunks" in src)
            a.check(f"{rel}: no hardcoded chunk_sec=2",
                    not re.search(r"AGENT_CHUNK_SEC\s*=\s*2(\.\d)?", src))

    # eval profiles (16k vs 32k)
    profile_path = ROOT / "scripts/eval/eval_profiles.py"
    if profile_path.exists():
        pt = profile_path.read_text()
        a.check("eval_profiles defines 16k profile", "16k" in pt or "16384" in pt)
        a.check("eval_profiles defines 32k profile", "32k" in pt or "32768" in pt)


# ===========================================================================
# Stage 8: cross-stage prompt + system identity
# ===========================================================================

def check_prompt_identity(a: Audit):
    a.section("8. Cross-stage prompt + system identity")

    from thinkstream.data.agent_protocol import (
        SYSTEM_PROMPT_V12, TOOLS_SCHEMA, build_assistant_content_v12,
    )

    # Same SYSTEM_PROMPT_V12 used by ALL stages — either directly imported,
    # or transitively via build_single_step_messages / build_per_timestep_messages_v12
    # (both use SYSTEM_PROMPT_V12 internally).
    paths_using_sysprompt = [
        # (path, accept_indirect)
        (ROOT / "thinkstream/sft/data_processor.py", False),
        (ROOT / "thinkstream/eval/streaming_vllm.py", True),   # via build_single_step_messages
        (ROOT / "thinkstream/model/agent_loop.py", False),
        (ROOT / "verl/recipe_thinkstream/streaming_agent_loop.py", False),
        (ROOT / "scripts/agent_data_v5/pass3c_samples.py", True),  # via build_assistant_content_v12
        (ROOT / "scripts/agent_data_v5/pass5_messages.py", False),
    ]
    indirect_markers = [
        "build_single_step_messages",
        "build_per_timestep_messages_v12",
        "build_assistant_content_v12",
    ]
    for p, accept_indirect in paths_using_sysprompt:
        if p.exists():
            txt = p.read_text()
            direct = (
                "SYSTEM_PROMPT_V12" in txt
                or "system_prompt_for_frame_protocol" in txt
            )
            indirect = accept_indirect and any(m in txt for m in indirect_markers)
            ok = direct or indirect
            a.check(
                f"{p.relative_to(ROOT)} uses protocol system prompt "
                f"({'direct' if direct else 'indirect'})",
                ok,
            )

    # Same TOOLS_SCHEMA
    paths_using_tools = [
        ROOT / "thinkstream/sft/data_processor.py",
        ROOT / "thinkstream/eval/streaming_vllm.py",
        ROOT / "thinkstream/eval/eval_baseline_vllm.py",
        ROOT / "verl/recipe_thinkstream/streaming_agent_loop.py",
    ]
    for p in paths_using_tools:
        if p.exists():
            txt = p.read_text()
            ok = "TOOLS_SCHEMA" in txt
            a.check(f"{p.relative_to(ROOT)} uses TOOLS_SCHEMA", ok)


# ===========================================================================
# Driver
# ===========================================================================

def main():
    print("=" * 78)
    print("ThinkStream end-to-end audit (Data → SFT → verl RL → Eval)")
    print("=" * 78)

    a = Audit()
    check_config(a)
    check_protocol(a)
    check_data_pipeline(a)
    check_sft(a)
    check_rl_reward_helpers(a)
    check_rl_verl(a)
    check_eval(a)
    check_prompt_identity(a)

    print("\n" + "=" * 78)
    print(f"PASSED: {a.passed}   FAILED: {a.failed}   WARNINGS: {a.warnings}")
    print("=" * 78)
    if a.failed:
        print("\nFailures:")
        for label, detail in a.failures:
            print(f"  ✗ {label} — {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
