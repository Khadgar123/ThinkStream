#!/bin/bash
# Unified eval matrix — base × agent × (test, OVO).
#
# Two evaluation regimes:
#
#   BASE (no streaming agent, no recall, no compress):
#     For each Qwen3-VL size (2B/4B/8B) on each frame budget {24, 64, 128,
#     256, 512, 1024}, run on test.jsonl AND ovo_bench. Streaming budget
#     (--mode streaming --max_frames 24) ≈ apples-to-apples vs the agent's
#     12-chunk visual window. Offline budgets answer "how much does more
#     visual context help a plain VLM?".
#
#   AGENT (full StreamingAgentLoop walk):
#     For each SFT/RL checkpoint, run on test.jsonl AND ovo_bench under
#     {bm25, hybrid} retriever × {system, self} compress_mode.
#       compress_mode=system  → trigger + fixed FIFO range, matches SFT data.
#       compress_mode=self    → no trigger, model decides; RL-only territory.
#     Recall backend toggle measures whether visual retrieval (siglip) helps
#     over pure lexical (bm25).
#
# Outputs land under each ckpt's eval/ folder, one JSON per config — easy
# to diff or aggregate later.
#
# Usage:
#   # Base only:
#   ./run_matrix.sh base
#   # Agent only (uses default SFT_CKPT and RL_CKPT below):
#   ./run_matrix.sh agent
#   # Both:
#   ./run_matrix.sh all
#
# Knobs are env-vars; see top of script.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Paths (override via env) ──────────────────────────────────────────────
TEST_JSONL="${TEST_JSONL:-data/agent_v5/final/test.jsonl}"
OVO_JSON="${OVO_JSON:-/home/tione/notebook/gaozhenkun/hzh/data/datasets/ovo_bench/ovo_bench_new.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data/datasets}"
FRAMES_ROOT="${FRAMES_ROOT:-data/agent_v5/frames}"

# ── Models (override via env) ─────────────────────────────────────────────
# For base eval — public Qwen3-VL checkpoints. Sized to study scaling.
QWEN3VL_2B="${QWEN3VL_2B:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN3VL_4B="${QWEN3VL_4B:-Qwen/Qwen3-VL-4B-Instruct}"
QWEN3VL_8B="${QWEN3VL_8B:-Qwen/Qwen3-VL-8B-Instruct}"

# For agent eval — your fine-tuned ckpts. Override with whatever you train.
SFT_CKPT="${SFT_CKPT:-output/agent-sft}"
RL_CKPT="${RL_CKPT:-output/agent-rl}"

# ── Eval scope (override via env) ─────────────────────────────────────────
# Smoke-test = small N for quick correctness check.
N_TEST="${N_TEST:-200}"               # samples per ckpt on test.jsonl
N_PER_OVO_TASK="${N_PER_OVO_TASK:-30}" # samples per OVO task per ckpt

# Frame budgets for base eval. Streaming uses fixed 24 (the agent's window);
# offline sweeps the global-context dimension.
BUDGETS_OFFLINE=(64 128 256 512 1024)

# Retriever × compress matrix for agent eval.
RETRIEVERS=(bm25 hybrid)
COMPRESS_MODES=(system self)

# Scoring matrix — both versions reported per ckpt:
#   strict  — response must come at the right time AND in exact format.
#             For agent eval: response within ask_chunk + 2.
#             For base eval:  response.strip() must equal gold token.
#   lenient — "if the model ever answered correctly, count it".
#             For agent eval: walk up to ask_chunk + 60.
#             For base eval:  first-matching-token wins (legacy behaviour).
SCORINGS=(strict lenient)

# Eval context profiles. 16k = SFT-aligned (safe default). 32k = extended,
# loosens queries cap / recall cap / max_new_tokens to use Qwen3-VL's
# native 32k context. See scripts/eval/eval_profiles.py for budget table.
PROFILES=(16k 32k)

# ── Helpers ───────────────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_base_one() {
    local ckpt="$1" mode="$2" budget="$3" bench="$4" scoring="$5"
    local label="$(basename "$ckpt")_${bench}_${mode}_${budget}_${scoring}"
    log "BASE ${label}"
    if [[ "$bench" == "test" ]]; then
        python scripts/eval/test_set_base.py \
            --ckpt "$ckpt" \
            --test_jsonl "$TEST_JSONL" \
            --video_root "$VIDEO_ROOT" \
            --mode "$mode" \
            --max_frames "$budget" \
            --scoring "$scoring" \
            --n "$N_TEST" \
            --out "${ckpt}/eval/test_base/${mode}_${budget}_${scoring}.json"
    else  # ovo
        python scripts/eval/ovo/base.py \
            --ckpt "$ckpt" \
            --benchmark_json "$OVO_JSON" \
            --video_root "$VIDEO_ROOT" \
            --frames_root "$FRAMES_ROOT" \
            --mode "$mode" \
            --max_frames "$budget" \
            --scoring "$scoring" \
            --n_per_task "$N_PER_OVO_TASK" \
            --out "${ckpt}/eval/ovo_base/${mode}_${budget}_${scoring}.json"
    fi
}

run_agent_one() {
    local ckpt="$1" retriever="$2" compress_mode="$3" bench="$4" scoring="$5" profile="$6"
    local label="$(basename "$ckpt")_${bench}_${retriever}_${compress_mode}_${scoring}_${profile}"
    log "AGENT ${label}"
    if [[ "$bench" == "test" ]]; then
        python scripts/eval/test_set_agent.py \
            --ckpt "$ckpt" \
            --test_jsonl "$TEST_JSONL" \
            --video_root "$VIDEO_ROOT" \
            --frames_root "$FRAMES_ROOT" \
            --retriever "$retriever" \
            --compress_mode "$compress_mode" \
            --scoring "$scoring" \
            --profile "$profile" \
            --n "$N_TEST" \
            --out "${ckpt}/eval/test_agent/${compress_mode}_${retriever}_${scoring}_${profile}.json"
    else
        python scripts/eval/ovo/eval_full.py \
            --ckpt "$ckpt" \
            --benchmark_json "$OVO_JSON" \
            --video_root "$VIDEO_ROOT" \
            --frames_root "$FRAMES_ROOT" \
            --retriever "$retriever" \
            --compress_mode "$compress_mode" \
            --scoring "$scoring" \
            --profile "$profile" \
            --n_per_task "$N_PER_OVO_TASK" \
            --out "${ckpt}/eval/ovo_agent/${compress_mode}_${retriever}_${scoring}_${profile}.json"
    fi
}

# ── Regimes ───────────────────────────────────────────────────────────────

run_base_matrix() {
    for ckpt in "$QWEN3VL_2B" "$QWEN3VL_4B" "$QWEN3VL_8B"; do
        for bench in test ovo; do
            for sc in "${SCORINGS[@]}"; do
                # Streaming = fixed 24-frame window (apples-to-apples vs agent).
                run_base_one "$ckpt" streaming 24 "$bench" "$sc"
                # Offline sweeps global context budget.
                for b in "${BUDGETS_OFFLINE[@]}"; do
                    run_base_one "$ckpt" offline "$b" "$bench" "$sc"
                done
            done
        done
    done
}

run_agent_matrix() {
    # Base / SFT / RL all run through the streaming-agent path. Base is
    # expected to score poorly (it doesn't know the agent protocol), but
    # the number is the apples-to-apples baseline against which fine-tuning
    # gains are measured. v9.4.2 fix (pixel budget + queries cap + recall
    # truncation) is what makes the streaming path NOT overflow on base —
    # before, base would crash at model_max_length on long videos.
    AGENT_CKPTS=()
    [[ -d "$SFT_CKPT" ]] && AGENT_CKPTS+=("$SFT_CKPT")
    [[ -d "$RL_CKPT"  ]] && AGENT_CKPTS+=("$RL_CKPT")
    # Public Qwen3-VL doesn't need a local dir; let HF resolve.
    for base in "$QWEN3VL_2B" "$QWEN3VL_4B" "$QWEN3VL_8B"; do
        AGENT_CKPTS+=("$base")
    done

    for ckpt in "${AGENT_CKPTS[@]}"; do
        for bench in test ovo; do
            for retr in "${RETRIEVERS[@]}"; do
                for cmp in "${COMPRESS_MODES[@]}"; do
                    for sc in "${SCORINGS[@]}"; do
                        for prof in "${PROFILES[@]}"; do
                            # Pure SFT × compress_mode=self is OOD (SFT data
                            # only has system-trigger samples). We still run
                            # it to MEASURE the OOD gap — that's a useful
                            # number — but expect lower acc. Same for lenient
                            # / 32k: each is an upper bound vs the SFT-aligned
                            # 16k baseline. Base × any agent config is also
                            # OOD by design.
                            run_agent_one "$ckpt" "$retr" "$cmp" "$bench" "$sc" "$prof"
                        done
                    done
                done
            done
        done
    done
}

# ── Main ──────────────────────────────────────────────────────────────────

case "${1:-all}" in
    base)  run_base_matrix ;;
    agent) run_agent_matrix ;;
    all)   run_base_matrix; run_agent_matrix ;;
    *) echo "Usage: $0 {base|agent|all}"; exit 1 ;;
esac

log "Done."
