#!/bin/bash
# =============================================================================
# 首批 50 视频 Agent 数据构造 — 端到端运行脚本
#
# 在 H20 8 卡节点上执行。Stage 2 通过 vLLM API 调用 AMD 节点上的 397B。
#
# 前提:
#   1. AMD 节点已启动 vLLM:
#      vllm serve Qwen/Qwen3.5-397B-A22B-FP8 --tensor-parallel-size 8 --port 8000
#   2. 数据集和视频文件已就位
#   3. pip install scenedetect whisper sentence-transformers faiss-cpu jieba
#
# Usage:
#   bash scripts/run_first_batch.sh <AMD_IP> <AMD_PORT>
#
# Example:
#   bash scripts/run_first_batch.sh 10.0.0.100 8000
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
AMD_IP="${1:?Usage: $0 <AMD_IP> <AMD_PORT>}"
AMD_PORT="${2:-8000}"
API_BASE="http://${AMD_IP}:${AMD_PORT}/v1"
TEACHER_MODEL="Qwen/Qwen3.5-397B-A22B-FP8"
CAPTION_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
VL_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# 数据路径 (根据实际环境修改)
STREAMO_ANNOTATION="${STREAMO_ANNOTATION:-/home/tione/notebook/gaozhenkun/hzh/data/Streamo/raw_data.json}"
THINKSTREAM_ANNOTATION="${THINKSTREAM_ANNOTATION:-/home/tione/notebook/gaozhenkun/hzh/data/ThinkStream/streaming_cot_cold_processed_5_20.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-/home/tione/notebook/gaozhenkun/hzh/data}"

# 输出目录
VIDEO_LIST="data/agent/video_list.json"
OUTPUT_VERSION="v0.1"

echo "============================================================"
echo "Agent Data Construction Pipeline — First Batch (50 videos)"
echo "============================================================"
echo "AMD API: ${API_BASE}"
echo "Teacher: ${TEACHER_MODEL}"
echo "Caption: ${CAPTION_MODEL}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 0: 选取首批视频
# ---------------------------------------------------------------------------
echo ""
echo "[Step 0] Selecting first batch of videos..."
python -m scripts.agent_data_pipeline.select_first_batch \
    --streamo_annotation "${STREAMO_ANNOTATION}" \
    --thinkstream_annotation "${THINKSTREAM_ANNOTATION}" \
    --video_root "${VIDEO_ROOT}" \
    --output "${VIDEO_LIST}" \
    --num_streamo 30 \
    --num_thinkstream 20 \
    --seed 42

# ---------------------------------------------------------------------------
# Step 1: Stage 0-1 预处理 (H20, ~28min)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 1] Stage 0-1: Video preprocessing + Event timeline..."
python -m scripts.agent_data_pipeline.run_pipeline \
    --video_list "${VIDEO_LIST}" \
    --stages 0,1 \
    --caption_model "${CAPTION_MODEL}"

echo ""
echo "[QC] Stage 0 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 0
echo ""
echo "[QC] Stage 1 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 1

# ---------------------------------------------------------------------------
# Step 2: Stage 2 Teacher 任务包 (AMD via API, ~67min)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2] Stage 2: Teacher task pack generation (via ${API_BASE})..."
python -m scripts.agent_data_pipeline.run_pipeline \
    --video_list "${VIDEO_LIST}" \
    --stages 2 \
    --api_base "${API_BASE}" \
    --teacher_model "${TEACHER_MODEL}" \
    --target_tasks 12

echo ""
echo "[QC] Stage 2 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 2 -v

# ---------------------------------------------------------------------------
# Step 3: Stage 3-5 展开 + 验证 (H20, ~24min)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3] Stage 3-5: Expansion + Query verification + 6-gate..."
python -m scripts.agent_data_pipeline.run_pipeline \
    --video_list "${VIDEO_LIST}" \
    --stages 3,4,5 \
    --vl_model "${VL_MODEL}"

echo ""
echo "[QC] Stage 3 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 3
echo ""
echo "[QC] Stage 4 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 4
echo ""
echo "[QC] Stage 5 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 5

# ---------------------------------------------------------------------------
# Step 4: Stage 6 样本组装 (CPU, ~5min)
# ---------------------------------------------------------------------------
echo ""
echo "[Step 4] Stage 6: Final sample assembly..."
python -m scripts.agent_data_pipeline.run_pipeline \
    --video_list "${VIDEO_LIST}" \
    --stages 6 \
    --version "${OUTPUT_VERSION}"

echo ""
echo "[QC] Stage 6 quality check..."
python -m scripts.agent_data_pipeline.quality_check --stage 6 -v

# ---------------------------------------------------------------------------
# Step 5: 全量质量报告
# ---------------------------------------------------------------------------
echo ""
echo "[Step 5] Full quality report..."
python -m scripts.agent_data_pipeline.quality_check --all

# ---------------------------------------------------------------------------
# 输出汇总
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "DATA CONSTRUCTION COMPLETE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  SFT data:       data/agent/sft_final/sft_${OUTPUT_VERSION}.jsonl"
echo "  Retriever data:  data/agent/retriever_train.jsonl"
echo "  RL data:         data/agent/rl_pool.jsonl"
echo ""
echo "Next steps:"
echo "  1. Review quality check output above"
echo "  2. Manually inspect 20 SFT samples for correctness"
echo "  3. If quality OK, proceed to training:"
echo "     bash scripts/sft_agent.sh"
echo "============================================================"
