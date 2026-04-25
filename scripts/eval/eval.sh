# === ThinkStream evaluation (streaming model) ===
bash ./scripts/eval/ovo/eval.sh \
--ckpt /your/ckpt/dir \
--model_type qwen2.5vl \
--ngpu 8

bash ./scripts/eval/rtvu/eval.sh \
--ckpt /your/ckpt/dir \
--model_type qwen2.5vl \
--ngpu 8

# === Baseline: offline (64 frames, full video, no streaming) ===
# Matches "Open-source Offline Models" in paper Table 2
# bash ./scripts/eval/baseline/eval_ovo_offline.sh \
# --ckpt Qwen/Qwen3-VL-8B \
# --model_type qwen3vl \
# --ngpu 8

# === Baseline: streaming (same engine, no think/action protocol) ===
# Matches "Open-source Online Models" in paper Table 2
# bash ./scripts/eval/baseline/eval_ovo.sh \
# --ckpt Qwen/Qwen3-VL-8B \
# --model_type qwen3vl \
# --ngpu 8
