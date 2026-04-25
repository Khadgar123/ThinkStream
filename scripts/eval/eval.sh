# ThinkStream evaluation (streaming model)
bash ./scripts/eval/ovo/eval.sh \
--ckpt /your/ckpt/dir \
--model_type qwen2.5vl \
--ngpu 8

bash ./scripts/eval/rtvu/eval.sh \
--ckpt /your/ckpt/dir \
--model_type qwen2.5vl \
--ngpu 8

# Baseline evaluation (vanilla VLM, no streaming)
# bash ./scripts/eval/baseline/eval_ovo.sh \
# --ckpt Qwen/Qwen3-VL-8B \
# --model_type qwen3vl \
# --ngpu 8
