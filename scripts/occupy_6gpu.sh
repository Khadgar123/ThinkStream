#!/bin/bash
# 6-GPU memory occupation wrapper — activates thinkstream conda env.
set -euo pipefail
cd /home/tione/notebook/gaozhenkun/hzh/ThinkStream
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
set +u
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate /home/tione/notebook/gaozhenkun/hzh/envs/thinkstream
set -u
python scripts/occupy_6gpu.py
