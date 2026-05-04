# Thinking in Streaming Video

This is the official repository for the paper "Thinking in Streaming Video".

## 📰 News

- [2026/03/25] We have released our Code and [ThinkStream dataset](https://huggingface.co/datasets/CASIA-IVA-Lab/ThinkStream).
- [2026/03/16] We have released our paper on arXiv [Thinking in Streaming Video](https://arxiv.org/abs/2603.12938v1). We are working on refactoring the codebase and conducting the final check. Please stay tuned!

## 📝 TODO
- [x] Release Paper
- [x] Release Code
- [x] Release Dataset
- [ ] Release Model

## 💡 Introduction
Real-time understanding of continuous video streams is essential for interactive assistants and multimodal agents operating in dynamic environments. However, most existing video reasoning approaches follow a batch paradigm that defers reasoning until the full video context is observed, resulting in high latency and growing computational cost that are incompatible with streaming scenarios.

To address this, we introduce **ThinkStream**, a framework for streaming video reasoning based on a Watch-Think-Speak paradigm that enables models to incrementally update their understanding as new video observations arrive. 

## ✨ Highlights
- **Streaming Watch-Think-Speak Paradigm**: We formulate streaming video understanding as an incremental reasoning and interaction process, driven by a novel Streaming RLVR (Reinforcement Learning with Verifiable Rewards) scheme to optimize reasoning updates and response timing. To maintain efficiency, we introduce Reasoning-Compressed Streaming Memory (RCSM), which replaces outdated visual tokens with compact intermediate reasoning traces, preserving essential context while drastically reducing inference costs.
- **True Training-Inference Consistency**: We provide robust support for irregular attention masks to ensure strict alignment between training and inference. During the autoregressive training phase, we utilize FlexAttention to handle flexible attention masking. For model inference (which also serves as the RL rollout phase), we completely re-implemented a highly efficient inference engine that natively supports dynamic KV cache processing. The entire codebase is designed to be highly extensible, aiming to facilitate future research in this direction.
- **High-Efficiency Streaming Inference**: We engineered a high-performance streaming inference backend that independently leverages CUDA Graph recording and replay for both the decoding phase and KV cache eviction. By integrating FlashAttention for core computations and FlashInfer to accelerate token sampling, we ultimately achieve extreme inference speeds to support scalable training and deployment.

## 📊 Main Results
Experiments on multiple streaming video benchmarks show that ThinkStream significantly outperforms existing online video models while maintaining low latency and memory usage.

* **OVO-Bench**: ThinkStream achieves a strong average score, significantly surpassing both its base model and competing open-source online models.
* **StreamingBench Real-Time**: ThinkStream attains highly competitive performance against proprietary models and vastly exceeds other open-source online MLLMs.
* **Efficiency**: Our framework successfully bounds latency as the processed video length increases, consistently staying below the required real-time thresholds.

## 📂 Directory Structure

```text
ThinkStream/
├── scripts/agent_data_v5/         # data construction passes: pass1...pass5
├── scripts/eval/                  # evaluation entrypoints
├── scripts/sft_per_timestep.sh    # production SFT launcher
├── scripts/grpo_train_verl.sh     # production GRPO launcher
├── thinkstream/
│   ├── data/agent_protocol.py     # shared pass/SFT/RL/eval prompt protocol
│   ├── sft/                       # SFT dataset loader and trainer
│   ├── eval/                      # shared eval engines
│   ├── model/                     # inference/runtime agent loop
│   └── trainer/                   # framework-agnostic reward/matcher helpers
├── verl/recipe_thinkstream/       # vendored verl recipe and RL agent loop
├── data/agent_v5/<batch_id>/      # generated batch roots (ignored by git)
├── docs/project_structure.md      # current path and batch-layout contract
├── requirements.txt
└── README.md
```

## 🚀 Get Started

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training

**Data Preparation:**
- Download the [ThinkStream dataset](https://huggingface.co/datasets/JohnCage/ThinkStream).
- Prepare the video sources: LLaVA-Video 178K, and the Charades / Kinetics-700 / ActivityNet subsets from Tarsier2.

*Note: The dataset path configurations are located in `thinkstream/data/__init__.py`, which follows a similar logic to `qwen-vl-finetune`.*

**Run Training (SFT → verl GRPO RL):**

Generated data should live under one batch root, for example
`data/agent_v5/batch2`. The pipeline emits SFT messages and RL trajectories
under `final/`.

```bash
# SFT on the batch's messages file
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
bash scripts/sft_per_timestep.sh
# → output/agent-sft/

# GRPO RL from the SFT checkpoint, using vendored verl
THINKSTREAM_DATA_ROOT=data/agent_v5/batch2 \
LLM=output/agent-sft/checkpoint-616 \
bash scripts/grpo_train_verl.sh
# → output/agent-grpo/  +  output/agent-grpo/audit/grpo_step.jsonl
```

GDPO RL uses NVIDIA-style per-reward decoupled advantage aggregation
([2601.05242](https://arxiv.org/abs/2601.05242)) over eight reward components
(`correctness`, `silent_quality`, `timing`, `recall_quality`, `recall_hit_rate`,
`range_tightness`, `format`, `overflow_pen`). See `thinkstream/trainer/gdpo_advantage.py`
and `docs/design.md` §8 for the full design (canonical current-state).

`scripts/grpo_train.sh` is only a backward-compatible forwarder. The active
RL implementation is `scripts/grpo_train_verl.sh` plus
`verl/recipe_thinkstream/run_thinkstream_grpo.sh`. See
`docs/project_structure.md`, `docs/design.md` §8, and
`docs/v12.14_recurrent_design.md` for the >180-chunk recurrent variant.

### Evaluation

First, prepare the official datasets for OVO-Bench and StreamingBench.

Run the respective `transfer_annotation_format.py` scripts under the `thinkstream/eval` folder to convert the format:
- `thinkstream/eval/ovo_bench/transfer_annotation_format.py`
- `thinkstream/eval/rtvu/transfer_annotation_format.py`

After conversion, run the v12 streaming agent eval. The canonical entry
is `scripts/eval/ovo/run_sft.sh`, which drives `StreamingAgentLoop`
(the same per-timestep, system-trigger-injecting, recall-orchestrating
runtime the model was trained against — chunk-by-chunk fresh-KV
inference with memory + queries text persistence).

```bash
bash scripts/eval/ovo/run_sft.sh \
    --benchmark_dir /path/to/ovo_bench \
    --model_path /path/to/ckpt \
    --model_type qwen3vl
```
*Note: You need to change the model checkpoint (`--model_path`) and
benchmark dir to your own paths.*

The legacy `eval.sh` script does NOT drive the agent loop and should
not be used for v12 evaluation. The retired `--use_agent_loop` flag
referenced in earlier README revisions has been removed; `run_sft.sh`
is now the only sanctioned path.

### Inference

Use Python to run `scripts/demo.py` for inference testing.

Before running, please change `MODEL_ID` and `VIDEO_PATH` in the code to your own paths. Meanwhile, you need to manually fill in the `content` (question/instruction) and `timestamp` in the `queries` list.

Then, simply run:
```bash
python scripts/demo.py
```
You will see the output results in the command line.

## ❤️ Acknowledgement

We would like to thank the following open-source projects for their valuable contributions:

* [deepslyme](https://github.com/Slymer-Tech/deepslyme)
* [qwen-vl-finetune](https://github.com/QwenLM/Qwen3-VL/)
* [FlashAttention](https://github.com/Dao-AILab/flash-attention)
* [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
* [FlexAttention](https://arxiv.org/abs/2412.05496)
* [Liger Kernel](https://github.com/linkedin/Liger-Kernel/)

## 📑 Citation
If you find this work helpful, you can cite the following papers:

```
@misc{liu2026thinkingstreamingvideo,
      title={Thinking in Streaming Video}, 
      author={Zikang Liu and Longteng Guo and Handong Li and Ru Zhen and Xingjian He and Ruyi Ji and Xiaoming Ren and Yanhao Zhang and Haonan Lu and Jing Liu},
      year={2026},
      eprint={2603.12938},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12938}, 
}
```
