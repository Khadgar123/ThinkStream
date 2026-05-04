# Paper Outline

## Abstract

One paragraph: streaming video problem, Watch-Think-Speak, memory/compression, training-inference consistency, benchmark gains.

## 1 Introduction

- Batch video reasoning observes full context and answers late.
- Streaming assistants need bounded latency and incremental decisions.
- Key challenge: maintain enough state without retaining all visual tokens.
- ThinkStream: Watch, Think, Speak with memory, recall, and compression.
- Contributions list.

## 2 Related Work

- Video-language models and video QA.
- Online / streaming video understanding.
- Multimodal agents and tool use.
- Memory-augmented LLMs and recurrent RL.
- RL with verifiable rewards / process rewards.

## 3 Task Formulation

- Streaming observations and chunk definition.
- Query timing and answer timing.
- Action space: silent, response, recall, compress.
- Metrics: correctness, timing, latency, memory.

## 4 Method

- Watch-Think-Speak loop.
- Reasoning-compressed streaming memory.
- Recall mechanism.
- Training data construction.
- SFT and RL objective.

## 5 Experiments

- Benchmarks and baselines.
- Main results.
- Ablations.
- Efficiency.

## 6 Analysis

- Qualitative trajectories.
- Error categories.
- Long-video behavior.

## 7 Conclusion

Short conclusion focused on streaming reasoning.

## Limitations

Required section. Do not add new experiments here.

