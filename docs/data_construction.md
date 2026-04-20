# Streaming Video Agent Data Construction

> Version: v2.0 | Date: 2026-04-20

## 1. Goal

Construct 3-action format (`<think>...<action>silent|response|recall</action>`) SFT + RL training data for the streaming video agent. The model learns to act at every 2s chunk:
- **Silent**: no question to answer, incremental observation
- **Response**: sufficient evidence to answer
- **Recall**: evidence has left the recent window (24s), retrieve historical frames

## 2. Infrastructure

| Node | Hardware | Purpose | vLLM Launch Command |
|------|----------|---------|-------------------|
| AMD | MI300X × 8 (192GB each) | 397B teacher | See below |
| H20 | H20 96GB × 8 | Training + verification | (Optional) deploy 35B for verification |

### vLLM Launch Command

```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --port 8000 \
    --enable-prefix-caching
```

### Stress Test Results (text-only baseline)

| Concurrency | Throughput (req/s) | Latency | Status |
|-------------|-------------------|---------|--------|
| 8 | 0.86 | 23.1s | Baseline |
| 16 | 8.52 | 4.7s | Good |
| 32 | 10.77 | 3.7s | Peak |
| 64 | 5.75 | 7.0s | Degraded (memory pressure) |

Peak throughput at concurrency 32 for text-only (16 input tokens). With vision tokens (~6.8K–37.5K per request), optimal concurrency is lower.

### Per-Step Concurrency Configuration

| Step | Tokens/Request | Concurrent | max_tokens | temperature |
|------|---------------|------------|------------|-------------|
| 2a (segment annotation) | ~6.8K (4 frames) | 32 | 512 | 0.3 |
| 2b (task design) | ~8K (text only) | 32 | 4096 | 0.7 |
| 2d (think generation) | ~37.5K (24 frames) | 16 | 128 | 0.5 |

### Token Budget Analysis

```
Step 2a: 4 frames × 1500 vision + 300 text = ~6.8K tokens/request
Step 2b: ~3K annotation text + 1K prompt = ~8K tokens (output ~4K)
Step 2d: 24 frames × 1500 vision + 1.5K text = ~37.5K tokens/request
         → requires --max-model-len >= 65536
```

## 3. Task Classification

### 3.1 Action Distribution (chunk-level targets)

| Action | Ratio | Description |
|--------|-------|-------------|
| Silent | 58-65% | Most chunks: observe silently |
| Response | 23-30% | Answer when evidence is available |
| Recall | 10-15% | Evidence outside window |

### 3.2 Sub-task Types

**Silent Sub-tasks**

| ID | Name | Description | Implementation |
|----|------|-------------|----------------|
| S1 | No-question observation | No user question | Naturally included in all episodes |
| S2 | Post-answer silence | Already answered | Naturally included |
| S3 | Standby waiting | Asked but event not occurred | S3_R2 silent chunks between ask→answer |
| S4 | Trigger monitoring | Condition not met | S4_R4 silent chunks between ask→answer |

**Response Sub-tasks**

| ID | Name | Description | Implementation |
|----|------|-------------|----------------|
| R1 | Immediate answer | Evidence in current frame | `assemble_episode` default path |
| R2 | Delayed answer | Event occurred after standby | S3_R2 answer chunk |
| R3 | Progressive answer | Answer unfolds over time, multiple responses | `_assemble_r3` |
| R4 | Trigger fired | Monitoring condition met | S4_R4 answer chunk |
| R5 | Continuous narration | Real-time commentary | Not implemented (deferred) |
| R6 | Post-recall answer | Answer after retrieval | Recall episode response chunk |
| R7 | False recall negative | Sounds like recall but isn't | `generate_triplets` false_negative |

**Recall Sub-tasks**

| ID | Name | What to ask | query pattern | Implementation |
|----|------|-------------|---------------|----------------|
| RC1 | Visual detail recall | Color/shape/appearance | entity+attribute | Generic RC handler |
| RC2 | Numeric recall | Price/number/text | OCR+number | Generic RC handler |
| RC3 | Procedural recall | Step details/order | action+step | Generic RC handler |
| RC4 | Cross-time comparison | What changed | entity+state | Generic RC handler |
| RC5 | Long causal recall | Why did this happen | event+cause | Generic RC handler |
| RC6 | Entity re-identification | Same person/object? | entity+appearance | Generic RC handler |
| RC7 | Multi-turn follow-up | Ask about prior conversation | dialogue+detail | `_assemble_rc7` |

### 3.3 Triplet Binding Rules

Each recall task (RC1-RC6) generates a triplet:
- 1 × **no-recall control** (R1): same question asked when evidence is visible
- 1 × **false-recall negative** (R7): same question with "earlier/before" phrasing when evidence is visible

**Exception**: RC7 is excluded from triplet binding because the control would reference a base conversation that doesn't exist in the control episode.

### 3.4 Task Counts per Video

| Task Type | Count | Condition |
|-----------|-------|-----------|
| R1 (immediate) | 3 | Always |
| S3_R2 (delayed) | 2 | Always |
| RC1 (visual detail) | 1-3 | `num_segments // 10` |
| RC2 (numeric) | 2 | Has OCR segments |
| RC3 (procedural) | 2 | >10 segments |
| RC4 (comparison) | 1 | >15 segments |
| RC5 (causal) | 1 | Always |
| RC6 (re-identification) | 1 | Always |
| RC7 (follow-up) | 1 | >15 segments |
| R3 (progressive) | 1 | >10 segments |
| S4_R4 (trigger) | 1 | Always |

### 3.5 Video Type → Task Mapping

| Video Type | Duration Req. | Suitable Tasks |
|------------|--------------|----------------|
| Tutorial/Cooking/Assembly | >120s | RC1, RC3, RC5 |
| Vlog/Long-take | >90s | RC4, RC6, R2, R3 |
| Screen recording/UI | >60s | RC2, R1 |
| Drama/Variety | >120s | RC5, RC7 |
| Sports/Outdoor | >90s | R4, RC4 |
| Short video (<30s) | any | R1, S1 (protocol only) |

## 4. Data Construction Pipeline

### 4.1 Overview

```
Step 1:  Select videos + extract frames          [CPU, ~5min]
Step 2a: 397B segment annotation                 [AMD vLLM, concurrent=32]
Step 2b: 397B task design (reference segment ID) [AMD vLLM, concurrent=32]
Step 2c: Rule validation + timing fix            [CPU, <1s]
Step 2d: 397B think generation                   [AMD vLLM, concurrent=16]
Step 3:  Triplet binding                         [CPU, <1s]
Step 4:  Chunk-level episode assembly             [CPU, ~1min]
Step 5:  Filtering                               [CPU + optional small model]
Step 6:  Sampling + final assembly               [CPU, <1s]
```

### 4.2 Step Details

**Step 1: Video Selection + Frame Extraction**
- Select from Streamo videos ≥60s, ranked by richness score (duration × type diversity × annotation density)
- Extract at 2fps, resize to 720px short edge (~1500 vision tokens/frame)
- Group into 4s segments, select 4 keyframes per segment (1fps coverage)
- Failed videos are skipped (logged, not crash)
- `duration_sec` updated from ffprobe (not annotation timestamps)
- Output: `video_registry.jsonl` + `frames/` + `segments/`

**Step 2a: 397B Segment Annotation**
- Each segment's 4 keyframes → 397B describes content in English
- Output per segment: `action`, `entities[]`, `visual_details[{entity, attributes}]`, `ocr`, `change`
- Prompt includes JSON example with exact schema
- Parse: try direct JSON, then balanced-bracket extraction, fallback to raw text
- Output: `segment_annotations.jsonl`

**Step 2b: 397B Task Design**
- All segment annotations (text) for one video → 397B designs 10-15 tasks
- Prompt specifies per-type counts, answer_type constraints, and few-shot examples (RC1, R3, RC7)
- References segment_id (not timestamps)
- Parse: balanced-bracket JSON array extraction
- Output: `task_candidates_raw.jsonl`

**Step 2c: Rule Validation**
- Resolve segment_id → precise timestamps
- Validation order: RC7 → R3 → RC* (generic) → S3_R2 → S4_R4 → else (R1)
- RC7: validate `base_ask_segment`, `base_question`, `base_answer`, gap ≥ 24s
- R3: validate `response_segments` (≥2, ascending order), pad `partial_answers`
- RC*: validate `support_segment`, gap ≥ 24s (auto-fix by pushing ask forward)
- General: reject empty question, empty answer, ask > duration
- Output: `task_candidates_verified.jsonl`

**Step 2d: 397B Think Generation**
- Send 24 actual frames from the 24s recent window (not text annotations)
- Token budget: 24 × 1500 + 1500 = ~37.5K → requires `--max-model-len 65536`
- For recall tasks: additional post-recall think with support frames + current frames
- Output: `task_pool.jsonl` (tasks with `think_at_ask` and `think_after_recall`)

**Step 3: Triplet Binding**
- Each RC1-RC6 recall task → 3 episodes:
  - `recall_positive`: original (need recall)
  - `control` (R1): same question, ask when evidence is visible
  - `false_negative` (R7): add "Earlier/Before, ..." phrasing, evidence still visible
- RC7 excluded (control would reference nonexistent base conversation)
- R3 excluded (not a recall task)
- Output: `task_triplets.jsonl`

**Step 4: Chunk-Level Assembly**

Dispatched by task type:

| Task Type | Assembly Function | Episode Structure |
|-----------|------------------|-------------------|
| R3 | `_assemble_r3` | question → response₁ → silent → response₂ → ... → responseₙ |
| RC7 | `_assemble_rc7` | base Q&A → (gap truncated to 3 context chunks) → follow-up recall → response |
| RC1-RC6 | default (recall path) | silent... → question+recall → recall_result → response |
| R1 | default (response path) | silent... → question+response |
| S3_R2 | default (delayed path) | silent... → question → waiting silents → response |
| S4_R4 | default (delayed path) | silent... → question → monitoring silents → response |

Key design decisions:
- Silent think: segment `action` description (annotation text), max 50 chars
- S3_R2/S4_R4 waiting silents: "User asked about this, but the event hasn't occurred yet. Waiting."
- RC7 gap truncation: only 3 silent chunks around each event (avoids 30+ filler silents)
- `answer_type` from 397B output, fallback to heuristic (yes/no → yesno, digit → number)
- Recall tasks without triplet binding (RC7) auto-promoted to `sample_type=recall_positive`

Output: `sft_episodes_raw.jsonl`

**Step 5: Filtering**
- Hard rules: ≥3 messages, all assistant messages match format regex, non-empty answer
- Format regex: `<think>.*</think><action>(silent|response|recall)</action>(<response>.*</response>|<query>{.*}</query>)?`
- Output: `sft_episodes_filtered.jsonl`

**Step 6: Sampling + Final Assembly**

| Dataset | Composition | Purpose |
|---------|-------------|---------|
| `sft_a.jsonl` | simple + 25% recall + 25% control | Protocol warmstart |
| `sft_b.jsonl` | all recall + control + false_neg + matched simple | Recall-heavy training |
| `rl_pool.jsonl` | verifiable answers only (yesno, number, entity, slot, multiple_choice) | RL training |

## 5. Persistence Table

| File | Content | When to Regenerate |
|------|---------|-------------------|
| `video_registry.jsonl` | Video index | Append when adding videos |
| `segments/*.json` | Per-video frame paths | Never (cached per video) |
| `segment_annotations.jsonl` | Per-segment 397B annotations | Never (most expensive) |
| `task_candidates_raw.jsonl` | 397B task designs | Delete to regenerate with new prompt |
| `task_candidates_verified.jsonl` | Rule-validated tasks | Auto-regenerated |
| `task_pool.jsonl` | Tasks with think content | Delete to regenerate |
| `task_triplets.jsonl` | Triplet bindings | Auto-derived |
| `sft_episodes_raw.jsonl` | Assembled episodes | Rerun when format changes |
| `sft_a.jsonl` / `sft_b.jsonl` | Training data | Rerun when ratios change |
| `rl_pool.jsonl` | RL training pool | Rerun when ratios change |
| `pipeline_stats.json` | Run statistics | Auto-generated |

## 6. Prompts

All prompts are in English. Key design choices:

- **SEGMENT_ANNOTATE_PROMPT**: includes JSON example with exact schema, `visual_details` as `[{entity, attributes}]` array
- **TASK_DESIGN_PROMPT**: per-type answer_type constraints, 3 few-shot examples (RC1, R3, RC7), query_candidates rules
- **THINK_PROMPT**: token-based length constraints (15-48 tokens), "Output English only"
- **AGENT_SYSTEM_PROMPT**: 3-action protocol definition, used as system message in all episodes

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| ffprobe/ffmpeg fails for a video | Skip video, log warning, continue |
| 397B returns no JSON (plain text) | Use raw text as `action`, other fields empty |
| 397B returns JSON with trailing text | Balanced-bracket extraction (not greedy regex) |
| `query_candidates` is null | Fallback to default query from expected_answer |
| Task validation fails | Reject task, log count |
| vLLM request fails | Retry 3× with exponential backoff (1s→2s→4s) |
| All videos fail | `print_statistics` handles empty list gracefully |

## 8. Training Plan

```
SFT-A (protocol alignment): datasets=sft_a, lr=1e-5, epochs=3
SFT-B (recall focus):        datasets=sft_b, lr=5e-6, epochs=3, from SFT-A ckpt
RL-A  (action calibration):  datasets=rl_pool, lr=2e-7, from SFT-B ckpt
```

## 9. Quality Metrics

| Metric | Pass Threshold |
|--------|---------------|
| Format compliance rate | ≥ 95% |
| OVO-Bench accuracy | Within -2% of base model |
| RTVU accuracy | Within -2% of base model |
| Recall precision | ≥ 70% |
| Recall specificity | ≥ 85% |

## 10. Usage

```bash
# 1. Start vLLM on AMD node (see Section 2)

# 2. Stress test
python -m scripts.agent_data_pipeline.generate_data stress_test \
    --api_base http://AMD_IP:8000/v1 --max_concurrent 8 --num_requests 20

# 3. Run full pipeline
python -m scripts.agent_data_pipeline.generate_data run \
    --api_base http://AMD_IP:8000/v1 \
    --streamo_dir /path/to/streamo \
    --video_root /path/to/videos \
    --output_dir data/agent \
    --max_concurrent 32 \
    --num_videos 200
```
