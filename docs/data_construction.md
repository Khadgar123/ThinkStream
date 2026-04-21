# Streaming Video Agent Data Construction

> **DEPRECATED**: This document describes the v3.0 multi-turn conversation format.
> The current design uses **per-timestep independent samples** (v5.6+).
> See `data_construction_zh.md` for the authoritative specification
> and `sft_engineering.md` for the SFT training design.

> Version: v3.0 | Date: 2026-04-20
> 
> Changelog:
> - v3.0: Fix 6 data-quality bugs (think leakage, query answer leakage, causal violation, format regex, recall noise, filter check). Add structured video content format. Add SFT consumer-side think stripping. See §11 for full analysis.
> - v2.0: Initial 3-action protocol pipeline.

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
- **Structured video content** (v3.0): User messages use `[{"type": "video", "video_start": t, "video_end": t+2}, {"type": "text", "text": question}]` instead of `<video>` string placeholder. This preserves explicit time ranges for non-contiguous episodes (RC7).
- **Generic silent think** (v3.0): Silent chunks use `"Observing the current scene."` instead of segment `action` text. This prevents causal violation: 4s segment annotations may describe events in the second 2s half, leaking future info to the first chunk.
- **Recall result noise** (v3.0): `_build_recall_result` injects realistic retrieval noise (70% oracle, 20% top-3, 5% distractor, 5% failure) instead of always-perfect oracle.
- **Query keyword extraction** (v3.0): When `query_candidates` is empty, fallback uses `_extract_query_keywords(question)` instead of `expected_answer`, preventing answer leakage into the model's query.
- S3_R2/S4_R4 waiting silents: "User asked about this, but the event hasn't occurred yet. Waiting."
- RC7 gap truncation: only 3 silent chunks around each event (avoids 30+ filler silents)
- `answer_type` from 397B output, fallback to heuristic (yes/no → yesno, digit → number)
- Recall tasks without triplet binding (RC7) auto-promoted to `sample_type=recall_positive`

Output: `sft_episodes_raw.jsonl`

**Step 5: Filtering**
- Hard rules: ≥3 messages, non-empty answer
- **Strict format regex** (v3.0): Three mutually exclusive patterns with `^...$` anchoring:
  - `<think>.*</think><action>silent</action>` (no trailing content)
  - `<think>.*</think><action>response</action><response>.*</response>` (response required)
  - `<think>.*</think><action>recall</action><query>{.*}</query>` (query required)
- **Query leakage check** (v3.0): `_check_query_leakage` rejects episodes where `<query>` text contains `expected_answer`
- Output: filtered episodes passed to Step 6

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
| `query_candidates` is null | **v3.0**: Fallback to `_extract_query_keywords(question)` (not `expected_answer`) |
| Task validation fails | Reject task, log count |
| vLLM request fails | Retry 3× with exponential backoff (1s→2s→4s) |
| All videos fail | `print_statistics` handles empty list gracefully |
| Recall result noise → no evidence found | Post-recall think adapted: "Retrieval did not return useful evidence" |
| Query contains expected answer | **v3.0**: Rejected by `_check_query_leakage` in Step 5 |

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

## 10. SFT Consumer-Side Protections (v3.0)

The SFT data processor (`thinkstream/data/stream_data_processor.py`) applies additional protections when consuming pipeline output:

### 10.1 Historical Think Stripping

**Problem**: In multi-turn episodes, the model sees all prior assistant `<think>` content. If chunk 0 has `<think>The person wears a red apron</think>`, a recall task at chunk 10 asking "what color apron?" becomes trivially solvable from text — no visual recall needed.

**Fix**: `_build_agent_messages` strips `<think>` content from all historical assistant messages, replacing `<think>detailed observation</think>` with `<think></think>`. Only the LAST assistant message (current training target) keeps its full think content.

**Implementation**: `_strip_think_content()` uses `re.sub(r'<think>.*?</think>', '<think></think>', text)`. A `last_asst_idx` is computed before the message loop; only `msg_idx >= last_asst_idx` retains original think.

### 10.2 Per-Chunk Video Loading

**Problem**: RC7 episodes have non-contiguous time ranges (base Q&A at t=12s, follow-up at t=60s). Loading a single contiguous video range [12, 62] includes gap frames that shouldn't be visible.

**Fix**: `preprocess_qwen_visual_agent` loads each video chunk independently via ghost messages to `process_vision_info`. Time ranges are read from the structured content (`video_start`/`video_end` per user message).

### 10.3 Loss Masking (LlamaFactory Agentic Convention)

Following the LlamaFactory agentic mode:
- **Loss ON**: assistant turns (think + action + response/query)
- **Loss OFF**: user turns (video chunks, questions, `<recall_result>` injection)

Implemented by `find_assistant_spans` which locates `assistant`...`<|im_end|>` token spans.

### 10.4 CE Weight for Mixed Data

The `<silent>` special token appears in cold-start data but NOT in 3-action agent data (which uses `<action>silent</action>` text). CE weight rebalancing is skipped when `n_silent == 0` to avoid downweighting `<response>`.

## 11. Data Quality Analysis (v3.0)

### 11.1 Bugs Fixed

| # | Bug | Severity | Root Cause | Fix |
|---|-----|----------|------------|-----|
| 1 | Historical `<think>` leaks visual details to future chunks | **Critical** | Multi-turn SFT sees all prior assistant reasoning | Strip historical think in SFT consumer |
| 2 | Query fallback uses `expected_answer` | **Critical** | Line 920: `task.get("expected_answer")` as query | Replace with `_extract_query_keywords(question)` |
| 3 | 4s segment annotation leaks to 2s chunk | **High** | Silent think uses `seg.get("action")[:50]` | Replace with generic `"Observing the current scene."` |
| 4 | Format regex allows invalid combinations | **High** | Optional trailing group `(response|query)?` | Split into 3 mutually exclusive anchored patterns |
| 5 | Recall result always oracle | **Medium** | Inline perfect support segment | `_build_recall_result` with 70/20/5/5 noise distribution |
| 6 | Filter misses query-answer leakage | **Medium** | No leakage check in `filter_episode` | `_check_query_leakage` rejects contaminated episodes |
| 7 | RC7 `video_end` not clamped | **Medium** | `t + AGENT_CHUNK_SEC` without `min(…, duration)` | Add `min()` clamp |
| 8 | CE weight mismatches 3-action format | **Medium** | `<silent>` token absent, `<response>` gets 0.5x weight | Skip reweighting when one class is absent |
| 9 | `fps` list handling fragile | **Low** | Scalar vs list not handled uniformly | Unified `[fps_val] * N` |

### 11.2 Remaining Design Considerations (Not Bugs)

**1. Recall failure still trains correct answer**

When `_build_recall_result` returns `found_correct=False` (10% of cases), the think says "no useful evidence" but the response still gives the correct answer. This may teach the model to hallucinate when recall fails. Mitigation: address in RL phase with `unsupported_answer_penalty`.

**2. Action distribution sampled at episode level**

`assemble_final` samples by episode type, not by chunk-level action counts. Different episode types have very different chunk counts (R1: ~14 chunks, S3_R2: ~20+ chunks). The actual action distribution may deviate from the 58/30/12 target. Mitigation: `print_statistics` reports per-message distribution; adjust episode type ratios empirically.

**3. Triplet ordering: think generated before triplet binding**

Step 2d generates `think_at_ask` for recall tasks, then Step 3 creates control/false_negative variants with hardcoded think overrides. Ideally, each variant would get its own teacher-generated think. Current approach is acceptable because:
- Control think is overwritten: "The answer is visible in the current window, responding directly."
- False-negative think is overwritten: "Although the user said 'earlier', the evidence is in the current window."
- Only `think_at_ask` is reused, but it's specific to the ask-time decision point.

**4. `validate_task` timing fix may not update segment**

When `fixed_ask` is computed to push ask forward, the `for s in segments` loop searches for a matching segment. If no segment contains `fixed_ask` (shouldn't happen with contiguous segments), `ask_segment` retains its old value. Low risk since segments are contiguous.

### 11.3 Data Format (v3.0 Structured Content)

User messages with video now use structured content:

```json
{
  "role": "user",
  "content": [
    {"type": "video", "video_start": 36.0, "video_end": 38.0},
    {"type": "text", "text": "What color was the apron?"}
  ]
}
```

Recall result messages remain plain strings:

```json
{
  "role": "user",
  "content": "<recall_result>\n<item rank=\"1\" start=\"8.0\" end=\"12.0\">caption: ... ocr: none</item>\n</recall_result>\nContinue following the protocol to respond."
}
```

Assistant messages are strings in the 3-action format:

```json
{
  "role": "assistant",
  "content": "<think>Cannot find evidence in current window.</think><action>recall</action><query>{\"query\":\"apron color beginning\",\"time_bias\":\"past_far\",\"target\":\"entity\",\"topk\":3}</query>"
}
```

### 11.4 Recall Result Noise Distribution

| Outcome | Probability | Description |
|---------|-------------|-------------|
| Oracle top-1 | 70% | Correct support segment at rank 1, single item |
| Correct in top-3 | 20% | Correct segment at random rank 1-3, with 2 distractor segments |
| Distractor only | 5% | 3 random non-support segments, no correct item |
| Retrieval failure | 5% | `<item>Not found</item>` |

When `found_correct=False`, the post-recall think is adapted to "Retrieval did not return useful evidence. Answering based on available context."

### 11.5 Test Coverage

`tests/test_agent_sft.py` — 64 test cases across 10 test classes:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestDataFormat | 7 | Structured content, no `<question>` tags, format compliance, JSON serialization |
| TestVideoTimeRanges | 5 | Ascending, contiguous, RC7 gap, duration clamp, 2s chunk width |
| TestAgenticActions | 9 | R1/RC1/S3_R2/R3/RC7 action sequences, recall_result user role |
| TestTriplets | 5 | Triplet generation, RC7 exclusion, false-negative phrasing |
| TestFiltering | 5 | Valid/invalid episodes, all task types pass |
| TestActionDistribution | 1 | All 3 actions present, silent dominates |
| TestBuildAgentMessages | 6 | Structured passthrough, legacy compat, recall_result string, video_meta |
| TestFindAssistantSpans | 3 | Span detection, multiple spans, no spans |
| TestEdgeCases | 4 | Short video, t=0 ask, near-end answer |
| TestP0Fixes | 13 | Think stripping, query leakage, causal violation, regex strictness, recall noise |

## 12. Usage

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

# 4. Run tests
python -m pytest tests/test_agent_sft.py -v

# 5. SFT training (two-stage)
STAGE=a bash scripts/sft_agent.sh                                    # Stage A
STAGE=b LLM=./output/agent-sft-a/checkpoint-best bash scripts/sft_agent.sh  # Stage B
```
