# Qwen3.5-397B-A17B-FP8 输出格式分析

> 基于 v5.5 pipeline test run (30 videos, 5 completed) 的实际输出数据 + API 行为测试

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.5-397B-A17B-FP8 (本地路径) |
| vLLM API | http://10.16.12.175:8000/v1 |
| max_model_len | 65536 |
| limit-mm-per-prompt | image=24 |
| enforce-eager | 已启用 |
| Pass 1 配置 | temperature=0.3, max_tokens=1024, thinking=True |

## 2. 核心发现：Qwen3.5 thinking 输出行为

### 2.1 API 返回结构

通过 API 健康检查确认：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *...",
      "reasoning": null
    }
  }]
}
```

关键事实：
- `reasoning` / `reasoning_content` 字段始终为 `null`
- thinking 内容直接混入 `content` 字段
- 模型不使用 `<think>...</think>` 标签包裹思考过程
- 即使 max_tokens=16，模型也优先输出 thinking 而非 JSON

### 2.2 content 字段的实际格式

模型输出结构为：**纯文本思考过程 + JSON 对象**（无任何标签分隔）

```
The user wants me to annotate a video chunk based on two frames.

**Frame Analysis:**
- **Frame 1 (t=0s):** Shows a woman with blonde hair, wearing a white collared shirt...
...

{"time": [0, 2], "visible_entities": [...], ...}
```

## 3. 已有输出样本统计（5 个视频，251 chunks）

### 3.1 总体解析成功率

| 视频 | 总 chunks | 成功 | 成功率 | 成功类型 |
|------|-----------|------|--------|----------|
| 0ySyffWMAVI | 51 | 1 | 2.0% | 空 JSON |
| EQnqHVzEM_A | 59 | 0 | 0.0% | — |
| xsFSYBJlrqA | 46 | 3 | 6.5% | 空 JSON |
| ytb_FYz91NRoAuM | 50 | 0 | 0.0% | — |
| ytb_zgfJuuCbhyw | 45 | 4 | 8.9% | 2 有内容 + 2 空 JSON |
| **总计** | **251** | **8** | **3.2%** | |

### 3.2 失败样本分析

- 共 243 个失败 chunk 保留了 `_raw` 字段
- **所有 `_raw` 均恰好 200 字符**（被 `raw[:200]` 截断）
- **0/243 个 `_raw` 中包含 `{` 字符** — 说明 JSON 部分在 200 字符之后
- 所有 `_raw` 均以 `"The user wants me to annotate"` 开头

典型 `_raw` 样本：

**样本 1（首 chunk，含 markdown 格式）：**
```
The user wants me to annotate a video chunk based on two frames.

**Frame Analysis:**
- **Frame 1 (t=0s):** Shows a woman with blonde hair, wearing a white collared shirt and a maroon vest. She is fac
```

**样本 2（后续 chunk，含编号分析）：**
```
The user wants me to annotate the video chunk from 52 to 54 seconds.

**1. Analyze the input frames:**
- The frames show a woman in a beige uniform carrying a large white bin filled with green leafy v
```

**样本 3（暗场景，自然语言描述）：**
```
The user wants me to annotate a video chunk based on two frames.
The first frame is extremely dark, almost black. It's hard to make out details, but there's a faint silhouette of a person.
The second
```

### 3.3 成功样本分析

8 个成功 chunk 分两类：

**类型 A：空 JSON（6/8）** — 模型直接返回空结构，无 thinking 前缀
```json
{
  "time": [86, 88],
  "visible_entities": [],
  "atomic_facts": [],
  "state_changes": [],
  "ocr": [],
  "spatial": "",
  "not_observable": []
}
```
推测原因：视频末尾或暗场景，模型认为无内容可标注，跳过了 thinking 直接输出 JSON。

**类型 B：有内容的 JSON（2/8，均来自 ytb_zgfJuuCbhyw chunk 29-30）**
```json
{
  "time": [58, 60],
  "visible_entities": [
    {
      "id": "woman_1",
      "attributes": ["blonde hair", "red gown", "red gloves", "sparkly waistband"],
      "action": "walking towards camera",
      "position": "center"
    }
  ],
  "atomic_facts": [
    {
      "fact": "A woman with long blonde hair is walking forward against a white background.",
      "confidence": 1.0,
      "support_level": "direct_current_chunk",
      "target_resolution_visible": true
    },
    {
      "fact": "She is wearing a floor-length red dress with a sequined belt.",
      "confidence": 1.0,
      "support_level": "direct_current_chunk",
      "target_resolution_visible": true
    },
    {
      "fact": "She is wearing long red gloves.",
      "confidence": 1.0,
      "support_level": "direct_current_chunk",
      "target_resolution_visible": true
    }
  ],
  "state_changes": ["Scene cuts from the demon character reading a book to the human woman walking forward."],
  "ocr": [],
  "spatial": "Single subject centered in the frame against a plain white background.",
  "not_observable": []
}
```
推测原因：场景简单（单人白背景），thinking 较短，JSON 在 200 字符截断前就开始了，或模型偶尔跳过 thinking。

### 3.4 thinking 输出的常见模式

从 `_raw` 首行统计（243 个失败样本）：

| 模式 | 出现次数 | 说明 |
|------|----------|------|
| `The user wants me to annotate a video chunk based on two frames.` | ~30 | 首 chunk |
| `The user wants me to annotate a video chunk from t=Xs to t=Ys.` | ~50 | 短时间戳格式 |
| `The user wants me to annotate the video chunk from X to Y seconds.` | ~160 | 长时间戳格式 |

thinking 内部结构通常包含：
- `**Frame Analysis:**` 或 `**1. Analyze the input:**` 标题
- 逐帧描述（`**Frame 1 (t=0s):**`）
- Markdown 格式（加粗、列表）
- 最终才输出 JSON（但被 200 字符截断看不到）

## 4. 问题根因链

```
Qwen3.5 thinking 行为
  → content 字段 = "thinking文本" + JSON（无标签分隔）
  → vLLM 不分离 reasoning_content（返回 null）
  → parse_evidence_result() 用 re.sub(r'<think>.*?</think>') 剥离 → 无匹配
  → json.loads(raw) 失败（raw 以 "The user wants..." 开头）
  → 找第一个 '{' → 在 thinking 的 markdown 中可能找到错误的 '{'
  → 或者 _raw[:200] 截断导致 JSON 部分丢失
  → parse_success = False
```

## 5. 修复方案对比

### 方案 A：从 raw 中提取最后一个完整 JSON 对象（推荐）

```python
def extract_json_from_raw(raw: str) -> Optional[dict]:
    """从 raw 输出中提取最后一个完整 JSON 对象。"""
    positions = [i for i, c in enumerate(raw) if c == '{']
    for start in reversed(positions):
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == '{': depth += 1
            elif raw[i] == '}': depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    break
    return None
```

优点：不依赖任何标签格式，兼容所有模型
缺点：如果 thinking 中包含 JSON 片段可能误匹配（但取最后一个可缓解）

### 方案 B：使用 `/no_think` 前缀关闭 thinking

在 prompt 前加 `/no_think\n\n`，让模型跳过 thinking 直接输出 JSON。

优点：从源头解决，输出更短更快
缺点：
- 需要确认 Qwen3.5 是否真的响应 `/no_think`（未验证）
- Pass 1 evidence 是结构化标注，thinking 可能提升标注质量

### 方案 C：vLLM chat_template 配置

通过 vLLM 的 `--chat-template` 参数配置 `<think>` 标签。

优点：标准化输出格式
缺点：需要重启 vLLM 服务，且需要找到/编写正确的 chat template

### 方案 D：组合方案（推荐实施）

1. **立即修复**：方案 A — 修改 `parse_evidence_result` 提取最后一个 JSON
2. **同时修复**：`_raw` 截断从 200 → 2000 字符（保留完整输出用于调试）
3. **后续优化**：测试 `/no_think` 对 Pass 1 质量的影响，如果质量不降则启用

## 6. 待验证问题

| # | 问题 | 验证方法 | 状态 |
|---|------|----------|------|
| 1 | Qwen3.5 的 thinking 输出完整长度是多少？ | 去掉 `_raw[:200]` 截断重跑 | 待做 |
| 2 | thinking 之后是否总是跟着有效 JSON？ | 同上 | 待做 |
| 3 | `/no_think` 是否对 Qwen3.5 生效？ | 单独 API 测试 | 待做（API 超时） |
| 4 | `/no_think` 是否影响标注质量？ | A/B 对比 | 待做 |
| 5 | Pass 2 的 `/no_think` 是否正常工作？ | 检查 Pass 2 输出 | 待做 |
| 6 | vLLM 是否支持 `reasoning_content` 分离？ | 检查 vLLM 版本和配置 | 已确认不支持 |

## 7. 附录：API 行为验证记录

### 测试 1：纯文本健康检查（成功）

请求：
```json
{
  "model": "/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8",
  "messages": [{"role": "user", "content": "Hello, respond with just OK"}],
  "max_tokens": 16,
  "temperature": 0.1
}
```

响应：
```json
{
  "choices": [{
    "message": {
      "content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *",
      "reasoning": null
    },
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 16, "completion_tokens": 16}
}
```

结论：
- 即使 max_tokens=16，模型也先输出 thinking（`"Thinking Process:\n\n1.  **Analyze..."`)
- `reasoning` 字段为 null，thinking 混在 content 中
- 没有 `<think>` 标签

### 测试 2：视觉 + evidence prompt（来自 pipeline test run）

请求：Pass 1 evidence prompt + 24 帧滑动窗口
响应：content 以 `"The user wants me to annotate..."` 开头，后跟 markdown 格式分析，最后才是 JSON

---

## 8. 实时 API 采样：各场景原始输出

> 以下为 2026-04-21 通过 streaming API 实时采集的 Qwen3.5-397B-A17B-FP8 原始输出。
> 所有输出均未做任何后处理，完整保留模型返回的 content 字段。

### 8.1 Evidence Graph Prompt（无 /no_think，max_tokens=1024）

请求 prompt（简化版）：
```
You are annotating a streaming video chunk. Scene: A woman with blonde hair in a news studio.
Output STRICT JSON: {"time": [0, 2], "visible_entities": [], ...}
Output JSON only:
```

原始输出（3959 chars，163.3s，finish_reason=length，1024 tokens 全部用完）：
```
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Annotate a streaming video chunk.
    *   Scene Description: A woman with blonde hair in a news studio.
    *   Output Format: STRICT JSON.
    *   Specific Schema: `{"time": [0, 2], "visible_entities": [], "atomic_facts": [], "state_changes": [], "ocr": [], "spatial": "", "not_observable": []}`
    *   Constraint: Output JSON only (no markdown, no explanations).

2.  **Analyze the Scene:**
    *   Subject: Woman.
    *   Attribute: Blonde hair.
    *   Location: News studio.
    *   Timeframe: 0 to 2 seconds (based on the schema example provided in the prompt, though usually, this would depend on the actual video chunk duration. I will stick to the provided example structure `[0, 2]`).

3.  **Drafting the Content:**
    *   `time`: [0, 2] (as per prompt example).
    *   `visible_entities`: ["woman", "blonde hair", "news studio", "desk", "background screens"] (inferring typical news studio elements).
    *   `atomic_facts`: ["A woman is present", "The woman has blonde hair", "The setting is a news studio", "The woman is facing the camera"].
    *   `state_changes`: [] (In a 2-second chunk, significant state changes are unlikely unless she moves dramatically, but usually static anchoring is assumed for this brief description).
    *   `ocr`: [] (No specific text mentioned in the prompt, though news studios often have tickers. I will leave empty or add generic placeholder if needed. Better to leave empty if not specified).
    *   `spatial`: "The woman is centered in the frame with the studio background behind her."
    *   `not_observable`: ["Audio content", "Full body below the desk", "Specific news topic being discussed"].

4.  **Refining for Strict JSON:**
    *   Ensure no trailing commas.
    *   Ensure double quotes.
    *   Ensure no markdown code blocks (```json ... ```).

5.  **Final Review:**
    *   Does it match the schema exactly? Yes.
    *   Is it valid JSON? Yes.
    *   Is there any extra text? No.

6.  **Constructing the JSON:**
    ```json
    {
      "time": [0, 2],
      "visible_entities": ["woman", "blonde hair", "news studio", "desk", "background monitors"],
      "atomic_facts": ["A woman with blonde hair is visible", "The setting is a news studio", "The woman is positioned in front of a desk"],
      "state_changes": [],
      "ocr": [],
      "spatial": "The woman is centered in the foreground with studio equipment and screens visible in the background",
      "not_observable": ["Audio content", "Text on background screens", "Lower body movement"]
    }
    ```

7.  **Final Polish:** Remove whitespace/newlines if necessary to ensure it's raw JSON, though pretty-printed JSON is still valid JSON. The prompt says "Output JSON only", usually implying raw text. I will provide compact or standard JSON. I'll stick to standard indentation for readability but ensure no markdown fences.

    Wait, I need to make sure I don't add markdown fences. Just the raw JSON string.

    Let's refine the content to be accurate to the *given* description ("A woman with blonde hair in a news studio") without over-hallucinating too much, but enough to fill the schema meaningfully.

    *   `visible_entities`: ["woman", "blonde hair", "news studio"]
    *   `atomic_facts`: ["A woman is present", "The woman has blonde hair", "The environment is a news studio"]
    *   `state_changes`: [] (2 seconds is too short for major state changes usually)
    *   `ocr`: [] (None specified)
    *   `spatial`: "Woman centered in frame"
    *   `not_observable`: ["Audio", "Content of news report"]

    Let's assemble.

    {"time": [0, 2], "visible_entities": ["woman", "blonde hair", "news studio"], "atomic_facts": ["A woman is visible", "The woman has blonde hair", "The setting is a news studio"], "state_changes": [], "ocr": [], "spatial": "The woman is centered in the frame with the studio background behind her", "not_observable": ["Audio
```

分析：
- **1024 tokens 全部被 thinking 消耗**，JSON 在 thinking 的第 6 步中以 markdown code block 形式出现，但最终的"干净 JSON"输出被截断
- thinking 包含 7 个编号步骤，含反复修改和自我纠正
- 模型在 thinking 中构造了 JSON 草稿，但最终输出时 tokens 已耗尽

### 8.2 Observation Prompt（/no_think 前缀，max_tokens=256）

请求 prompt：
```
/no_think

Describe what is NEW in the latest 2 seconds of a news broadcast in 40-60 tokens.
Scene: anchor continues speaking, ticker updates to show stock prices.

Output (one paragraph, 40-60 tokens):
```

原始输出（1080 chars，41.0s）：
```
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Describe what is NEW in the latest 2 seconds of a news broadcast.
    *   Constraint 1: Length must be between 40 and 60 tokens.
    *   Constraint 2: Output must be one paragraph.
    *   Scene details: Anchor continues speaking, ticker updates to show stock prices.
    *   Constraint 3: "/no_think" (This usually means I should not output my thought process, just the final answer).

2.  **Drafting Content:**
    *   Focus on the *change* in the last 2 seconds.
    *   Anchor: Continues speaking (so maybe a new sentence or gesture, ...
```

关键发现：**模型识别到了 `/no_think` 指令**（在 thinking 中说 "This usually means I should not output my thought process"），**但仍然输出了完整的 thinking**。

### 8.3 Compress Prompt（/no_think 前缀，max_tokens=256）

请求 prompt：
```
/no_think

Compress these observations into a structured summary.
[0-2] Woman anchor speaking in studio.
[2-4] Ticker shows weather. Anchor gestures.
[4-6] Camera shifts. Map graphic appears.

Output JSON: {"time_range": [0, 6], "text": "..."}
```

原始输出（964 chars，40.9s）：
```
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A list of observations with timestamps (e.g., [0-2] Woman anchor speaking in studio...).
    *   Task: Compress these observations into a structured summary.
    *   Constraint: Output JSON format `{"time_range": [start, end], "text": "..."}`.
    *   Constraint: `/no_think` (This implies I should not output my internal thinking process, just the final result).

2.  **Analyze the Input Data:**
    *   [0-2] Woman anchor speaking in studio.
    *   [2-4] Ticker shows weather. Anchor gestures.
    *   [4-6] Camera shifts. Map graphic ...
```

### 8.4 Task Question Prompt（无 /no_think，max_tokens=512）

原始输出（2035 chars，83.5s）：
```
Thinking Process:

1.  **Analyze the Request:**
    *   Input: Visual evidence description (Entity: anchor_1, Fact: Anchor gestures with right hand while explaining weather graphic).
    *   Task: Generate a JSON object with specific fields: "question", "concise_answer", "answer_type".
    *   Constraint: The output must be valid JSON.
    *   Content: Based on the provided fact about the anchor's gesture.

2.  **Analyze the Visual Evidence/Fact:**
    *   Entity: anchor_1
    *   Action: Gestures with right hand
    *   Context: While explaining weather graphic

3.  **Draft the Question:** ...
```

### 8.5 纯文本健康检查（max_tokens=32）

请求：`"Say OK"`

原始输出：
```
Thinking Process:

1.  **Analyze the Request:**
    *   Input: "Say OK"
    *   Intent: The user wants
```

即使只给 32 tokens，模型也优先输出 thinking 而非回答。

## 9. 综合结论

### 9.1 Qwen3.5 输出格式规律

| 特征 | 值 |
|------|-----|
| thinking 起始标记 | `"Thinking Process:\n\n1.  **Analyze the Request:**"` （固定模式） |
| thinking 结束标记 | **无**（直接接 JSON 或最终回答，无分隔标签） |
| `<think>` 标签 | **从不使用** |
| `reasoning_content` 字段 | **始终为 null** |
| `/no_think` 效果 | **无效** — 模型识别但不遵守 |
| thinking 长度 | 约 500-3000 tokens（视任务复杂度） |
| JSON 位置 | thinking 之后，通常在输出末尾 |
| thinking 内部格式 | Markdown（编号列表、加粗、code block） |

### 9.2 对 pipeline 各 Pass 的影响

| Pass | Prompt 类型 | /no_think | thinking 影响 | 严重程度 |
|------|-------------|-----------|---------------|----------|
| Pass 1 Evidence | JSON 输出 | 无 | thinking 消耗大量 tokens，JSON 可能被截断 | **P0** |
| Pass 2 Observation | 短文本输出 | 有（无效） | thinking 消耗 tokens，实际观察文本被截断 | **P0** |
| Pass 2 Compress | JSON 输出 | 有（无效） | 同 Pass 1 | **P0** |
| Pass 3 Task Question | JSON 输出 | 无 | thinking 消耗 tokens | **P1** |
| Pass 3 Recall Query | JSON 输出 | 有（无效） | 同上 | **P1** |
| Pass 3 Response | 文本输出 | 有（无效） | thinking 混入 response 文本 | **P1** |

### 9.3 修复优先级

1. **P0（立即）**：所有 parse 函数改为从 raw 中提取最后一个完整 JSON（方案 A）
2. **P0（立即）**：`max_tokens` 需要大幅增加以容纳 thinking 开销（当前 1024 不够）
3. **P0（立即）**：`_raw` 截断从 200 → 完整保留（至少 4000 chars）
4. **P1（短期）**：研究 vLLM 配置是否能启用 `reasoning_content` 分离（需要 `--enable-reasoning` 参数？）
5. **P1（短期）**：研究 Qwen3.5 的 chat_template 是否支持 `enable_thinking=False` 参数
6. **P2（中期）**：考虑换用支持 thinking 分离的 vLLM 版本或配置

### 9.4 thinking 文本的潜在利用价值

虽然 thinking 导致了解析问题，但 thinking 内容本身包含有价值的信息：
- 模型对场景的逐步分析
- 实体识别和属性推理过程
- 自我纠正和质量检查

如果能正确分离 thinking 和 JSON，thinking 部分可以用于：
- 数据质量审计（检查模型推理是否合理）
- 训练数据增强（thinking 作为 CoT 样本）

---

*文档生成时间：2026-04-21*
*数据来源：commit 33388aa (v5.5 pipeline test run) + 实时 API 采样*
