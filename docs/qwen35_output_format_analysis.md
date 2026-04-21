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

*文档生成时间：2026-04-21*
*数据来源：commit 33388aa (v5.5 pipeline test run)*
