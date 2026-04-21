# Pipeline Test Run 问题分析

## 测试环境
- vLLM: 视觉模式, http://10.16.12.175:8000/v1
- 模型: /home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8 (本地路径)
- 视频: 30条, 60-180s, 6个数据集各5条
- 环境: /home/tione/notebook/gaozhenkun/hzh/envs/thinkstream

## 运行结果
- Pass 1 跑了约3.5小时, 完成5/30个视频的evidence
- 552个API请求完成, 全部200 OK
- 速度: ~2.8 req/min (视觉+thinking模式)

## 核心问题: Evidence Graph parse_success 极低 (3.2%)

### 统计
| 视频 | 总chunk | 成功 | 成功率 |
|------|---------|------|--------|
| 0ySyffWMAVI | 51 | 1 | 2% |
| EQnqHVzEM_A | 59 | 0 | 0% |
| xsFSYBJlrqA | 46 | 3 | 7% |
| ytb_FYz91NRoAuM | 50 | 0 | 0% |
| ytb_zgfJuuCbhyw | 45 | 4 | 9% |
| **总计** | **251** | **8** | **3.2%** |

### 根因: Qwen3.5 thinking 输出格式不匹配

代码预期的 thinking 格式:
```
<think>思考过程</think>
{"time": [0, 2], "visible_entities": [...], ...}
```

Qwen3.5 实际输出格式:
```
The user wants me to annotate a video chunk...

**Frame Analysis:**
- **Frame 1 (t=0s):** Shows a very bright...
...

{"time": [0, 2], "visible_entities": [...], ...}
```

模型没有使用 `<think>...</think>` 标签, 而是直接在 content 里输出自然语言思考过程, 然后才是 JSON。

### 代码问题定位

`pass1_evidence.py:parse_evidence_result()` 第99行:
```python
raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
```
这个正则完全没匹配到, 因为 Qwen3.5 不用 `<think>` 标签。

之后尝试 `json.loads(raw)` 失败(因为 raw 以 "The user wants..." 开头), 然后尝试找 `{` 开始的 JSON, 但 `_raw` 被截断到 200 字符, 可能截断了 JSON 部分。

### 少数成功的 chunk 分析

成功的 8 个 chunk 全部是空 JSON (所有字段为空列表/空字符串), 说明:
1. 模型对这些 chunk 没有输出思考过程, 直接返回了空 JSON
2. 或者思考过程很短, JSON 在 200 字符截断之前就开始了

## 需要修复的问题

### P0: thinking 输出解析
1. `parse_evidence_result` 需要处理 Qwen3.5 的实际 thinking 格式
2. 方案A: 不依赖 `<think>` 标签, 直接从 raw 中提取最后一个完整 JSON 对象
3. 方案B: 使用 vLLM 的 `chat_template` 参数让模型用 `<think>` 标签
4. 方案C: 在 prompt 中加 `/no_think` 前缀关闭 thinking (Pass 1 当前开着 thinking)

### P1: _raw 截断问题
`parse_evidence_result` 第109行 `default["_raw"] = raw[:200]` 截断太短, 导致无法事后分析完整输出。应该保留更多内容用于调试。

### P2: 模型名配置
`config.py` 的 `VLLM_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"` 和实际 vLLM 注册的模型名 `/home/tione/notebook/gaozhenkun/model/Qwen3.5-397B-A17B-FP8` 不匹配, 需要通过 `--model` 参数覆盖。

### P3: Pass 2 同样受影响
`pass2_rollout.py:parse_observation_result()` 和 `parse_compress_result()` 也有相同的 `<think>` 剥离逻辑, 虽然 Pass 2 用 `/no_think` 前缀, 但需要确认 Qwen3.5 是否真的响应 `/no_think`。

## 建议修复方案

最稳妥的方案: 修改所有 parse 函数, 不依赖 `<think>` 标签, 而是从 raw 输出中提取最后一个完整的 JSON 对象:

```python
def extract_json_from_raw(raw: str) -> Optional[dict]:
    """Extract the last complete JSON object from raw output."""
    # Find all { positions, try parsing from each (last first)
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

同时 Pass 1 可以考虑关闭 thinking (用 `/no_think`), 因为 evidence graph 是结构化标注, 不需要推理链。
