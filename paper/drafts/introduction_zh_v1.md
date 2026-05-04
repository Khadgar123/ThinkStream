# ChronoStream 正文中文稿与引用计划

标题：

**ChronoStream: Agentic Temporal Reasoning for Streaming Video Understanding**

本文档只保留当前要进入论文正文的中文稿、引用插入位置和参考文献分析。英文版本已对应写入 `paper/submission/main.tex`。

## 1. Introduction 中文正文稿

视频大语言模型已经显著推进通用视频问答、长视频理解和时序定位，但流视频理解仍然不同于离线视频推理。离线设置通常假设完整视频在回答前可见，模型可以先收集全局上下文，再定位证据并生成答案。流视频没有这一前提，视频内容按时间连续到达，query 可能在任意时刻出现，答案所需信息也可能位于 query 之前、query 发生时，或 query 之后。模型因此必须只依据已经到达的视频内容，在有限上下文和实时响应约束下持续维持对场景的时序理解。

为使视频模型在线运行，现有方法主要形成两类范式。响应控制方法通过输出状态或就绪度预测，使模型在连续观察中判断当前信息是否足以回答。状态维护方法则通过缓存窗口或层次记忆，将不断增长的视频历史收束到固定预算内，使长视频推理保持可计算。这些方法分别改善了回答时机和上下文规模，但通常仍把历史作为随时间前向写入的状态。若某段早期画面只有在后续 query 出现后才需要被精确使用，模型缺少按时间锚召回、重组并利用该片段的机制。

因此，强流视频模型需要学会 **thinking with time**，把时间从被动上下文转化为可操作的推理维度。第一，它应在固定预算内持续更新长期记忆，而不是依赖无限增长的视觉上下文。第二，它应将连续片段整合为保留来源的紧凑摘要，使长程语义能够在预算内递归更新。第三，它应允许模型在当前 query 需要早期关键线索时，按时间锚召回历史片段。这样的记忆应是有界的、来源可追溯的、时间可召回的长期状态。

基于这一观点，我们提出 **ChronoStream**，一个用于流视频理解的 agentic temporal reasoning 框架。ChronoStream 将前向流式状态扩展为可召回的时序记忆，并用同一条逐 chunk agent 轨迹协调当前观察、记忆整合、历史召回和答案生成。每一步中，模型基于当前视觉窗口、带时间锚的记忆和待回答 query 更新当前理解；当当前状态不足以支持回答时，模型可以发起带时间范围的历史召回；当记忆达到预算压力时，系统触发时段整合，由模型将指定时间范围写成来源可追溯的摘要；当证据充分时，模型对当前 query 输出答案。

本文贡献概括如下。第一，我们提出 ChronoStream，将流视频理解表述为 agentic temporal reasoning，使 VideoLLM 能在有限视觉窗口外维护可召回的时序工作记忆。第二，我们设计统一的逐 chunk agent 轨迹，将来源可追溯的时段整合、带时间锚的历史召回和 query 回答纳入同一生成过程。第三，我们将在 OVO-Bench、StreamingBench 等流视频基准上评估 ChronoStream，并从回答准确率、响应时机、长程记忆和效率维度验证其有效性。TODO: needs experiment.

## 2. Related Work 中文正文稿

**VideoLLMs and Streaming Video Understanding.** VideoLLM 已从图文对齐扩展到通用视频问答、长视频理解和时序定位。Video-MME、LongVideoBench、MVBench 等影响力文献说明当前模型已经能够处理复杂视频语义和长程时序线索。与这些通常在回答前读取完整视频的设置不同，StreamingBench 和 OVO-Bench 强调流视频中 query 的时间戳、实时理解、历史证据召回和主动响应能力。VideoLLM-online、LiveCC、StreamReady、Streamo 和 AURA 等方法进一步通过 streaming EOS、readiness 机制或统一响应状态建模回答时机。ChronoStream 沿着这一问题设置前进，但关注点不是单独的输出时机，而是回答前的时间记忆如何被维护、召回并参与推理。

**Streaming Context and Memory Management.** 流视频的长期运行要求模型在有限上下文、显存和延迟预算内保留历史。AURA 将这一问题表述为 video stream context management；StreamingVLM 和 StreamKV 通过 compact KV cache、attention sinks、近期视觉窗口或 segment-level cache 复用历史状态；FluxMem 和 StreamForest 将视频历史组织为层次视觉记忆或事件记忆；ThinkStream 和 VST 则把观察或推理轨迹转写为 streaming thoughts / textual semantic memory，以减少 query 到达后的推理延迟。这些方法解决的是流视频如何持续可计算，但历史内容通常由窗口、缓存、相似度压缩或前向摘要决定；当后续 query 需要某个早期短暂细节时，模型缺少按时间锚召回并重新组织历史片段的生成式接口。

**Agentic Reasoning with Tools and Memory.** 近期 agentic reasoning 研究开始将证据获取和记忆维护从外部流程转为模型生成轨迹的一部分。DeepEyes 和 DeepEyesV2 使模型在推理过程中交错执行视觉操作、代码执行或搜索，并把中间观察继续纳入后续推理。与此并行，MemAgent 将长文档视为连续片段流，并维护动态更新的固定长度 memory；ReMemR1 则在 memorize-while-reading 范式上加入 callback / memory retrieval，使模型学习何时召回、召回哪些历史记忆。上述工作共同说明，强推理模型不应只被动消费静态上下文，而应能主动操作推理所需的证据。ChronoStream 将这种证据操作范式放入流视频场景，其中证据具有时间锚，query 可能异步到达，模型需要在同一条 streaming trajectory 中协调连续观察、记忆整合、时间锚召回和答案生成。

## 3. 引用插入位置

| 正文位置 | 推荐 BibTeX key | 用途 |
|---|---|---|
| Introduction 第 1 段：离线视频理解基础 | `videomme2025,longvideobench2024,mvbench2024` | 支撑 VideoLLM / 长视频 / 视频问答背景。 |
| Introduction 第 1 段：流视频与离线视频差异 | `streamingbench2024,ovobench2025` | 支撑 query 时间戳、实时理解、历史/未来证据。 |
| Introduction 第 2 段：回答时机方法 | `videollm_online2024,livecc2025,streamready2026,streamo2026,aura2026` | 支撑 streaming EOS、readiness、response-state、统一流式交互。 |
| Introduction 第 2 段：上下文与记忆管理 | `streamingvlm2025,fluxmem2026,streamforest2025,thinkstream2026,vst2026` | 支撑 KV/cache、层次视觉记忆、语言化 streaming memory。 |
| Introduction 第 3 段：agentic reasoning / memory | `deepeyes2026,deepeyesv22026,memagent2025,rememr12026` | 支撑主动证据操作、固定长度记忆和 recall/callback。 |
| Related Work 三小节 | 同上按段复用 | 避免每句堆 citation，按范式引用代表文献。 |

## 4. 参考文献分析

| 方向 | 代表文献 | 这些文献怎么写 | 对 ChronoStream 的用法 |
|---|---|---|---|
| 流视频问题定义 | StreamingBench, OVO-Bench | 从 offline benchmark 的完整视频假设出发，强调 streaming query 可以在任意时间出现，并需要实时、历史和未来相关能力。 | 用于第一段建立任务差异。 |
| 回答时机控制 | VideoLLM-online, LiveCC, StreamReady, Streamo, AURA | 从 EOS、readiness 或 response-state 角度建模何时回答。AURA 进一步强调统一流式交互。 | 用于说明已有工作主要解决 `when to speak`。 |
| 流式上下文与记忆管理 | StreamingVLM, FluxMem, StreamForest, ThinkStream, VST | 从 compact KV cache、视觉 token 压缩、层次事件记忆或 textual semantic memory 控制成本和延迟。 | 用于说明已有工作主要解决 `how to keep the stream affordable`。 |
| Agentic 工具推理 | DeepEyes, DeepEyesV2 | 不把工具作为预处理，而是放入 reasoning trajectory，使模型主动获取并整合证据。 | 用于支撑 “time as actionable evidence” 的方法范式。 |
| 长上下文记忆 agent | MemAgent, ReMemR1 | MemAgent 使用动态固定长度 memory；ReMemR1 用 callback / memory retrieval 解决 forward-only memory 的证据遗漏。 | 用于支撑时间锚召回和可召回记忆。 |

## 5. BibTeX 状态

已将上述正文中使用的引用写入：

`paper/submission/references.bib`

已将英文 Introduction 和 Related Work 写入：

`paper/submission/main.tex`

当前 BibTeX 是工作稿级别：条目主要根据本地 PDF 首页、抽取文本、项目 README 的 citation block，以及已知会议/arXiv 信息手工整理，能够通过 LaTeX 编译，但还不是最终相机版引用库。终稿前需要逐条替换或核验为官方 BibTeX，优先来源为 ACL Anthology、CVF Open Access、OpenReview / ICLR、NeurIPS proceedings、arXiv official metadata 和项目 README 中作者提供的 citation。当前风险主要有三类：部分 2025/2026 arXiv 论文的年份应按首版、最新版还是会议版写法统一；部分待正式 proceedings 的论文暂时只能写 arXiv；部分长作者列表使用 `and others`，终稿前应视 venue 要求展开或保留。

## 6. 流视频相关工作备忘：两类范式如何具体工作

这一节用于连接 Related Work 和 Figure 1。它不作为正文直接粘贴，而是帮助我们把已有方法抽象成可画、可比较的机制。现有流视频方法并不是简单的若干模块堆叠，而是大体围绕两个可计算对象展开：一类把 **输出状态** 作为核心变量，解决 `when to speak`；另一类把 **有界上下文** 作为核心变量，解决 `how to sustain the stream`。ChronoStream 的落点应当是第三种对象：把带时间锚的长期记忆变成模型可以维护、整合、召回并用于回答的推理状态。

### 6.1 Response-Control Paradigm：输出状态是决策对象

图中总结句：

> predicts whether to emit; history is consumed as context rather than operated on.

这一类方法把流视频理解转化为逐时间步的发言决策。视频帧或 chunk 连续进入系统，模型在每个时刻根据当前观察和 query 判断是否已经具备输出条件。如果条件不足，系统继续观察；如果条件满足，则生成 answer、caption 或 commentary。具体实现通常有两种代表形式：一种使用独立的 trigger / readiness 模块监控视频流，到达阈值后再调用生成模型；另一种把 Silence、Standby、Response 或 streaming EOS 等状态写入自回归序列，使“何时输出”和“输出什么”由同一模型生成。

这一范式的优势是直观且有效：它直接惩罚过早输出和延迟输出，使流视频助手能够在事件完成、证据出现或用户提问后及时响应。但它的决策对象主要是用户侧输出状态。回答前的历史如何进入上下文、如何被压缩、如何在后续 query 到来时重新使用，通常仍由滑动窗口、缓存、外部记忆或前向摘要决定。换言之，这类方法学习的是“现在该不该说”，而不是“说之前应当怎样操作历史”。

Figure 1(a) 可以抽象成：

`video stream + query -> response-state / readiness decision -> keep watching or answer`

图中核心模块：

- timeline 上连续到达的视频 chunk；
- query 出现后，每个时间点连接到 `silent / standby / response` 或 `ready?`；
- 触发后生成 answer / commentary；
- 底部标注 `controls when to speak; temporal evidence remains passive`。

### 6.2 Streaming Context Management Paradigm：有界上下文是维护对象

图中总结句：

> compresses an unbounded stream into a bounded state; later queries read that state as given.

这一类方法的核心问题不是何时输出，而是视频流持续变长后如何让上下文、显存和延迟保持可承受。系统通常把不断到达的视频历史折叠成一个固定预算内的状态。低层实现可以是 KV/cache 复用、视觉 token 剪枝或聚合；高层实现可以是层次事件记忆、视觉记忆树或语言化摘要。无论表示形式不同，基本流程都是相似的：新片段进入，近期细节被保留，较早历史被合并、降采样、摘要或移入长期状态，query 到来时再读取当前保留下来的有界状态。

这一范式解决了流式运行的底层瓶颈，使模型不会随着视频长度增长而线性堆积全部视觉 token。但它的历史状态多由固定策略或当时的局部判断向前维护，query 到来时只能使用已经形成的表示。如果某个早期短暂细节在当前时刻未被保留为可定位内容，后续 query 需要它时，模型往往只能依赖压缩后的粗粒度状态，而缺少由模型生成的时间定位、召回和重新组织过程。

Figure 1(b) 可以抽象成：

`video stream -> window / cache / token aggregation / summary -> bounded context -> answer`

图中核心模块：

- timeline 上旧帧不断进入状态维护通道；
- sliding window 保留最近视觉细节；
- cache、token clusters、event memory 或 summary 表示长期历史；
- query 到达时读取当前 bounded context；
- 底部标注 `bounds cost; early details may become coarse or inaccessible`。

### 6.3 ChronoStream：时间记忆是可操作推理状态

图中总结句：

> manages time before answering: maintain, consolidate, recall, and answer in one trajectory.

ChronoStream 不应画成“响应控制 + 记忆压缩”的拼接，而应画成一个统一的逐 chunk agent trajectory。每一步中，同一模型同时看到当前视觉窗口、带时间锚的长期记忆和待回答 query，并生成结构化的内部操作或答案。当前窗口负责近处细节，长期记忆负责跨窗口状态，时段整合把指定时间范围写成来源可追溯的摘要，时间锚召回则使模型在 query 需要早期证据时重新取回对应历史。

因此，ChronoStream 的对比点不是单纯“也会压缩”或“也会判断何时回答”，而是把流视频中的历史状态变成一个可由模型操作的推理对象。它仍然保持有限视觉窗口和固定记忆预算，但不会把回答前的历史处理完全交给被动缓存或一次性摘要；模型可以在同一条生成轨迹中维护记忆、整合时段、召回历史，并在证据充分后生成 query 的答案。

Figure 1(c) 可以抽象成：

`current window + temporal memory + query -> agent trajectory -> consolidate / recall / answer`

图中核心模块：

- 当前视觉窗口；
- 带时间锚的 bounded temporal memory；
- source-linked interval summary；
- time-anchored recall arrow 指回历史片段；
- agent box 同时接收 current window、temporal memory、query 和 recalled evidence；
- 底部标注 `maintains and recalls temporal memory before answering`。

## 7. 两类流视频范式的逐篇机制备忘

这一节用于你快速理解每篇论文到底做了什么，以及它们为什么被放进 Figure 1 的两类对比中。这里不是最终 Related Work 文本，而是写作备忘：先讲机制，再讲核心模块，最后讲和 ChronoStream 的关系。

### 7.1 Response-Control Paradigm：学习何时输出

**VideoLLM-online.** 这篇可以视为早期 streaming VideoLLM 的代表。它的核心问题是把离线 VideoLLM 改成能在视频播放过程中交互的在线模型，而不是等完整视频结束后再回答。方法上，它把离线标注转成 streaming dialogue data，并引入 streaming EOS 训练目标：模型在每个时间步要么生成 EOS 表示继续观察，要么生成文本表示当前应该响应。核心模块包括 2 FPS 左右的流式视觉编码、由离线数据合成的 streaming dialogue、streaming EOS loss，以及视频编码、LLM forward、文本生成并行化的实时推理框架。它适合放在 Figure 1(a) 作为“token-driven response control”的基础例子，因为它学习的是逐帧是否输出，而不是显式维护可召回的历史记忆。

**LiveCC.** LiveCC 更偏实时视频解说和大规模流式语言监督。它利用 streaming speech transcription / ASR 构造大规模 video-ASR 训练数据，使 VideoLLM 学会在视频播放过程中持续输出 commentary。它的关键不是复杂的问答推理，而是把视频帧和按时间对齐的语言转写组织成流式训练信号，从而降低构建 streaming supervision 的成本。核心模块包括 Live-CC-5M 预训练数据、Live-WhisperX-526K 指令数据、streaming commentary 训练格式和实时生成评测。它可以放入 response-control 范式，是因为模型需要决定何时继续静默、何时发出解说文本；但在我们的叙事中应谨慎表述，它主要证明了流式语言监督可扩展，而不是专门解决历史证据召回。

**StreamReady.** StreamReady 直接把 streaming QA 中的回答时机作为中心问题。它指出 proactive setting 中 query 可能先于证据出现，模型不能只追求答对，还必须在证据窗口出现时回答。方法上，它提出 Answer Readiness Score，把早答和晚答用非对称惩罚纳入 effective accuracy；模型内部加入 readiness token 和 readiness head，用分数判断当前是否已经看到足够证据。核心模块包括 answer evidence window、ARS 指标、readiness token/head、ProReady-QA 数据集，以及长短期视觉推理结构。它适合放在 Figure 1(a) 的 `ready? -> answer` 路线中，因为它最清楚地代表“证据是否足够输出”的判别式时机控制。

**Streamo.** Streamo 的重点是把多种 streaming video task 统一成 instruction tuning。它不再用外部 trigger 单独判断时机，而是在自回归输出中显式建模 Silence、Standby、Response 三类状态，使“何时说”和“说什么”由同一模型在一个序列中生成。核心模块包括 Streamo-Instruct-465K、多任务时间边界标注、response state tokens、focal-weighted loss，以及实时 narration、action caption、event caption、event grounding、time-sensitive QA 等任务格式。它和我们较接近，因为它已经是端到端状态生成；但它的状态主要描述输出粒度和响应时机，历史如何被重新组织或按时间召回不是它的核心接口。

**AURA.** AURA 是一个更系统化的 always-on streaming visual interaction 框架。它先把现有方法分成 decoupled trigger-response pipeline 和 unified architecture，然后提出端到端 streaming visual assistant，同时支持实时问答和 proactive response。核心模块包括 Interactive Video Stream Context Management、Coarse-to-Fine Data Engine、Silent-Speech Balanced Loss 和 Real-time Streaming Inference Framework。AURA 横跨 response control 和 context management：它既关心 silent/response 的长期稳定，也关心 unbounded video-text input 如何放进有限上下文。Figure 1 中如果只画两类对比，可以把它放在 (a) 和 (b) 的边界处；在正文中更适合作为“统一流式交互系统”的代表，而不是我们要否定的简单 gate 方法。

### 7.2 Streaming Context Management Paradigm：维护可承受的历史状态

**StreamingVLM.** StreamingVLM 的问题设定是 infinite video streams 下如何保持实时、稳定和低延迟。它批评 full attention 会带来二次复杂度，普通 sliding window 会破坏连贯性或引入重复计算，因此提出 streaming-aware KV cache：保留 attention sinks、短视觉窗口和较长文本窗口，并用 contiguous RoPE 保持位置一致。训练上，它用短重叠 chunk 的 full attention SFT 模拟推理时的注意力模式。核心模块包括 attention sink、short vision window、long text window、KV reuse、contiguous RoPE 和 overlapped-chunk SFT。它适合放在 Figure 1(b) 的 `KV/cache + sliding window` 路线中，因为它解决的是无限流的效率和训练推理对齐，而不是 query 到来后由模型主动召回过去时间段。

**FluxMem.** FluxMem 是训练无关的视觉记忆压缩方法，关注流视频中视觉 token 如何分层保留。它把视觉历史组织成 short-term、mid-term、long-term 三层记忆：近期信息完整保留，中期通过 Temporal Adjacency Selection 去除相邻帧冗余，长期通过 Spatial Domain Consolidation 合并空间重复区域，并用自适应阈值根据场景统计决定压缩强度。核心模块包括 hierarchical memory、TAS、SDC、自适应压缩阈值和视觉 token budget。它适合放在 Figure 1(b) 的 `token aggregation / visual memory` 路线中，因为它让历史更紧凑，但压缩过程主要由视觉冗余和预算驱动，而不是由后续 query 生成式地重新组织历史。

**StreamForest.** StreamForest 把流视频历史组织为 Persistent Event Memory Forest。它不是简单保留帧，而是把连续视频聚合成多个事件级树结构，并用时间距离、内容相似度和合并次数等 penalty 控制节点合并；同时保留 Fine-grained Spatiotemporal Window 捕捉当前场景细节。核心模块包括 persistent event memory forest、event node merging、penalty-guided compression、fine-grained spatiotemporal window、OnlineIT 数据和 ODV-Bench。它适合放在 Figure 1(b) 的 `hierarchical event memory` 路线中，因为它对长程历史建模比普通 cache 更语义化，但历史仍主要作为事件记忆被向前维护，query 触发的时间锚召回不是核心生成动作。

**ThinkStream.** ThinkStream 已经不只是传统上下文压缩，它提出 Watch-Think-Speak，使模型在 chunk 到达时生成 `<think>`，再输出 `<silent>` 或 `<response>`。它的 Reasoning-Compressed Streaming Memory 将中间 reasoning trace 作为 compact semantic memory，用文本思考替代被淘汰的视觉 token，从而降低长视频流的视觉上下文成本，并用 streaming RLVR 优化结构化 reasoning、回答正确性和响应时机。核心模块包括 Watch-Think-Speak、streaming thoughts、RCSM、silent/response 动作和 verifiable reward。它和 ChronoStream 非常接近，因此不能被简单说成“只做压缩”。我们更稳的差异应写成：ThinkStream 主要把历史转写为前向累积的 reasoning memory，而 ChronoStream 进一步把带时间锚的长期记忆变成可整合、可召回、可用于 query 解答的 agent trajectory。

**VST.** VST 的核心观点是 VideoLLM 可以 watch and think simultaneously，把原本 query 之后才发生的 CoT 推理前置到视频播放过程中，从而摊销回答延迟。它把 streaming thinking 建模为多轮对话，模型观察视频 clip 时写入 textual thoughts 到外部 memory，并通过 VST-SFT 和 VST-RL 学习因果流式推理。核心模块包括 thinking while watching、streaming CoT、textual semantic memory、VST-SFT、VST-RL 和基于 video knowledge graph 的训练数据合成。它适合放在 Figure 1(b) 和 Figure 1(c) 的过渡处：它已经强调语言化时序记忆和端到端 RL，但重点仍是持续思考和延迟摊销；ChronoStream 要突出的是 query 需要早期证据时的时间锚召回和时段整合，而不是单纯更早生成 thought。

### 7.3 对 Figure 1 的归纳

如果 Figure 1 只允许三块，建议不要在图中列出所有论文名，而是画机制：

**(a) Response-Control Streaming.** 画一条视频时间线，query 到达后每个时刻输出 `silent / standby / response` 或 `ready?`，最后触发 answer。旁边标注代表机制：streaming EOS、response-state token、readiness head。

**(b) Streaming Context Management.** 画同一条视频时间线，旧帧进入 window、KV/cache、token clusters、event memory 或 summary，query 到达时读取 bounded context。旁边标注代表机制：KV reuse、visual token aggregation、hierarchical/event memory、textual semantic memory。

**(c) ChronoStream.** 画当前窗口、带时间锚的 bounded temporal memory、source-linked interval summary、time-anchored recall arrow 和 agent box。重点不是画更多模块，而是表现同一模型在一条 trajectory 中维护记忆、整合时段、召回历史并生成 query 答案。
