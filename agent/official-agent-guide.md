# Codex、Claude Code、LangChain、OpenCode 与 DeepSeek Harness 官方指南

调研与核验日期：2026-09-02

> 本文只用厂商官网、官方文档站和官方 GitHub 组织下的仓库解释各产品能力。文中的“设计建议”和“五层 Harness”是基于这些资料做的横向归纳，不是任何一家厂商的原文。产品迭代很快，涉及版本、默认权限和实验功能时，应点击原文复核。

## 先读结论

这五个名字并不处在同一层：

- **Codex、Claude Code、OpenCode** 是开箱即用的编码智能体。它们已经提供代码搜索、文件修改、命令执行、权限控制、项目指令、技能和子智能体等完整工作环境。
- **LangChain** 是 Agent 框架，**LangGraph** 是有状态编排运行时，**Deep Agents** 是在二者之上的成品 Harness。它们适合把 Agent 嵌入自己的产品。
- **DeepSeek Harness（`dsh`）** 是 DeepSeek 官方的插件化 Agent Harness。其卖点不是某个固定 Agent，而是“所有部分都可替换”的 Cordis 插件架构。目前仍是开发者预览。

如果目标是“更好地使用 Agent”，先精通 Codex 或 Claude Code；如果目标是“设计自己的 Agent 产品”，再学习 LangGraph/Deep Agents 和 DeepSeek Harness 的架构。

## 一张表建立全局认识

| 对象 | 它是什么 | 项目指令/记忆 | 扩展方式 | 多智能体 | 更适合 |
| --- | --- | --- | --- | --- | --- |
| [Codex](https://learn.chatgpt.com/docs/codex/cli) | OpenAI 编码智能体与 CLI/SDK | `AGENTS.md` 分层加载 | Skills、MCP、Hooks、SDK、App Server | Subagents，主 Agent 汇总结果 | 日常研发、自动化、基于 Codex 二次开发 |
| [Claude Code](https://code.claude.com/docs/en/overview) | Anthropic 编码智能体 | `CLAUDE.md`、Rules、Auto memory | Skills、MCP、Hooks、Plugins、Agent SDK | Subagents；实验性 Agent Teams | 日常研发、长任务、团队并行研究/评审 |
| [LangChain](https://docs.langchain.com/oss/python/langchain/agents) | Agent 构建框架 | Agent state、短期/长期 memory | Tools、Middleware、MCP、LangGraph | Subagents、Handoffs、Skills、Router 等模式 | 构建业务 Agent 和可控工作流 |
| [OpenCode](https://opencode.ai/docs/) | 开源编码智能体，TUI/客户端-服务器架构 | `AGENTS.md`、instructions | Agents、Skills、MCP、Plugins、Custom tools、SDK | Primary agents + Subagents | 自托管、可换模型、深度定制编码助手 |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | DeepSeek 官方插件化 Harness | 会话事件日志、持久化插件、可接记忆 MCP | Cordis Service/Event/Plugin、Profile、Patch | Subagent 与实验性 Agent Teams 子系统 | 研究插件化 Harness、开发可替换的 Agent 平台 |

## Agent 与 Harness 到底是什么

LangChain 官方给出了很实用的定义：[Agent 是“模型循环调用工具，直到任务完成”](https://docs.langchain.com/oss/python/langchain/agents)；Harness 是包在循环外面的 prompt、tools、middleware 等运行设施。可以把完整系统理解为：

```text
Agent = Model + Agent Loop

Harness = Instructions + Context + Tools + State/Memory
        + Permissions + Verification + Lifecycle + Observability
```

模型决定推理能力，Harness 决定它在真实工程里能否稳定完成任务。一个强模型配上混乱上下文、过宽权限和没有测试的 Harness，仍然会不可靠。

## 1. Codex：从正确使用到二次开发

### 官方一键入口

- [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)：安装、交互、恢复会话、图片输入和常用能力总览。
- [用 `AGENTS.md` 提供项目指令](https://learn.chatgpt.com/docs/agent-configuration/agents-md)：全局、仓库、子目录指令的发现和覆盖规则。
- [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)：内置角色、自定义角色、并发与权限。
- [Agent Skills](https://learn.chatgpt.com/docs/build-skills)：按需加载的可复用流程、脚本和参考资料。
- [MCP](https://learn.chatgpt.com/docs/extend/mcp)：连接外部工具、数据和服务。
- [Hooks](https://learn.chatgpt.com/docs/hooks)：在会话、工具调用、审批、压缩和子智能体事件上运行确定性逻辑。
- [安全、沙箱与审批](https://learn.chatgpt.com/docs/agent-approvals-security)：区分“技术上能做什么”和“何时需要询问”。
- [非交互模式 `codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode)：用于脚本和 CI，支持 JSONL 和结构化输出。
- [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk) 与 [App Server](https://learn.chatgpt.com/docs/app-server)：把同一个 Codex Agent 嵌入自己的程序或客户端。
- [OpenAI 官方 Codex 仓库](https://github.com/openai/codex)：源码、Issue 和发布信息。

### 推荐使用路径

1. 在仓库根目录运行 Codex，先让它只读探索项目，要求输出证据和计划。
2. 把反复说明的构建命令、目录边界、编码规范、验证门槛写入仓库的 `AGENTS.md`。
3. 把只对某个子目录生效的规则放到该目录的 `AGENTS.md` 或 `AGENTS.override.md`，不要让根指令无限膨胀。
4. 把“有固定步骤、参考资料或脚本”的工作提炼成 Skill；把外部系统能力接成 MCP。
5. 对独立、读多写少、会产生大量上下文的任务使用 Subagent；多个 Agent 同时改同一个文件时要谨慎。
6. 用测试、lint、build、截图比较或独立评审 Agent 构成验证闭环，再考虑无人值守的 `codex exec`。

### 最值得理解的机制

**项目指令是分层上下文，不是强制安全边界。** Codex 从全局配置开始，再从仓库根目录沿当前工作目录向下加载项目指令；更接近当前目录的内容后出现并可覆盖前面的内容。规则应短、具体、可验证。真正需要阻止的行为应交给沙箱、审批、权限和 Hooks。

**Skill 用渐进式披露节约上下文。** 启动时只暴露技能名称与描述，匹配任务后才加载完整 `SKILL.md`；脚本、模板和长参考资料也只在需要时读取。不要把所有领域知识塞进 `AGENTS.md`。

**Subagent 的首要价值是上下文隔离。** 搜索结果、日志和长文件留在子上下文，只把结论交回主 Agent。适合并行研究、代码库探索和独立评审，不适合强顺序依赖或高冲突写入。

**沙箱和审批是两个维度。** 沙箱规定进程能够访问的文件、网络和系统能力；审批策略规定何时暂停请求人类许可。自动模式并不等于无限权限，完全访问模式也不应在不可信仓库或宿主机上使用。

### 从 Codex 学习设计自己的 Agent

- 用层级指令解决 monorepo 的局部上下文问题。
- 用 Skills 把“发现能力”和“加载完整能力”分开。
- 把读写权限、联网能力和人工审批设计成可组合而非单一开关。
- 为自动化提供事件流、结构化输出、恢复 thread 和明确的失败语义。
- Hooks 是确定性护栏，但不能代替操作系统级隔离。

## 2. Claude Code：重点理解上下文、验证和团队协作

### 官方一键入口

- [Claude Code 概览](https://code.claude.com/docs/en/overview) 与 [最佳实践](https://code.claude.com/docs/en/best-practices)。
- [Claude Code 的工作原理](https://code.claude.com/docs/en/how-claude-code-works)：Agent loop、工具和上下文的基础模型。
- [`CLAUDE.md` 与 Auto memory](https://code.claude.com/docs/en/memory)：持久指令、路径规则和自动记忆。
- [Subagents](https://code.claude.com/docs/en/sub-agents)：为专门任务定义独立上下文、模型、工具和权限。
- [Agent Teams](https://code.claude.com/docs/en/agent-teams)：Lead、Teammate、共享任务和直接通信。
- [Skills](https://code.claude.com/docs/en/skills)、[MCP](https://code.claude.com/docs/en/mcp)、[Hooks 指南](https://code.claude.com/docs/en/hooks-guide) 和 [Plugins](https://code.claude.com/docs/en/plugins)。
- [权限](https://code.claude.com/docs/en/permissions) 与 [沙箱](https://code.claude.com/docs/en/sandboxing)。
- [非交互模式](https://code.claude.com/docs/en/headless) 与 [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)。

### 官方最佳实践的核心

Claude Code 官方把**上下文窗口**视为最重要的资源：对话、读取的文件和命令输出都会占用上下文，越接近上限越容易遗漏早期信息。有效做法是：

- 给 Agent 一个能自己执行的验证条件，例如测试、构建、lint、固定样例或截图对比。
- 按“探索 → 计划 → 实现 → 验证/提交”拆阶段，先在 Plan mode 中理解问题。
- Prompt 说明目标、相关路径、约束和验收标准，但不要替 Agent 预写所有实现步骤。
- 及时纠偏；需要保留探索结果时交给 Subagent，主上下文只保留决策。
- 要求展示验证证据，而不是只说“已完成”。

### `CLAUDE.md`、Rules、Memory、Skill 怎么分工

| 机制 | 放什么 | 不应放什么 |
| --- | --- | --- |
| `CLAUDE.md` | 每次会话都应知道的构建命令、架构约束、团队规范 | 很长的教程、偶尔才用的流程、强制安全策略 |
| `.claude/rules/` | 仅对指定目录或文件类型生效的规则 | 全仓库通用规则的重复副本 |
| Auto memory | Claude 从纠正和偏好中积累的跨会话经验 | 必须由团队审查和强制执行的政策 |
| Skill | 按需加载的专业流程、参考资料、脚本和模板 | 每次会话都必须加载的最小事实 |
| Hook/Permission | 必须确定性阻止、审批或审计的动作 | 只靠自然语言就能清楚表达的偏好 |

官方建议让单个 `CLAUDE.md` 尽量简洁（文档给出的目标是 200 行以内），并定期清理冲突和过时规则。它是上下文，不是强制执行机制。

### Subagent 与 Agent Teams 不要混用概念

| 维度 | Subagent | Agent Teams |
| --- | --- | --- |
| 运行关系 | 在一次主会话内由主 Agent 委派 | 多个独立 Claude Code 会话组成团队 |
| 通信 | 主要把最终结果交回调用者 | Teammate 可彼此直接通信，并使用共享任务列表 |
| 成本 | 较低，重工作被压缩后回传 | 较高，每个 Teammate 都是独立实例 |
| 适用 | 聚焦探索、搜索、分析、独立检查 | 需要讨论、挑战假设、跨层并行协作的复杂任务 |
| 避免 | 需要持续互相协商的任务 | 顺序任务、多人改同一文件、依赖很密的任务 |

截至核验日，[Agent Teams](https://code.claude.com/docs/en/agent-teams) 仍标为实验功能，默认关闭，需要设置 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`；它只在交互式会话中生成 Teammate，非交互 `-p`/Agent SDK 运行不会生成团队成员。官方也明确提醒它会显著增加 token 成本。

实用的团队任务应满足三个条件：可以独立推进、边界清晰、合并点清晰。例如三人分别做架构调查、风险审查和验证方案；不宜让三人同时改同一个核心文件。

### 从 Claude Code 学习设计自己的 Agent

- 让 Agent 拥有可运行的完成判据，而不是依赖主观“看起来完成”。
- 主上下文是稀缺资源；用子上下文隔离探索，用摘要保留决策。
- 团队协作需要身份、任务状态、消息通道、所有权和退出协议，不能只做并行调用。
- 将指令、记忆、权限和 Hook 分层；“建议做什么”和“绝不能做什么”不是同一种机制。

## 3. LangChain、LangGraph 与 Deep Agents：构建自己的 Agent

### 先分清三层

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)：高层 Agent 框架，提供模型、工具、system prompt、structured output、state 和 middleware。
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)：低层、有状态的编排运行时，重点是 durable execution、streaming、persistence 和 human-in-the-loop。
- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)：基于 LangChain/LangGraph 的成品 Harness，内置文件系统、上下文管理、规划、Subagents 和人工介入。
- [LangSmith Observability](https://docs.langchain.com/langsmith/observability)：追踪运行、调试工具调用和观察 Agent 行为。

选择原则：只需常见 tool-calling loop，使用 `create_agent`；需要精确状态机、持久恢复和确定性/智能步骤混编，使用 LangGraph；希望直接获得较完整的 coding/research Harness，优先评估 Deep Agents。

### 最小 Agent 和真正可用的 Harness

LangChain 的 `create_agent(model, tools, ...)` 会运行“模型选择工具 → 执行工具 → 把结果交回模型”的循环，直到模型不再调用工具。生产系统还需要补上：

- structured output，避免下游解析自然语言；
- checkpointer 保存单个 thread 的短期状态；
- store 保存跨 thread 的长期记忆；
- middleware 做日志、重试、模型回退、限流、PII 检查、摘要和人工审批；
- LangSmith trace/evaluation 观察成功率、错误路径、成本和延迟。

官方资料：[`create_agent` 与 Harness 配置](https://docs.langchain.com/oss/python/langchain/agents)、[Middleware](https://docs.langchain.com/oss/python/langchain/middleware)、[短期记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory)、[长期记忆](https://docs.langchain.com/oss/python/langchain/long-term-memory)、[Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)。

### Multi-agent 不是默认答案

[LangChain Multi-agent 官方指南](https://docs.langchain.com/oss/python/langchain/multi-agent) 明确指出：很多复杂任务用一个具有动态工具和正确 prompt 的 Agent 就能完成。真正需要 Multi-agent 的常见原因是：

- **上下文管理**：不同专业知识不能同时塞入一个窗口；
- **团队边界**：不同团队独立开发和维护能力；
- **并行化**：子任务可以并发推进；
- **工具过载**：工具太多导致主模型路由质量下降。

官方归纳的主要模式包括：

- **Subagents**：主 Agent 把子 Agent 当工具，路由集中、结果回主 Agent；
- **Handoffs**：状态变化触发角色、prompt 或工具集合切换；
- **Skills**：仍是单 Agent，只按需加载知识和流程；
- 更复杂的 Router 或自定义 LangGraph 工作流。

先问“谁应该看到哪些上下文、谁拥有最终决策、状态在哪里保存”，再选择模式。Multi-agent 的核心不是 Agent 数量，而是上下文工程和控制流。

### Deep Agents 已经内置什么

[Deep Agents 概览](https://docs.langchain.com/oss/python/deepagents/overview) 将 Harness 能力分成四组：

1. **执行环境**：自定义工具、MCP、虚拟文件系统、可插拔后端、文件权限、可选沙箱和代码解释器。
2. **上下文管理**：Skills、`AGENTS.md` memory、摘要、长工具结果卸载、长期存储和部分模型的 prompt cache。
3. **委派**：可选 todo 规划，以及用内置 `task` 工具启动一次性 Subagent。
4. **Steering**：在敏感工具调用前通过 LangGraph interrupt 暂停并接受批准、修改或拒绝。

Deep Agents 的 Subagent 每次获得新上下文，独立运行，最后只返回一次结果；默认是无状态的，不适合需要持续对话的团队协作。文件权限仅约束 Harness 文件工具；如果开放了任意 shell/sandbox 执行，还必须在执行环境层单独隔离。

### 从 LangChain 体系学习设计自己的 Agent

- 把 LLM 决策步骤和确定性业务步骤放进同一张可审计的状态图。
- 中断时持久化状态，使审批、故障和长任务可以安全恢复。
- Middleware 是 Agent loop 的扩展面，适合跨工具的限流、重试、摘要、审计与策略。
- 记忆至少分 thread 内短期状态和跨 thread 长期存储，二者不能混为一谈。
- 在创建 Multi-agent 前先解决 context routing；有时一个 Skill 或动态工具集更简单、更稳定。

## 4. OpenCode：开源、可换模型的编码智能体

### 官方一键入口

- [OpenCode 文档](https://opencode.ai/docs/) 与 [官方 GitHub 仓库](https://github.com/anomalyco/opencode)。
- [CLI](https://opencode.ai/docs/cli)：TUI、`run`、`serve`、session、MCP、ACP 等命令。
- [配置](https://opencode.ai/docs/config)：JSON/JSONC、多层配置合并和覆盖顺序。
- [`AGENTS.md` Rules](https://opencode.ai/docs/rules)：项目指令、初始化和兼容指令来源。
- [Agents](https://opencode.ai/docs/agents)：Primary agents、Subagents 与自定义 Agent。
- [Tools](https://opencode.ai/docs/tools)、[Permissions](https://opencode.ai/docs/permissions) 和 [Custom tools](https://opencode.ai/docs/custom-tools)。
- [Skills](https://opencode.ai/docs/skills)、[MCP servers](https://opencode.ai/docs/mcp-servers) 和 [Plugins](https://opencode.ai/docs/plugins)。
- [Server](https://opencode.ai/docs/server) 与 [JavaScript/TypeScript SDK](https://opencode.ai/docs/sdk)。

### 使用思路

OpenCode 默认启动 TUI，也可用 `opencode run` 非交互运行。它采用客户端-服务器结构：TUI 是客户端，后端暴露 OpenAPI；`opencode serve` 可以单独启动无头服务器，SDK 可以启动或连接服务器。因此它很适合自建界面、编辑器集成或服务化调用。

OpenCode 的配置来源会合并而不是整份替换，组织远程配置、全局配置、自定义配置、项目 `opencode.json` 和 `.opencode` 目录按优先级叠加。设计团队配置时要检查最终合并值，而不是只看仓库文件。

### Agent、Skill、Tool、Plugin 的分工

| 扩展 | 作用 |
| --- | --- |
| Primary agent | 用户直接切换的主角色；内置 Build 和只读倾向的 Plan |
| Subagent | 由主 Agent 自动委派或用 `@` 手动调用的专门角色 |
| Skill | 通过 `SKILL.md` 按需加载的流程与知识，兼容 `.opencode`、`.claude`、`.agents` 路径 |
| Custom tool | 用 TypeScript/JavaScript 定义模型可调用函数，内部可调用其他语言脚本 |
| MCP server | 通过标准协议连接外部工具；工具描述也会占用上下文 |
| Plugin | 监听事件、修改行为、注入环境或增加工具的运行时扩展 |
| SDK/Server | 以 API 方式管理 session、消息、文件、Agent 和事件 |

OpenCode 将权限动作解析为 `allow`、`ask` 或 `deny`，支持工具级和模式级规则。`--auto` 只自动批准原本需要询问的动作，显式 `deny` 仍然生效。不要因为使用 auto mode 就省略 deny 规则。

### 从 OpenCode 学习设计自己的 Agent

- 客户端和 Agent 运行服务器解耦，便于 TUI、Web、IDE 共用同一能力。
- 配置采用可解释的分层合并，并提供 schema，适合组织、用户和项目三级治理。
- Agent、Skill、Tool、Plugin 是不同扩展层，应保持清晰边界。
- MCP 工具列表本身有上下文成本；需要按 Agent 启用，而不是全局堆积。

## 5. DeepSeek Harness：研究“一切皆插件”的 Agent 架构

### 先看状态和边界

[DeepSeek 官方仓库](https://github.com/deepseek-ai/deepseek-harness) 将 DeepSeek Harness（`dsh`）定义为开源 Agent Harness，底层由 Cordis 驱动，采用“Everything is a Plugin”架构。它与 LangChain 的 Deep Agents 不是同一个项目。

截至核验日，官方仍把它标为**开发者预览**，并警告未来会有破坏兼容性的变更。[官方安全说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.zh.md) 还指出项目尚未经过安全审计，能够执行模型生成的命令、加载第三方插件并访问被开放的文件、网络、进程和凭据；沙箱、审批和权限只能降低风险，不能保证隔离。实验时优先使用一次性虚拟机、容器或专用环境，并坚持最小权限。

### 官方一键入口

- [中文 README 与运行方法](https://github.com/deepseek-ai/deepseek-harness/blob/master/README.zh.md)：`npx @deepseek-ai/dsh web` 或从源码构建。
- [官方文档站](https://deepseek-harness.github.io/deepseek-harness/) 与 [Web UI 指南](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/index.zh.md)。
- [架构总览](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md)：Cordis、Profile、Bundle、Patch、Agent loop 与能力 seam。
- [Cordis 入门](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.zh.md) 与 [进入 Harness 的插件教程](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-tutorial/07-into-the-harness.zh.md)。
- [第一个插件](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/index.zh.md)、[开发工具](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/tool.zh.md) 和 [插件配置](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/config.zh.md)。
- [Service Definition / Provider / Consumer 三角色设计](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/practice/index.zh.md)。
- [Subagent 子系统](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.zh.md)、[实验性 Agent Teams 子系统](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/agent-team.zh.md)。
- [审批子系统](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/approval.zh.md)、[权限预设](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/permission-presets.zh.md) 和 [会话持久化](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.zh.md)。
- [Python SDK 指南](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/python-sdk.zh.md) 与 [连接记忆 MCP 的官方示例](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/mcp-memory.zh.md)。

### 最小体验

安装 Node.js 后可直接启动：

```bash
npx @deepseek-ai/dsh web
```

默认 Web UI 地址是 `http://127.0.0.1:3080`。在设置中配置模型，选择工作区后即可让 Agent 读取/编辑文件、运行命令、委派工作和维护计划。正式试验前先阅读安全说明；不要把包含敏感凭据的宿主目录直接暴露给它。

### Cordis 的五个核心概念

1. **Plugin**：实现能力的对象或 `Service` 子类，加载和卸载由框架管理。
2. **Context**：服务容器；消费者通过稳定的 `ctx.<key>` 找服务，而不是导入具体实现。
3. **Dependency injection**：插件用 `inject` 声明依赖，依赖未就绪时等待，不手写启动顺序。
4. **Typed events**：服务用 `emit`、`waterfall`、`parallel`、`serial`、`bail` 等明确语义通信。
5. **Reversible effects**：工具、prompt 片段、adapter、provider、listener 的注册都应有 disposer，支持 reload 和 teardown。

这使模型适配器、工具注册表、会话日志、Agent loop 本身都可由配置替换，而不是修改一个特权内核。

### Profile、Bundle 与 Patch

运行中的 `dsh` 是一棵插件树：Profile 表示一个具名组装，Bundle 分发一组 Cordis 配置，Patch/overlay 对上层配置进行替换或插入。官方提供 `web`、`headless`、`sdk`、`sdk-minimal` 和 `acp` 等 Profile。可用下面的命令查看最终配置树：

```bash
dsh --profile web --dump-config
```

这种设计很适合多前端、多个运行环境共享能力：底层 Agent loop 和持久化不变，只替换 Web、SDK、ACP 或模型/工具 Provider。

### 能力的三角色拆分

DeepSeek Harness 官方以 Bash 能力为例拆成：

- **Service Definition**：定义稳定接口与请求/结果类型；
- **Service Provider**：提供具体实现，例如本地执行或远端沙箱；
- **Consumer**：把能力暴露成模型工具或给其他服务使用。

Provider 可以替换，Definition 和 Consumer 不需要同步改动。设计自己的 Harness 时，这比让工具直接依赖某个执行库更利于演进、测试和安全替换。

### 状态、安全和协作上的设计启发

- 会话采用只追加事件日志作为真源，持久化后端可替换；flush、崩溃恢复和被中断轮次有明确语义。
- 审批结果是闭合集合，缺少应答者或异常时 fail closed，而不是默认放行。
- 沙箱模式与审批策略是两个独立 knob，权限预设只是组合它们，不能替代真实执行层的强制措施。
- Subagent Provider 公开能力声明；请求不被支持时明确失败，避免静默降级。
- Agent Teams 不只是并行调用，还建模身份、持久 mailbox、共享任务 DAG 和回放。
- 每个插件注册都可逆，支持热重载和可靠 teardown；这是长期运行 Agent 服务经常缺失的一层。

## 6. 五种方案的多智能体模型对比

| 系统 | 拓扑 | 状态/通信 | 最适合 | 关键风险 |
| --- | --- | --- | --- | --- |
| Codex Subagents | 主 Agent → 子 Agent → 汇总 | 子上下文隔离，结果回主 Agent | 并行调研、代码探索、独立评审 | 并发写冲突、额外 token |
| Claude Subagents | 主会话集中委派 | 独立上下文，通常单次结果回传 | 专业角色、减少主上下文污染 | 描述过多也会占启动上下文 |
| Claude Agent Teams | Lead + 对等 Teammates | 共享任务、直接消息、独立会话 | 需要互相挑战和协调的研究/开发 | 实验性、成本高、文件冲突 |
| LangChain/Deep Agents | Subagent-as-tool 或自定义图 | graph state、checkpointer/store、interrupt | 嵌入业务产品的可控工作流 | 架构复杂度和可观察性成本 |
| OpenCode | Primary agent + Subagent | 主会话委派，可自定义角色和权限 | 开源编码助手的专门角色 | 工具/MCP 过多导致上下文拥挤 |
| DeepSeek Harness | 可插拔 Subagent Provider；实验 Team | 事件日志、持久 mailbox、任务 DAG | 研究可替换的多 Agent 基础设施 | 预览期、API 变化、安全未审计 |

选择顺序建议：

1. 一个 Agent + 少量合适工具能否完成？
2. 是否只需按需加载 Skill，而不是新 Agent？
3. 是否只需一个隔离上下文、单次回传结果的 Subagent？
4. 任务是否真的需要 Agent 之间持续通信、共享任务和互相挑战？只有此时再上 Team。

## 7. 设计自己的 Agent：五层 Harness

### 7.1 Instructions：告诉 Agent 怎样工作

- 仓库根指令只放稳定、通用、可验证的规则。
- 子目录规则只在相关代码范围加载。
- 一次性需求留在用户任务里；可复用复杂流程做成 Skill。
- 强制安全策略不要只写成自然语言，必须落到权限、沙箱或 Hook。

### 7.2 State：让长任务可以恢复

至少区分四类状态：

- 当前 thread 的消息和工作状态；
- 跨会话的项目记忆与用户偏好；
- 可机读任务清单及状态；
- 只追加运行事件、工具证据和检查点。

不要把聊天摘要当作唯一真源。对长任务，状态应有 schema、稳定 ID、原子写入、恢复规则和失败后的重放/补偿策略。

### 7.3 Verification：定义什么时候才算完成

验证门槛应能被 Agent 自己执行：

- 单元/集成测试、build、lint、typecheck；
- 结构化 schema 校验；
- UI 截图或视觉 diff；
- 关键行为的固定样例；
- 独立 Reviewer 尝试推翻实现结论。

让 Agent 回报命令、退出码和关键输出。没有证据时，状态只能是“声称完成”。

### 7.4 Scope：限制 Agent 能影响什么

- 默认最小工具集、最小目录和最小网络域名。
- 研究/规划角色优先只读；写入与外部副作用单独授权。
- 高风险动作采用 `ask/approve/edit/reject` 或同等审批模型。
- Subagent 权限不得因为继承而意外放大。
- shell 执行和 MCP 服务都应视为新的信任边界。

### 7.5 Lifecycle：可靠地启动、交接、停止和恢复

成熟 Agent 要明确：

- 启动时加载哪些配置、指令、工具和状态；
- 如何做环境自检与依赖检查；
- 工具、插件和后台任务如何取消与清理；
- 会话压缩或重启前如何写 handoff；
- 崩溃后从哪个 checkpoint 恢复；
- 如何判定 complete、blocked、failed，避免无限循环或虚假完成。

## 8. 一个可落地的仓库级 Harness 骨架

下面是产品中立的参考，不要求所有工具采用相同文件名：

```text
repo/
├── AGENTS.md                 # 稳定的项目规则、命令、边界和验收要求
├── agent/
│   ├── feature_list.json    # 可机读任务/功能状态与验收项
│   ├── progress.md          # 面向人的进展、证据、问题和决策
│   ├── session-handoff.md   # 下次会话恢复所需最小信息
│   ├── init.sh              # 幂等环境自检/初始化入口
│   └── evals/               # 固定样例、回归集、评分与审查脚本
├── .agents/skills/          # 可跨兼容 Agent 使用的渐进加载 Skills
└── src/ ...
```

建议的停止协议：只有验收命令通过、证据记录完整、任务状态原子更新后才能标记完成；需要用户选择时写清 blocker 和候选项；不得因为上下文快满或时间较长而伪造完成。

## 9. 推荐学习与实践顺序

### 第一阶段：把一个编码 Agent 用好

1. 读 [Claude Code 最佳实践](https://code.claude.com/docs/en/best-practices) 和 [Codex `AGENTS.md`](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。
2. 为一个熟悉仓库写 50～150 行项目指令，包含真实的 build/test 命令。
3. 连续完成三个小任务，每次都要求 Agent 自己运行验证并展示证据。

### 第二阶段：掌握上下文与复用

1. 把高噪声研究交给 Subagent，对比主上下文消耗和结果质量。
2. 把一个重复流程做成 Skill；只在匹配任务时加载。
3. 接入一个 MCP 服务，记录工具描述带来的上下文成本和权限边界。

### 第三阶段：做持久、可控的 Agent

1. 用 LangChain `create_agent` 做最小 loop。
2. 增加 structured output、checkpointer、store、middleware 和 HITL。
3. 用 LangSmith trace 找一次工具选择错误或恢复错误，并做回归用例。

### 第四阶段：研究 Harness 架构

1. 用 Deep Agents 快速体验文件系统、Subagent、Skills 和 approval。
2. 阅读 DeepSeek Harness 的 Cordis、Profile/Patch 和三角色能力设计。
3. 自己实现一个可替换 Provider：同一 Service Definition 下切换本地实现和沙箱实现。
4. 只有明确需要通信和共享任务时，再实验 Claude Agent Teams 或自定义多 Agent 图。

## 10. 设计/评审检查表

### 使用现成 Agent

- [ ] 目标、范围、相关路径和验收条件是否明确？
- [ ] Agent 是否先探索和计划，再修改？
- [ ] 项目指令是否短、无冲突、命令可执行？
- [ ] 是否给了测试/build/lint/截图等完成信号？
- [ ] 是否使用了最小权限和最小工具集？
- [ ] 并行 Agent 是否拥有互不冲突的文件或只读任务？
- [ ] 最终结果是否包含可核验的证据？

### 设计自己的 Agent

- [ ] Agent loop、deterministic workflow 和 UI/transport 是否解耦？
- [ ] 工具 schema 是否清楚，错误、超时、取消和重试语义是否明确？
- [ ] thread state、long-term memory、task state、event log 是否分层？
- [ ] 中断、审批、崩溃和上下文压缩后能否恢复？
- [ ] 权限、审批、沙箱是否各自建模并 fail closed？
- [ ] Skills/工具是否渐进暴露，避免上下文和工具过载？
- [ ] 是否有 trace、指标、回归集和独立验证？
- [ ] 插件和后台任务是否有可逆注册、取消与 teardown？
- [ ] Multi-agent 是否解决了明确问题，而不是为了形式上的“多个 Agent”？

## 总结

真正高质量的 Agent 系统，不是“更长的提示词”或“更多的 Agent”，而是把正确上下文在正确时间交给模型，同时让状态可恢复、工具有边界、结果可验证、生命周期可控制。

- 从 Codex 学分层指令、Skills、沙箱/审批和自动化接口。
- 从 Claude Code 学上下文管理、验证闭环以及 Subagent/Team 的使用边界。
- 从 LangChain/LangGraph/Deep Agents 学 Agent loop、状态图、持久化、Middleware 和 HITL。
- 从 OpenCode 学开源客户端-服务器架构、多层配置与扩展面的划分。
- 从 DeepSeek Harness 学插件容器、能力 seam、可替换 Provider、事件日志和可逆生命周期。

先把单 Agent 的指令、状态、验证、范围和生命周期做扎实，再引入 Multi-agent，通常会得到更简单、更便宜、也更可靠的系统。
