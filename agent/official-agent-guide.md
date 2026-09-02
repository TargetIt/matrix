# Codex、Claude Code、LangChain、OpenCode 与 DeepSeek Harness 官方文档全景指南

调研与逐链接核验日期：2026-09-02

> 本文只用厂商官网、官方文档站和官方 GitHub 组织下的仓库说明产品事实。正文严格分为两部分：第一部分按公司/产品纵向梳理，第二部分按 Agent 能力横向比较。文中的“选型判断”和“设计建议”是基于官方资料的综合分析，不冒充厂商原话。

## 调研边界与覆盖方法

这五个对象不在同一抽象层：

- **Codex、Claude Code、OpenCode、Deep Agents Code** 是可以直接完成编码任务的 Agent 产品。
- **LangChain** 是 Agent 框架，**LangGraph** 是状态化编排运行时，**Deep Agents SDK** 是更完整的 Agent Harness。
- **DeepSeek Harness（`dsh`）** 是 DeepSeek 官方的插件化 Agent Harness，目前仍处于开发者预览。

本次没有只看 Overview，而是先审计官方文档目录，再选择每个维度的概念页、指南页和参考页交叉核对：

| 对象 | 官方目录基线 | 本次覆盖 |
| --- | --- | --- |
| Codex | [官方 `llms.txt`](https://learn.chatgpt.com/llms.txt)，当前索引 160 个入口 | CLI、IDE、Cloud、Prompt、配置、指令、Memory、Skill、Plugin、MCP、Hook、权限、沙箱、Goal、Review、Worktree、自动化、SDK、App Server、管理 |
| Claude Code | [官方文档索引](https://code.claude.com/docs/llms.txt)，当前列出 166 篇英文页面 | 工作原理、Prompt、Context、Cache、Session、Memory、Skill、Hook、MCP、权限、Goal、Subagent、Agent View、Team、Workflow、SDK、部署 |
| LangChain | [总索引](https://docs.langchain.com/llms.txt)、[LangChain 76 页](https://docs.langchain.com/oss/python/langchain/llms.txt)、[LangGraph 43 页](https://docs.langchain.com/oss/python/langgraph/llms.txt)、[Deep Agents 40 页](https://docs.langchain.com/oss/python/deepagents/llms.txt) | Framework、Runtime、Harness、Coding Agent、State、Memory、Middleware、HITL、Multi-agent、Testing、Observability、Deployment |
| OpenCode | [官方文档](https://opencode.ai/docs/) 和 [官方文档源码目录](https://github.com/anomalyco/opencode/tree/dev/packages/web/src/content/docs)，当前 36 个顶层页面 | CLI/TUI/Web、配置、Provider、Model、Agent、Rule、Command、Tool、Skill、MCP、Plugin、Permission、Policy、ACP、SDK、Server、CI |
| DeepSeek Harness | [官方文档站](https://deepseek-harness.github.io/deepseek-harness/)、[官方仓库文档](https://github.com/deepseek-ai/deepseek-harness/tree/master/docs)，含 17 篇中文用户/开发指南和 52 个子系统页 | Web/SDK、Cordis、Profile/Patch、Prompt、Tool、Session、Persistence、Compaction、Skill、Plan、Goal、Subagent、Team、Sandbox、Approval、Telemetry |

## 快速结论

- 想提高日常研发效率：先精通 Codex 或 Claude Code；重视开源、自托管和多模型时看 OpenCode。
- 想把 Agent 嵌入自己的业务：先用 LangChain `create_agent`，需要可靠恢复和复杂控制流时下沉 LangGraph，需要成品 Harness 时用 Deep Agents。
- 想研究 Agent Runtime 的内部模块化：DeepSeek Harness 对 Service、Provider、Consumer、事件、日志和生命周期的拆分最值得阅读，但不应忽略其开发者预览和未完成安全审计的状态。
- Multi-agent 不是成熟度标志。项目指令、状态、验证、权限和恢复机制没有做好时，增加 Agent 数量只会放大成本与冲突。

# 第一部分：按公司与产品纵向梳理

## 1. OpenAI Codex

### 1.1 官方产品地图

Codex 不是只有一个 CLI。官方把主要工作面拆成：

- [Codex CLI](https://learn.chatgpt.com/docs/codex/cli)：终端内探索、修改、执行、审查和委派。
- [Codex IDE Extension](https://learn.chatgpt.com/docs/codex/ide)：利用编辑器当前文件、选区和工作区上下文。
- [Codex Cloud](https://learn.chatgpt.com/docs/cloud)：在隔离云环境中并行委派任务并审查结果。
- ChatGPT 桌面端中的 Codex：本地项目、Goal、Worktree、Review 和应用集成。
- [`codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode)：脚本、CI 和结构化输出。
- [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk) 与 [App Server](https://learn.chatgpt.com/docs/app-server)：把 Codex thread/turn/event/approval 能力嵌入自己的客户端。

### 1.2 官方文档维度全表

| 维度 | 官方说明与关键结论 | 一键原文 |
| --- | --- | --- |
| 工作方式 | 从结果出发，让 Agent 探索、实施、验证；任务运行中可 steer 当前轮次或 queue 下一轮 | [Prompting](https://learn.chatgpt.com/docs/prompting) |
| 长任务 | Goal 应写 Outcome、Constraints、Verification；Goal 不扩大原有权限 | [Long-running work](https://learn.chatgpt.com/docs/long-running-work) |
| CLI/命令 | 覆盖 plan、goal、review、diff、compact、resume、fork、agent、skills、hooks、MCP 等命令 | [Developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli) |
| 基础配置 | 用户 `~/.codex/config.toml`、可信项目 `.codex/config.toml`、Profile、系统层和 CLI override 分层解析 | [Config basics](https://learn.chatgpt.com/docs/config-file/config-basic) |
| 高级配置 | Provider、模型参数、Shell 环境、MCP、Hook、Agent roles、History、OTel、通知和 TUI | [Advanced configuration](https://learn.chatgpt.com/docs/config-file/config-advanced) |
| 完整字段 | `config.toml` 与管理员 `requirements.toml` 的可搜索字段表 | [Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference) |
| 登录/Provider | ChatGPT 登录、API Key、Access Token、Headless 登录、凭据存储和替代 Provider | [Authentication](https://learn.chatgpt.com/docs/auth) |
| 模型 | 模型、推理强度和本地/云默认值的选择 | [Models](https://learn.chatgpt.com/docs/models) |
| 项目指令 | `AGENTS.md` 从全局到仓库/子目录分层加载；更具体的目录指令更靠后 | [`AGENTS.md`](https://learn.chatgpt.com/docs/agent-configuration/agents-md) |
| 命令规则 | `.rules` 控制哪些命令可在沙箱外运行，并提供规则测试机制 | [Rules](https://learn.chatgpt.com/docs/agent-configuration/rules) |
| Memory | 本地 Memory 与团队规则分离；Memory 是辅助召回，不是必须遵守规则的唯一载体 | [Memories](https://learn.chatgpt.com/docs/customization/memories) |
| Skill | `SKILL.md` + scripts/references/assets；启动时看元数据，使用时再加载全文 | [Build skills](https://learn.chatgpt.com/docs/build-skills) |
| Plugin | 可安装和分发的能力包；可捆绑 Skills、MCP、Hooks 等 | [Plugins](https://learn.chatgpt.com/docs/plugins)、[Build plugins](https://learn.chatgpt.com/docs/build-plugins) |
| MCP | 连接 stdio/HTTP 外部工具、资源和 Prompt，支持 OAuth 与 Plugin 提供的 MCP | [MCP](https://learn.chatgpt.com/docs/extend/mcp) |
| Hooks | Session、Prompt、Tool、Permission、Compaction、Subagent、Stop 等生命周期事件 | [Hooks](https://learn.chatgpt.com/docs/hooks) |
| Subagent | 自定义角色、模型、推理强度、工具/MCP、沙箱；强调上下文隔离和独立任务 | [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents) |
| 并行隔离 | 独立任务可并行；会写代码的并行 chat 应使用独立 Worktree | [Worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees) |
| 权限/沙箱 | 沙箱规定技术访问边界，审批规定何时询问；二者独立 | [Agent approvals & security](https://learn.chatgpt.com/docs/agent-approvals-security) |
| 权限 Profile | 可定义文件、网络、Unix socket、私网等更细的 Permission Profile | [Permissions](https://learn.chatgpt.com/docs/permissions) |
| 网络 | Cloud 默认在 Agent 阶段限制网络，可用域名和 HTTP 方法白名单开放 | [Cloud internet access](https://learn.chatgpt.com/docs/cloud/internet-access) |
| 环境 | Cloud setup、环境变量/Secret、缓存；Local setup script/actions；Git Worktree | [Cloud environment](https://learn.chatgpt.com/docs/environments/cloud-environment)、[Local environment](https://learn.chatgpt.com/docs/environments/local-environment) |
| Review | 可选择审查范围、查看 Finding、处理 inline comment、PR 和本地变更 | [Code review](https://learn.chatgpt.com/docs/code-review) |
| 自动化 | `codex exec`、GitHub Action、Scheduled Tasks/事件触发 | [`codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode)、[GitHub Action](https://learn.chatgpt.com/docs/github-action)、[Scheduled tasks](https://learn.chatgpt.com/docs/automations) |
| 程序化接口 | SDK 提供 thread start/continue/resume；App Server 暴露 thread、turn、event、approval、skill、auth 等协议 | [SDK](https://learn.chatgpt.com/docs/codex-sdk)、[App Server](https://learn.chatgpt.com/docs/app-server) |
| 观测/管理 | 可选 OTel、History、管理员约束和 Managed config | [Security/telemetry](https://learn.chatgpt.com/docs/agent-approvals-security#monitoring-and-telemetry)、[Managed configuration](https://learn.chatgpt.com/docs/enterprise/managed-configuration) |

### 1.3 Codex 官方最核心的使用建议

官方 Prompt 指南并不要求固定模板，而是建议根据任务选择四类信息：

1. **Goal**：最终要得到什么。
2. **Context**：哪些文件、来源、截图或系统会改变答案。
3. **Output**：结果格式、受众和细节程度。
4. **Boundaries**：必须保持不变、禁止做或需要先询问的事项。

重点是“描述结果而不是微管理全部步骤”。较大任务再加入验收检查；运行中发现方向不对时 steer，而不是等整个任务结束。

官方对定制层的建议顺序也很明确：先用 `AGENTS.md` 固化仓库惯例，再用 Plugin/Skill 复用流程，用 MCP 接外部系统，最后才用 Subagent 隔离高噪声或专门任务。详见 [Customization](https://learn.chatgpt.com/docs/customization/overview)。

### 1.4 Codex 容易混淆的边界

- `AGENTS.md` 是持久指令；Memory 是从旧会话生成的辅助召回；Skill 是按需流程；MCP 是外部能力；Hook 是确定性生命周期动作。
- Rules 主要决定命令何时允许越出沙箱，不等同于 `AGENTS.md` 的行为建议。
- Goal 是持续完成条件，不等同于 Plan；Goal 不会自动扩大沙箱或网络权限。
- SDK 适合直接控制本地 Agent；需要完整客户端协议、审批和事件流时使用 App Server。
- 自动化应默认最小权限。非交互运行遇到必须新增审批的动作会失败，而不是偷偷放行。

### 1.5 Codex 推荐阅读顺序

1. [CLI](https://learn.chatgpt.com/docs/codex/cli) → [Prompting](https://learn.chatgpt.com/docs/prompting) → [`AGENTS.md`](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。
2. [Configuration](https://learn.chatgpt.com/docs/config-file/config-basic) → [Security](https://learn.chatgpt.com/docs/agent-approvals-security)。
3. [Skills](https://learn.chatgpt.com/docs/build-skills) → [MCP](https://learn.chatgpt.com/docs/extend/mcp) → [Hooks](https://learn.chatgpt.com/docs/hooks) → [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)。
4. [`codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode) → [SDK](https://learn.chatgpt.com/docs/codex-sdk) → [App Server](https://learn.chatgpt.com/docs/app-server)。

## 2. Anthropic Claude Code

### 2.1 官方产品地图

Claude Code 已覆盖 [CLI、Desktop、VS Code、JetBrains、Web、Mobile、Chrome、Slack 和 CI/CD](https://code.claude.com/docs/en/platforms)。其核心是 Agentic loop：读取上下文、调用工具、观察结果、继续决策。官方还把并行能力拆为四个不同层次：

- [Subagents](https://code.claude.com/docs/en/sub-agents)：单会话内的隔离工作者。
- [Agent View](https://code.claude.com/docs/en/agent-view)：从一个界面调度和观察多个独立会话。
- [Agent Teams](https://code.claude.com/docs/en/agent-teams)：Lead + Teammates + 共享任务 + 直接通信。
- [Dynamic Workflows](https://code.claude.com/docs/en/workflows)：Claude 编写脚本，大规模扇出 Subagent 并交叉验证。

[Run agents in parallel](https://code.claude.com/docs/en/agents) 是这四类能力的官方选型入口。

### 2.2 官方文档维度全表

| 维度 | 官方说明与关键结论 | 一键原文 |
| --- | --- | --- |
| 工作原理 | Agentic loop、内置工具、环境、Session、Context、Checkpoint、Permission | [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works) |
| Prompt | 指定文件/场景/测试偏好，指向来源和现有模式，描述症状与修复判据 | [Best practices](https://code.claude.com/docs/en/best-practices)、[Prompt library](https://code.claude.com/docs/en/prompt-library) |
| 推荐流程 | Explore → Plan → Implement → Verify/Commit；小改动可跳过 Plan | [Best practices](https://code.claude.com/docs/en/best-practices#explore-first-then-plan-then-code) |
| 功能选型 | 比较 `CLAUDE.md`、Rules、Skill、Subagent、Workflow、MCP、Hook、Plugin | [Features overview](https://code.claude.com/docs/en/features-overview) |
| 配置目录 | `.claude` 下 settings、rules、agents、skills、hooks、workflows、memory 的作用 | [Explore `.claude`](https://code.claude.com/docs/en/claude-directory) |
| Settings | User/Project/Local/Managed 等 Scope、合并和优先级 | [Settings](https://code.claude.com/docs/en/settings) |
| Model | 模型、effort、extended thinking/context、auto-compact、fallback 和 Prompt Cache | [Model configuration](https://code.claude.com/docs/en/model-config) |
| 项目指令 | `CLAUDE.md`、`CLAUDE.local.md`、嵌套文件、import 和路径 Rules | [Memory and `CLAUDE.md`](https://code.claude.com/docs/en/memory) |
| Auto memory | Claude 自动保存纠正和偏好；可审计编辑；与人工维护的规则分离 | [Auto memory](https://code.claude.com/docs/en/memory#auto-memory) |
| Context | 显示自动加载内容、文件读取成本、Compaction 前后保留内容 | [Context window](https://code.claude.com/docs/en/context-window) |
| Prompt Cache | 解释缓存层次、失效行为、TTL、Subagent cache 和命中率 | [Prompt caching](https://code.claude.com/docs/en/prompt-caching) |
| Session | Resume、Name、Branch、Export、Transcript；Checkpoint 支持 Rewind | [Sessions](https://code.claude.com/docs/en/sessions)、[Checkpointing](https://code.claude.com/docs/en/checkpointing) |
| 大仓库 | 嵌套指令、路径 Rules、Sparse Worktree、Code intelligence、按包 Skill | [Large codebases](https://code.claude.com/docs/en/large-codebases) |
| Skills | 按需知识/流程、支持工具预授权、动态上下文、Subagent context 和 Evals | [Skills](https://code.claude.com/docs/en/skills) |
| Plugins | 捆绑 Skill、Agent、Hook、MCP、LSP、Monitor 和默认设置 | [Plugins](https://code.claude.com/docs/en/plugins) |
| MCP | Local/Project/User scope、OAuth、Tool search、Resources、Prompts、Elicitation | [MCP](https://code.claude.com/docs/en/mcp) |
| Hooks | Command/HTTP/MCP/Prompt/Agent Hook；事件覆盖 Session、Tool、Task、Team、Worktree、Compact、Model switch | [Hooks guide](https://code.claude.com/docs/en/hooks-guide)、[Hooks reference](https://code.claude.com/docs/en/hooks) |
| Goal/验证 | `/goal` 由独立 Evaluator 检查完成条件；Stop Hook 可做确定性门禁 | [Goal](https://code.claude.com/docs/en/goal)、[Best practices](https://code.claude.com/docs/en/best-practices#give-claude-a-way-to-verify-its-work) |
| Subagent | 独立 Context、工具、模型、权限、Hook、Skill；支持显式/自动委派和 Fork | [Subagents](https://code.claude.com/docs/en/sub-agents) |
| 多会话 | Agent View、Cross-session messaging、Worktree isolation | [Agent View](https://code.claude.com/docs/en/agent-view)、[Messaging](https://code.claude.com/docs/en/cross-session-messaging)、[Worktrees](https://code.claude.com/docs/en/worktrees) |
| Agent Teams | Lead、Teammate、共享 Task、直接消息和 Team hooks；当前仍为实验功能 | [Agent Teams](https://code.claude.com/docs/en/agent-teams) |
| Dynamic Workflow | 适合大量文件审计、迁移、迭代修复和多来源研究 | [Dynamic workflows](https://code.claude.com/docs/en/workflows) |
| Permission | Tool allow/ask/deny、模式、Managed policy、Workspace trust | [Permissions](https://code.claude.com/docs/en/permissions)、[Permission modes](https://code.claude.com/docs/en/permission-modes) |
| Sandbox | 文件/网络 OS 隔离、Secret mask、组织强制策略和局限 | [Sandboxing](https://code.claude.com/docs/en/sandboxing) |
| Review | 本地 Diff Review、PR Code Review、多 Agent 分析和定制 `REVIEW.md` | [Code Review](https://code.claude.com/docs/en/code-review) |
| 自动化 | Headless、GitHub Actions、Cloud Routine、Session 内 Schedule | [Headless](https://code.claude.com/docs/en/headless)、[GitHub Actions](https://code.claude.com/docs/en/github-actions)、[Routines](https://code.claude.com/docs/en/routines)、[Scheduled tasks](https://code.claude.com/docs/en/scheduled-tasks) |
| SDK | Agent loop、Session、External storage、Structured output、Custom tools、Subagent、Hook、Permission | [Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) |
| 部署/观测 | Subprocess、容器、Session 持久化、扩缩容、多租户、OTel、Secret/Network | [Hosting](https://code.claude.com/docs/en/agent-sdk/hosting)、[Observability](https://code.claude.com/docs/en/agent-sdk/observability)、[Secure deployment](https://code.claude.com/docs/en/agent-sdk/secure-deployment) |

### 2.3 Claude Code 官方最核心的使用建议

1. **给验证信号。** 测试、build、lint、固定 fixture 或截图对比必须能让 Claude 自己读取 Pass/Fail。
2. **先探索再计划。** 不熟悉或多文件任务先在 Plan mode 研究；能用一句话描述 Diff 的小任务无需强制规划。
3. **Prompt 指向具体上下文。** 命名文件、症状、已有实现模式和测试偏好，比抽象地说“优化代码”更可靠。
4. **证据优于声明。** 让 Claude 给出执行的命令、退出结果或截图，而不是只说完成。
5. **主动管理 Context。** 高噪声调查交给 Subagent，及时 compact/branch/resume，不让无关日志长期污染主会话。

### 2.4 Claude Code 的功能分工

官方 [Features overview](https://code.claude.com/docs/en/features-overview) 给出了非常清晰的选择规则：

- 每个会话都必须知道：`CLAUDE.md`。
- 只对某类路径生效：`.claude/rules/`。
- 偶尔使用的知识或重复流程：Skill。
- 外部数据/动作：MCP。
- 高噪声、独立上下文：Subagent。
- 大量扇出并需要交叉验证：Dynamic Workflow。
- 每次事件都必须执行：Hook。
- 跨项目分发整套能力：Plugin。

### 2.5 Agent Teams 的准确边界

截至核验日，Agent Teams 仍默认关闭，需要 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`。它适合研究、竞争假设、跨层模块等可独立推进且需要互相交流的任务；不适合严格顺序任务、强依赖任务和多人同时改同一文件。非交互 `-p` 与 Agent SDK 不生成 Teammate。官方明确提醒它的 Token 成本高于 Subagent。

### 2.6 Claude Code 推荐阅读顺序

1. [How it works](https://code.claude.com/docs/en/how-claude-code-works) → [Best practices](https://code.claude.com/docs/en/best-practices)。
2. [Features overview](https://code.claude.com/docs/en/features-overview) → [Memory](https://code.claude.com/docs/en/memory) → [Context](https://code.claude.com/docs/en/context-window)。
3. [Permissions](https://code.claude.com/docs/en/permissions) → [Sandbox](https://code.claude.com/docs/en/sandboxing) → [Hooks](https://code.claude.com/docs/en/hooks-guide)。
4. [Subagents](https://code.claude.com/docs/en/sub-agents) → [Parallel agents](https://code.claude.com/docs/en/agents) → [Agent Teams](https://code.claude.com/docs/en/agent-teams)。
5. [Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) → [Agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) → [Hosting](https://code.claude.com/docs/en/agent-sdk/hosting)。

## 3. LangChain、LangGraph、Deep Agents 与 LangSmith

### 3.1 先分清官方产品层级

| 层 | 定位 | 官方入口 |
| --- | --- | --- |
| LangChain | 模型、Message、Tool、Agent loop、Middleware 等高层构件 | [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview) |
| LangGraph | 长运行、状态化、可恢复的编排运行时 | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) |
| Deep Agents SDK | 在 LangChain/LangGraph 上封装文件、Skill、Memory、Subagent、Sandbox、HITL 的 Harness | [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview) |
| Deep Agents Code | 基于 Deep Agents SDK 的终端编码 Agent 产品 | [Deep Agents Code](https://docs.langchain.com/oss/deepagents/code/overview) |
| LangSmith | Trace、Evaluation、Studio 和 Deployment | [Observability](https://docs.langchain.com/oss/python/langchain/observability)、[Deployment](https://docs.langchain.com/oss/python/langgraph/deploy) |

### 3.2 LangChain 官方文档维度

| 维度 | 官方说明与关键结论 | 一键原文 |
| --- | --- | --- |
| Agent 定义 | Agent 是模型循环调用工具直到任务结束；Harness 是 Prompt、Tool、Middleware 等外围 | [Agents](https://docs.langchain.com/oss/python/langchain/agents) |
| 核心构件 | Model、Tool、System prompt、Structured output、State、Runtime | [Agents](https://docs.langchain.com/oss/python/langchain/agents)、[Component architecture](https://docs.langchain.com/oss/python/langchain/component-architecture) |
| Context engineering | 分 Model context、Tool context、Lifecycle context；控制 System Prompt、Messages、Tools、Model 和 Response format | [Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering) |
| Model | Provider、Tool calling、Reasoning、Multimodal、Cache、Rate limit、Dynamic selection | [Models](https://docs.langchain.com/oss/python/langchain/models) |
| Tools | Schema、Context/State/Store、Error、Dynamic selection、Headless tools | [Tools](https://docs.langchain.com/oss/python/langchain/tools) |
| Structured output | Provider-native 或 Tool strategy，支持错误处理 | [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) |
| Runtime | 在 Tool/Middleware 中访问 Context、Store、Execution info | [Runtime](https://docs.langchain.com/oss/python/langchain/runtime) |
| Middleware | Retry、Fallback、Summarization、HITL、Call limit、PII、Todo、Tool selector、Filesystem、Subagent、Rubric | [Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview)、[Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) |
| 短期记忆 | Checkpointer 管理 Thread 内状态；长上下文可 trim/delete/summarize | [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) |
| 长期记忆 | Store 以 namespace/key 跨 Thread 保存 JSON 文档 | [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory) |
| Guardrail/HITL | PII、自定义前后置 Guardrail；敏感 Tool call 可 approve/edit/reject/respond | [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails)、[HITL](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) |
| MCP | Tools、Resources、Prompts、HTTP/stdio、Session、Interceptor、Progress、Elicitation | [MCP](https://docs.langchain.com/oss/python/langchain/mcp) |
| Multi-agent | Subagents、Handoffs、Skills、Router、自定义 Workflow | [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent) |
| Streaming | Agent progress、Token、Tool call、Reasoning、Subagent、State 和自定义更新 | [Streaming](https://docs.langchain.com/oss/python/langchain/streaming)、[Event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming) |
| 测试/Eval | Unit、Integration、Trajectory match、LLM-as-judge、LangSmith Evals | [Testing](https://docs.langchain.com/oss/python/langchain/test)、[Agent Evals](https://docs.langchain.com/oss/python/langchain/test/evals) |
| Observability | 自动 Trace Agent 各步骤，支持项目和 Metadata | [LangSmith observability](https://docs.langchain.com/oss/python/langchain/observability) |

### 3.3 LangGraph 官方文档维度

LangGraph 不是“另一个 Chat Agent API”，而是把流程建模为 State + Node + Edge，并提供持久执行：

| 维度 | 官方要点 | 一键原文 |
| --- | --- | --- |
| 设计方法 | 先画业务步骤，再区分 LLM/Data/Action/User-input 节点，最后设计原始 State | [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph) |
| API | Graph API 适合显式状态图；Functional API 适合保留普通控制流 | [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)、[Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api) |
| Persistence | Checkpoint 保存 Thread 状态；Store 保存跨 Thread 数据 | [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) |
| Checkpointer | State history、Replay、Update、Durability mode、Custom saver | [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers) |
| Interrupt | 暂停并持久化，后续用同一 Thread 恢复；副作用必须幂等 | [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) |
| Fault tolerance | Retry、Timeout、Drain、Resume-safe failure、Subgraph failure | [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance) |
| Time travel | 从 Checkpoint Replay、Fork 或修改 State 后继续 | [Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) |
| Subgraph | Stateful/Stateless 子图、继承 Checkpointer、流式查看子图 | [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) |
| Testing | 单独测试 Node/Edge，支持 Partial execution | [Test](https://docs.langchain.com/oss/python/langgraph/test) |
| Deployment | 通过 LangSmith Deployment 托管 | [Deployment](https://docs.langchain.com/oss/python/langgraph/deploy) |

### 3.4 Deep Agents SDK 官方文档维度

[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) 把成品 Harness 分成四块：Execution environment、Context management、Delegation、Steering。

| 维度 | 官方要点 | 一键原文 |
| --- | --- | --- |
| 定制 | Model、Tool、System prompt、Middleware stack、Subagent、Backend、HITL、Skill、Memory、Profile、Structured output | [Customization](https://docs.langchain.com/oss/python/deepagents/customization) |
| Backend | State、Filesystem、Local shell、Store、ContextHub、Composite routing、自定义协议 | [Backends](https://docs.langchain.com/oss/python/deepagents/backends) |
| Sandbox | Thread/Assistant scope，Agent-in-sandbox 与 Sandbox-as-tool 两种模式 | [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes) |
| Permission | 路径/操作规则按顺序 first-match-wins，可暂停审批，Subagent 可更窄 | [Permissions](https://docs.langchain.com/oss/python/deepagents/permissions) |
| Context | Input/Runtime/State、Offload、Summarization、Subagent isolation、Long-term memory | [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering) |
| Skill | 动态/命名空间 Skill、Backend 加载、Subagent 专属 Skill、读写权限和脚本 | [Skills](https://docs.langchain.com/oss/python/deepagents/skills) |
| Memory | Agent/User/Organization scope、Episodic memory、Background consolidation、并发写入 | [Memory](https://docs.langchain.com/oss/python/deepagents/memory) |
| Subagent | 默认/自定义/Compiled/Dynamic Subagent，工具/模型/Prompt/Skill 可独立 | [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents) |
| Async Subagent | Worker pool、ASGI/HTTP transport、单部署/拆分部署/混合拓扑 | [Async subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents) |
| HITL | Checkpointer + Thread ID；支持 Tool 和 Subagent 内部中断 | [HITL](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) |
| Fault tolerance | Provider rate limit、Call limit、Retry、Fallback、Error handling | [Fault tolerance](https://docs.langchain.com/oss/python/deepagents/fault-tolerance) |
| Event stream | 单独流式观察 Subagent 生命周期、Message、Tool call 和嵌套工作 | [Event streaming](https://docs.langchain.com/oss/python/deepagents/event-streaming) |
| Production | 多租户、Async、Durability、Memory scope、Sandbox、Guardrail、Privacy、Frontend | [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production) |

### 3.5 Deep Agents Code：原稿遗漏的重要产品

Deep Agents Code 是 LangChain 官方终端编码 Agent，不应只讨论 SDK：

- [Quickstart](https://docs.langchain.com/oss/deepagents/code/quickstart)：交互、非交互、Piping 和 LangSmith Trace。
- [Configuration](https://docs.langchain.com/oss/deepagents/code/configuration)：配置优先级、Managed policy、Auto memory、History、诊断和数据目录。
- [Memory and Skills](https://docs.langchain.com/oss/deepagents/code/memory-and-skills)：`AGENTS.md`、Auto memory、Skill 发现与 Headless 调用。
- [Subagents](https://docs.langchain.com/oss/deepagents/code/subagents)：用带 YAML frontmatter 的 `AGENTS.md` 定义角色。
- [Goals and rubrics](https://docs.langchain.com/oss/deepagents/code/goals-and-rubrics)：持续目标与独立评分标准。
- [Approval modes](https://docs.langchain.com/oss/deepagents/code/approval-modes)：Manual、Auto、YOLO；Auto 使用分类模型并在副作用前重新验证。
- [Hooks](https://docs.langchain.com/oss/deepagents/code/hooks)、[MCP](https://docs.langchain.com/oss/deepagents/code/mcp-tools)、[Plugins](https://docs.langchain.com/oss/deepagents/code/plugins)、[Python extensions](https://docs.langchain.com/oss/deepagents/code/extensions)。
- [Remote sandboxes](https://docs.langchain.com/oss/deepagents/code/remote-sandboxes)：LangSmith、AgentCore、Daytona、Modal、Runloop、Vercel、E2B 等执行后端。

### 3.6 LangChain 体系的官方选型逻辑

- 普通 Tool-calling Agent：`create_agent`。
- 需要明确状态机、持久恢复、Interrupt、Replay：LangGraph。
- 需要文件系统、Skill、Memory、Subagent、Sandbox 的成品 Harness：Deep Agents SDK。
- 直接想用终端编码 Agent：Deep Agents Code。
- 生产 Trace、Eval、Studio、Deployment：LangSmith。

官方 Multi-agent 文档还特别提醒：单 Agent + 动态 Tool/Skill 经常已经足够。只有上下文隔离、组织边界、并行化或工具过载是真问题时，才引入 Multi-agent。

## 4. OpenCode

### 4.1 官方产品地图

OpenCode 是开源编码 Agent，底层采用 Client/Server 结构：TUI 是客户端，Server 提供 OpenAPI；还支持 Web、IDE 终端、ACP、GitHub/GitLab 自动化和 JavaScript/TypeScript SDK。

### 4.2 官方文档维度全表

| 维度 | 官方说明与关键结论 | 一键原文 |
| --- | --- | --- |
| 入门 | 安装、Provider、`/init`、问答、修改、Undo、Share、Customize | [Overview](https://opencode.ai/docs/) |
| CLI/TUI/Web | `run`、`serve`、`attach`、session、MCP、ACP；TUI 与 Web 都连接 Server | [CLI](https://opencode.ai/docs/cli)、[TUI](https://opencode.ai/docs/tui)、[Web](https://opencode.ai/docs/web) |
| 配置 | Remote → Global → Custom → Project → `.opencode` 分层合并；支持 Managed settings | [Config](https://opencode.ai/docs/config) |
| Provider/Model | 大量云/本地 Provider、Custom Provider、模型 Variant 和默认模型 | [Providers](https://opencode.ai/docs/providers)、[Models](https://opencode.ai/docs/models) |
| 项目指令 | `/init` 生成/更新 `AGENTS.md`；支持 Global/Project/Claude-compatible 指令 | [Rules](https://opencode.ai/docs/rules) |
| Prompt 复用 | Command 模板支持参数、Shell 输出、文件引用、指定 Agent/Model/Subtask | [Commands](https://opencode.ai/docs/commands) |
| Agent | Primary 与 Subagent；内置 Build、Plan、General、Explore、Scout 及内部 Agent | [Agents](https://opencode.ai/docs/agents) |
| Agent 配置 | JSON 或 Markdown 定义 description、prompt、model、steps、permission、mode、task permission | [Agent configuration](https://opencode.ai/docs/agents#configure) |
| Tool | Bash、Edit、Write、Read、Grep、Glob、LSP、Patch、Skill、Todo、Web、Question | [Tools](https://opencode.ai/docs/tools) |
| Custom Tool | `.opencode/tools` 中用 JS/TS 定义 schema/execute，可转调任意语言脚本 | [Custom tools](https://opencode.ai/docs/custom-tools) |
| Skill | `.opencode`、`.claude`、`.agents` 路径发现；元数据先加载，正文按需读取 | [Skills](https://opencode.ai/docs/skills) |
| MCP | Local/Remote/OAuth，支持全局和按 Agent 管理；提醒工具会占 Context | [MCP servers](https://opencode.ai/docs/mcp-servers) |
| Plugin | 项目/全局 JS/TS 或 npm 包；监听事件、注入 Context/环境、增加 Tool | [Plugins](https://opencode.ai/docs/plugins) |
| Permission | `allow`/`ask`/`deny`，支持 Pattern、外部目录和 Agent override；`--auto` 不绕过 deny | [Permissions](https://opencode.ai/docs/permissions) |
| Policy | 针对具名 Resource 的实验性策略层，按规则顺序匹配 | [Policies](https://opencode.ai/docs/policies) |
| Context/Compaction | Config 提供 compaction；TUI 支持 `/compact`；Plugin 有 compaction hook | [Config](https://opencode.ai/docs/config#compaction)、[TUI](https://opencode.ai/docs/tui#compact)、[Plugins](https://opencode.ai/docs/plugins#compaction-hooks) |
| 外部上下文 | Reference 可挂载项目外目录或 Git 仓库，并说明用途 | [References](https://opencode.ai/docs/references) |
| 代码质量 | LSP 诊断和代码导航，Formatter 在写入后格式化 | [LSP](https://opencode.ai/docs/lsp)、[Formatters](https://opencode.ai/docs/formatters) |
| Session/Share | Session 可 export/import；Share 创建公开链接，官方强调隐私和保留策略 | [CLI sessions](https://opencode.ai/docs/cli#session)、[Share](https://opencode.ai/docs/share) |
| GitHub/GitLab | Issue/PR 评论触发、定时任务和 CI/CD | [GitHub](https://opencode.ai/docs/github)、[GitLab](https://opencode.ai/docs/gitlab) |
| ACP | 对接 Zed、JetBrains、Avante、CodeCompanion | [ACP](https://opencode.ai/docs/acp) |
| SDK/Server | SDK 管理 Project、Session、File、TUI、Auth、Event；Server 暴露 OpenAPI/SSE | [SDK](https://opencode.ai/docs/sdk)、[Server](https://opencode.ai/docs/server) |
| 网络/企业 | Proxy、证书、Central config、SSO、内部 Gateway、自托管 | [Network](https://opencode.ai/docs/network)、[Enterprise](https://opencode.ai/docs/enterprise) |
| 排错 | Log、Storage、Plugin/cache、Server connection、Provider/Auth 常见故障 | [Troubleshooting](https://opencode.ai/docs/troubleshooting) |

### 4.3 OpenCode 官方文档透露的设计重点

- **配置统一。** `opencode.json/jsonc` 是模型、Agent、Permission、MCP、Plugin、Compaction 等能力的汇合点；各层是 merge 而不是整份替换。
- **Primary 与 Subagent 分开。** 用户直接切换 Primary；主 Agent 自动委派或用户 `@` 调用 Subagent。
- **Plan 是受限角色。** 用于分析代码和评审建议；Build 用于真正修改。
- **Context 成本显式暴露。** MCP 文档提醒过多 Tool schema 会快速占满 Context；应按 Agent 启用。
- **Client/Server 解耦。** 同一个 Agent Runtime 可被 TUI、Web、SDK、IDE/ACP 使用。

### 4.4 OpenCode 当前官方文档的明确空白

为避免把别家的做法强加给 OpenCode，需要说明：当前官方顶层文档没有独立的“Prompt best practices”、自动长期 Memory、Agent Team/共享任务板、Goal evaluator 或 Agent Eval 指南。它提供的是 Command 模板、`AGENTS.md`、Session、Compaction、Primary/Subagent 和 SDK/Event。横向比较时应标为“官方未提供专门机制/指南”，不能自行补齐概念。

### 4.5 OpenCode 推荐阅读顺序

1. [Overview](https://opencode.ai/docs/) → [Config](https://opencode.ai/docs/config) → [Providers](https://opencode.ai/docs/providers)。
2. [Rules](https://opencode.ai/docs/rules) → [Agents](https://opencode.ai/docs/agents) → [Permissions](https://opencode.ai/docs/permissions)。
3. [Tools](https://opencode.ai/docs/tools) → [Skills](https://opencode.ai/docs/skills) → [MCP](https://opencode.ai/docs/mcp-servers) → [Plugins](https://opencode.ai/docs/plugins)。
4. [Server](https://opencode.ai/docs/server) → [SDK](https://opencode.ai/docs/sdk) → [ACP](https://opencode.ai/docs/acp)。

## 5. DeepSeek Harness

### 5.1 先确认成熟度与风险

[DeepSeek 官方 README](https://github.com/deepseek-ai/deepseek-harness/blob/master/README.zh.md) 把 `dsh` 定义为开源 Agent Harness，强调 “Everything is a Plugin”。截至核验日仍是开发者预览，未来会有破坏兼容性变更。

[官方安全说明](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.zh.md) 明确指出：项目尚未经过安全审计，能够执行模型生成的代码/命令、加载第三方插件并访问开放给它的文件、网络、进程和凭据；沙箱、审批、权限不能保证隔离。实验应优先使用一次性 VM、容器或专用环境，备份可访问文件并坚持最小权限。

### 5.2 官方产品与架构地图

- [Web UI](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/index.zh.md)：配置模型、选择 Workspace、运行任务。
- `web`、`headless`、`sdk`、`sdk-minimal`、`acp` Profile：不同入口复用同一插件树，见 [Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md)。
- [Python SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/python-sdk.zh.md)：通过 NDJSON-RPC/stdio 驱动打包 Runtime。
- [Cordis](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.zh.md)：Plugin、Context、Dependency injection、Typed events、Reversible effects。

### 5.3 官方文档维度全表

| 维度 | 官方说明与关键结论 | 一键原文 |
| --- | --- | --- |
| 运行 | `npx @deepseek-ai/dsh web` 或源码构建 | [README](https://github.com/deepseek-ai/deepseek-harness/blob/master/README.zh.md) |
| Model/Provider | DeepSeek、目录 Provider、自定义 OpenAI-compatible endpoint、模态和请求兼容性 | [Providers](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.zh.md) |
| 配置组合 | Profile + Bundle + Profile patch + Home patch + `--patch` overlay | [Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md#profile-与组合包) |
| Plugin 生命周期 | Fiber 状态、依赖驱动加载、自动清理、嵌套 Context、Dispose、HMR | [Plugin lifecycle](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/framework/index.zh.md) |
| Service/Event | Definition/Provider/Consumer；emit/bail/serial/waterfall typed events | [Services](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/framework/service.zh.md)、[Events](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/framework/events.zh.md) |
| System Prompt | 有序 PromptSection、动态 PromptContext、Tool schema、变量和 scoped shadow | [System prompt](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/system-prompt.zh.md) |
| Agent loop | Turn/Step/Request/Tool 的持久事件和实时 waterfall 扩展点 | [Architecture: turn flow](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md#轮次流程) |
| Tool | 统一 JSON Schema、参数/输出校验、Scope restriction、Pre/Execute/Post/Result pipeline | [Tools](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/tools.zh.md) |
| Shell/Sandbox | Shell 是可替换 Service；Sandbox 对每次进程调用包装 argv 并 fail closed | [Shell](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/shell.zh.md)、[Sandbox](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/sandbox.zh.md) |
| Approval | `ask/never` 策略；结果为 allowed-once/rejected/cancelled/unavailable，失败时拒绝 | [Approval](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/approval.zh.md) |
| Permission preset | 将独立的 Sandbox mode 与 Approval policy 组合为 UI 可选预设 | [Permission presets](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/permission-presets.zh.md) |
| Session | Append-only `SessionEvent` 日志是真源；模型历史、UI、Fork、Telemetry 从日志投影 | [Session](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.zh.md) |
| Persistence | 可替换后端、Flush checkpoint、崩溃恢复、Session header、JSONL provider | [Persistence](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.zh.md) |
| Context compression | Compaction 对历史压缩，Tool result pruner；Spill 把大结果移出主上下文 | [Compaction](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/compaction.zh.md)、[Spill](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/spill.zh.md) |
| Memory | 当前官方提供第三方 Memory MCP 参考配置，不把它等同于 Session persistence | [Memory MCP](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/mcp-memory.zh.md) |
| Skill | Registry、Provider、本地发现优先级、摘要/候选/全文分层、Session catalog | [Skills](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/skills.zh.md) |
| Plan/Goal/Todo | Plan mode 可恢复；Goal 记录生命周期并续跑；Todo 是持久列表事件 | [Plan](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/plan.zh.md)、[Goal](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/goal.zh.md)、[Todo](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/todo.zh.md) |
| Subagent | 可插拔 Provider；支持一次性与可继续子 Agent、能力协商、工具过滤、Persona、Depth | [Subagent](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.zh.md) |
| Agent Teams | 持久 Roster、Mailbox、共享 Task DAG 和 Replay；目前为实验性私有显式启用 seam | [Agent Teams](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/agent-team.zh.md) |
| Workflow/Job/Schedule | Workflow run、后台 Job registry、Session 内持久 Schedule | [Workflow](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/workflow.zh.md)、[Jobs](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/jobs.zh.md)、[Schedule](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/schedule.zh.md) |
| LLM Adapter | StreamChunk 协议、GenerateOptions、Adapter registry、Error handling | [LLM adapter](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/practice/llm-adapter.zh.md)、[LLM streaming](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/llm-streaming.zh.md) |
| Telemetry/Credentials | 遥测后端可替换并有脱敏 waterfall；凭据解析和描述分离 | [Telemetry](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session-telemetry.zh.md)、[Credentials](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/credentials.zh.md) |
| 插件开发 | 第一个 Plugin、Config schema、Tool、打包安装、三角色能力设计 | [Developer guide](https://github.com/deepseek-ai/deepseek-harness/tree/master/docs/user/develop) |

### 5.4 DeepSeek Harness 最值得学习的架构原则

1. **没有特权内核。** Model adapter、Tool registry、Session log、Agent loop 都是插件。
2. **能力按三角色设计。** Service Definition 固定契约，Provider 可替换，Consumer 决定怎样暴露给模型。
3. **注册必须可逆。** `ctx.effect()`/`ctx.on()` 注册的 Prompt、Tool、Listener、Provider 都要有 Dispose，保证 HMR/Teardown。
4. **模型可见即已记录。** 能进入模型的内容必须能从 Session 日志重建。
5. **能力协商要显式失败。** Subagent Provider 不支持某选项时拒绝启动，不能静默降级。
6. **沙箱和审批分离。** Permission preset 只是两个独立执行 knob 的组合，不能代替真实强制执行。

### 5.5 DeepSeek Harness 当前官方文档的明确空白

官方当前更重视 Runtime/Plugin 架构，没有单独的终端 Prompt 最佳实践、内置长期 Memory 产品指南、Agent Eval 教程或成熟生产部署安全承诺。System Prompt 页讲的是“怎样组合 Prompt section/context/tool schema”，不是“用户应该怎样写任务 Prompt”。这些空白必须如实标注。

### 5.6 DeepSeek Harness 推荐阅读顺序

1. [README](https://github.com/deepseek-ai/deepseek-harness/blob/master/README.zh.md) → [Safety](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.zh.md) → [Web UI](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/index.zh.md)。
2. [Cordis primer](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.zh.md) → [Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.zh.md)。
3. [First plugin](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/index.zh.md) → [Tool](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/tool.zh.md) → [Capability roles](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/practice/index.zh.md)。
4. [Session](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.zh.md) → [Tools](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/tools.zh.md) → [Subagent](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.zh.md) → [Sandbox/Approval](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/permission-presets.zh.md)。

# 第二部分：按 Agent 能力横向比较

## 6. Prompt：官方分别建议怎样写任务

### Codex

[官方 Prompting](https://learn.chatgpt.com/docs/prompting) 推荐按需提供 Goal、Context、Output、Boundaries。先说需要的结果，只有过程本身重要时才规定步骤；添加真正会影响结果的来源；边界只保留能避免真实损失的 1～2 项；重要任务要求最终检查。

适合 Codex 的任务结构：

```text
目标结果：修复登录超时后的刷新失败。
上下文：重点看 src/auth/，以现有 session 测试为模式。
边界：不要改变 token 格式，不要修改登录接口。
验收：先写失败测试，修复后运行 auth 测试和 typecheck，报告命令结果。
```

### Claude Code

[Best practices](https://code.claude.com/docs/en/best-practices#provide-specific-context-in-your-prompts) 更强调具体性：指定文件、场景、测试偏好；指向 Git history 或现有实现模式；描述症状和“修好是什么样”。同时要求可执行验证，并在复杂任务采用 Explore → Plan → Implement。

Claude 的官方例子体现“委派目标，不微管理实现”：告诉它找根因、复现、遵守已有模式和运行测试，而不是预写每一行代码。

### LangChain

LangChain 面向的是开发者构建 Agent。[Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering) 把 Prompt 视为 Model Context 的一个组成：System Prompt、Messages、Tools、Model、Response format 共同决定行为。官方建议动态内容通过 Runtime/Middleware 生成，不要把所有状态硬编码进一个静态 Prompt；LangGraph 还建议 State 保存原始数据，真正调用模型时再格式化 Prompt。

### OpenCode

当前官网没有独立 Prompt 最佳实践页。官方支持两层：一次性任务直接输入 TUI/`opencode run`；可复用 Prompt 用 [Commands](https://opencode.ai/docs/commands) 模板化，并插入参数、Shell 输出和文件引用。长期约束放 [`AGENTS.md`](https://opencode.ai/docs/rules)，不能把第三方 Prompt 方法冒充为 OpenCode 官方建议。

### DeepSeek Harness

当前官网没有面向最终用户的 Prompt 写作指南。[System prompt subsystem](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/system-prompt.zh.md) 解决的是 Runtime 组装：插件贡献有序 Section、动态 Context、变量和 Tool schema，Scope 可覆盖全局定义。对自建 Harness 的启发是把 Persona、Policy、Runtime facts、Tools 分段注册，而不是拼接一个不可维护的大字符串。

### 横向结论

- 使用产品时：Codex 的 Goal/Context/Output/Boundary 最通用，Claude 的“具体上下文 + 验证 + 分阶段”最工程化。
- 设计 Runtime 时：LangChain 强调 Context engineering，DeepSeek 强调 Prompt registry/section/scope。
- OpenCode 当前只提供 Command/Rule 机制，没有官方 Prompt 方法论，应避免过度推断。

## 7. 持久项目指令：每次会话都应知道什么

- **Codex**：[`AGENTS.md`](https://learn.chatgpt.com/docs/agent-configuration/agents-md)；全局和仓库/子目录层叠，目录越近越具体。只放稳定命令、约束和验证要求。
- **Claude Code**：[`CLAUDE.md`](https://code.claude.com/docs/en/memory) + `.claude/rules/`；官方建议简洁，当前给出的经验目标是单文件 200 行以内，路径规则可按需加载。
- **LangChain**：`system_prompt` 是 Agent 构造参数；动态用户/租户信息应通过 [Runtime/Middleware](https://docs.langchain.com/oss/python/langchain/runtime) 注入。Deep Agents 的 Always-on Memory 可加载 `AGENTS.md`。
- **OpenCode**：[`AGENTS.md`](https://opencode.ai/docs/rules)；`/init` 会扫描仓库生成或改善内容，还支持 `opencode.json` 的 instructions 和 Claude-compatible 文件。
- **DeepSeek Harness**：[Prompt section/context registry](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/system-prompt.zh.md)；这是插件级组装，不是约定一个仓库根 Prompt 文件。

共同原则：每次都需要的最小事实才进入 Always-on 指令；长流程放 Skill；强制安全规则放 Permission/Sandbox/Hook。

## 8. Context、Compaction、Memory 与 Session

| 系统 | 短期上下文 | 跨会话状态 | 官方重点 |
| --- | --- | --- | --- |
| Codex | Thread、History、`/compact`、Goal | 可选本地 Memory；团队规则仍放 `AGENTS.md` | [Memory](https://learn.chatgpt.com/docs/customization/memories) 默认关闭，生成会避开短/活跃会话并尝试脱敏 |
| Claude Code | Context window、Auto compact、Session resume/fork | Auto memory + `CLAUDE.md`；SDK 可接 External SessionStore | [Context](https://code.claude.com/docs/en/context-window)、[Session storage](https://code.claude.com/docs/en/agent-sdk/session-storage) |
| LangChain | Checkpointer 保存 Thread state | Store 保存 namespace/key 长期数据 | [Short-term](https://docs.langchain.com/oss/python/langchain/short-term-memory) 与 [Long-term](https://docs.langchain.com/oss/python/langchain/long-term-memory) 明确分层 |
| OpenCode | Session、Compaction、Export/Import | 当前官方无独立自动长期 Memory 指南 | [Config compaction](https://opencode.ai/docs/config#compaction)、[Session CLI](https://opencode.ai/docs/cli#session) |
| DeepSeek Harness | Append-only Session log、Compaction、Spill | Session persistence；长期语义 Memory 通过外部 MCP 示例 | [Persistence](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.zh.md)、[Memory MCP](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/mcp-memory.zh.md) |

不要把四个概念混为一谈：Context 是本轮模型可见内容；Compaction 是压缩；Session persistence 是恢复运行；Long-term memory 是跨 Thread 的语义召回。

## 9. Tool 与 MCP

### 共同点

五家都把 Tool schema 视为 Agent 能力接口，也都支持或连接 MCP。MCP 不只带来功能，还扩大信任边界和 Context 成本。

### 差异

- **Codex**：[MCP](https://learn.chatgpt.com/docs/extend/mcp) 同时提供 Tool、Resource、Prompt；可用 Skill 描述如何组合 MCP 工具。
- **Claude Code**：[MCP](https://code.claude.com/docs/en/mcp) 覆盖本地/远端传输、OAuth、Resource、Prompt、Elicitation、Tool search、动态更新、Managed policy；工具规模大时用 Tool search 延迟加载。
- **LangChain**：[Tools](https://docs.langchain.com/oss/python/langchain/tools) 可访问 State/Context/Store/Stream；[MCP](https://docs.langchain.com/oss/python/langchain/mcp) 支持 Interceptor、Progress、Logging 和 Elicitation。
- **OpenCode**：[Tool](https://opencode.ai/docs/tools) 内置 Read/Edit/Bash/LSP/Web/Skill/Todo；[Custom tool](https://opencode.ai/docs/custom-tools) 用 JS/TS 包装任意语言；MCP 文档直接提醒过多 Tool 描述会挤占 Context。
- **DeepSeek Harness**：[Tool Runtime](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/tools.zh.md) 提供 Pre/Execute/Post/Result waterfall、参数和输出双向校验、Scope 过滤；能力通过 Service Provider 可替换。

设计建议：Tool schema 要窄、描述要能区分、错误要结构化；读写和并发安全按单次调用判断；对大量工具采用按 Agent/Skill/Tool-search 渐进暴露。

## 10. Skill、Plugin 与 Hook

| 系统 | Skill | Plugin | Hook/等价扩展 |
| --- | --- | --- | --- |
| Codex | `SKILL.md` 渐进加载 | 分发 Skill/MCP/Hook 的安装单元 | 生命周期 Command/MCP Hook；项目 Hook 受 Trust 控制 |
| Claude Code | 知识或动作 Skill，可做 Eval、动态 Context、Subagent context | 打包 Skill/Agent/Hook/MCP/LSP/Monitor | Command/HTTP/MCP/Prompt/Agent Hook，事件最丰富 |
| LangChain | Deep Agents Skill 支持 Backend、Namespace、Permission、Script | Deep Agents Code Plugin | Middleware 是 Loop 内扩展；Deep Agents Code 另有命令 Hook |
| OpenCode | 兼容 `.opencode/.claude/.agents` Skill | JS/TS/npm Plugin | Plugin Event 即运行时 Hook，可注入 Context、保护 `.env`、监听 Compaction |
| DeepSeek Harness | Registry/Provider/Catalog 分层 | 所有能力本身就是 Cordis Plugin | Typed Event + reversible effect 是核心扩展机制 |

原文：[Codex Skills](https://learn.chatgpt.com/docs/build-skills)、[Claude Features](https://code.claude.com/docs/en/features-overview)、[Deep Agents Skills](https://docs.langchain.com/oss/python/deepagents/skills)、[OpenCode Plugins](https://opencode.ai/docs/plugins)、[Cordis primer](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.zh.md)。

判断方法：

- 需要模型理解和灵活执行：Skill。
- 需要每次事件确定性触发：Hook/Event/Middleware。
- 需要跨仓库安装和版本管理：Plugin。
- 需要外部系统数据或动作：MCP。

## 11. Plan、Goal、Task 与 Verification

- **Codex**：`/plan` 用于先研究方案；`/goal` 把第一条 Prompt 同时作为完成条件；官方要求 Outcome、Constraint、Verification，支持独立 Review 和 Worktree 并行。原文：[Long-running work](https://learn.chatgpt.com/docs/long-running-work)。
- **Claude Code**：Plan mode 只读研究；`/goal` 由独立 Evaluator 每轮检查；Stop Hook 可阻止结束；验证 Subagent/Workflow 可提供第二意见。原文：[Best practices](https://code.claude.com/docs/en/best-practices)、[Goal](https://code.claude.com/docs/en/goal)。
- **LangChain**：Todo/Rubric Middleware、LangGraph State/Interrupt、Deep Agents Code Goal 与独立 Rubric；AgentEvals 支持轨迹匹配和 LLM judge。原文：[Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)、[Goals and rubrics](https://docs.langchain.com/oss/deepagents/code/goals-and-rubrics)、[Agent Evals](https://docs.langchain.com/oss/python/langchain/test/evals)。
- **OpenCode**：Plan Primary Agent、Todo tool、LSP/Formatter、Undo/Redo；当前官网无持续 Goal evaluator 或专门 Agent Eval。原文：[Agents](https://opencode.ai/docs/agents)、[Tools](https://opencode.ai/docs/tools)。
- **DeepSeek Harness**：Plan/Goal/Todo 都是独立持久子系统；Goal 管生命周期和续跑，Todo 维护事件不变量。原文：[Plan](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/plan.zh.md)、[Goal](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/goal.zh.md)。

最可靠的完成门槛仍是 Agent 可运行的测试、build、lint、schema、视觉 diff 和独立审查证据。Todo 只证明“记录过任务”，不证明“结果正确”。

## 12. Subagent 与 Multi-agent

### 官方模型对比

| 系统 | 拓扑与状态 | 官方推荐场景 | 限制/风险 |
| --- | --- | --- | --- |
| Codex | 主 Agent 管理 Subagent thread，结果回主上下文；可定制角色 | 独立研究、探索、评审、专门工具/模型 | 并发写冲突；Subagent 增加 Token；需控制工具和权限 |
| Claude Subagent | 单 Session 委派，独立 Context，可后台运行或 Fork | 高噪声搜索、专门角色、Fresh review | 描述本身也占 Context；Fork 继承更多上下文成本 |
| Claude Agent View | 多独立 Session，由 Supervisor 管理和 Worktree 隔离 | 同时管理许多任务 | 不是 Teammate 协作网络 |
| Claude Agent Teams | Lead + Teammate + Task list + Direct message | 竞争假设、研究评审、跨层模块 | 实验性、成本高、协调/文件冲突 |
| Claude Workflow | Script 扇出大量 Subagent 并汇总/交叉验证 | 全仓审计、大迁移、大规模研究 | 需要控制规模、成本和重试 |
| LangChain | Subagent-as-tool、Handoff、Skill、Router、自定义 Graph | 业务路由、角色切换、并发 Workflow | 首先要解决 Context routing 和 State ownership |
| Deep Agents | 默认/自定义/动态/异步 Subagent，支持独立 Backend/Skill/Permission | 长任务隔离和部署级 Worker pool | 同步子 Agent 通常单次结果；异步需任务状态/传输 |
| OpenCode | Primary + Subagent，支持自动委派和 `@` 显式调用 | Explore、General、Scout、自定义专门角色 | 当前无官方 Team/共享任务板 |
| DeepSeek Harness | 多 Provider、一次性/可继续 Subagent；实验 Team 有持久 Mailbox/DAG | 研究可插拔 Provider、跨 Runtime 委派 | 预览期；Provider 能力不一致需显式协商 |

原文入口：[Codex Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)、[Claude parallel agents](https://code.claude.com/docs/en/agents)、[LangChain Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)、[OpenCode Agents](https://opencode.ai/docs/agents)、[DSH Subagent](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.zh.md)。

### 选择顺序

1. 单 Agent + 正确 Tool/Prompt 能否完成？
2. 是否只是偶尔加载知识，用 Skill 即可？
3. 是否只需隔离一次高噪声任务，用 Subagent？
4. 是否需要大量同构扇出和交叉验证，用 Workflow？
5. 是否真的需要成员持续通信、共享任务和互相挑战，才用 Team？

## 13. Permission、Approval、Sandbox 与 Network

四个概念应分开：

- **Permission**：允许访问哪些资源或调用哪些工具。
- **Approval**：什么动作必须暂停让人/分类器决定。
- **Sandbox**：操作系统或远端执行环境真正强制的文件/进程/网络边界。
- **Network policy**：可访问的域、地址、协议和方法。

各家官方路线：

- **Codex**：Sandbox + Approval policy + Rules + Permission profile + Network proxy；项目 `.codex` 只在 Trust 后加载。原文：[Security](https://learn.chatgpt.com/docs/agent-approvals-security)。
- **Claude Code**：Tool rules + Permission mode + Sandboxed Bash + Managed policy；Hook 能额外阻断，但 Hook 本身也需 Trust。原文：[Permissions](https://code.claude.com/docs/en/permissions)、[Sandbox](https://code.claude.com/docs/en/sandboxing)。
- **LangChain/Deep Agents**：HITL Middleware 与 Backend Permission；任意 Shell/Sandbox execution 需要执行环境自己的隔离。原文：[Deep Agents Permissions](https://docs.langchain.com/oss/python/deepagents/permissions)、[Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes)。
- **OpenCode**：Tool Permission 为 allow/ask/deny；`--auto` 只放行 ask，不绕过 deny；Policy 是实验性资源规则。原文：[Permissions](https://opencode.ai/docs/permissions)。
- **DeepSeek Harness**：Sandbox mode 和 Approval policy 是独立 knob；Approval 无应答或异常时 fail closed；官方不保证 Sandbox 绝对隔离。原文：[Permission presets](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/permission-presets.zh.md)、[Safety](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.zh.md)。

共同建议：默认最小权限；Research/Plan 角色只读；写文件、外部发送、删除、Secret 和公网访问分开授权；不可信插件/MCP/Shell 是新的信任边界。

## 14. State、Persistence、Resume 与 Lifecycle

- **Codex**：Thread/Turn、Resume/Fork、History、Goal、Compaction；App Server 提供完整生命周期事件和中断。原文：[App Server lifecycle](https://learn.chatgpt.com/docs/app-server#lifecycle-overview)。
- **Claude Code**：Session Resume/Fork、Checkpoint、Transcript；Agent SDK 可外接 SessionStore，Host 文档要求处理 Subprocess、取消、扩缩容和多租户。原文：[Sessions](https://code.claude.com/docs/en/sessions)、[SDK storage](https://code.claude.com/docs/en/agent-sdk/session-storage)。
- **LangGraph**：Checkpointer 是 Thread 内恢复基础，Store 是跨 Thread 数据；Interrupt/Replay/Time travel/Drain 构成最完整的 Durable execution 模型。原文：[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)。
- **OpenCode**：Server 管理 Session/Message/Event，CLI 可 export/import；当前官网没有 LangGraph 式 Checkpoint/Replay 语义说明。原文：[Server](https://opencode.ai/docs/server)。
- **DeepSeek Harness**：Append-only Session Event 是真源；Persistence 有 Flush checkpoint、冷恢复和 interrupted turn 修复；Cordis Effect 确保 Dispose/HMR。原文：[Persistence](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.zh.md)、[Lifecycle](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/framework/index.zh.md)。

设计自己的 Agent 时，至少要有稳定 ID、显式状态机、原子/追加写、取消信号、崩溃恢复、后台任务清理和可解释终态。只保存聊天摘要不等于可恢复执行。

## 15. 自动化、CI、定时与外部触发

| 系统 | 非交互 | CI/代码托管 | 定时/事件 |
| --- | --- | --- | --- |
| Codex | [`codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode) | [GitHub Action](https://learn.chatgpt.com/docs/github-action) | [Scheduled tasks](https://learn.chatgpt.com/docs/automations) |
| Claude Code | [`claude -p`](https://code.claude.com/docs/en/headless) | [GitHub Actions](https://code.claude.com/docs/en/github-actions)、GitLab CI | [Cloud Routines](https://code.claude.com/docs/en/routines)、[Session schedules](https://code.claude.com/docs/en/scheduled-tasks) |
| LangChain | Agent invoke/stream/API | LangGraph/LangSmith Deployment | 由应用/平台调度；不是 CLI 固定功能 |
| OpenCode | `opencode run` | [GitHub](https://opencode.ai/docs/github)、[GitLab](https://opencode.ai/docs/gitlab) | GitHub Event/Schedule 示例 |
| DeepSeek Harness | `headless`/SDK Profile | [GitHub Webhook review](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/github-review.zh.md) | [Session schedule](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/schedule.zh.md)、Webhook/Workflow/Job 子系统 |

自动化必须显式提供认证、最小权限、结构化输出、超时/取消、幂等和失败通知；交互模式中“可以问用户”的路径，在 CI 中通常必须改为预配置或失败。

## 16. SDK、Server、Protocol、Streaming 与 Structured Output

- **Codex SDK**：TypeScript/Python 控制 Thread；[App Server](https://learn.chatgpt.com/docs/app-server) 是更完整的 JSON-RPC 协议，覆盖 Thread、Turn、Event、Review、Approval、Skill、Auth、Filesystem。
- **Claude Agent SDK**：把 Claude Code Agent loop 作为库，支持 Session、Custom tools、MCP、Tool search、Subagent、Hook、Permission、Structured output、OTel 和 Hosting。原文：[SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)。
- **LangChain/LangGraph**：`create_agent`/Graph 是应用 SDK；Streaming 可投影 Token、Tool、State、Subagent；Structured output 可用 Provider/Tool strategy。LangSmith Agent Server 提供部署 API。
- **OpenCode**：Server 暴露 OpenAPI 3.1/SSE，SDK 可启动 Server 或只连接 Client，并提供 Session/File/TUI/Auth/Event 和 JSON Schema output。原文：[Server](https://opencode.ai/docs/server)、[SDK](https://opencode.ai/docs/sdk)。
- **DeepSeek Harness**：Python SDK 通过 NDJSON-RPC/stdio 驱动同版本 `dsh --profile sdk`；ACP 是另一自动化入口；LLM/Tool/Session 都有类型化事件协议。原文：[Python SDK](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/python-sdk.zh.md)。

选型时不要只看“有没有 SDK”，还要看：是否能恢复 Session、是否有事件流、审批怎样回传、Structured output 失败怎样处理、工具取消和并发怎样建模、状态是否可外置。

## 17. Observability、Evaluation 与 Production

- **Codex**：本地可选 OTel 事件/Metric，管理员有 Managed config/Analytics；自动化应保留 JSONL 和验证证据。原文：[Monitoring and telemetry](https://learn.chatgpt.com/docs/agent-approvals-security#monitoring-and-telemetry)。
- **Claude**：Agent SDK 用 OpenTelemetry 导出 Trace/Metric/Event，可标注最终用户并控制敏感数据；Hosting 单独讨论 Session、Cost、Concurrency 和 Multi-tenant isolation。原文：[Observability](https://code.claude.com/docs/en/agent-sdk/observability)。
- **LangChain**：LangSmith 是最完整的一体化 Trace/Eval/Studio/Deployment 路线；AgentEvals 支持轨迹和 Judge。原文：[Observability](https://docs.langchain.com/oss/python/langchain/observability)、[Evals](https://docs.langchain.com/oss/python/langchain/test/evals)。
- **OpenCode**：Server/SDK 有 Event/Logging，CLI 有 stats；当前官方顶层文档没有专门 Agent Eval 体系。原文：[Server APIs](https://opencode.ai/docs/server#apis)。
- **DeepSeek Harness**：Session telemetry 是可替换后端且有脱敏 waterfall；仓库开发 CI 有门禁，但产品仍未安全审计，不能据此宣称 Production-ready。原文：[Telemetry](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session-telemetry.zh.md)、[Development](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/development.zh.md)。

生产评估至少应覆盖：任务成功率、工具选择/参数正确率、轨迹、恢复成功率、人工拒绝率、延迟、Token/费用、权限违规、错误分类和代表性回归集。

## 18. 从五家官方文档归纳出的 Agent Harness 设计框架

### 18.1 Instructions

一次性要求放任务 Prompt；稳定仓库事实放 `AGENTS.md/CLAUDE.md`；路径特定规则局部加载；长流程做 Skill；强制策略落到 Permission/Hook/Sandbox。

### 18.2 State

分开保存 Thread/Session state、跨会话 Memory、Task/Goal state、Append-only event/trace。所有状态有 Schema、稳定 ID、作用域、版本和恢复语义。

### 18.3 Verification

让 Agent 能自己运行 Test/Build/Lint/Schema/Visual diff；记录命令和结果；重要任务用新 Context Reviewer 或 Eval 推翻结论。没有证据只能叫“声称完成”。

### 18.4 Scope

默认最小目录、最小 Tool、最小网络和最小 Secret；Research 只读；并行 Worker 有明确文件所有权；外部 MCP/Plugin/CLI 均作为信任边界。

### 18.5 Lifecycle

明确 Bootstrap、Trust、Run、Pause、Approval、Cancel、Compact、Checkpoint、Resume、Handoff、Complete/Failed/Blocked 和 Teardown。后台任务和插件注册必须能清理。

## 19. 实践路线

1. 选择 Codex 或 Claude Code，在熟悉仓库完成三个小任务，每个都写清验收并让 Agent 自证。
2. 把重复纠正沉淀成简短 `AGENTS.md/CLAUDE.md`，把偶尔使用的长流程做成 Skill。
3. 接一个 MCP，检查 Tool schema、Context 成本、认证和 Permission。
4. 用一个只读 Subagent 做代码库研究，再用独立 Reviewer 验证，观察上下文与成本。
5. 用 LangChain `create_agent` 做最小 Agent，加 Structured output、Middleware、Checkpointer、Store 和 HITL。
6. 用 LangGraph 实现一次可暂停、可恢复、可 Replay 的有状态流程。
7. 阅读 DeepSeek Harness 的 Session、Tool、Service/Event、Provider 和 Lifecycle 设计，尝试替换一个 Provider。
8. 只有明确需要共享任务和直接通信时，再实验 Agent Team。

## 20. 最终检查表

### 使用 Agent

- [ ] Prompt 是否说明结果、上下文、边界和验收？
- [ ] 是否先研究/计划了真正复杂的改动？
- [ ] 项目指令是否短、稳定、无冲突并能确认已加载？
- [ ] Context 是否只含相关材料，长日志是否隔离/压缩？
- [ ] 是否使用最小 Tool、目录、网络和 Secret？
- [ ] Agent 是否实际执行验证并给出证据？
- [ ] 并行任务是否有 Worktree 或互斥文件所有权？
- [ ] 最终状态是否区分 Complete、Failed、Blocked？

### 设计 Agent

- [ ] Agent loop、Deterministic workflow、UI/Transport 是否解耦？
- [ ] System Prompt、Runtime context、Tool schema、Response format 是否分层？
- [ ] Tool error、Timeout、Cancel、Retry、Idempotency 是否明确？
- [ ] Thread state、Long-term memory、Task state、Event log 是否分层？
- [ ] Permission、Approval、Sandbox、Network 是否分别建模并 fail closed？
- [ ] Interrupt、Crash、Compaction、Deploy 后是否可恢复？
- [ ] Skill/Tool 是否渐进暴露，避免 Context/Tool overload？
- [ ] 是否有 Trace、Metric、Eval、Regression 和安全审计？
- [ ] Plugin/Hook/Background task 是否可 Dispose/Drain？
- [ ] Multi-agent 是否解决了清晰问题，并限制 Depth、Concurrency、Spend 和写冲突？

## 总结

五家官方文档虽然产品形态不同，但共同指向同一件事：可靠 Agent 的关键不是更长 Prompt 或更多 Agent，而是正确的 Context、清晰的 Tool、持久的 State、可执行的 Verification、最小的 Scope 和完整的 Lifecycle。

从使用角度，Codex 的定制层次和 Claude Code 的工程闭环最值得直接照做；OpenCode 的开源 Client/Server 与配置系统适合自托管。从构建角度，LangChain/LangGraph/Deep Agents 提供最完整的应用框架路线；DeepSeek Harness 则提供最细的插件、事件、会话日志和可替换能力 seam 设计参考。
