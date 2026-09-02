# AI Coding Agent 学习资源调研

调研日期：2026-08-25

官方专题：[Codex、Claude Code、LangChain、OpenCode 与 DeepSeek Harness 官方指南](official-agent-guide.md)（2026-09-02 核验，含使用路径、多智能体对比和自建 Harness 设计方法）。

参考对象：[AIHero](https://www.aihero.dev/)。

AIHero 的核心定位不是泛泛讲 AI，而是把 coding agents 当成真实工程工具来使用：从需求澄清、上下文收集、计划、切片、测试、代码审查到交付，强调 engineering fundamentals、可安装 skills、AGENTS.md/CLAUDE.md、上下文管理和 human-in-the-loop。

## 筛选标准

本次筛选优先考虑和 AIHero 高度相似的资源：

- 面向开发者使用 Claude Code、Codex、Cursor、Amp 等 coding agents。
- 关注真实工程流程，而不是只做 prompt tips 或演示型 toy app。
- 覆盖上下文工程、计划模式、TDD、代码审查、skills、MCP、AGENTS.md/CLAUDE.md 等可落地方法。
- 内容结构适合系统学习或沉淀团队规范。

## 高度相似资源

| 资源 | 链接 | 相似点 | 适合用途 |
| --- | --- | --- | --- |
| Codegen: Build Your First AI Coding Agent | https://codegen.com/courses/build-your-first-agent/ | 免费 6 课，覆盖 CLAUDE.md、skills、MCP、code review agent，和 AIHero 的“可执行工作流/技能”路线很接近。 | 快速理解 coding agent 的基础构件，并搭一个 review agent。 |
| DeepLearning.AI: Claude Code: A Highly Agentic Coding Assistant | https://www.deeplearning.ai/courses/claude-code-a-highly-agentic-coding-assistant | 与 Anthropic 合作的短课，讲 Claude Code 的 agentic coding best practices。 | 系统学习 Claude Code 的官方推荐用法。 |
| Anthropic Claude Code Best Practices | https://code.claude.com/docs/en/best-practices | 官方实践指南，覆盖 Claude Code 在代码库探索、计划、实现和验证中的工作方式。 | 制定团队使用 Claude Code 的基础规范。 |
| HumanLayer: Advanced Context Engineering for Coding Agents | https://www.humanlayer.dev/blog/advanced-context-engineering | 重点是先研究、计划、架构，再让 agent 写代码；强调人类在设计决策中的位置。 | 复杂代码库、长任务、上下文工程和 human-in-the-loop。 |
| Tactical Agentic Coding | https://agenticengineer.com/tactical-agentic-coding | 进阶 agent orchestration、parallel execution、production deployment 取向。 | 已经熟悉 Claude Code/Codex 后，学习更高阶的 agentic engineering 技巧。 |

## 标准与技能生态

| 资源 | 链接 | 价值 |
| --- | --- | --- |
| AGENTS.md | https://agents.md/ | 面向 coding agents 的开放项目说明格式，可理解为 agent 版 README。 |
| OpenAI: Custom instructions with AGENTS.md | https://learn.chatgpt.com/docs/agent-configuration/agents-md | Codex 官方 AGENTS.md 配置说明，适合沉淀项目级指令。 |
| MCP Servers: Claude Skills & Agent Skills Library | https://mcpservers.org/agent-skills | 可浏览 Claude Code、Codex、Cursor 等可复用 agent skills。 |
| VoltAgent awesome-agent-skills | https://github.com/VoltAgent/awesome-agent-skills | GitHub 上的大规模 agent skills 集合，适合找工作流模板。 |
| Amp Owner's Manual | https://ampcode.com/manual | Amp 官方手册，包含 skills、MCP、上下文和工具使用说明。 |

## 厂商实践型内容

| 资源 | 链接 | 价值 |
| --- | --- | --- |
| Cursor Learn | https://cursor.com/learn | Cursor 官方学习入口，适合理解 AI coding mental model 和常见实践。 |
| Cursor Agent Best Practices | https://cursor.com/blog/agent-best-practices | 讲 agent + TDD + 测试闭环，适合把 agent 输出约束到可验证结果。 |
| Anthropic: Effective Context Engineering for AI Agents | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | 更偏 agent 系统层面的上下文压缩、保留和工具设计。 |
| Coursera: Claude Code in Action | https://www.coursera.org/learn/claude-code-in-action | 偏长会话、hands-off、hooks、rules、cloud routines 等实践。 |

## 个人与团队博客

| 资源 | 链接 | 价值 |
| --- | --- | --- |
| Hamel Husain: Coding Agents Notes | https://hamel.dev/ | 高质量工程经验，偏工具判断和真实使用反馈。 |
| Hamel Husain: Amp Notes | https://hamel.dev/notes/coding-agents/amp.html | 分析 Amp 的 MCP、AGENTS.md、权限、headless mode 等能力。 |
| Builder.io: How I Use Claude Code | https://www.builder.io/blog/claude-code | Claude Code 作为日常主力工作界面的实践经验。 |
| Builder.io: Cursor Tips | https://www.builder.io/blog/cursor-tips | Cursor 实战技巧，包含 TDD、debug、自动迭代。 |
| Martin Fowler: Building Your Own CLI Coding Agent | https://martinfowler.com/articles/build-own-coding-agent.html | 从工程角度拆解如何构建 CLI coding agent。 |

## 更宽泛但有补充价值

| 资源 | 链接 | 价值 |
| --- | --- | --- |
| roadmap.sh AI Agents | https://roadmap.sh/ai-agents | AI agent 学习路线，适合补齐基础概念。 |
| roadmap.sh AI Engineer | https://roadmap.sh/ai-engineer | AI engineer 路线图，覆盖 LLM 应用、RAG、工具调用等基础。 |
| Langfuse AI Engineering Library | https://langfuse.com/library | AI engineering 资源集合，偏 LLM 应用、evaluation、observability。 |
| Scrimba AI Engineer Path | https://scrimba.com/ai-engineer-path-c02v | 面向 Web/JavaScript 开发者的 AI engineer 学习路径。 |
| DataCamp: Introduction to Agent Skills | https://www.datacamp.com/courses/introduction-to-agent-skills | 面向 agent skills 的课程化入门。 |
| DeepLearning.AI: Agent Skills with Anthropic | https://www.deeplearning.ai/courses/agent-skills-with-anthropic | 专门讲 Anthropic agent skills 的短课。 |

## 优先级建议

如果目标是复刻 AIHero 的价值，而不是泛泛学习 AI，建议按以下顺序看：

1. AIHero：先建立“real engineering with coding agents”的主线。
2. Codegen Build Your First AI Coding Agent：快速落地 CLAUDE.md、skills、MCP 和 review agent。
3. Anthropic Claude Code Best Practices / DeepLearning.AI Claude Code：补官方实践。
4. HumanLayer Advanced Context Engineering：理解复杂任务如何前置研究、计划和架构。
5. AGENTS.md / OpenAI AGENTS.md：沉淀项目级规范，服务 Codex、Claude Code、Cursor、Copilot 等工具。
6. Skills libraries：按需挑选可复用 workflow，形成团队自己的 skill set。

## 结论

和 AIHero 最接近的不是传统 AI 课程，而是“agentic software engineering”资源：它们共同关注如何让 agent 在真实代码库里稳定产出，而不是单次 prompt 的惊艳效果。

最值得优先跟进的五个资源是：

- Codegen: Build Your First AI Coding Agent
- DeepLearning.AI: Claude Code: A Highly Agentic Coding Assistant
- Anthropic Claude Code Best Practices
- HumanLayer: Advanced Context Engineering for Coding Agents
- AGENTS.md
