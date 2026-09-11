# agent100

**从模型 API 到工具调用：一个持续迭代的 Python Agent CLI。**  
**From model APIs to tool use: an evolving Python Agent CLI.**

Version **0.3.0** · Python **3.11+** · Command **`agent100`**

[中文](#中文) · [English](#english)

## 中文

### 项目与求职方向

我正在寻找 **AI Agent / LLM 应用开发相关岗位**。agent100 是我的个人学习与求职作品，用来展示如何把模型 API 接入推进到可运行、可追踪、有执行边界的 Agent 应用。

项目随我的 100 天 Agent 学习计划迭代。核心循环、工具协议、失败恢复和上下文管理使用 Python 显式实现，便于阅读代码和讨论工程取舍。目前是持续迭代的个人项目，尚未进行生产环境验证。

如果你正在招聘 Agent 开发者，欢迎通过邮箱 [1774364027w@gmail.com](mailto:1774364027w@gmail.com) 联系我，也可以访问 [GitHub 个人主页](https://github.com/sysoebuilder) 或在本仓库发起技术交流。我希望围绕工具调用、工作流编排和 LLM 应用可靠性开展工作。

### 已实现的能力

| 能力 | 实现内容 |
| --- | --- |
| 多轮对话 | GLM 流式输出、终端 Markdown 展示、本地会话保存与加载 |
| Agent 循环 | 模型选择工具、执行工具、回传结果并继续推理；限制最大模型调用轮数 |
| 工具系统 | 统一描述、注册表、调用与结果结构；JSON Schema 参数校验 |
| 内置工具 | 当前时间、网络搜索、QQ SMTP 邮件发送、Windows PowerShell 命令执行 |
| 执行控制 | 风险等级检查；聊天模式下，带副作用的工具执行前展示参数并请求确认 |
| 失败恢复 | 区分临时失败、永久失败与结果未知，结合副作用和幂等能力决定是否重试 |
| 上下文管理 | 增量估算 Token，达到阈值后请求精确计数，压缩旧消息并保留近期对话 |
| 任务规划 | 结构化计划、工具与依赖校验、计划确认、步骤状态跟踪及结果保存 |
| 工程基础 | 结构化日志、模块化代码、pytest 测试、wheel 与源码发行包构建 |

### 快速开始

以下使用 Windows PowerShell。当前命令执行工具仅支持 Windows；工作目录不构成文件访问沙箱。

```powershell
git clone https://github.com/sysoebuilder/agent100.git
cd agent100
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

编辑 `.env`，填写自己的配置：

```dotenv
ZAI_API_KEY=your_api_key
GLM_REASONING_MODEL=GLM-5.3-flash
```

当前 CLI 使用智谱 GLM；可按账号可用模型调整名称，所选模型和服务需支持项目使用的工具调用、结构化输出和 Token 计数接口。缺少 API Key 时，CLI 也会提示输入并保存到本地 `data/.env`。

```powershell
.\.venv\Scripts\agent100.exe --version
.\.venv\Scripts\agent100.exe chat
.\.venv\Scripts\agent100.exe taskplan "搜索 Python 日志最佳实践并整理学习提纲"
```

邮件工具另需配置 `QQ_SMTP_USERNAME` 和 `QQ_SMTP_AUTH_CODE`，后者是 SMTP 授权码，不是邮箱登录密码。模型和搜索调用使用你自己的服务账号，可能产生费用。

会话、日志和计划默认保存在本地 `data/`；可用 `LLM_CLI_DATA_DIR` 指定其他目录。不要提交真实 `.env` 或个人会话数据。

### 面试时可以讨论的工程问题

- **模型与执行器如何分工？** 模型提出调用，程序负责参数校验、执行控制和结果封装。
- **为什么失败不能一律重试？** 查询失败与邮件发送结果未知需要不同的恢复策略。
- **如何管理长对话？** 用本地增量估算减少计数请求，再按阈值压缩历史。
- **如何检查执行过程？** 使用结构化工具结果、步骤状态和日志记录排查问题。
- **当前边界在哪里？** 人工确认不等于操作系统沙箱；搜索结果中的恶意指令和错误信息仍需进一步测试与防护。

### 代码导航

| 路径 | 内容 |
| --- | --- |
| `src/ai_agent_learning/agent/` | Agent 循环、状态、停止策略与上下文管理 |
| `src/ai_agent_learning/tools/` | 工具协议、注册、执行与恢复策略 |
| `src/ai_agent_learning/planning/` | 计划生成、校验、执行与持久化 |
| `src/ai_agent_learning/model.py` | GLM 模型 API 适配与流式生成 |
| `src/ai_agent_learning/ui/` | 终端展示与工具确认 |
| `tests/` | 模型适配、Agent、工具、运行时与终端相关测试 |
| `docs/` | 实现说明与设计细节 |
| `days/`、`projects/` | 学习练习与里程碑目录；部分目录仍为占位说明 |

实现说明：[GLM 接入](docs/glm-model-client.md) · [搜索工具](docs/web-search.md) · [上下文估算](docs/context-token-estimation.md) · [终端界面](docs/terminal-ui.md)

### 测试与打包

```powershell
.\.venv\Scripts\python.exe -m pip install pytest build
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m build
```

构建产物位于 `dist/`：`.whl` 用于安装，`.tar.gz` 为源码发行包。项目尚未提供独立 EXE。

## English

### Project and career interests

I am looking for **AI Agent / LLM application engineering roles**. agent100 is my personal learning and portfolio project, demonstrating the progression from a model API integration to a runnable agent with execution controls and traceable behavior.

The project evolves alongside my 100-day Agent learning plan. The agent loop, tool contracts, recovery policies, and context management are implemented explicitly in Python so that the code and engineering tradeoffs can be reviewed. This is an actively developed personal project; it has not been validated for production use.

If you are hiring Agent developers, please contact me at [1774364027w@gmail.com](mailto:1774364027w@gmail.com), visit my [GitHub profile](https://github.com/sysoebuilder), or start a technical discussion in this repository. My interests include tool use, workflow orchestration, and reliable LLM applications.

### Implemented features

| Capability | Implementation |
| --- | --- |
| Multi-turn chat | GLM streaming, terminal Markdown rendering, local session persistence and loading |
| Agent loop | Model-selected tools, execution, result feedback, and a model-call limit |
| Tool system | Shared specifications, registry, call/result contracts, and JSON Schema validation |
| Built-in tools | Current time, web search, QQ SMTP email, and Windows PowerShell commands |
| Execution controls | Risk checks; chat mode displays arguments and requests approval before side-effecting tools run |
| Failure recovery | Transient, permanent, and unknown-outcome failures; retry decisions account for side effects and idempotency |
| Context management | Incremental token estimates, threshold-triggered exact counting, and history compaction retaining recent messages |
| Task planning | Structured plans, tool and dependency validation, plan approval, step status tracking, and saved results |
| Engineering foundations | Structured logs, modular code, pytest tests, and wheel/source distribution builds |

### Quick start

The commands below use Windows PowerShell. The command execution tool currently requires Windows; its working directory is not a filesystem sandbox.

```powershell
git clone https://github.com/sysoebuilder/agent100.git
cd agent100
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
```

Set `ZAI_API_KEY` in `.env`. The current CLI uses GLM, with `GLM_REASONING_MODEL=GLM-5.3-flash` as the default. You can choose an available model, provided the model and service support the tool calling, structured output, and token counting interfaces used by the project. If the key is missing, the CLI can prompt for it and save it locally in `data/.env`.

```powershell
.\.venv\Scripts\agent100.exe --version
.\.venv\Scripts\agent100.exe chat
.\.venv\Scripts\agent100.exe taskplan "Search for Python logging practices and prepare a study outline"
```

Email requires `QQ_SMTP_USERNAME` and `QQ_SMTP_AUTH_CODE` (an SMTP authorization code, not your mailbox password). Model and search requests use your own service account and may incur charges.

Sessions, logs, and plans are stored in `data/` by default. Set `LLM_CLI_DATA_DIR` to override the location. Keep real `.env` files and personal session data out of version control. The current terminal interface is primarily in Chinese.

### Engineering topics for an interview

- **Separating model decisions from execution:** the model proposes calls; application code validates, controls, and executes them.
- **Retrying safely:** failed lookups and email sends with unknown outcomes require different recovery policies.
- **Managing long conversations:** incremental estimates reduce counting requests; threshold-based compaction limits context growth.
- **Inspecting execution:** structured results, step states, and logs help diagnose failures.
- **Understanding the limits:** human approval is not an OS sandbox; malicious instructions and false information in search results require further testing and defenses.

### Code map and development

`agent/` contains the loop, state, policies, and context manager; `tools/` contains contracts and execution; `planning/` handles plans and step execution; `ui/` handles terminal interaction. These modules live under `src/ai_agent_learning/`. `model.py` contains the GLM adapter and streaming generation used by the CLI.

`tests/` covers model adapters, agent behavior, tools, runtime, and terminal behavior. `docs/` contains implementation notes, primarily in Chinese. `days/` and `projects/` track learning exercises and milestones; some directories are placeholders rather than implemented applications.

Implementation notes: [GLM integration](docs/glm-model-client.md) · [Web search](docs/web-search.md) · [Context estimates](docs/context-token-estimation.md) · [Terminal UI](docs/terminal-ui.md)

```powershell
.\.venv\Scripts\python.exe -m pip install pytest build
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m build
```

Build artifacts are written to `dist/`: a `.whl` installation package and a `.tar.gz` source distribution. A standalone executable is not currently provided.
