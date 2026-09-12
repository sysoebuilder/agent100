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
| 多轮对话 | GLM / DeepSeek 流式输出、终端 Markdown 展示、本地会话保存与加载 |
| Agent 循环 | 模型选择工具、执行工具、回传结果并继续推理；限制最大模型调用轮数 |
| 工具系统 | 统一描述、注册表、调用与结果结构；JSON Schema 参数校验 |
| 内置工具 | 当前时间、基于 GLM 搜索引擎的网络搜索、QQ SMTP 邮件发送、Windows PowerShell 命令执行 |
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
GLM_MODEL=GLM-5.3-flash
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

运行 `agent100 config` 选择 GLM 或 DeepSeek，并输入模型名称（回车使用 `.env` 中的名称，不校验模型名称）。命令把选定模型和对应 Key、Base URL 写入 `data/active-model.env`，不会修改源 `.env`。之后 `chat` 和 `taskplan` 只使用当前配置；配置缺失时提示执行 `agent100 config` 并退出。修改源 `.env` 后需再次运行 `config` 才会同步。详见 [模型配置](docs/model-configuration.md)。

```powershell
.\.venv\Scripts\agent100.exe --version
.\.venv\Scripts\agent100.exe config
.\.venv\Scripts\agent100.exe chat
.\.venv\Scripts\agent100.exe taskplan "搜索 Python 日志最佳实践并整理学习提纲"
```

Web Search 功能基于 GLM 搜索引擎实现，必须配置 GLM API Key（环境变量名为 `ZAI_API_KEY`）；即使当前对话模型选择 DeepSeek，该配置仍不可省略。邮件工具另需配置 `QQ_SMTP_USERNAME` 和 `QQ_SMTP_AUTH_CODE`，后者是 SMTP 授权码，不是邮箱登录密码。模型和搜索调用使用你自己的服务账号，可能产生费用。

会话、日志和计划默认保存在本地 `data/`；可用 `LLM_CLI_DATA_DIR` 指定其他目录。不要提交真实 `.env` 或个人会话数据。

### 代码导航

| 路径 | 内容 |
| --- | --- |
| `src/ai_agent_learning/agent/` | Agent 循环、状态、停止策略与上下文管理 |
| `src/ai_agent_learning/tools/` | 工具协议、注册、执行与恢复策略 |
| `src/ai_agent_learning/planning/` | 计划生成、校验、执行与持久化 |
| `src/ai_agent_learning/model.py` | GLM、DeepSeek 模型 API 适配与流式生成 |
| `src/ai_agent_learning/ui/` | 终端展示与工具确认 |
| `tests/` | 模型适配、Agent、工具、运行时与终端相关测试 |
| `docs/` | 实现说明与设计细节 |
| `days/` | 学习练习与里程碑目录；部分目录仍为占位说明 |

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
| Multi-turn chat | GLM / DeepSeek streaming, terminal Markdown rendering, local session persistence and loading |
| Agent loop | Model-selected tools, execution, result feedback, and a model-call limit |
| Tool system | Shared specifications, registry, call/result contracts, and JSON Schema validation |
| Built-in tools | Current time, GLM-powered web search, QQ SMTP email, and Windows PowerShell commands |
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

Manually configure the provider credentials, model names, and base URLs in project `.env` using `.env.example`. Run `agent100 config` to choose GLM or DeepSeek and enter a model name (Enter keeps the name from `.env`; names are not validated). The command copies the selection to `data/active-model.env` without editing the source `.env`. Chat and task planning only use this snapshot; missing configuration directs you to `agent100 config`. Source changes take effect only after running `config` again. See [model configuration](docs/model-configuration.md).

```powershell
.\.venv\Scripts\agent100.exe --version
.\.venv\Scripts\agent100.exe config
.\.venv\Scripts\agent100.exe chat
.\.venv\Scripts\agent100.exe taskplan "Search for Python logging practices and prepare a study outline"
```

Web Search is powered by the GLM search engine and requires a GLM API key (`ZAI_API_KEY`); this configuration is mandatory even when DeepSeek is selected as the chat model. Email requires `QQ_SMTP_USERNAME` and `QQ_SMTP_AUTH_CODE` (an SMTP authorization code, not your mailbox password). Model and search requests use your own service account and may incur charges.

Sessions, logs, and plans are stored in `data/` by default. Set `LLM_CLI_DATA_DIR` to override the location. Keep real `.env` files and personal session data out of version control. The current terminal interface is primarily in Chinese.

### Code map and development

`agent/` contains the loop, state, policies, and context manager; `tools/` contains contracts and execution; `planning/` handles plans and step execution; `ui/` handles terminal interaction. These modules live under `src/ai_agent_learning/`. `model.py` contains the GLM and DeepSeek clients used by the CLI.

`tests/` covers model adapters, agent behavior, tools, runtime, and terminal behavior. `docs/` contains implementation notes, primarily in Chinese. `days/` tracks learning exercises and milestones; some directories are placeholders rather than implemented applications.

Implementation notes: [GLM integration](docs/glm-model-client.md) · [Web search](docs/web-search.md) · [Context estimates](docs/context-token-estimation.md) · [Terminal UI](docs/terminal-ui.md)

```powershell
.\.venv\Scripts\python.exe -m pip install pytest build
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m build
```

Build artifacts are written to `dist/`: a `.whl` installation package and a `.tar.gz` source distribution. A standalone executable is not currently provided.
