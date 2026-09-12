# 模型配置

只有两个固定客户端：`glm` 对应 `GlmModelClient`，`deepseek` 对应 `DeepSeekClient`。模型名称由用户提供，不校验名称或调用 API 验证可用性。

## 1. 手动维护候选配置

编辑项目根目录 `.env`，填写需要使用的提供商：

```dotenv
ZAI_API_KEY=your_glm_key
GLM_MODEL=GLM-5.3-flash
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

代码不使用 `DEFAULT_MODEL` 补齐模型名称，也不再读取 `GLM_REASONING_MODEL`。

## 2. 选择当前模型

```powershell
agent100 config
```

选择提供商后输入模型名称；回车使用该提供商 `.env` 中的名称。命令只读取 `.env`，不会要求输入 Key 或修改源文件。Key 和 Base URL 缺失时，请先手动补全源文件。

选择后完整替换 `data/active-model.env`，内容只有四项：

```dotenv
MODEL_PROVIDER='deepseek'
MODEL_NAME='deepseek-v4-flash'
MODEL_API_KEY='your_deepseek_key'
MODEL_BASE_URL='https://api.deepseek.com'
```

`LLM_CLI_DATA_DIR` 可覆盖数据目录；默认数据目录位于当前项目根目录下。从项目根目录运行命令，确保配置和启动使用同一目录。当前文件包含明文 Key，不要提交或分享。

## 3. 启动

`agent100 chat` 和 `agent100 taskplan` 由 `create_runtime()` 读取当前配置。四个模型字段只来自 `active-model.env`，不受系统环境变量或源 `.env` 中同名字段覆盖。配置文件缺失或连接配置无效时提示执行 `agent100 config` 并退出，不自动进入配置流程。

修改源 `.env` 不会自动更换运行中的模型或下次启动的模型。需要再次执行 `config` 才会更新当前快照；正在运行的会话保持原来的连接。

源 `.env` 中的 `ZAI_API_KEY`、QQ SMTP 变量仍供搜索、邮件等工具使用，与当前模型的四个字段分开。旧的 `data/.env` 首次启动配置流程已移除。

DeepSeek 的 `tokenization()` 使用最小流式请求读取输入用量，会产生 API 费用。

## English

Manually maintain provider credentials and base URLs in project `.env`. Use `GLM_MODEL` and `DEEPSEEK_MODEL` for model names. Run `agent100 config`, choose a provider, and enter a model name or press Enter to keep the source value. Names are not validated.

The command writes exactly four fields to `data/active-model.env`: `MODEL_PROVIDER`, `MODEL_NAME`, `MODEL_API_KEY`, and `MODEL_BASE_URL`. It never edits the source `.env` or asks for a key. Runtime model settings come exclusively from this snapshot. Run `config` again to sync source changes. Missing active configuration directs you to `agent100 config` without opening setup automatically.
