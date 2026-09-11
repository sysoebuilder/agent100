# 统一搜索工具

`web_search` 与 `current_time` 一样，通过 `ToolRegistry` 注册，由 `ToolExecutor` 执行。
GlmModelClient 把它转换为普通函数工具，不再自动追加 `type="web_search"` 的服务端工具。

调用流程：模型返回 `ToolCall` → 执行器校验并调用搜索 handler → 生成 `ToolResult` → 模型使用搜索结果回答。

## 参数和结果

模型只需传入搜索词：

```json
{"query": "智谱 GLM 官方文档"}
```

`query` 必填，不能全部为空白，最长 70 字符。`count` 可选，范围 1–50，默认 5。
返回 `query` 和 `results`；每条结果包含 `title`、`url`、`summary`、`source`、`publish_date`。
来源和日期可以为空。空结果列表表示未找到结果；响应格式错误会作为工具失败记录。
网页内容作为资料使用，回答时引用来源链接。

## 注册与配置

`main.create_runtime()` 为聊天和任务规划创建搜索 HTTP 连接，使用现有 `ZAI_API_KEY`，并调用：

```python
search_client = WebSearchClient(search_http_client)
registry = ToolRegistry(tools=build_builtin_tools(search_client))
```

`search_http_client` 由应用注入，配置标准 API 地址和 Bearer 认证。工具不自行读取环境变量。
搜索引擎默认 `search_pro`，可在构造 WebSearchClient 时指定 `search_engine="search_std"`；模型不能修改密钥、API 地址或搜索引擎。

独立搜索请求使用 `POST https://open.bigmodel.cn/api/paas/v4/web_search`，将 `query` 映射为 `search_query`，设置 `search_intent=false`、`content_size="medium"`。
这是独立的搜索服务，收费规则见[官方联网搜索说明](https://docs.bigmodel.cn/cn/guide/tools/web-search)，不根据所选对话模型判断搜索费用。

## 执行与生命周期

- `ToolExecutor` 统一管理参数校验、风险检查、超时与最多 3 次重试。handler 不叠加重试；401 等永久错误直接失败。
- 聊天中搜索调用会记录在 Agent 步骤和工具历史中，并计入工具调用统计。
- Planner 可以生成 `executor.type="tool"`、`tool_name="web_search"` 的步骤；规划阶段只提供描述，不执行搜索。计划确认后，由 PlanExecutor 执行搜索，后续模型步骤通过依赖结果获取资料。
- `AsyncExitStack` 在正常退出、初始化失败、取消或运行异常时关闭搜索和模型连接。
- 已移除两个模型客户端的 `include_web_search` 参数。是否提供搜索能力由注册工具列表控制；模型执行步骤传入 `tools=[]`。

协议字段依据：[智谱网络搜索 API](https://docs.bigmodel.cn/api-reference/工具-api/网络搜索)。
