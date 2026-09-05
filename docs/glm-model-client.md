# GlmModelClient

使用项目已有的 `AsyncOpenAI` 接入智谱官方兼容接口，无需额外 SDK。
`ArkModelClient` 的构造、工具定义、对话、任务规划、Token 计数、上下文压缩和关闭方法均已实现。

```python
import asyncio
import os

from ai_agent_learning.model import GlmModelClient
from ai_agent_learning.schema import Message, ModelRequest


async def main():
    client = GlmModelClient(
        api_key=os.environ["ZAI_API_KEY"],
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        thinking="enabled",  # 可改为 disabled
    )
    try:
        response = await client.create_response(
            ModelRequest(
                model="GLM-5.3-flash",
                messages=[Message(role="user", content="解释一下 Agent 的工具调用流程")],
            ),
            tools=[],
        )
        if response.message is not None:
            print(response.message.content)
    finally:
        await client.close()


asyncio.run(main())
```

AgentLoop、ContextManager、ConversationService、Planner 和 PlanExecutor 均接受该客户端。
`run_chat` 和 `run_taskplan` 均使用 GlmModelClient，读取 `ZAI_API_KEY` 和 `GLM_REASONING_MODEL`，默认模型为 `GLM-5.3-flash`；缺少 API Key 时会提示输入并保存到 `data/.env`。上下文预算设为 128,000 tokens，规划、执行和分词均使用所配置的 GLM 模型。

- `create_response`：只转换传入的工具定义，不再自动添加原生搜索配置。`tools=[]` 表示不提供工具；联网搜索通过注册的普通 `web_search` 工具执行，见 [搜索工具说明](web-search.md)。ArkModelClient 同样采用此行为，已移除 `include_web_search` 参数。
- 工具续调：把返回的 `response.tool_calls` 和执行产生的 `ToolResult` 按顺序传入下一次调用的 `tool_history`。`ToolCall.assistant_message` 保存原始参数和思考内容，需保留该字段，避免重建 ToolCall 时丢失。
- `create_task_plan`：使用 JSON 模式，提供 TaskPlan schema 和工具描述，再通过 Pydantic 校验返回结果。规划请求不注册可执行工具；无效计划会抛出校验异常。
- `tokenization`：请求 `tokenizer` 并读取 `usage.total_tokens`；网络或可重试状态码最多尝试 3 次。
- `compact_context`：返回带有“历史对话摘要”前缀的 system 消息。
- `close`：释放 HTTP 和 SDK 连接。

示例采用标准 API 地址。自定义 base_url 时，所选服务需支持 Chat Completions 和 tokenizer 接口；模型也需支持所用的思考、JSON 输出和函数调用能力。搜索使用单独的智谱 HTTP 客户端，不依赖模型的原生联网搜索能力。

官方依据：[深度思考](https://docs.bigmodel.cn/cn/guide/capabilities/thinking)、[思考模式与工具续调](https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode)、[OpenAI 兼容接口](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)、[结构化输出](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output)、[文本分词器](https://docs.bigmodel.cn/api-reference/模型-api/文本分词器)、[联网搜索](https://docs.bigmodel.cn/cn/guide/tools/web-search)。
