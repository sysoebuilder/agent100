import json
from typing import cast

import httpx
from openai import AsyncOpenAI
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ToolParam,
)

from ai_agent_learning.planning.schema import TaskPlan
from ai_agent_learning.retry import retry_async
from ai_agent_learning.schema import Message, ModelRequest, ModelResponse
from ai_agent_learning.tools.contracts import ToolCall, ToolResult, ToolSpec


class ArkModelClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                30.0,
                connect=5.0,
            ),
        )

    @staticmethod
    def _build_api_tools(
        tools: list[ToolSpec],
        *,
        include_web_search: bool,
    ) -> list[ToolParam]:
        function_tools: list[FunctionToolParam] = [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
                "strict": True,
            }
            for spec in tools
        ]
        api_tools: list[ToolParam] = [*function_tools]

        if include_web_search:
            api_tools.append(
                cast(
                    ToolParam,
                    {
                        "type": "web_search",
                        "max_keyword": 10,
                    },
                )
            )

        return api_tools

    async def create_response(
        self,
        request: ModelRequest,
        tools: list[ToolSpec],
        tool_history: list[ToolResult | ToolCall] | None = None,
        *,
        include_web_search: bool = True,
    ) -> ModelResponse:
        api_tools = self._build_api_tools(
            tools,
            include_web_search=include_web_search,
        )

        model_input: ResponseInputParam = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        for history_item in tool_history or []:
            if isinstance(history_item, ToolCall):
                model_input.append(
                    {
                        "type": "function_call",
                        "call_id": history_item.id,
                        "name": history_item.name,
                        "arguments": json.dumps(
                            history_item.arguments,
                            ensure_ascii=False,
                        ),
                    }
                )

            elif isinstance(history_item, ToolResult):
                model_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": history_item.call_id,
                        "output": json.dumps(
                            {
                                "success": history_item.success,
                                "data": history_item.data,
                                "error": history_item.error,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

        if api_tools:
            response = await self._client.responses.create(
                model=request.model,
                store=False,
                input=model_input,
                tools=api_tools,
                tool_choice="auto",
            )
        else:
            response = await self._client.responses.create(
                model=request.model,
                store=False,
                input=model_input,
            )

        tool_calls: list[ToolCall] = []

        for output_item in response.output:
            if not isinstance(
                output_item,
                ResponseFunctionToolCall,
            ):
                continue

            arguments = json.loads(output_item.arguments)

            if not isinstance(arguments, dict):
                raise TypeError("工具调用参数必须是 JSON 对象")

            tool_calls.append(
                ToolCall(
                    id=output_item.call_id,
                    name=output_item.name,
                    arguments=arguments,
                )
            )

        output_text = response.output_text.strip()

        message = (
            Message(
                role="assistant",
                content=output_text,
            )
            if output_text
            else None
        )

        return ModelResponse(
            message=message,
            tool_calls=tool_calls,
        )

    async def create_task_plan(
        self,
        request: ModelRequest,
        tools: list[ToolSpec],
    ) -> TaskPlan:
        api_tools = self._build_api_tools(
            tools,
            include_web_search=False,
        )
        response = await self._client.responses.parse(
            model=request.model,
            store=False,
            input=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            text_format=TaskPlan,
            tools=api_tools,  # type: ignore[arg-type]
            tool_choice="none",
        )

        if response.output_parsed is None:
            raise ValueError("response.output_parsed is None")

        return response.output_parsed

    async def tokenization(self, messages: list[Message], model: str) -> int:
        async def create_request() -> httpx.Response:
            response = await self._http_client.post(
                "tokenization",
                json={
                    "model": model,
                    "text": [
                        f"{message.role}: {message.content}" for message in messages
                    ],
                },
            )
            response.raise_for_status()
            return response

        response = await retry_async(create_request, max_attempts=3)

        result = response.json()
        total_tokens = sum(item["total_tokens"] for item in result["data"])
        return total_tokens

    async def compact_context(self, messages: list[Message], model: str) -> Message:
        compression_input = [
            {
                "role": "system",
                "content": (
                    "你的任务是压缩随后提供的历史对话。"
                    "请保留重要事实、用户偏好、已有结论、"
                    "未完成任务和必要细节。"
                    "不要回答历史问题，只输出摘要。"
                ),
            },
            *[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        ]

        response = await self._client.responses.create(
            model=model,
            store=False,
            input=compression_input,  # type: ignore
        )
        if response.output_text is None:
            raise ValueError("response.output_text is None")

        return Message(role="system", content=f"历史对话摘要: {response.output_text}")

    async def close(self) -> None:
        await self._http_client.aclose()
        await self._client.close()
