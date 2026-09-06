import json
from collections.abc import Callable
from typing import Any, Literal, cast

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
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
    ) -> list[ToolParam]:
        function_tools: list[FunctionToolParam] = [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
                # ToolSpec 允许可选参数；本地执行器负责 JSON Schema 校验。
                "strict": False,
            }
            for spec in tools
        ]
        api_tools: list[ToolParam] = [*function_tools]

        return api_tools

    async def create_response(
        self,
        request: ModelRequest,
        tools: list[ToolSpec],
        tool_history: list[ToolResult | ToolCall] | None = None,
    ) -> ModelResponse:
        api_tools = self._build_api_tools(tools)

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
        # 规划阶段只展示工具说明，不注册可执行工具。
        instructions = Message(
            role="system",
            content=(
                "请生成任务计划，不要执行工具。可用于计划的工具定义：\n"
                + json.dumps(self._build_api_tools(tools), ensure_ascii=False)
            ),
        )
        response = await self._client.responses.parse(
            model=request.model,
            store=False,
            input=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in [instructions, *request.messages]
            ],
            text_format=TaskPlan,
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


class GlmModelClient:
    """通过智谱的 OpenAI 兼容接口实现与 Ark 相同的异步操作。

    文档：https://docs.bigmodel.cn/cn/guide/capabilities/thinking
    工具续调：https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        *,
        thinking: Literal["enabled", "disabled"] = "enabled",
    ) -> None:
        if thinking not in ("enabled", "disabled"):
            raise ValueError("thinking 必须是 enabled 或 disabled")
        self._thinking = thinking
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    @staticmethod
    def _build_api_tools(
        tools: list[ToolSpec],
    ) -> list[dict[str, Any]]:
        api_tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                },
            }
            for spec in tools
        ]
        return api_tools

    @staticmethod
    def _build_messages(
        messages: list[Message],
        tool_history: list[ToolResult | ToolCall] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        api_messages: list[dict[str, Any]] = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        emitted_calls: set[str] = set()
        pending_assistant: dict[str, Any] | None = None
        for item in tool_history or []:
            if isinstance(item, ToolCall):
                if item.id in emitted_calls:
                    continue
                if item.assistant_message is not None:
                    # 同一轮的多个工具共用原始消息；保留参数字符串和思考内容。
                    api_messages.append(item.assistant_message)
                    emitted_calls.update(
                        call["id"] for call in item.assistant_message["tool_calls"]
                    )
                    pending_assistant = None
                else:
                    if pending_assistant is None:
                        pending_assistant = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [],
                        }
                        api_messages.append(pending_assistant)
                    pending_assistant["tool_calls"].append(
                        {
                            "id": item.id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": json.dumps(
                                    item.arguments, ensure_ascii=False
                                ),
                            },
                        }
                    )
                    emitted_calls.add(item.id)
            else:
                pending_assistant = None
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": json.dumps(
                            {
                                "success": item.success,
                                "data": item.data,
                                "error": item.error,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
        return cast(list[ChatCompletionMessageParam], api_messages)

    async def create_response(
        self,
        request: ModelRequest,
        tools: list[ToolSpec],
        tool_history: list[ToolResult | ToolCall] | None = None,
    ) -> ModelResponse:
        api_tools = self._build_api_tools(tools)
        tool_options: dict[str, Any] = {}
        if api_tools:
            tool_options = {"tools": api_tools, "tool_choice": "auto"}
        response = await self._client.chat.completions.create(
            model=request.model,
            messages=self._build_messages(request.messages, tool_history),
            extra_body={"thinking": {"type": self._thinking}},
            **tool_options,
        )
        if not response.choices:
            raise ValueError("GLM 响应没有 choices")
        return self._parse_response(response.choices[0].message)

    async def create_response_stream(
        self,
        request: ModelRequest,
        tools: list[ToolSpec],
        tool_history: list[ToolResult | ToolCall] | None = None,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """接收流式响应，实时通知正文片段，完整接收后返回可执行的工具调用。"""
        api_tools = self._build_api_tools(tools)
        tool_options: dict[str, Any] = {}
        if api_tools:
            tool_options = {"tools": api_tools, "tool_choice": "auto"}
        stream = await self._client.chat.completions.create(
            model=request.model,
            messages=self._build_messages(request.messages, tool_history),
            extra_body={"thinking": {"type": self._thinking}},
            stream=True,
            **tool_options,
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        # 正常结束、网络异常或调用方取消时，都关闭本次响应流。
        async with stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    if choice.index != 0:
                        continue
                    delta = choice.delta
                    if delta.content is not None:
                        content_parts.append(delta.content)
                        if delta.content and on_text_delta is not None:
                            on_text_delta(delta.content)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning is not None:
                        reasoning_parts.append(reasoning)
                        if reasoning and on_reasoning_delta is not None:
                            on_reasoning_delta(reasoning)
                    for part in delta.tool_calls or []:
                        call = calls.setdefault(
                            part.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if part.id:
                            call["id"] = part.id
                        if part.type:
                            call["type"] = part.type
                        if part.function is not None:
                            if part.function.name:
                                call["function"]["name"] = part.function.name
                            if part.function.arguments is not None:
                                call["function"]["arguments"] += part.function.arguments
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason

        if finish_reason not in {"stop", "tool_calls"}:
            raise ValueError(
                f"GLM 流式响应未正常完成：{finish_reason or '缺少结束标记'}"
            )
        ordered_calls = [calls[index] for index in sorted(calls)]
        for call in ordered_calls:
            if not call["id"] or not call["function"]["name"]:
                raise ValueError("GLM 流式工具调用缺少 id 或函数名")
        output: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) if content_parts else None,
            "tool_calls": ordered_calls,
        }
        if reasoning_parts:
            output["reasoning_content"] = "".join(reasoning_parts)
        return self._parse_response(ChatCompletionMessage.model_validate(output))

    @staticmethod
    def _parse_response(output: ChatCompletionMessage) -> ModelResponse:
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": output.content,
            "tool_calls": [
                call.model_dump(exclude_none=True) for call in output.tool_calls or []
            ],
        }
        reasoning = getattr(output, "reasoning_content", None)
        if reasoning is not None:
            assistant_message["reasoning_content"] = reasoning
        tool_calls: list[ToolCall] = []
        for call in output.tool_calls or []:
            if call.type != "function":
                continue
            arguments = json.loads(call.function.arguments)
            if not isinstance(arguments, dict):
                raise TypeError("工具调用参数必须是 JSON 对象")
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                    assistant_message=assistant_message,
                )
            )
        content = (output.content or "").strip()
        return ModelResponse(
            message=Message(role="assistant", content=content) if content else None,
            tool_calls=tool_calls,
        )

    async def create_task_plan(
        self,
        request: ModelRequest,
        tools: list[ToolSpec],
    ) -> TaskPlan:
        # GLM 的 JSON 模式不接受 Responses API 的 text_format。
        # 仅提供工具说明，不注册可执行工具，避免规划阶段触发调用。
        instructions = Message(
            role="system",
            content=(
                "请生成任务计划，只输出符合以下 JSON Schema 的 JSON 对象，"
                "不要输出 Markdown，不要执行工具。\n"
                + json.dumps(TaskPlan.model_json_schema(), ensure_ascii=False)
                + "\n可用于计划的工具定义：\n"
                + json.dumps(
                    self._build_api_tools(tools),
                    ensure_ascii=False,
                )
            ),
        )
        response = await self._client.chat.completions.create(
            model=request.model,
            messages=self._build_messages([instructions, *request.messages]),
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": self._thinking}},
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("GLM 没有返回任务计划")
        return TaskPlan.model_validate_json(response.choices[0].message.content)

    async def tokenization(self, messages: list[Message], model: str) -> int:
        async def create_request() -> httpx.Response:
            response = await self._http_client.post(
                "tokenizer",
                json={
                    "model": model,
                    "messages": self._build_messages(messages),
                },
            )
            response.raise_for_status()
            return response

        response = await retry_async(create_request, max_attempts=3)
        return int(response.json()["usage"]["total_tokens"])

    async def compact_context(self, messages: list[Message], model: str) -> Message:
        instructions = Message(
            role="system",
            content=(
                "你的任务是压缩随后提供的历史对话。"
                "请保留重要事实、用户偏好、已有结论、"
                "未完成任务和必要细节。"
                "不要回答历史问题，只输出摘要。"
            ),
        )
        response = await self._client.chat.completions.create(
            model=model,
            messages=self._build_messages([instructions, *messages]),
            extra_body={"thinking": {"type": self._thinking}},
        )
        if (
            not response.choices
            or not (response.choices[0].message.content or "").strip()
        ):
            raise ValueError("GLM 没有返回历史对话摘要")
        return Message(
            role="system",
            content=f"历史对话摘要: {response.choices[0].message.content}",
        )

    async def close(self) -> None:
        try:
            await self._http_client.aclose()
        finally:
            await self._client.close()
