import json
import logging
from collections.abc import Sequence

from ai_agent_learning.model import ArkModelClient, GlmModelClient
from ai_agent_learning.schema import Message, ModelRequest
from ai_agent_learning.tools.contracts import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ContextLimitExceeded(RuntimeError):
    """压缩后仍无法放入模型上下文窗口。"""


class ContextManager:
    def __init__(
        self,
        client: ArkModelClient | GlmModelClient,
        tokenizer_model: str,
        *,
        context_limit: int = 1_000_000,
        compact_threshold: float = 0.8,
        keep_recent: int = 6,
    ) -> None:
        if not tokenizer_model.strip():
            raise ValueError("tokenizer_model 不能为空")

        if context_limit <= 0:
            raise ValueError("context_limit 必须大于 0")

        if not 0 < compact_threshold < 1:
            raise ValueError("compact_threshold 必须在 0 和 1 之间")

        if keep_recent <= 0:
            raise ValueError("keep_recent 必须大于 0")

        self._client = client
        self._tokenizer_model = tokenizer_model
        self._context_limit = context_limit
        self._compact_limit = int(context_limit * compact_threshold)
        self._keep_recent = keep_recent

    @staticmethod
    def _serialize_tool_item(
        item: ToolCall | ToolResult,
    ) -> str:
        if isinstance(item, ToolCall):
            payload = {
                "type": "function_call",
                "call_id": item.id,
                "name": item.name,
                "arguments": item.arguments,
            }
            # GLM 会回传原始 assistant 内容，计数时也应包含它。
            # 一轮多个工具共用这段内容，只在第一个工具上计入。
            if (
                item.assistant_message is not None
                and item.assistant_message["tool_calls"][0]["id"] == item.id
            ):
                payload["assistant_message"] = item.assistant_message
        else:
            payload = {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": {
                    "success": item.success,
                    "data": item.data,
                    "error": item.error,
                },
            }

        return json.dumps(
            payload,
            ensure_ascii=False,
        )

    def _build_token_messages(
        self,
        request: ModelRequest,
        tool_history: Sequence[ToolCall | ToolResult],
    ) -> list[Message]:
        messages = list(request.messages)

        messages.extend(
            Message(
                role="system",
                content=self._serialize_tool_item(item),
            )
            for item in tool_history
        )

        return messages

    async def _count_tokens(
        self,
        request: ModelRequest,
        tool_history: Sequence[ToolCall | ToolResult],
    ) -> int:
        messages = self._build_token_messages(
            request,
            tool_history,
        )

        return await self._client.tokenization(
            messages,
            self._tokenizer_model,
        )

    def _raise_if_over_limit(self, input_tokens: int) -> None:
        if input_tokens > self._context_limit:
            raise ContextLimitExceeded(
                f"上下文在压缩后仍超过模型上限：{input_tokens} > {self._context_limit}"
            )

    async def compact_if_needed(
        self,
        request: ModelRequest,
        tool_history: Sequence[ToolCall | ToolResult],
    ) -> ModelRequest:
        input_tokens = await self._count_tokens(
            request,
            tool_history,
        )

        logger.debug(
            "Agent 上下文 Token 计算完成",
            extra={
                "event": "agent.context.tokens.counted",
                "input_tokens": input_tokens,
                "context_limit": self._context_limit,
            },
        )

        if input_tokens <= self._compact_limit:
            return request

        old_messages = request.messages[: -self._keep_recent]
        recent_messages = request.messages[-self._keep_recent :]

        if not old_messages:
            self._raise_if_over_limit(input_tokens)
            return request

        logger.info(
            "开始压缩 Agent 历史上下文",
            extra={
                "event": "agent.context.compaction.started",
                "input_tokens_before": input_tokens,
                "compacted_messages": len(old_messages),
                "retained_messages": len(recent_messages),
            },
        )

        summary = await self._client.compact_context(
            old_messages,
            request.model,
        )

        # 就地更新，调用方可以把压缩后的消息同步回 Session。
        request.messages[:] = [
            summary,
            *recent_messages,
        ]

        tokens_after = await self._count_tokens(
            request,
            tool_history,
        )

        logger.info(
            "Agent 历史上下文压缩完成",
            extra={
                "event": "agent.context.compaction.completed",
                "input_tokens_before": input_tokens,
                "input_tokens_after": tokens_after,
                "compacted_messages": len(old_messages),
                "retained_messages": len(recent_messages),
            },
        )

        self._raise_if_over_limit(tokens_after)
        return request
