import logging
import secrets
import time

from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.planning.schema import TaskPlan
from ai_agent_learning.schema import (
    Message,
    ModelRequest,
    Session,
)
from ai_agent_learning.tools.contracts import ToolSpec

logger = logging.getLogger(__name__)


class ConversationService:
    def __init__(
        self,
        client: ArkModelClient,
        reasoning_model: str,
        tokenizer_model: str,
        context_limit: int = 1_000_000,
        compact_threshold: float = 0.8,
        keep_recent: int = 6,
    ) -> None:
        if not 0 < compact_threshold < 1:
            raise ValueError("compact_threshold 必须在 0 和 1 之间")

        if keep_recent <= 0:
            raise ValueError("keep_recent 必须大于 0")

        self._client = client
        self._reasoning_model = reasoning_model
        self._tokenizer_model = tokenizer_model
        self._context_limit = context_limit
        self._compact_threshold = compact_threshold
        self._keep_recent = keep_recent

    @staticmethod
    def _generate_trace_id() -> str:
        while True:
            trace_id = secrets.token_hex(16)
            if trace_id != "0" * 32:
                return trace_id

    def _log_fields(
        self,
        *,
        session_id: str,
        trace_id: str,
        event: str,
        mode: str,
        **fields: object,
    ) -> dict[str, object]:
        return {
            "session_id": session_id,
            "trace_id": trace_id,
            "event": event,
            "mode": mode,
            **fields,
        }

    async def _compact_if_needed(
        self,
        session: Session,
        trace_id: str,
        mode: str,
    ) -> int:
        input_tokens = await self._client.tokenization(
            session.messages,
            self._tokenizer_model,
        )

        logger.debug(
            "上下文 Token 计算完成",
            extra=self._log_fields(
                session_id=session.session_id,
                trace_id=trace_id,
                event="context.tokens.counted",
                mode=mode,
                input_tokens=input_tokens,
                context_limit=self._context_limit,
            ),
        )

        compact_limit = int(self._context_limit * self._compact_threshold)

        if input_tokens <= compact_limit:
            return input_tokens

        old_messages = session.messages[: -self._keep_recent]
        recent_messages = session.messages[-self._keep_recent :]

        if not old_messages:
            logger.warning(
                "上下文接近上限，但没有可压缩的旧消息",
                extra=self._log_fields(
                    session_id=session.session_id,
                    trace_id=trace_id,
                    event="context.compaction.skipped",
                    mode=mode,
                    input_tokens=input_tokens,
                    context_limit=self._context_limit,
                    reason="no_old_messages",
                ),
            )
            return input_tokens

        logger.warning(
            "开始压缩历史上下文",
            extra=self._log_fields(
                session_id=session.session_id,
                trace_id=trace_id,
                event="context.compaction.started",
                mode=mode,
                input_tokens_before=input_tokens,
                compacted_messages=len(old_messages),
                retained_messages=len(recent_messages),
            ),
        )

        summary = await self._client.compact_context(
            old_messages,
            self._reasoning_model,
        )

        session.messages[:] = [
            summary,
            *recent_messages,
        ]

        tokens_after = await self._client.tokenization(
            session.messages,
            self._tokenizer_model,
        )

        logger.info(
            "历史上下文压缩完成",
            extra=self._log_fields(
                session_id=session.session_id,
                trace_id=trace_id,
                event="context.compaction.completed",
                mode=mode,
                input_tokens_before=input_tokens,
                input_tokens_after=tokens_after,
                compacted_messages=len(old_messages),
                retained_messages=len(recent_messages),
            ),
        )

        return tokens_after

    async def _prepare_request(
        self,
        session: Session,
        content: str,
        trace_id: str,
        mode: str,
    ) -> tuple[ModelRequest, int]:
        content = content.strip()

        if not content:
            raise ValueError("消息不能为空")

        session.messages.append(Message(role="user", content=content))

        input_tokens = await self._compact_if_needed(
            session=session,
            trace_id=trace_id,
            mode=mode,
        )

        request = ModelRequest(
            messages=session.messages,
            model=self._reasoning_model,
        )

        return request, input_tokens

    async def create_task_plan(
        self,
        session: Session,
        content: str,
        tools: list[ToolSpec],
    ) -> TaskPlan:
        trace_id = self._generate_trace_id()
        started_at = time.perf_counter()

        logger.info(
            "任务计划生成开始",
            extra=self._log_fields(
                session_id=session.session_id,
                trace_id=trace_id,
                event="task_plan.started",
                mode="task_plan",
                model=self._reasoning_model,
            ),
        )

        try:
            request, input_tokens = await self._prepare_request(
                session=session,
                content=content,
                trace_id=trace_id,
                mode="task_plan",
            )

            plan = await self._client.create_task_plan(
                request,
                tools=tools,
            )

            session.messages.append(
                Message(
                    role="assistant",
                    content=plan.model_dump_json(),
                )
            )

            duration_ms = round(
                (time.perf_counter() - started_at) * 1000,
                2,
            )

            logger.info(
                "任务计划生成完成",
                extra=self._log_fields(
                    session_id=session.session_id,
                    trace_id=trace_id,
                    event="task_plan.completed",
                    mode="task_plan",
                    model=request.model,
                    input_tokens=input_tokens,
                    duration_ms=duration_ms,
                ),
            )

            return plan

        except Exception:
            duration_ms = round(
                (time.perf_counter() - started_at) * 1000,
                2,
            )

            logger.exception(
                "任务计划生成失败",
                extra=self._log_fields(
                    session_id=session.session_id,
                    trace_id=trace_id,
                    event="task_plan.failed",
                    mode="task_plan",
                    model=self._reasoning_model,
                    duration_ms=duration_ms,
                ),
            )
            raise
