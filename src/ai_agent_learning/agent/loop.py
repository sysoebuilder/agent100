import asyncio
from collections.abc import Callable

from ai_agent_learning.agent.context import ContextManager
from ai_agent_learning.agent.policy import (
    AgentPolicy,
    PolicyAction,
)
from ai_agent_learning.agent.state import (
    AgentRunResult,
    AgentState,
    AgentStep,
)
from ai_agent_learning.model import DeepSeekClient, GlmModelClient
from ai_agent_learning.schema import ModelRequest, ModelResponse
from ai_agent_learning.tools.contracts import ToolCall, ToolResult
from ai_agent_learning.tools.executor import ToolExecutor
from ai_agent_learning.tools.registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        model_client: GlmModelClient | DeepSeekClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        policy: AgentPolicy,
        context_manager: ContextManager,
    ) -> None:
        self._model_client = model_client
        self._registry = registry
        self._executor = executor
        self._context_manager = context_manager
        self._policy = policy

    async def run(
        self,
        request: ModelRequest,
        *,
        on_text_delta: Callable[[str], None] | None = None,
        on_message_end: Callable[[], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        on_response_end: Callable[[], None] | None = None,
    ) -> AgentRunResult:
        tool_history: list[ToolCall | ToolResult] = []
        state = AgentState()
        response: ModelResponse | None = None

        while True:
            decision = self._policy.before_model_call(state)
            if decision.action is not PolicyAction.CONTINUE:
                if decision.status is None or decision.stop_reason is None:
                    raise RuntimeError("终止决策缺少状态或原因")
                state.status = decision.status
                state.stop_reason = decision.stop_reason

                return AgentRunResult(
                    response=response
                    if decision.action is PolicyAction.COMPLETE
                    else None,
                    state=state,
                )

            step_index = len(state.steps) + 1

            request = await self._context_manager.compact_if_needed(
                request=request,
                tool_history=tool_history,
            )

            response = await self._model_client.create_response_stream(
                request=request,
                tools=self._registry.list_specs(),
                tool_history=tool_history,
                on_text_delta=on_text_delta,
                on_reasoning_delta=on_reasoning_delta,
            )

            if response.message is not None and on_message_end is not None:
                on_message_end()
            # 工具轮可能只有思考内容，没有正文，也需要结束临时思考展示。
            if on_response_end is not None:
                on_response_end()

            state.model_calls_used += 1
            agent_step = AgentStep(
                index=step_index,
                message=response.message,
                tool_calls=list(response.tool_calls),
            )
            state.steps.append(agent_step)

            decision = self._policy.after_model_response(
                state,
                response,
            )

            if decision.action is not PolicyAction.CONTINUE:
                if decision.status is None or decision.stop_reason is None:
                    raise RuntimeError("终止决策缺少状态或原因")
                state.status = decision.status
                state.stop_reason = decision.stop_reason

                return AgentRunResult(
                    response=response
                    if decision.action is PolicyAction.COMPLETE
                    else None,
                    state=state,
                )

            tool_history.extend(response.tool_calls)

            # 同一轮并发执行，gather 按调用顺序返回结果。
            results = await asyncio.gather(
                *(
                    self._executor.execute_with_recovery(call=tool_call)
                    for tool_call in response.tool_calls
                )
            )
            state.tool_calls_used += len(results)
            tool_history.extend(results)
            agent_step.tool_results.extend(results)
