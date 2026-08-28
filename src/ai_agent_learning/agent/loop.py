from ai_agent_learning.agent.context import ContextManager
from ai_agent_learning.agent.state import (
    AgentRunResult,
    AgentState,
    AgentStatus,
    AgentStep,
    StopReason,
)
from ai_agent_learning.agent.policy import (
    PolicyAction,
    PolicyDecision,
    AgentPolicy,
)
from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.schema import ModelRequest
from ai_agent_learning.tools.contracts import ToolCall, ToolResult
from ai_agent_learning.tools.executor import ToolExecutor
from ai_agent_learning.tools.registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        model_client: ArkModelClient,
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

    async def run(self, request: ModelRequest) -> AgentRunResult:
        tool_history: list[ToolCall | ToolResult] = []
        state = AgentState()

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

            response = await self._model_client.create_response(
                request=request,
                tools=self._registry.list_specs(),
                tool_history=tool_history,
            )

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

            for tool_call in response.tool_calls:
                result = await self._executor.execute(tool_call)
                state.tool_calls_used += 1
                tool_history.append(result)
                agent_step.tool_results.append(result)

            
