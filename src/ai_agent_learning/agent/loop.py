from ai_agent_learning.agent.context import ContextManager
from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.schema import ModelRequest, ModelResponse
from ai_agent_learning.tools.contracts import ToolCall, ToolResult
from ai_agent_learning.tools.executor import ToolExecutor
from ai_agent_learning.tools.registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        model_client: ArkModelClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        context_manager: ContextManager,
        *,
        max_steps: int = 10,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")

        self._model_client = model_client
        self._registry = registry
        self._executor = executor
        self._context_manager = context_manager
        self._max_steps = max_steps

    async def run(self, request: ModelRequest) -> ModelResponse:
        tool_history: list[ToolCall | ToolResult] = []

        for _ in range(self._max_steps):
            request = await self._context_manager.compact_if_needed(
                request=request,
                tool_history=tool_history,
            )

            response = await self._model_client.create_response(
                request=request,
                tools=self._registry.list_specs(),
                tool_history=tool_history,
            )

            if not response.tool_calls:
                if response.message is None:
                    raise RuntimeError("模型没有返回任何有效内容")

                return response

            tool_history.extend(response.tool_calls)

            for tool_call in response.tool_calls:
                result = await self._executor.execute(tool_call)
                tool_history.append(result)

        raise RuntimeError("Agent 执行达到最大步数")
