from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.planning.schema import TaskPlan
from ai_agent_learning.planning.validator import validate_plan
from ai_agent_learning.schema import ModelRequest
from ai_agent_learning.tools.registry import ToolRegistry


class Planner:
    def __init__(
        self,
        model_client: ArkModelClient,
        registry: ToolRegistry,
    ) -> None:
        self._model_client = model_client
        self._registry = registry

    async def create_plan(self, request: ModelRequest) -> TaskPlan:
        plan = await self._model_client.create_task_plan(
            request,
            tools=self._registry.list_specs(),
        )
        validate_plan(plan, self._registry)
        return plan
