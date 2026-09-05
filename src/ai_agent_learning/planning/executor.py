import json
from collections.abc import Callable

from pydantic import JsonValue

from ai_agent_learning.model import ArkModelClient, GlmModelClient
from ai_agent_learning.planning.schema import ExecutorType, PlanStep
from ai_agent_learning.planning.state import (
    PlanExecution,
    PlanStatus,
    StepExecution,
    StepStatus,
)
from ai_agent_learning.schema import Message, ModelRequest
from ai_agent_learning.tools.contracts import ToolCall, ToolResult
from ai_agent_learning.tools.executor import ToolExecutor

StepStatusCallback = Callable[[PlanStep, StepExecution], None]


class PlanExecutor:
    def __init__(
        self,
        model_client: ArkModelClient | GlmModelClient,
        model: str,
        tool_executor: ToolExecutor,
        *,
        on_step_status: StepStatusCallback | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model 不能为空")

        self._model_client = model_client
        self._model = model
        self._tool_executor = tool_executor
        self._on_step_status = on_step_status

    async def execute(self, execution: PlanExecution) -> PlanExecution:
        if execution.status is not PlanStatus.RUNNING:
            raise ValueError("只有用户确认并进入 running 的计划才能执行")

        plan_step_ids = [step.step_id for step in execution.plan.steps]
        execution_step_ids = [step.step_id for step in execution.steps]

        if execution_step_ids != plan_step_ids:
            raise ValueError("计划步骤与执行状态不一致")

        execution_by_id = {
            step.step_id: step
            for step in execution.steps
        }

        for index, plan_step in enumerate(execution.plan.steps):
            step_execution = execution.steps[index]

            if step_execution.status is not StepStatus.WAITING:
                raise ValueError(
                    f"步骤 {plan_step.step_id} 必须从 waiting 状态开始执行"
                )

            incomplete_dependencies = [
                dependency_id
                for dependency_id in plan_step.depends_on
                if execution_by_id[dependency_id].status
                is not StepStatus.COMPLETED
            ]

            if incomplete_dependencies:
                step_execution.status = StepStatus.SKIPPED
                step_execution.error = (
                    "依赖步骤未完成：" + ", ".join(incomplete_dependencies)
                )
                self._notify_step_status(plan_step, step_execution)
                self._mark_remaining_steps_skipped(execution, index + 1)
                execution.status = PlanStatus.FAILED
                return execution

            step_execution.status = StepStatus.RUNNING
            self._notify_step_status(plan_step, step_execution)

            try:
                result = await self._execute_step(
                    execution=execution,
                    plan_step=plan_step,
                    execution_by_id=execution_by_id,
                )
            except Exception as error:  # noqa: BLE001
                step_execution.status = StepStatus.FAILED
                step_execution.error = str(error) or type(error).__name__
                self._notify_step_status(plan_step, step_execution)
                self._mark_remaining_steps_skipped(execution, index + 1)
                execution.status = PlanStatus.FAILED
                return execution

            step_execution.result = result
            step_execution.error = None
            step_execution.status = StepStatus.COMPLETED
            self._notify_step_status(plan_step, step_execution)

        execution.status = PlanStatus.COMPLETED
        return execution

    async def _execute_step(
        self,
        execution: PlanExecution,
        plan_step: PlanStep,
        execution_by_id: dict[str, StepExecution],
    ) -> JsonValue:
        if plan_step.executor.type is ExecutorType.TOOL:
            return await self._execute_tool_step(
                execution=execution,
                plan_step=plan_step,
                tool_name=plan_step.executor.tool_name,
            )

        return await self._execute_model_step(
            execution=execution,
            plan_step=plan_step,
            execution_by_id=execution_by_id,
        )

    async def _execute_tool_step(
        self,
        execution: PlanExecution,
        plan_step: PlanStep,
        tool_name: str,
    ) -> JsonValue:
        tool_call = ToolCall(
            id=f"{execution.plan_id}:{plan_step.step_id}",
            name=tool_name,
            arguments=plan_step.inputs,
        )
        result = await self._tool_executor.execute_with_recovery(tool_call)

        if not result.success:
            raise RuntimeError(self._tool_error_message(result))

        return result.data

    async def _execute_model_step(
        self,
        execution: PlanExecution,
        plan_step: PlanStep,
        execution_by_id: dict[str, StepExecution],
    ) -> JsonValue:
        dependency_results = {
            dependency_id: execution_by_id[dependency_id].result
            for dependency_id in plan_step.depends_on
        }
        payload = {
            "goal": execution.plan.goal,
            "current_step": plan_step.model_dump(mode="json"),
            "dependency_results": dependency_results,
        }
        request = ModelRequest(
            model=self._model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "你是计划执行器中的模型步骤。"
                        "只执行给出的当前步骤，不得新增、删除或修改计划步骤，"
                        "不得调用任何工具。只返回当前步骤的最终结果。"
                    ),
                ),
                Message(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False),
                ),
            ],
        )
        response = await self._model_client.create_response(
            request=request,
            tools=[],
        )

        if response.tool_calls:
            raise RuntimeError("模型步骤不得调用工具")

        if response.message is None:
            raise RuntimeError("模型步骤没有返回结果")

        return response.message.content

    @staticmethod
    def _tool_error_message(result: ToolResult) -> str:
        if result.error is None:
            return "工具执行失败"

        message = result.error.get("message")

        if isinstance(message, str) and message.strip():
            return message

        return json.dumps(result.error, ensure_ascii=False, default=str)

    def _mark_remaining_steps_skipped(
        self,
        execution: PlanExecution,
        start_index: int,
    ) -> None:
        remaining_steps = zip(
            execution.plan.steps[start_index:],
            execution.steps[start_index:],
            strict=True,
        )

        for plan_step, step_execution in remaining_steps:
            if step_execution.status is StepStatus.WAITING:
                step_execution.status = StepStatus.SKIPPED
                step_execution.error = "前置步骤执行失败，当前步骤未执行"
                self._notify_step_status(plan_step, step_execution)

    def _notify_step_status(
        self,
        plan_step: PlanStep,
        step_execution: StepExecution,
    ) -> None:
        if self._on_step_status is not None:
            self._on_step_status(plan_step, step_execution)
