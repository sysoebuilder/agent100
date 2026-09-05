import json
import sys
from collections.abc import Mapping
from typing import TextIO

from pydantic import JsonValue

from ai_agent_learning.planning.schema import (
    ExecutorType,
    PlanStep,
    TaskPlan,
)
from ai_agent_learning.planning.state import StepExecution, StepStatus

STEP_STATUS_LABELS: Mapping[StepStatus, str] = {
    StepStatus.WAITING: "等待执行",
    StepStatus.RUNNING: "执行中",
    StepStatus.COMPLETED: "已完成",
    StepStatus.FAILED: "执行失败",
    StepStatus.SKIPPED: "已跳过",
}


def show_task_plan(
    plan: TaskPlan,
    *,
    output: TextIO | None = None,
) -> None:
    """在终端中展示一份等待用户确认的完整计划。"""
    output = output or sys.stdout

    print("\n任务计划", file=output)
    print(f"目标：{plan.goal}", file=output)
    print(f"步骤数：{len(plan.steps)}", file=output)

    for index, step in enumerate(plan.steps, start=1):
        print(file=output)
        print(f"{index}. {step.title}", file=output)
        print(f"   步骤编号：{step.step_id}", file=output)
        print(f"   执行方式：{_format_executor(step)}", file=output)
        print(f"   输入：{_format_json(step.inputs)}", file=output)
        print(f"   预期结果：{step.expected_result}", file=output)
        print(f"   完成条件：{step.completion_condition}", file=output)
        print(f"   依赖步骤：{_format_dependencies(step)}", file=output)

    print(file=output, flush=True)


def show_step_status(
    step: PlanStep,
    execution: StepExecution,
    *,
    output: TextIO | None = None,
) -> None:
    """输出步骤的当前执行状态，供 Executor 在状态变化时调用。"""
    if execution.step_id != step.step_id:
        raise ValueError(
            "计划步骤与执行状态不匹配："
            f"{step.step_id} != {execution.step_id}"
        )

    output = output or sys.stdout
    status_label = STEP_STATUS_LABELS[execution.status]

    print(
        f"[{status_label}] {step.step_id}：{step.title}",
        file=output,
        flush=True,
    )

    if execution.status is StepStatus.COMPLETED:
        print(
            f"  结果：{_format_json(execution.result)}",
            file=output,
            flush=True,
        )
    elif execution.error:
        print(
            f"  原因：{execution.error}",
            file=output,
            flush=True,
        )


def _format_executor(step: PlanStep) -> str:
    if step.executor.type is ExecutorType.TOOL:
        return f"本地工具（{step.executor.tool_name}）"

    return "模型"


def _format_dependencies(step: PlanStep) -> str:
    if not step.depends_on:
        return "无"

    return "、".join(step.depends_on)


def _format_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(", ", ": "),
    )
