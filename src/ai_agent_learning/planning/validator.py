from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from ai_agent_learning.planning.schema import ExecutorType, TaskPlan
from ai_agent_learning.tools.registry import ToolRegistry


class PlanValidationError(ValueError):
    pass


def validate_plan(
    plan: TaskPlan,
    registry: ToolRegistry,
) -> None:
    errors: list[str] = []

    if not plan.goal.strip():
        errors.append("goal 不能只包含空格")

    step_ids = [step.step_id for step in plan.steps]
    known_step_ids = set(step_ids)

    if len(step_ids) != len(known_step_ids):
        errors.append("step_id 不能重复")

    seen_step_ids: set[str] = set()

    for index, step in enumerate(plan.steps):
        step_path = f"steps[{index}]"
        expected_step_id = f"step_{index + 1}"

        if step.step_id != expected_step_id:
            errors.append(
                f"{step_path}.step_id 必须是 {expected_step_id}，"
                f"实际为 {step.step_id}"
            )

        if not step.title.strip():
            errors.append(f"{step_path}.title 不能只包含空格")

        if not step.expected_result.strip():
            errors.append(f"{step_path}.expected_result 不能只包含空格")

        if not step.completion_condition.strip():
            errors.append(
                f"{step_path}.completion_condition 不能只包含空格"
            )

        if len(step.depends_on) != len(set(step.depends_on)):
            errors.append(f"{step_path}.depends_on 不能包含重复步骤")

        for dependency_id in step.depends_on:
            if dependency_id not in known_step_ids:
                errors.append(
                    f"{step_path}.depends_on 引用了不存在的步骤 "
                    f"{dependency_id}"
                )
            elif dependency_id not in seen_step_ids:
                errors.append(
                    f"{step_path}.depends_on 只能引用当前步骤之前的步骤，"
                    f"实际引用 {dependency_id}"
                )

        if step.executor.type is ExecutorType.TOOL:
            tool_name = step.executor.tool_name

            if not tool_name.strip():
                errors.append(
                    f"{step_path}.executor.tool_name 不能只包含空格"
                )
            else:
                try:
                    tool = registry.get_tool(tool_name)
                except KeyError:
                    errors.append(
                        f"{step_path}.executor.tool_name 工具未注册："
                        f"{tool_name}"
                    )
                else:
                    input_validator = Draft202012Validator(
                        tool.spec.input_schema
                    )
                    input_errors = sorted(
                        input_validator.iter_errors(step.inputs),
                        key=lambda error: tuple(
                            str(item) for item in error.absolute_path
                        ),
                    )

                    for input_error in input_errors:
                        input_path = ".".join(
                            str(item) for item in input_error.absolute_path
                        )
                        full_path = f"{step_path}.inputs"

                        if input_path:
                            full_path = f"{full_path}.{input_path}"

                        errors.append(
                            f"{full_path}: {input_error.message}"
                        )

        seen_step_ids.add(step.step_id)

    if errors:
        raise PlanValidationError("计划验证失败：" + "；".join(errors))
