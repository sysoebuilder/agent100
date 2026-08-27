from jsonschema import Draft202012Validator

from ai_agent_learning.schema import TaskPlan

TASK_PLAN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TaskPlan",
    "description": "完成一个目标所需的结构化任务计划",
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "计划最终要达成的目标",
            "minLength": 1,
        },
        "steps": {
            "type": "array",
            "description": "按照执行顺序排列的步骤",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "string",
                        "description": "步骤的唯一编号，例如 step_1",
                        "pattern": "^step_[1-9][0-9]*$",
                    },
                    "title": {
                        "type": "string",
                        "description": "该步骤要完成的事情",
                        "minLength": 1,
                        "maxLength": 300,
                    },
                    "completion_condition": {
                        "type": "string",
                        "description": "可检查、可判断的完成条件",
                        "minLength": 1,
                        "maxLength": 300,
                    },
                },
                "required": ["step_id", "title", "completion_condition"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal", "steps"],
    "additionalProperties": False,
}


def validate_task_plan(task_plan: TaskPlan) -> None:
    plan = task_plan.model_dump()

    validator = Draft202012Validator(TASK_PLAN_SCHEMA)
    schema_errors = sorted(
        validator.iter_errors(plan), key=lambda error: list(error.absolute_path)
    )

    if schema_errors:
        messages = []

        for error in schema_errors:
            path = ".".join(str(item) for item in error.absolute_path)
            path = path or "<root>"
            messages.append(f"{path}: {error.message}")

        raise ValueError("Schema 验证失败：" + "；".join(messages))

    semantic_errors = []

    if not plan["goal"].strip():
        semantic_errors.append("goal 不能只包含空格")

    step_ids = [step["step_id"] for step in plan["steps"]]

    if len(step_ids) != len(set(step_ids)):
        semantic_errors.append("step_id 不能重复")

    for index, step in enumerate(plan["steps"]):
        if not step["title"].strip():
            semantic_errors.append(f"steps[{index}].title 不能只包含空格")

        if not step["completion_condition"].strip():
            semantic_errors.append(
                f"steps[{index}].completion_condition 不能只包含空格"
            )

    if semantic_errors:
        raise ValueError("语义验证失败：" + "；".join(semantic_errors))


def show_task_plan(plan: TaskPlan) -> None:
    print(f"目标: {plan.goal}")

    for step in plan.steps:
        print(
            f"步骤 {step.step_id}: {step.title}\n"
            f"  完成条件: {step.completion_condition}"
        )
