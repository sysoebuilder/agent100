from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ExecutorType(str, Enum):
    TOOL = "tool"
    MODEL = "model"


class ToolExecutorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ExecutorType.TOOL] = Field(
        description="使用已注册工具执行当前步骤。"
    )
    tool_name: str = Field(
        min_length=1,
        max_length=100,
        description="当前步骤获准调用的工具注册名。",
    )


class ModelExecutorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ExecutorType.MODEL] = Field(
        description="使用 Executor 配置的模型执行当前步骤。"
    )


ExecutorSpec = Annotated[
    ToolExecutorSpec | ModelExecutorSpec,
    Field(
        discriminator="type",
        description="当前步骤使用的执行器类型。",
    ),
]


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(
        min_length=1,
        pattern=r"^step_[1-9][0-9]*$",
        description="步骤的唯一编号，按照 step_1、step_2 的格式递增。",
    )
    title: str = Field(
        min_length=1,
        max_length=300,
        description="当前步骤需要完成的具体任务。",
    )
    executor: ExecutorSpec
    inputs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="传给执行器的结构化输入。",
    )
    expected_result: str = Field(
        min_length=1,
        max_length=500,
        description="当前步骤预期产生的结果。",
    )
    completion_condition: str = Field(
        min_length=1,
        max_length=500,
        description="用于判断当前步骤是否完成的可检查条件。",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="当前步骤依赖的先前步骤编号。",
    )


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=1,
        max_length=1000,
        description="计划最终需要达成的用户目标。",
    )
    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=10,
        description="按照执行顺序排列的计划步骤。",
    )
