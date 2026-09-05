from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ai_agent_learning.planning.schema import TaskPlan


class StepStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step_id: str = Field(
        min_length=1,
        pattern=r"^step_[1-9][0-9]*$",
        description="对应计划步骤的唯一编号。",
    )
    status: StepStatus = Field(
        default=StepStatus.WAITING,
        description="当前步骤的运行状态。",
    )
    result: JsonValue | None = Field(
        default=None,
        description="当前步骤成功后产生的可持久化结果。",
    )
    error: str | None = Field(
        default=None,
        description="当前步骤失败时的错误说明。",
    )


class PlanExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    plan_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="当前计划执行实例的唯一编号。",
    )
    plan: TaskPlan = Field(
        description="本次执行所依据的完整计划。"
    )
    status: PlanStatus = Field(
        default=PlanStatus.AWAITING_APPROVAL,
        description="整份计划当前所处的运行状态。",
    )
    steps: list[StepExecution] = Field(
        min_length=1,
        description="与计划步骤一一对应的运行记录。",
    )
