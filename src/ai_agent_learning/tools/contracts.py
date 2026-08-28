from dataclasses import dataclass
from typing import Any
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ToolSpec():
    name: str
    description: str
    risk_level: RiskLevel
    input_schema: dict[str, Any]
    has_side_effects: bool = False

@dataclass
class ToolCall():
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult():
    call_id: str
    success: bool
    data: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


    