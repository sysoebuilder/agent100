from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from ai_agent_learning.tools.contracts import ToolSpec

ToolHandler: TypeAlias = Callable[
    ...,
    Awaitable[dict[str, Any]],
]


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    def __init__(
        self,
        tools: tuple[RegisteredTool, ...],
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}

        for tool in tools:
            name = tool.spec.name

            if name in self._tools:
                raise ValueError(f"工具名称重复：{name}")

            self._tools[name] = tool

    def get_tool(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"工具未注册：{name}") from None

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]
