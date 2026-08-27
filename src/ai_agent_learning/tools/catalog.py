from ai_agent_learning.tools.builtin import current_time
from ai_agent_learning.tools.registry import RegisteredTool

BUILTIN_TOOLS: tuple[RegisteredTool, ...] = (
    RegisteredTool(
        spec=current_time.SPEC,
        handler=current_time.handler,
    ),
)
