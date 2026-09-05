from ai_agent_learning.tools.builtin import current_time, web_search
from ai_agent_learning.tools.registry import RegisteredTool

BUILTIN_TOOLS: tuple[RegisteredTool, ...] = (
    RegisteredTool(
        spec=current_time.SPEC,
        handler=current_time.handler,
    ),
)


def build_builtin_tools(
    search_client: web_search.WebSearchClient,
) -> tuple[RegisteredTool, ...]:
    """将带外部依赖的工具与静态工具一起注册。"""
    return (
        *BUILTIN_TOOLS,
        RegisteredTool(spec=web_search.SPEC, handler=search_client.handler),
    )
