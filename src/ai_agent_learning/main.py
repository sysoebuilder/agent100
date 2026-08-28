import json
import os
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv, set_key

from ai_agent_learning.agent.context import ContextManager
from ai_agent_learning.agent.loop import AgentLoop
from ai_agent_learning.agent.state import AgentRunResult
from ai_agent_learning.context import save_context, select_context
from ai_agent_learning.conversation_service import ConversationService
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.model import ArkModelClient
from ai_agent_learning.paths import get_data_dir
from ai_agent_learning.schema import Message, ModelRequest
from ai_agent_learning.session import create_session_name, save_session, select_session
from ai_agent_learning.taskplan import show_task_plan, validate_task_plan
from ai_agent_learning.tools.catalog import BUILTIN_TOOLS
from ai_agent_learning.tools.contracts import ToolCall, ToolResult
from ai_agent_learning.tools.executor import ToolExecutor
from ai_agent_learning.tools.registry import ToolRegistry

API_KEY_ENV = "ARK_API_KEY"
REASONING_MODEL_ENV = "ARK_REASONING_MODEL"
TOKENIZER_MODEL = "doubao-seed-evolving"
CONTEXT_LIMIT = 1_000_000
KEEP_RECENT = 6


def tool_call_to_message(call: ToolCall) -> Message:
    return Message(
        role="assistant",
        content=json.dumps(
            {
                "event": "agent.tool_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            },
            ensure_ascii=False,
        ),
    )


def tool_result_to_message(result: ToolResult) -> Message:
    return Message(
        role="assistant",
        content=json.dumps(
            {
                "event": "agent.tool_result",
                "call_id": result.call_id,
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "metadata": result.metadata,
            },
            ensure_ascii=False,
        ),
    )


def agent_run_result_to_messages(
    result: AgentRunResult,
) -> list[Message]:
    messages: list[Message] = []

    for step in result.state.steps:
        if step.message is not None:
            messages.append(step.message)

        messages.extend(
            tool_call_to_message(call)
            for call in step.tool_calls
        )
        messages.extend(
            tool_result_to_message(tool_result)
            for tool_result in step.tool_results
        )

    return messages


def ask_required(
    prompt: str,
    *,
    secret: bool = False,
) -> str:
    reader = getpass if secret else input

    while True:
        try:
            value = reader(prompt).strip()
        except EOFError as error:
            raise SystemExit("\n无法读取输入，配置已取消。") from error

        if value:
            return value

        print("输入不能为空，请重新输入。")


def save_env_value(
    env_file: Path,
    name: str,
    value: str,
) -> None:
    try:
        env_file.touch(exist_ok=True)

        set_key(
            str(env_file),
            name,
            value,
            quote_mode="always",
        )
    except OSError as error:
        raise SystemExit(f"无法保存配置文件：{env_file}") from error

    # 让本次运行立即使用刚输入的配置。
    os.environ[name] = value


def load_or_create_config() -> tuple[str, str]:
    env_file = get_data_dir() / ".env"

    # 系统环境变量优先，data/.env 只补充缺失配置。
    load_dotenv(
        env_file,
        override=False,
    )

    api_key = os.getenv(API_KEY_ENV, "").strip()
    reasoning_model = os.getenv(
        REASONING_MODEL_ENV,
        "",
    ).strip()

    if api_key and reasoning_model:
        return api_key, reasoning_model

    print("\n首次使用 llm-cli，需要完成模型配置。")
    print(f"配置将保存到：{env_file}")
    print("请勿将该文件提交到 Git。\n")

    if not api_key:
        api_key = ask_required(
            "请输入 ARK API Key（输入内容不会显示）：",
            secret=True,
        )
        save_env_value(
            env_file,
            API_KEY_ENV,
            api_key,
        )

    if not reasoning_model:
        reasoning_model = ask_required(
            "请输入 ARK 模型推理点 ID：",
        )
        save_env_value(
            env_file,
            REASONING_MODEL_ENV,
            reasoning_model,
        )

    print("配置保存成功。\n")

    return api_key, reasoning_model


async def run_chat() -> None:
    configure_logging()
    api_key, reasoning_id = load_or_create_config()

    client = ArkModelClient(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    tool_registry = ToolRegistry(
        tools=BUILTIN_TOOLS,
    )

    tool_executor = ToolExecutor(
        registry=tool_registry,
    )

    context_manager = ContextManager(
        client=client,
        tokenizer_model=TOKENIZER_MODEL,
        context_limit=CONTEXT_LIMIT,
        keep_recent=KEEP_RECENT,
    )

    agent_loop = AgentLoop(
        model_client=client,
        registry=tool_registry,
        executor=tool_executor,
        context_manager=context_manager,
        max_steps=10,
    )

    session = select_session()
    context = select_context(session.session_id)
    try:
        while True:
            content = input("请输入消息: ").strip()
            if not content:
                print("消息不能为空")
                continue

            if content in {"exit", "quit"}:
                break

            if not session.name.strip():
                session.name = create_session_name(
                    content,
                    session_id=session.session_id,
                )

            session.messages.append(
                Message(
                    role="user",
                    content=content,
                ),
            )

            context.messages.append(
                Message(
                    role="user",
                    content=content,
                ),
            )

            request = ModelRequest(
                messages=context.messages,
                model=reasoning_id,
            )
            agent_result = await agent_loop.run(request)
            run_messages = agent_run_result_to_messages(agent_result)

            session.messages.extend(run_messages)
            context.messages[:] = request.messages
            context.messages.extend(run_messages)

            if (
                agent_result.response is None
                or agent_result.response.message is None
            ):
                raise RuntimeError("Agent 没有返回最终文本")

            print(f"助手: {agent_result.response.message.content}")

        if session.messages:
            save_session(session)
            save_context(context)

    finally:
        await client.close()


async def run_taskplan(goal: str | None = None) -> None:
    configure_logging()
    api_key, reasoning_id = load_or_create_config()

    client = ArkModelClient(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    conversation_service = ConversationService(
        client=client,
        reasoning_model=reasoning_id,
        tokenizer_model=TOKENIZER_MODEL,
        keep_recent=KEEP_RECENT,
    )

    session = select_session()

    try:
        if not goal:
            goal = input("请输入任务目标: ").strip()
            if not goal:
                print("任务目标不能为空")
                return

        if goal in {"exit", "quit"}:
            return

        if not session.name.strip():
            session.name = create_session_name(
                goal,
                session_id=session.session_id,
            )

        plan = await conversation_service.create_task_plan(
            session=session,
            content=goal,
        )
        validate_task_plan(plan)
        show_task_plan(plan)

        if session.messages:
            save_session(session)

    finally:
        await client.close()


if __name__ == "__main__":
    from ai_agent_learning.cli import main

    main()
