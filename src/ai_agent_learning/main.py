import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from getpass import getpass
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key

from ai_agent_learning.agent.context import ContextManager
from ai_agent_learning.agent.loop import AgentLoop
from ai_agent_learning.agent.policy import AgentLimits, AgentPolicy
from ai_agent_learning.agent.state import AgentRunResult, StopReason
from ai_agent_learning.context import save_context, select_context
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.model import GlmModelClient
from ai_agent_learning.paths import get_data_dir
from ai_agent_learning.planning.executor import PlanExecutor
from ai_agent_learning.planning.planner import Planner
from ai_agent_learning.planning.renderer import (
    show_step_status,
    show_task_plan,
)
from ai_agent_learning.planning.state import (
    PlanExecution,
    PlanStatus,
    StepExecution,
)
from ai_agent_learning.schema import Message, ModelRequest
from ai_agent_learning.session import create_session_name, save_session, select_session
from ai_agent_learning.tools.builtin.web_search import WebSearchClient
from ai_agent_learning.tools.catalog import build_builtin_tools
from ai_agent_learning.tools.contracts import ToolCall, ToolResult
from ai_agent_learning.tools.executor import ToolExecutor
from ai_agent_learning.tools.registry import ToolRegistry

API_KEY_ENV = "ZAI_API_KEY"
REASONING_MODEL_ENV = "GLM_REASONING_MODEL"
DEFAULT_MODEL = "GLM-5.3-flash"
CONTEXT_LIMIT = 128_000
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

        messages.extend(tool_call_to_message(call) for call in step.tool_calls)
        messages.extend(
            tool_result_to_message(tool_result) for tool_result in step.tool_results
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
    reasoning_model = (
        os.getenv(
            REASONING_MODEL_ENV,
            "",
        ).strip()
        or DEFAULT_MODEL
    )

    if api_key and reasoning_model:
        return api_key, reasoning_model

    print("\n首次使用 llm-cli，需要完成模型配置。")
    print(f"配置将保存到：{env_file}")
    print("请勿将该文件提交到 Git。\n")

    if not api_key:
        api_key = ask_required(
            "请输入 GLM API Key（输入内容不会显示）：",
            secret=True,
        )
        save_env_value(
            env_file,
            API_KEY_ENV,
            api_key,
        )

    save_env_value(env_file, REASONING_MODEL_ENV, reasoning_model)

    print("配置保存成功。\n")

    return api_key, reasoning_model


@asynccontextmanager
async def create_runtime(
    api_key: str,
) -> AsyncIterator[tuple[GlmModelClient, ToolRegistry]]:
    """统一管理模型与搜索连接，包括初始化失败和用户取消时的清理。"""
    async with AsyncExitStack() as stack:
        client = GlmModelClient(api_key=api_key)
        stack.push_async_callback(client.close)
        search_http_client = await stack.enter_async_context(
            httpx.AsyncClient(
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        )
        search_client = WebSearchClient(search_http_client)
        registry = ToolRegistry(tools=build_builtin_tools(search_client))
        yield client, registry


async def run_chat() -> None:
    configure_logging()
    api_key, reasoning_id = load_or_create_config()

    async with create_runtime(api_key) as (client, tool_registry):
        tool_executor = ToolExecutor(
            registry=tool_registry,
        )

        context_manager = ContextManager(
            client=client,
            tokenizer_model=reasoning_id,
            context_limit=CONTEXT_LIMIT,
            keep_recent=KEEP_RECENT,
        )

        agent_policy = AgentPolicy(limits=AgentLimits())

        agent_loop = AgentLoop(
            model_client=client,
            registry=tool_registry,
            executor=tool_executor,
            context_manager=context_manager,
            policy=agent_policy,
        )

        session = select_session()
        context = select_context(session.session_id)
        context_manager.initialize_token_estimate(context.messages)

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
            streaming_started = False

            def print_text_delta(text: str) -> None:
                nonlocal streaming_started
                if not text:
                    return
                if not streaming_started:
                    print("助手: ", end="", flush=True)
                    streaming_started = True
                print(text, end="", flush=True)

            def finish_message() -> None:
                nonlocal streaming_started
                if streaming_started:
                    print(flush=True)
                    streaming_started = False

            try:
                agent_result = await agent_loop.run(
                    request,
                    on_text_delta=print_text_delta,
                    on_message_end=finish_message,
                )
            finally:
                # 中断或网络异常时，避免下一个终端提示接在残缺正文后面。
                finish_message()
            if agent_result.state.stop_reason is not StopReason.FINAL_RESPONSE:
                if agent_result.state.stop_reason is StopReason.MAX_STEPS:
                    print("Agent 已超过最大步骤数")
                    continue
                if agent_result.state.stop_reason is StopReason.TOOL_FAILURE_LIMIT:
                    print("Agent 已超过最大工具调用次数")
                    continue
                if agent_result.state.stop_reason is StopReason.INVALID_MODEL_RESPONSE:
                    print("Agent 返回无效模型响应")
                    continue
                if agent_result.state.stop_reason is StopReason.REPEATED_TOOL_CALL:
                    print("Agent 重复调用工具")
                    continue
                print("Agent 已停止，未知原因")

            run_messages = agent_run_result_to_messages(agent_result)
            session.messages.extend(run_messages)
            context.messages[:] = request.messages
            context.messages.extend(run_messages)

            if agent_result.response is None or agent_result.response.message is None:
                raise RuntimeError("Agent 没有返回最终文本")

        if session.messages:
            save_session(session)
            save_context(context)


async def run_taskplan(goal: str | None = None) -> None:
    configure_logging()
    api_key, reasoning_id = load_or_create_config()

    async with create_runtime(api_key) as (client, tool_registry):
        context_manager = ContextManager(
            client=client,
            tokenizer_model=reasoning_id,
            context_limit=CONTEXT_LIMIT,
            keep_recent=KEEP_RECENT,
        )
        planner = Planner(
            model_client=client,
            registry=tool_registry,
        )
        plan_executor = PlanExecutor(
            model_client=client,
            model=reasoning_id,
            tool_executor=ToolExecutor(registry=tool_registry),
            on_step_status=show_step_status,
        )

        session = select_session()
        context = select_context(session.session_id)
        context_manager.initialize_token_estimate(context.messages)

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

        user_message = Message(
            role="user",
            content=goal,
        )
        session.messages.append(user_message)
        context.messages.append(user_message)

        request = ModelRequest(
            messages=context.messages,
            model=reasoning_id,
        )
        request = await context_manager.compact_if_needed(
            request=request,
            tool_history=(),
        )
        context.messages[:] = request.messages

        plan = await planner.create_plan(request)
        plan_message = Message(
            role="assistant",
            content=plan.model_dump_json(),
        )
        session.messages.append(plan_message)
        context.messages.append(plan_message)

        execution = PlanExecution(
            plan=plan,
            steps=[StepExecution(step_id=step.step_id) for step in plan.steps],
        )

        show_task_plan(plan)

        approval = input("是否执行该计划？[y/N]: ").strip().lower()

        if approval not in {"y", "yes"}:
            execution.status = PlanStatus.CANCELLED
            print("计划已取消，未执行任何步骤。")
        else:
            execution.status = PlanStatus.RUNNING
            print("\n开始执行计划：")
            execution = await plan_executor.execute(execution)

            if execution.status is PlanStatus.COMPLETED:
                print("\n计划执行完成。")
            else:
                print("\n计划执行失败。")

        if session.messages:
            save_session(session)
            save_context(context)


if __name__ == "__main__":
    from ai_agent_learning.cli import main

    main()
