import asyncio
import json
from contextlib import asynccontextmanager

import pytest

import ai_agent_learning.main as main_module
from ai_agent_learning.agent.state import (
    AgentRunResult,
    AgentState,
    AgentStatus,
    AgentStep,
    StopReason,
)
from ai_agent_learning.main import agent_run_result_to_messages
from ai_agent_learning.model_config import create_model_config
from ai_agent_learning.schema import Context, Message, ModelResponse, Session
from ai_agent_learning.tools.contracts import ToolCall, ToolResult


def test_agent_run_result_to_messages_preserves_tool_events() -> None:
    calls = [
        ToolCall(
            id="call-1",
            name="first_tool",
            arguments={"value": 1},
        ),
        ToolCall(
            id="call-2",
            name="second_tool",
            arguments={"value": 2},
        ),
    ]
    results = [
        ToolResult(
            call_id="call-1",
            success=True,
            data={"answer": 10},
        ),
        ToolResult(
            call_id="call-2",
            success=False,
            error={"code": "FAILED", "message": "工具失败"},
        ),
    ]
    final_message = Message(
        role="assistant",
        content="工具执行完成。",
    )
    final_response = ModelResponse(message=final_message)
    state = AgentState(
        status=AgentStatus.COMPLETED,
        current_step=2,
        model_calls_used=2,
        tool_calls_used=2,
        stop_reason=StopReason.FINAL_RESPONSE,
        steps=[
            AgentStep(
                index=1,
                tool_calls=calls,
                tool_results=results,
            ),
            AgentStep(
                index=2,
                message=final_message,
            ),
        ],
    )
    run_result = AgentRunResult(
        response=final_response,
        state=state,
    )

    messages = agent_run_result_to_messages(run_result)

    tool_events = [json.loads(message.content) for message in messages[:-1]]
    assert [event["event"] for event in tool_events] == [
        "agent.tool_call",
        "agent.tool_call",
        "agent.tool_result",
        "agent.tool_result",
    ]
    assert [event["call_id"] for event in tool_events] == [
        "call-1",
        "call-2",
        "call-1",
        "call-2",
    ]
    assert messages[-1] == final_message
    assert messages.count(final_message) == 1


def test_run_config_bootstraps_when_source_is_missing(monkeypatch, tmp_path) -> None:
    selected = create_model_config("glm", "secret-key")
    saved_source = []
    saved_active = []

    class FakeUI:
        def show_notice(self, message):
            pass

        def show_error(self, message):
            pytest.fail(message)

        def setup_first_model(self):
            return selected

    monkeypatch.setattr(main_module, "TerminalUI", FakeUI)
    monkeypatch.setattr(main_module, "source_config_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(main_module, "save_source_model", saved_source.append)
    monkeypatch.setattr(
        main_module, "save_active_model", lambda config: saved_active.append(config) or tmp_path / "active-model.env",
    )

    main_module.run_config()

    assert saved_source == [selected]
    assert saved_active == [selected]


@pytest.mark.parametrize("error", [None, RuntimeError, asyncio.CancelledError])
def test_chat_prints_deltas_before_run_returns(monkeypatch, capsys, error):
    session = Session(name="流式测试")
    context = Context(context_id=session.session_id)
    saved = []
    partial = Message(role="assistant", content="我先查询。")
    final = Message(role="assistant", content="结果如下。")

    class FakeLoop:
        def __init__(self, **kwargs):
            pass

        async def run(
            self, request, *, on_text_delta, on_reasoning_delta, on_response_end
        ):
            capsys.readouterr()  # 清除启动标题和输入提示。
            on_reasoning_delta("临时思考")
            on_text_delta("我先")
            # run 尚未结束，终端就已经收到了第一块文字。
            assert capsys.readouterr().out == "助手: 我先"
            if error:
                raise error("中断")
            on_text_delta("查询。")
            on_response_end()
            assert capsys.readouterr().out == "查询。\n"
            on_text_delta("结果")
            assert capsys.readouterr().out == "助手: 结果"
            on_text_delta("如下。")
            on_response_end()
            return AgentRunResult(
                response=ModelResponse(message=final),
                state=AgentState(
                    status=AgentStatus.COMPLETED,
                    stop_reason=StopReason.FINAL_RESPONSE,
                    steps=[
                        AgentStep(index=1, message=partial),
                        AgentStep(index=2, message=final),
                    ],
                ),
            )

    @asynccontextmanager
    async def runtime():
        yield object(), object(), "glm"

    inputs = iter(["你好", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "create_runtime", runtime)
    monkeypatch.setattr(main_module, "AgentLoop", FakeLoop)
    monkeypatch.setattr(main_module, "select_session", lambda **kwargs: session)
    monkeypatch.setattr(main_module, "select_context", lambda session_id: context)
    monkeypatch.setattr(main_module, "save_session", saved.append)
    monkeypatch.setattr(main_module, "save_context", saved.append)

    if error is asyncio.CancelledError:
        with pytest.raises(error):
            asyncio.run(main_module.run_chat())
        assert capsys.readouterr().out == "\n"
    elif error:
        asyncio.run(main_module.run_chat())
        output = capsys.readouterr().out
        assert "错误 · 中断" in output
        assert "临时思考" not in output
    else:
        asyncio.run(main_module.run_chat())
        output = capsys.readouterr().out
        assert output.startswith("如下。\n")
        assert "结果如下。" not in output
        assert session.messages[-2:] == [partial, final]
        assert context.messages[-2:] == [partial, final]
        assert saved == [session, context]
