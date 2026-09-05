import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_agent_learning import main


@pytest.mark.parametrize("failure_stage", [None, "registration", "body", "cancel"])
def test_runtime_closes_model_and_search_connections(failure_stage, monkeypatch):
    async def run():
        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        )
        model = SimpleNamespace(close=AsyncMock())
        monkeypatch.setattr(main, "GlmModelClient", lambda **kwargs: model)

        def create_http_client(**kwargs):
            assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
            return http_client

        monkeypatch.setattr(main.httpx, "AsyncClient", create_http_client)
        if failure_stage == "registration":

            def fail_registration(search_client):
                raise RuntimeError("registration failed")

            monkeypatch.setattr(main, "build_builtin_tools", fail_registration)

        async def use_runtime():
            async with main.create_runtime("test-key") as (client, registry):
                assert client is model
                assert {spec.name for spec in registry.list_specs()} == {
                    "current_time",
                    "web_search",
                }
                if failure_stage == "body":
                    raise RuntimeError("body failed")
                if failure_stage == "cancel":
                    raise asyncio.CancelledError()

        if failure_stage == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await use_runtime()
        elif failure_stage:
            with pytest.raises(RuntimeError):
                await use_runtime()
        else:
            await use_runtime()
        model.close.assert_awaited_once()
        assert http_client.is_closed

    asyncio.run(run())
