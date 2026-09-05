import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from ai_agent_learning.tools.builtin.web_search import WebSearchClient
from ai_agent_learning.tools.catalog import build_builtin_tools
from ai_agent_learning.tools.contracts import ToolCall
from ai_agent_learning.tools.executor import ToolExecutor
from ai_agent_learning.tools.registry import ToolRegistry


def execute_search(handler, arguments):
    async def run():
        async with httpx.AsyncClient(
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            headers={"Authorization": "Bearer test-key"},
            transport=httpx.MockTransport(handler),
        ) as http_client:
            registry = ToolRegistry(
                tools=build_builtin_tools(WebSearchClient(http_client))
            )
            return await ToolExecutor(registry).execute_with_recovery(
                ToolCall("search-1", "web_search", arguments)
            )

    return asyncio.run(run())


def test_search_maps_request_and_keeps_citable_sources():
    def handler(request):
        assert request.url == "https://open.bigmodel.cn/api/paas/v4/web_search"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "search_query": "智谱官方文档",
            "search_engine": "search_pro",
            "search_intent": False,
            "count": 1,
            "content_size": "medium",
        }
        return httpx.Response(
            200,
            json={
                "search_result": [
                    {
                        "title": "文档",
                        "link": "https://docs.bigmodel.cn/",
                        "content": "说明",
                        "media": "智谱",
                        "publish_date": "2026-09-05",
                        "icon": "ignored",
                    },
                    {
                        "title": "额外结果",
                        "link": "https://example.com/",
                        "content": "内容",
                    },
                ]
            },
        )

    result = execute_search(handler, {"query": "智谱官方文档", "count": 1})
    assert result.success
    assert result.data == {
        "query": "智谱官方文档",
        "results": [
            {
                "title": "文档",
                "url": "https://docs.bigmodel.cn/",
                "summary": "说明",
                "source": "智谱",
                "publish_date": "2026-09-05",
            },
        ],
    }
    assert len(result.metadata["attempts"]) == 1


def test_default_count_and_empty_results_are_successful():
    def handler(request):
        assert json.loads(request.content)["count"] == 5
        return httpx.Response(200, json={"search_result": []})

    result = execute_search(handler, {"query": "no results"})
    assert result.success and result.data["results"] == []


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": ""},
        {"query": "  "},
        {"query": "a" * 71},
        {"query": "a", "count": 0},
        {"query": "a", "count": 51},
        {"query": "a", "count": "5"},
        {"query": "a", "api_key": "model-key"},
    ],
)
def test_invalid_arguments_do_not_make_network_requests(arguments):
    def handler(request):
        pytest.fail("Invalid tool input must be rejected before making an HTTP request")

    result = execute_search(handler, arguments)
    assert not result.success
    assert result.error["code"] == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("status,expected_attempts", [(429, 3), (503, 3), (401, 1)])
def test_executor_handles_http_errors_without_nested_retries(
    status, expected_attempts, monkeypatch
):
    monkeypatch.setattr(
        "ai_agent_learning.tools.executor.wait_before_retry", AsyncMock(return_value=0)
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "unavailable"}})

    result = execute_search(handler, {"query": "天气"})
    assert not result.success
    assert result.error["code"] == f"HTTP_{status}"
    assert len(requests) == len(result.metadata["attempts"]) == expected_attempts


def test_executor_recovers_from_timeout(monkeypatch):
    monkeypatch.setattr(
        "ai_agent_learning.tools.executor.wait_before_retry", AsyncMock(return_value=0)
    )
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("search timed out", request=request)
        return httpx.Response(200, json={"search_result": []})

    result = execute_search(handler, {"query": "天气"})
    assert result.success
    assert len(result.metadata["attempts"]) == 2


@pytest.mark.parametrize(
    "payload", [{}, {"search_result": "invalid"}, {"search_result": [{}]}]
)
def test_malformed_responses_are_failures(payload):
    result = execute_search(
        lambda request: httpx.Response(200, json=payload), {"query": "天气"}
    )
    assert not result.success
    assert len(result.metadata["attempts"]) == 1
