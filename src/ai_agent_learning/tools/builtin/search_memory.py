"""记忆检索工具定义。"""

from typing import Any

from ai_agent_learning.memory.embedding import GlmEmbeddingClient
from ai_agent_learning.memory.schema import MemorySearchResult
from ai_agent_learning.memory.store import QdrantMemoryStore
from ai_agent_learning.tools.contracts import RiskLevel, ToolSpec

KEY_LIMIT = 10
VECTOR_LIMIT = 5
SCORE_THRESHOLD = 0.6

SPEC = ToolSpec(
    name="search_memory",
    description=(
        "按需检索与当前对话有关的用户长期记忆。"
        "在回答需要依赖以往记忆时调用。"
        "queries 应描述需要查找的信息，并结合当前对话消除‘之前’‘这个’等模糊指代；"
        "不得猜测用户未表达过的事实。不同的信息需求可拆成多条查询。"
    ),
    risk_level=RiskLevel.LOW,
    input_schema={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "pattern": r"\S",
                },
                "description": "需要从长期记忆中查找的信息，最多五条。",
            },
            "key": {
                "type": "string",
                "minLength": 1,
                "pattern": r"^[a-z][a-z0-9_]*$",
                "description": (
                    "用于精确匹配记忆的稳定英文 snake_case 标识，"
                    "例如 response_style。已知时填写，不确定时省略，"
                    "不要把查询关键词当作 key。"
                ),
            },
        },
        "required": ["queries"],
        "additionalProperties": False,
    },
    has_side_effects=False,
)


async def handler(
    queries: list[str],
    key: str | None = None,
) -> dict[str, Any]:
    """合并 key 精确查询与向量查询结果，并按记忆 ID 去重。"""
    normalized_queries = list(dict.fromkeys(query.strip() for query in queries))
    normalized_queries = [query for query in normalized_queries if query]
    if not normalized_queries:
        raise ValueError("至少需要一条非空记忆查询")

    normalized_key = key.strip() if key else None
    embedding = GlmEmbeddingClient()
    store = QdrantMemoryStore(vector_size=embedding.dimensions)
    memories_by_id: dict[str, dict[str, Any]] = {}

    try:
        await store.initialize()

        if normalized_key:
            key_matches = await store.find_by_key(normalized_key, limit=KEY_LIMIT)
            for memory in key_matches:
                memories_by_id[memory.id] = {
                    "memory": memory.model_dump(mode="json"),
                    "score": None,
                }

        vectors = await embedding.embed(normalized_queries)
        vector_matches_by_id: dict[str, MemorySearchResult] = {}
        for vector in vectors:
            vector_matches = await store.search(
                vector,
                limit=VECTOR_LIMIT,
                score_threshold=SCORE_THRESHOLD,
            )
            for result in vector_matches:
                previous = vector_matches_by_id.get(result.memory.id)
                if previous is None or result.score > previous.score:
                    vector_matches_by_id[result.memory.id] = result

        best_vector_matches = sorted(
            vector_matches_by_id.values(),
            key=lambda result: result.score,
            reverse=True,
        )
        for result in best_vector_matches:
            item = memories_by_id.setdefault(
                result.memory.id,
                {
                    "memory": result.memory.model_dump(mode="json"),
                    "score": result.score,
                },
            )
            previous_score = item["score"]
            if previous_score is None or result.score > previous_score:
                item["score"] = result.score

        return {"memories": list(memories_by_id.values())}
    finally:
        try:
            await store.close()
        finally:
            await embedding.close()
