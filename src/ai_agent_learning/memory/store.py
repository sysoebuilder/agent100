"""使用 Qdrant 本地模式持久化、查询和管理长期记忆。"""

import asyncio
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from qdrant_client import AsyncQdrantClient, models

from ai_agent_learning.memory.schema import (
    MemoryCandidate,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    StoredMemory,
)
from ai_agent_learning.paths import get_data_dir

DEFAULT_COLLECTION_NAME = "memories"
DEFAULT_VECTOR_SIZE = 512


def default_memory_database_path() -> Path:
    """返回默认的本地 Qdrant 数据目录。"""
    return get_data_dir() / "memory" / "qdrant"


class QdrantMemoryStore:
    """单用户长期记忆的本地向量存储。"""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size 必须大于 0")
        if not collection_name.strip():
            raise ValueError("collection_name 不能为空")

        database_path = Path(path or default_memory_database_path()).resolve()
        database_path.mkdir(parents=True, exist_ok=True)
        self._client = AsyncQdrantClient(path=str(database_path))
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_size(self) -> int:
        return self._vector_size

    async def initialize(self) -> None:
        """创建集合；集合已存在时校验向量配置。"""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return
            if await self._client.collection_exists(self._collection_name):
                info = await self._client.get_collection(self._collection_name)
                vector_config = info.config.params.vectors
                if not isinstance(vector_config, models.VectorParams):
                    raise ValueError("记忆集合必须使用单一向量配置")
                if vector_config.size != self._vector_size:
                    raise ValueError(
                        "记忆集合向量维度不一致："
                        f"数据库为 {vector_config.size}，配置为 {self._vector_size}"
                    )
                if vector_config.distance != models.Distance.COSINE:
                    raise ValueError("记忆集合必须使用 Cosine 距离")
            else:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            self._initialized = True

    async def add(
        self,
        candidate: MemoryCandidate,
        vector: Sequence[float],
        *,
        memory_id: str | None = None,
    ) -> StoredMemory:
        """把经过决策的候选记忆作为正式记忆写入数据库。"""
        now = datetime.now(UTC)
        memory = StoredMemory(
            **candidate.model_dump(),
            id=memory_id or str(uuid4()),
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
        )
        await self.upsert(memory, vector)
        return memory

    async def upsert(
        self,
        memory: StoredMemory,
        vector: Sequence[float],
    ) -> None:
        """新增或完整覆盖一条正式记忆及其 content 向量。"""
        await self.initialize()
        normalized_vector = self._validate_vector(vector)
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=memory.id,
                    vector=normalized_vector,
                    payload=memory.model_dump(mode="json"),
                )
            ],
            wait=True,
        )

    async def get(self, memory_id: str) -> StoredMemory | None:
        """按 ID 读取一条正式记忆。"""
        await self.initialize()
        records = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return self._memory_from_payload(records[0].payload)

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        scope: str | None = None,
        memory_type: MemoryType | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> list[MemorySearchResult]:
        """按 content 向量检索相似的正式记忆。"""
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        await self.initialize()
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=self._validate_vector(vector),
            query_filter=self._build_filter(
                scope=scope,
                memory_type=memory_type,
                status=status,
            ),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [
            MemorySearchResult(
                memory=self._memory_from_payload(point.payload),
                score=point.score,
            )
            for point in response.points
        ]

    async def find_by_key(
        self,
        key: str,
        *,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        limit: int = 10,
    ) -> list[StoredMemory]:
        """按结构化 key 精确查找记忆，用于更新和去重。"""
        if not key.strip():
            raise ValueError("key 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        await self.initialize()
        conditions: list[models.Condition] = [
            self._match_condition("key", key),
            self._match_condition("status", status.value),
        ]
        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [self._memory_from_payload(record.payload) for record in records]

    async def set_status(
        self,
        memory_id: str,
        status: MemoryStatus,
    ) -> StoredMemory | None:
        """更新记忆状态，并同步更新时间。"""
        memory = await self.get(memory_id)
        if memory is None:
            return None
        now = datetime.now(UTC)
        await self._client.set_payload(
            collection_name=self._collection_name,
            points=[memory_id],
            payload={"status": status.value, "updated_at": now.isoformat()},
            wait=True,
        )
        return memory.model_copy(update={"status": status, "updated_at": now})

    async def confirm(self, memory_id: str) -> StoredMemory | None:
        """记录一次重复确认，不修改正文和向量。"""
        memory = await self.get(memory_id)
        if memory is None:
            return None
        now = datetime.now(UTC)
        confirmation_count = memory.confirmation_count + 1
        await self._client.set_payload(
            collection_name=self._collection_name,
            points=[memory_id],
            payload={
                "updated_at": now.isoformat(),
                "last_confirmed_at": now.isoformat(),
                "confirmation_count": confirmation_count,
            },
            wait=True,
        )
        return memory.model_copy(
            update={
                "updated_at": now,
                "last_confirmed_at": now,
                "confirmation_count": confirmation_count,
            }
        )

    async def delete(self, memory_id: str) -> bool:
        """永久删除一条记忆；返回该记忆原先是否存在。"""
        if await self.get(memory_id) is None:
            return False
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[memory_id]),
            wait=True,
        )
        return True

    async def count(
        self,
        *,
        status: MemoryStatus | None = None,
    ) -> int:
        """统计记忆数量，可按状态过滤。"""
        await self.initialize()
        count_filter = None
        if status is not None:
            count_filter = models.Filter(
                must=[self._match_condition("status", status.value)]
            )
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=count_filter,
            exact=True,
        )
        return result.count

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self._vector_size:
            raise ValueError(
                f"向量维度必须为 {self._vector_size}，实际为 {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("向量不能包含 NaN 或无穷值")
        return values

    @staticmethod
    def _match_condition(key: str, value: str) -> models.FieldCondition:
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )

    @classmethod
    def _build_filter(
        cls,
        *,
        scope: str | None,
        memory_type: MemoryType | None,
        status: MemoryStatus,
    ) -> models.Filter:
        conditions: list[models.Condition] = [
            cls._match_condition("status", status.value)
        ]
        if memory_type is not None:
            conditions.append(cls._match_condition("type", memory_type.value))
        return models.Filter(must=conditions)

    @staticmethod
    def _memory_from_payload(payload: dict[str, Any] | None) -> StoredMemory:
        if payload is None:
            raise ValueError("Qdrant 记忆记录缺少 payload")
        return StoredMemory.model_validate(payload)
