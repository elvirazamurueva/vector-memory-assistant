"""Клиент для работы с уже существующим индексом Pinecone (nemo).

Используется актуальный официальный Python SDK `pinecone` (версии 6+).
Индекс НЕ создается — подключение идет к существующему.
"""

from typing import Any

from pinecone import Pinecone

from app.config import (
    EXPECTED_EMBEDDING_DIM,
    get_pinecone_api_key,
    get_pinecone_index_name,
    get_pinecone_proxy_url,
)


class PineconeClient:
    """Обертка над индексом Pinecone: upsert, поиск, fetch, delete, stats."""

    def __init__(self) -> None:
        # proxy_url применяется только к Pinecone (если задан PINECONE_PROXY_URL);
        # на Telegram и GenAPI он не влияет.
        self._pc = Pinecone(
            api_key=get_pinecone_api_key(),
            proxy_url=get_pinecone_proxy_url(),
        )
        # Подключаемся к уже существующему индексу (новый не создаем)
        self._index = self._pc.Index(name=get_pinecone_index_name())

    def upsert_vector(
        self,
        vector_id: str,
        values: list[float],
        text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Записывает один вектор с метаданными (text, source).

        Повторный upsert с тем же vector_id перезаписывает запись,
        поэтому дубликаты при повторном запуске не появляются.
        Возвращает количество записанных векторов.
        """
        if len(values) != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Длина вектора {len(values)} не равна {EXPECTED_EMBEDDING_DIM}"
            )

        full_metadata: dict[str, Any] = {"text": text, "source": source}
        if metadata:
            full_metadata.update(metadata)

        result = self._index.upsert(vectors=[(vector_id, values, full_metadata)])
        return result.upserted_count

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Поиск ближайших векторов. Возвращает список {id, score, metadata}."""
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
            }
            for match in response.matches
        ]

    def fetch(self, vector_id: str) -> dict[str, Any] | None:
        """Достает запись по ID. Возвращает None, если записи нет."""
        response = self._index.fetch(ids=[vector_id])
        vectors = response.vectors or {}
        if vector_id not in vectors:
            return None
        record = vectors[vector_id]
        return {
            "id": record.id,
            "values": record.values,
            "metadata": record.metadata or {},
        }

    def delete(self, vector_id: str) -> None:
        """Удаляет вектор по ID."""
        self._index.delete(ids=[vector_id])

    def stats(self) -> dict[str, Any]:
        """Возвращает статистику индекса (количество векторов и т.п.)."""
        return self._index.describe_index_stats().to_dict()
