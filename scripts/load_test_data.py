"""Загрузка тестовых фраз в существующий индекс Pinecone (nemo).

Для каждой фразы получается настоящий embedding через GenAPI,
затем вектор записывается через upsert со стабильным ID (car-001, car-002, ...).
Повторный запуск НЕ создает дубликаты — записи перезаписываются по тем же ID.

Запуск:
    .venv\\Scripts\\Activate.ps1
    python scripts/load_test_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings import get_embedding
from app.pinecone_client import PineconeClient
from data.test_phrases import TEST_PHRASES

SOURCE = "data/test_phrases.py"


def main() -> None:
    client = PineconeClient()
    loaded = 0

    for index, phrase in enumerate(TEST_PHRASES, start=1):
        vector_id = f"car-{index:03d}"

        # Настоящий embedding через API (никаких случайных векторов)
        vector = get_embedding(phrase)

        client.upsert_vector(
            vector_id=vector_id,
            values=vector,
            text=phrase,
            source=SOURCE,
        )
        loaded += 1
        print(f"{vector_id}: записано — {phrase[:60]}")

    print(f"\nУспешно загружено записей: {loaded}")
    stats = client.stats()
    # Векторы лежат в namespace по умолчанию ("__default__" в SDK 10+)
    namespaces = stats.get("namespaces") or {}
    default_ns = namespaces.get("__default__") or namespaces.get("") or {}
    total = default_ns.get("vector_count", stats.get("total_vector_count", "?"))
    print(f"Всего векторов в индексе: {total}")


if __name__ == "__main__":
    main()
