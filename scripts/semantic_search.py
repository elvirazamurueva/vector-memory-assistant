"""Семантический поиск по индексу Pinecone (nemo).

5 поисковых запросов, сформулированных ИНАЧЕ, чем исходные фразы.
Для каждого запроса: embedding -> поиск top-5 -> вывод ID, score, текст.

Запуск (после load_test_data.py):
    .venv\\Scripts\\Activate.ps1
    python scripts/semantic_search.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings import get_embedding
from app.pinecone_client import PineconeClient

SEARCH_QUERIES = [
    "какая машина подходит для поездок зимой по заснеженной дороге",
    "автомобиль с наименьшим расходом топлива в городе",
    "что выбрать для большой семьи и дальних путешествий",
    "машина с современными электронными помощниками водителя",
    "мощный звук двигателя и спортивный характер",
]

TOP_K = 5


def main() -> None:
    client = PineconeClient()

    for number, query in enumerate(SEARCH_QUERIES, start=1):
        print("=" * 70)
        print(f"Запрос {number}: {query}")
        print("-" * 70)

        query_vector = get_embedding(query)
        results = client.search(query_vector, top_k=TOP_K)

        if not results:
            print("Ничего не найдено. Сначала выполните scripts/load_test_data.py")
            continue

        for rank, hit in enumerate(results, start=1):
            text = hit["metadata"].get("text", "(нет текста в metadata)")
            print(f"{rank}. ID={hit['id']}  score={hit['score']:.4f}  {text}")

    print("=" * 70)


if __name__ == "__main__":
    main()
