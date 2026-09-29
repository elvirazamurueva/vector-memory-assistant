"""Проверка получения embedding через GenAPI.

Запуск:
    .venv\\Scripts\\Activate.ps1
    python scripts/test_embedding.py
"""

import sys
from pathlib import Path

# Делаем пакет app/ импортируемым при запуске из любой директории
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EXPECTED_EMBEDDING_DIM, get_embedding_model
from app.embeddings import get_embedding

TEST_PHRASE = "Nissan Juke — компактный кроссовер"


def main() -> None:
    print(f"Модель: {get_embedding_model()}")
    print(f"Тестовая фраза: {TEST_PHRASE}")

    vector = get_embedding(TEST_PHRASE)

    print("Запрос успешен: OK")
    print(f"Размерность embedding: {len(vector)} (ожидалась {EXPECTED_EMBEDDING_DIM})")
    # Первые 5 чисел — только чтобы показать, что вектор настоящий.
    print(f"Первые 5 значений: {[round(x, 4) for x in vector[:5]]}")


if __name__ == "__main__":
    main()
