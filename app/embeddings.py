"""Получение embedding через GenAPI.

У GenAPI два разных API, и для embeddings нужен именно второй:
  * чат (LLM) — OpenAI-совместимый endpoint из GENAPI_URL;
  * embeddings — нативный API генераций: EMBEDDINGS_URL
    (https://api.gen-api.ru/api/v1/networks/embeddings).

Нативный endpoint принимает POST с телом
    {"input": ["текст"], "model": "text-embedding-3-small", "is_sync": true}
и сразу возвращает вектор в response.data[0].embedding.

Ключ (GENAPI_API_KEY) берется из .env, в коде его нет и в логи он не пишется.
"""

import requests

from app.config import (
    EXPECTED_EMBEDDING_DIM,
    get_embedding_model,
    get_embeddings_url,
    get_genapi_api_key,
)

# Генерация embedding занимает ~3 секунды, даем запас на медленный ответ
REQUEST_TIMEOUT = 60


class EmbeddingError(RuntimeError):
    """Ошибка получения embedding (сеть, API, некорректный ответ)."""


def get_embedding(text: str) -> list[float]:
    """Возвращает embedding для текста, проверив размерность 1536.

    Полный вектор в консоль не выводится — только размерность.
    Ключ API никогда не попадает ни в сообщение об ошибке, ни в лог.
    """
    if not text or not text.strip():
        raise ValueError("Текст для embedding не должен быть пустым")

    payload = {
        "input": [text.strip()],
        "model": get_embedding_model(),
        "is_sync": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {get_genapi_api_key()}",
    }

    try:
        response = requests.post(
            get_embeddings_url(),
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        # str(exc) у requests не содержит заголовков с ключом
        raise EmbeddingError(f"Сеть недоступна при запросе embedding: {exc}") from exc

    if response.status_code != 200:
        raise EmbeddingError(
            f"GenAPI вернул HTTP {response.status_code} на запрос embedding"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise EmbeddingError("GenAPI вернул не-JSON ответ на запрос embedding") from exc

    if isinstance(data.get("error"), dict):
        message = str(data["error"].get("message", ""))[:200]
        raise EmbeddingError(f"GenAPI вернул ошибку: {message}")

    try:
        vector = data["response"]["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingError("Неожиданная структура ответа GenAPI") from exc

    if len(vector) != EXPECTED_EMBEDDING_DIM:
        raise EmbeddingError(
            f"Неожиданная размерность embedding: {len(vector)} "
            f"(ожидалось {EXPECTED_EMBEDDING_DIM})"
        )

    return vector
