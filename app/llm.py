"""Обращение к языковой модели через GenAPI (OpenAI-совместимый API).

Модель берется из переменной окружения CHAT_MODEL (.env).
API-ключ и адрес берутся из существующей конфигурации GenAPI —
в этом модуле нет ни ключей, ни захардкоженных адресов.

Сетевой proxy для Telegram к этой части НЕ относится: запросы в GenAPI
идут напрямую, через GENAPI_URL.
"""

from openai import OpenAI

from app.config import get_chat_model, get_genapi_api_key, get_genapi_base_url

SYSTEM_PROMPT = (
    "Ты — ассистент проекта Vector Memory Assistant. "
    "Отвечай на языке пользователя, кратко и по делу. "
    "Опирайся на найденные в векторном хранилище записи (контекст ниже). "
    "Если контекста недостаточно, честно скажи об этом и не выдумывай факты."
)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Возвращает (кэшированный) OpenAI-совместимый клиент для GenAPI."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_genapi_api_key(),
            base_url=get_genapi_base_url(),
        )
    return _client


def generate_answer(user_message: str, memories: list[str]) -> str:
    """Возвращает текст ответа LLM по сообщению пользователя и найденному контексту.

    Args:
        user_message: исходное сообщение пользователя из Telegram.
        memories: тексты (metadata.text) записей, найденных в Pinecone.

    Raises:
        ValueError: пустое сообщение пользователя.
        RuntimeError: API вернул некорректный (пустой) ответ.
    """
    if not user_message or not user_message.strip():
        raise ValueError("Сообщение пользователя не должно быть пустым")

    context_block = (
        "\n".join(f"- {line}" for line in memories)
        if memories
        else "(ничего не найдено)"
    )

    prompt = (
        f"Найденные похожие записи:\n{context_block}\n\n"
        f"Вопрос пользователя:\n{user_message.strip()}"
    )

    client: OpenAI = get_client()
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    if not response.choices:
        raise RuntimeError("GenAPI вернул пустой ответ (отсутствуют choices)")

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("GenAPI вернул пустой текст ответа")

    return answer
