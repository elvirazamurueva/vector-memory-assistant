"""Telegram-бот: сообщение -> embedding -> поиск в Pinecone -> контекст -> LLM-ответ.

Модуль связывает компоненты проекта:
    app.embeddings.get_embedding   — векторизация текста через GenAPI
    app.pinecone_client.PineconeClient — поиск и запись в индекс nemo
    app.llm.generate_answer        — генерация ответа через GenAPI (CHAT_MODEL)

Секреты берутся только из переменных окружения (.env), в коде их нет.

Запуск:
    python -m app.bot

Сетевой proxy (PROXY_URL) относится ИСКЛЮЧИТЕЛЬНО к соединению
бота с Telegram API. GenAPI (embeddings + LLM) работает через свои настройки,
а Pinecone использует отдельный PINECONE_PROXY_URL.

Токен бота, ключи API и пароль proxy не пишутся ни в консоль, ни в лог.
"""

import logging
import os
import time
import uuid

import requests
import telebot
from telebot import apihelper
from telebot.types import Message

from app.config import (
    get_chat_model,
    get_pinecone_index_name,
    get_telegram_bot_token,
    get_telegram_proxy_url,
    proxy_log_label,
)
from app.embeddings import get_embedding
from app.llm import generate_answer
from app.pinecone_client import PineconeClient

logger = logging.getLogger("vma.bot")

# Сколько похожих записей брать из Pinecone для контекста
TOP_K = 5

# Единое сообщение об ошибке: без traceback и без технических деталей
FRIENDLY_ERROR = (
    "Произошла внутренняя ошибка, попробуй ещё раз чуть позже. "
    "Технические детали записаны в лог бота."
)

# Все переменные, чьи значения нельзя показывать в логах и чатах
_SECRET_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "PROXY_URL",
    "PINECONE_PROXY_URL",
    "TELEGRAM_PROXY_PASSWORD",
    "GENAPI_API_KEY",
    "PINECONE_API_KEY",
)

_store: PineconeClient | None = None


def get_store() -> PineconeClient:
    """Ленивый singleton клиента Pinecone (существующий индекс из .env)."""
    global _store
    if _store is None:
        _store = PineconeClient()
    return _store


def redact(text: object) -> str:
    """Вырезывает значения секретных переменных из любого текста (логи, ответы)."""
    result = str(text)
    for name in _SECRET_ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value and len(value) >= 4:
            result = result.replace(value, "***")
    return result


def configure_telegram_proxy() -> bool:
    """Настраивает сетевой proxy для pyTelegramBotAPI ДО создания TeleBot.

    PROXY_URL=user:pass@host:port  ->  proxy_url = "http://" + PROXY_URL

    Актуальный способ для telebot 4.x: модуль telebot.apihelper хранит атрибут
    proxy (словарь requests), который подставляется в proxies каждого запроса
    к api.telegram.org. Возвращает True, если proxy применён.

    В лог попадает только host:port: логин, пароль и токен не пишутся.
    """
    proxy_url = get_telegram_proxy_url()
    if not proxy_url:
        logger.info("Proxy для Telegram не задан — подключаюсь к api.telegram.org напрямую")
        return False

    logger.info("Подключаю Telegram через сетевой proxy: %s", proxy_log_label(proxy_url))

    # requests ожидает словарь {"http": url, "https": url}: api.telegram.org — https
    apihelper.proxy = {"http": proxy_url, "https": proxy_url}

    # requests по умолчанию (trust_env=True) учитывает переменные окружения
    # NO_PROXY / no_proxy. Если в системе стоит no_proxy=*, такой session
    # ИГНОРИРУЕТ даже явно заданный proxy, и подключение к Telegram не пройдет.
    # Поэтому фиксируем сессию с trust_env=False: telebot использует
    # apihelper.session для всех запросов, а proxy остаётся явным.
    pinned = requests.Session()
    pinned.trust_env = False
    apihelper.session = pinned
    return True


def collect_memory_texts(matches: list[dict]) -> list[str]:
    """Достает из результатов поиска только тексты (metadata.text)."""
    texts: list[str] = []
    for item in matches:
        text = (item.get("metadata") or {}).get("text")
        if text:
            texts.append(str(text).strip())
    return texts


def handle_remember(text: str) -> str:
    """Сохраняет воспоминание: embedding -> upsert в Pinecone (source=telegram).

    Возвращает ID новой записи.
    """
    vector = get_embedding(text)
    memory_id = f"tg-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

    get_store().upsert_vector(
        vector_id=memory_id,
        values=vector,
        text=text,
        source="telegram",
    )
    logger.info("Сохранено воспоминание id=%s (длина текста %s симв.)", memory_id, len(text))
    return memory_id


def answer_question(question: str) -> str:
    """Полный RAG-цикл: embedding -> top-5 из Pinecone -> контекст -> LLM."""
    vector = get_embedding(question)
    matches = get_store().search(vector, top_k=TOP_K)
    memories = collect_memory_texts(matches)
    logger.info("Поиск в Pinecone: %s похожих записей (top_k=%s)", len(memories), TOP_K)

    answer = generate_answer(user_message=question, memories=memories)
    # Страховка: если модель вдруг повторит ключ, в чат он не уйдет
    return redact(answer)


def register_handlers(bot: telebot.TeleBot) -> None:
    """Регистрирует /start, /help, /remember и обработчик обычного текста."""

    @bot.message_handler(commands=["start", "help"])
    def cmd_start(message: Message) -> None:
        logger.info("Команда /%s от chat_id=%s", message.text.split()[0].lstrip("/"), message.chat.id)
        bot.reply_to(
            message,
            "Привет! Я бот с векторной памятью (Pinecone + GenAPI).\n\n"
            "• /remember <текст> — сохранить воспоминание\n"
            "• обычное сообщение — найду похожие записи и отвечу на их основе\n"
            "• /start, /help — эта справка",
        )

    @bot.message_handler(commands=["remember"])
    def cmd_remember(message: Message) -> None:
        text = (message.text or "").partition(" ")[2].strip()
        if not text:
            bot.reply_to(message, "Использование: /remember <текст, который нужно сохранить>")
            return

        logger.info("/remember от chat_id=%s (длина %s симв.)", message.chat.id, len(text))
        try:
            memory_id = handle_remember(text)
        except Exception:  # noqa: BLE001 — бот не должен падать из-за ошибки API
            logger.exception("Не удалось сохранить воспоминание (chat_id=%s)", message.chat.id)
            bot.reply_to(message, FRIENDLY_ERROR)
            return

        bot.reply_to(message, f"Сохранено. ID: {memory_id}")

    @bot.message_handler(func=lambda message: bool((message.text or "").strip()))
    def cmd_text(message: Message) -> None:
        question = message.text.strip()
        logger.info(
            "Сообщение от chat_id=%s (длина %s симв.): %s",
            message.chat.id,
            len(question),
            redact(question[:100]),
        )
        try:
            answer = answer_question(question)
        except Exception:  # noqa: BLE001 — бот не должен падать из-за ошибки API
            logger.exception("Ошибка обработки сообщения (chat_id=%s)", message.chat.id)
            bot.reply_to(message, FRIENDLY_ERROR)
            return

        bot.reply_to(message, answer)


def create_bot() -> telebot.TeleBot:
    """Создает TeleBot: сначала proxy, затем токен, затем обработчики."""
    configure_telegram_proxy()

    # Токен уходит только в конструктор — наружу (в лог/консоль) он не печатается
    bot = telebot.TeleBot(get_telegram_bot_token(), parse_mode=None, skip_pending=True)
    register_handlers(bot)
    return bot


def main() -> None:
    """Точка входа: python -m app.bot"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Служебные логи SDK (httpx, pinecone) на INFO слишком шумные — оставляем
    # только предупреждения и ошибки, чтобы видеть логи самого бота.
    for noisy in ("httpx", "httpcore", "pinecone", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Обязательные настройки проверяем до подключения к Telegram
    chat_model = get_chat_model()

    bot = create_bot()
    me = bot.get_me()
    logger.info(
        "Бот запущен: @%s (модель ответов: %s, индекс: %s)",
        me.username,
        chat_model,
        get_pinecone_index_name(),
    )

    logger.info("Ожидаю сообщения... (Ctrl+C — остановка)")
    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=20, interval=1.0, none_stop=True)
    except KeyboardInterrupt:
        logger.info("Остановка бота (Ctrl+C).")


if __name__ == "__main__":
    main()
