"""Загрузка и проверка настроек проекта из переменных окружения (.env).

Важное разделение настроек:
  * GenAPI  — embeddings и языковая модель (GENAPI_*, EMBEDDING_MODEL, CHAT_MODEL);
  * Pinecone — векторное хранилище (PINECONE_*);
  * Telegram — токен бота и СЕТЕВОЙ proxy для доступа к api.telegram.org.

Сетевой proxy относится ТОЛЬКО к соединению бота с Telegram API
и никак не связан с GenAPI / Pinecone.
"""

import os
from urllib.parse import quote

from dotenv import load_dotenv

# Загружаем .env из корня проекта (один уровень вверх от app/)
load_dotenv()

# Ожидаемая размерность embedding (индекс nemo создан с 1536 измерениями)
EXPECTED_EMBEDDING_DIM = 1536


def _require(name: str) -> str:
    """Возвращает значение переменной окружения или понятная ошибка, если она пуста."""
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Переменная {name} не задана. "
            "Заполните её в файле .env (образец — в .env.example)."
        )
    return value


def get_genapi_api_key() -> str:
    return _require("GENAPI_API_KEY")


def get_genapi_url() -> str:
    return _require("GENAPI_URL")


def get_genapi_base_url() -> str:
    """Базовый URL GenAPI для OpenAI-совместимого клиента.

    OpenAI-клиент сам добавляет к base_url путь /chat/completions, поэтому
    если в .env указан полный путь (.../v1/chat/completions, как в примерах
    GenAPI), лишний суффикс здесь отрезается — иначе будет 404.
    """
    url = get_genapi_url().rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()


# Endpoint embeddings у GenAPI отдельный от чата: нативный API генераций,
# а не OpenAI-совместимый /v1/embeddings (тот для этой модели отдает 500).
DEFAULT_EMBEDDINGS_URL = "https://api.gen-api.ru/api/v1/networks/embeddings"


def get_embeddings_url() -> str:
    return os.getenv("EMBEDDINGS_URL", DEFAULT_EMBEDDINGS_URL).strip()


def get_pinecone_api_key() -> str:
    return _require("PINECONE_API_KEY")


def get_pinecone_index_name() -> str:
    return os.getenv("PINECONE_INDEX_NAME", "nemo").strip()


def get_pinecone_proxy_url() -> str | None:
    """Сетевой proxy ТОЛЬКО для Pinecone (data-plane индекса) или None.

    У Pinecone отдельный домен для данных индекса, который может быть
    недоступен напрямую из некоторых сетей. Здесь задается независимая
    настройка PINECONE_PROXY_URL — она НЕ влияет ни на Telegram, ни на GenAPI.

    Формат как у PROXY_URL: user:pass@host:port (схема http:// добавится).
    Если пусто — Pinecone работает напрямую.
    """
    proxy_url = os.getenv("PINECONE_PROXY_URL", "").strip()
    if not proxy_url:
        return None
    if "://" in proxy_url:
        return proxy_url
    return f"http://{proxy_url}"


# --- Настройки Telegram-бота -------------------------------------------------


def get_telegram_bot_token() -> str:
    return _require("TELEGRAM_BOT_TOKEN")


def get_chat_model() -> str:
    """Модель языковой модели для ответов (GenAPI, OpenAI-совместимый API)."""
    return _require("CHAT_MODEL")


def get_telegram_proxy_url() -> str | None:
    """Возвращает URL сетевого proxy для Telegram API или None (работать напрямую).

    Основной способ — одна переменная:
        PROXY_URL=user:pass@host:port
    Схема в .env не указывается, здесь добавляется http:// автоматически.
    (Proxy — HTTP-туннель: схема http:// корректна и для https-запросов
    к api.telegram.org.)

    Запасной способ — детальные переменные:
        TELEGRAM_PROXY_TYPE / _HOST / _PORT / _USERNAME / _PASSWORD

    ВАЖНО: этот proxy относится только к соединению «бот -> api.telegram.org».
    GenAPI (embeddings и LLM) и Pinecone работают через свои обычные настройки,
    без него.
    """
    proxy_url = os.getenv("PROXY_URL", "").strip()
    if proxy_url:
        if "://" in proxy_url:
            # Схема уже указана — оставляем её как есть
            return proxy_url
        return f"http://{proxy_url}"

    return _build_proxy_from_parts()


def _build_proxy_from_parts() -> str | None:
    """Собирает proxy из детальных переменных TELEGRAM_PROXY_* (запасной способ)."""
    host = os.getenv("TELEGRAM_PROXY_HOST", "").strip()
    port = os.getenv("TELEGRAM_PROXY_PORT", "").strip()
    if not host or not port:
        return None

    proxy_type = os.getenv("TELEGRAM_PROXY_TYPE", "").strip().lower() or "http"

    # socks5/socks4 -> вариант с резолвом DNS на стороне proxy
    if proxy_type == "socks5":
        proxy_type = "socks5h"
    elif proxy_type == "socks4":
        proxy_type = "socks4a"

    if proxy_type not in {"http", "https", "socks5h", "socks4a"}:
        raise RuntimeError(
            f"Неподдерживаемый TELEGRAM_PROXY_TYPE='{proxy_type}'. "
            "Допустимые значения: http, https, socks5, socks4."
        )

    username = os.getenv("TELEGRAM_PROXY_USERNAME", "").strip()
    password = os.getenv("TELEGRAM_PROXY_PASSWORD", "")

    auth = ""
    if username:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    elif password:
        raise RuntimeError(
            "TELEGRAM_PROXY_PASSWORD задан без TELEGRAM_PROXY_USERNAME — "
            "укажите логин proxy или очистите пароль."
        )

    return f"{proxy_type}://{auth}{host}:{port}"


def proxy_log_label(proxy_url: str) -> str:
    """Безопасная строка для лога: только host:port без логина и пароля.

    Отрезает всё до последнего «@», поэтому credentials не попадают в лог.
    """
    return proxy_url.split("@")[-1]

