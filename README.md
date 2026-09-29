# Vector Memory Assistant

**Семантическая память на Pinecone и GenAPI**

Локальный Telegram-бот с семантической памятью. Он принимает сообщения
пользователя, ищет смыслово похожие записи в векторном хранилище Pinecone,
использует найденный контекст при генерации ответа через LLM и умеет
сохранять новую информацию по команде `/remember`.

Что умеет бот:

- принимать текстовые сообщения пользователя в Telegram;
- превращать их в embedding и искать семантически похожие записи в Pinecone;
- передавать найденный контекст языковой модели вместе с вопросом;
- сохранять новые записи в память командой `/remember <текст>`.

---

## Архитектура

Фактическая схема обработки сообщения:

```
Пользователь
   ↓
Telegram
   ↓  сетевой proxy (PROXY_URL)
Telegram Bot (app/bot.py)
   ↓
embedding через GenAPI (app/embeddings.py)
   ↓
Pinecone — поиск top-5 похожих записей (app/pinecone_client.py)
   ↓
семантически похожие записи (metadata.text)
   ↓
LLM через GenAPI (app/llm.py)
   ↓
ответ пользователю в Telegram
```

Роли компонентов:

| Компонент | Назначение |
|---|---|
| **GenAPI** | Два независимых API: embeddings (векторизация текста) и LLM (генерация ответа) |
| **Pinecone** | Векторное хранилище и семантический поиск (индекс `nemo`) |
| **Telegram** | Интерфейс пользователя: приём сообщений и отправка ответов |
| **`PROXY_URL`** | Сетевой proxy **только для Telegram** |
| **`PINECONE_PROXY_URL`** | Сетевой proxy **только для Pinecone** |

### Про proxy

`PROXY_URL` и `PINECONE_PROXY_URL` — это **сетевые** proxy, используемые
исключительно для установления сетевого соединения:

- `PROXY_URL` — соединение бота с `api.telegram.org`, если Telegram недоступен
  напрямую;
- `PINECONE_PROXY_URL` — соединение с доменом данных индекса Pinecone, который
  может быть недоступен напрямую из некоторых сетей.

Это не API-провайдеры и не endpoint-ы языковой модели. GenAPI (embeddings и LLM)
работает через свои собственные настройки (`GENAPI_URL`, `EMBEDDINGS_URL`) и
указанные proxy не использует. Настройки не смешиваются: у каждого компонента
свой proxy.

---

## Возможности

- **Embeddings** — модель `text-embedding-3-small`, размерность вектора 1536.
- **Семантический поиск** — поиск по смыслу (косинусное сходство), а не по
  точному совпадению слов.
- **Pinecone index `nemo`** — работа с существующим индексом.
- **Top-5** — для контекста берутся 5 наиболее похожих записей.
- **RAG-контекст** — тексты найденных записей передаются в LLM вместе с вопросом.
- **Генерация ответа через GenAPI** — модель задаётся переменной `CHAT_MODEL`.
- **`/remember`** — сохранение новой записи в векторную память
  (`source=telegram`).
- **Telegram-бот** — на библиотеке pyTelegramBotAPI, с поддержкой сетевого proxy.
- **Логирование** — запуск, входящие сообщения, результаты поиска, ошибки API,
  сохранение записей. Секреты в лог не попадают.
- **Обработка ошибок** — при недоступности Pinecone или LLM пользователь
  получает понятное сообщение, технические детали уходят только в лог.

---

## Технологии

| Технология | Назначение |
|---|---|
| Python 3.10+ | Язык реализации |
| [pyTelegramBotAPI](https://pypi.org/project/pyTelegramBotAPI/) | Telegram-бот |
| [openai](https://pypi.org/project/openai/) | OpenAI-совместимый клиент для чата GenAPI |
| [pinecone](https://pypi.org/project/pinecone/) | Официальный SDK Pinecone |
| [requests](https://pypi.org/project/requests/) | HTTP-запросы к embeddings GenAPI |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Загрузка настроек из `.env` |
| [PySocks](https://pypi.org/project/PySocks/) | Поддержка SOCKS-proxy |

---

## Структура проекта

```
vector-memory-assistant/
├── .env                  # реальные секреты (не коммитится)
├── .env.example          # шаблон настроек без секретов
├── .gitignore
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── config.py             # настройки из .env
│   ├── embeddings.py         # get_embedding() через GenAPI
│   ├── pinecone_client.py    # PineconeClient: upsert/search/fetch/delete/stats
│   ├── llm.py                # generate_answer() через GenAPI
│   └── bot.py                # Telegram-бот
├── data/
│   └── test_phrases.py       # 30 тестовых фраз об автомобилях
└── scripts/
    ├── test_embedding.py     # проверка embeddings
    ├── load_test_data.py     # загрузка тестовых данных в Pinecone
    └── semantic_search.py    # проверка семантического поиска
```

---

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## Настройка

Скопируйте `.env.example` в `.env` и заполните значения:

```dotenv
# GenAPI — embeddings и LLM
GENAPI_API_KEY=
GENAPI_URL=https://proxy.gen-api.ru/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDINGS_URL=https://api.gen-api.ru/api/v1/networks/embeddings
CHAT_MODEL=gpt-4o-mini

# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=nemo
PINECONE_PROXY_URL=

# Telegram
TELEGRAM_BOT_TOKEN=
PROXY_URL=
```

| Переменная | Описание |
|---|---|
| `GENAPI_API_KEY` | Ключ API GenAPI (используется и для embeddings, и для LLM) |
| `GENAPI_URL` | Базовый URL чата GenAPI (OpenAI-совместимый) |
| `EMBEDDING_MODEL` | Модель embeddings (`text-embedding-3-small`) |
| `EMBEDDINGS_URL` | Endpoint embeddings GenAPI |
| `CHAT_MODEL` | Модель для генерации ответов, например `gpt-4o-mini` |
| `PINECONE_API_KEY` | Ключ API Pinecone |
| `PINECONE_INDEX_NAME` | Имя индекса Pinecone (`nemo`) |
| `PINECONE_PROXY_URL` | Сетевой proxy для Pinecone (необязательно) |
| `TELEGRAM_BOT_TOKEN` | Токен бота от @BotFather |
| `PROXY_URL` | Сетевой proxy для Telegram (необязательно) |

> **Про embeddings GenAPI.** У сервиса два разных API: чат (OpenAI-совместимый,
> `GENAPI_URL`) и embeddings (нативный API генераций, `EMBEDDINGS_URL`).
> Embeddings идут через `EMBEDDINGS_URL` и возвращают вектор размерностью 1536.
> Ключ `GENAPI_API_KEY` — общий для обоих API.

### Proxy (необязательно)

Формат для обоих proxy — `user:pass@host:port` без схемы (`http://` добавляется
кодом). Если proxy без авторизации — `host:port`.

```dotenv
PROXY_URL=user:pass@host:port
PINECONE_PROXY_URL=user:pass@host:port
```

Обе переменные независимы: `PROXY_URL` влияет только на Telegram,
`PINECONE_PROXY_URL` — только на Pinecone. Если оставить их пустыми, оба
компонента работают напрямую.

Запасной способ настройки proxy для Telegram — детальные переменные
`TELEGRAM_PROXY_TYPE` (`http` / `https` / `socks5` / `socks4`),
`TELEGRAM_PROXY_HOST`, `TELEGRAM_PROXY_PORT`, `TELEGRAM_PROXY_USERNAME`,
`TELEGRAM_PROXY_PASSWORD`. Они используются, только если `PROXY_URL` пуст.

---

## Запуск

Все команды выполняются из корня проекта в активированном `.venv`.

| Команда | Назначение |
|---|---|
| `python scripts/test_embedding.py` | Проверяет, что embeddings работают, и выводит размерность вектора (ожидается 1536) |
| `python scripts/load_test_data.py` | Загружает 30 тестовых фраз в Pinecone (upsert по стабильным ID — повторный запуск не создаёт дубликаты) |
| `python scripts/semantic_search.py` | Выполняет 5 семантических запросов и выводит top-5 результатов (ID, score, текст) |
| `python -m app.bot` | Запускает Telegram-бота |

Остановить бота — `Ctrl+C`.

---

## Использование

**Сохранение записи:**

```
вы: /remember Я предпочитаю японские автомобили
бот: Сохранено. ID: tg-1730256381000-a1b2c3d4
```

**Обычный запрос:**

```
вы: Что мне подойдёт для города?
бот: Судя по сохранённым записям, вам подойдут японские автомобили —
     они надёжны и хорошо ведут себя в городском потоке.
```

Что происходит после сообщения пользователя:

1. бот получает текст;
2. текст превращается в embedding через GenAPI;
3. выполняется поиск top-5 похожих записей в Pinecone;
4. из найденных записей берётся текст (`metadata.text`) и формируется контекст;
5. контекст и вопрос передаются в LLM через GenAPI;
6. готовый ответ отправляется пользователю.

Команды `/start` и `/help` показывают справку.

---

## Тестовые данные

В файле `data/test_phrases.py` содержится список `TEST_PHRASES` — **30 фраз
об автомобилях**. Они используются для проверки семантического поиска: каждая
фраза векторизуется и записывается в Pinecone со стабильным ID
(`car-001` … `car-030`) и метаданными `text` и `source`.

---

## Проверка семантического поиска

Скрипт `scripts/semantic_search.py` выполняет **5 поисковых запросов**,
сформулированных иначе, чем исходные фразы, и для каждого выводит **top-5**
результатов с ID, score и текстом.

Пример реального вывода (запрос про мощный двигатель и спортивный характер):

```
Запрос: мощный звук двигателя и спортивный характер
1. ID=car-013  score=0.4906  Ford Mustang — классический американский маслкар с ревущим V8
2. ID=car-005  score=0.4754  BMW X5 — представительский кроссовер с мощным мотором и полным приводом
3. ID=car-012  score=0.4561  Mazda CX-5 — кроссовер с фирменным дизайном и аудиосистемой Bose
4. ID=car-018  score=0.4492  Пикап Mitsubishi L200 сочетает грузовой характер и проходимость
5. ID=car-009  score=0.4233  Volkswagen Golf — легендарный хэтчбек, эталон своего класса
```

Результаты показывают, что поиск работает по смыслу: на запрос о мощном
двигателе первыми выходят записи о Mustang и BMW X5.

---

## Безопасность

- `.env` находится в `.gitignore` и не попадает в репозиторий;
- `.venv/` находится в `.gitignore`;
- API-ключи не хранятся в коде — только в переменных окружения;
- Telegram-токен не хранится в коде;
- credentials proxy не хранятся в коде;
- секреты не выводятся в лог: в логах виден только `host:port` proxy без логина
  и пароля;
- сообщения об ошибках не раскрывают ключи и токены — технические детали пишутся
  только в лог, а пользователь получает общее сообщение.
