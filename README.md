# telegram-scraper

`telegram-scraper` is a Telegram channel ingestion tool for developers who want a reusable pipeline rather than a fixed data source list. It scrapes user-supplied Telegram channels, normalizes messages, can translate and title content with pluggable AI providers, stores records in MongoDB, and can forward inserted articles to downstream webhooks.

## Features

- Scrape one or more Telegram channels with checkpointed resume state.
- Filter by `--since` and `--until` using ISO timestamps or durations such as `24h` and `7d`.
- Generate titles with `heuristic`, `ollama`, or `genai`.
- Optionally translate content to English before enrichment.
- Store normalized records in MongoDB or write JSONL as a fallback.
- Run sentiment and topic enrichment with Hugging Face models.
- Forward inserted records to optional downstream webhooks.

## Requirements

- Python `>=3.10,<3.15`
- A Telegram API ID and hash from https://my.telegram.org
- Poetry for dependency management

## Installation

```bash
poetry install
cp .env.example .env
```

Fill in `.env` with your own values before running the scraper.

## Configuration

Required:

- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `TELEGRAM_CHANNELS`

`TELEGRAM_CHANNELS` accepts comma-separated or newline-separated Telegram URLs. The project does not provide default channels.

Common optional settings:

- `AI_PROVIDER=heuristic|ollama|genai`
- `TRANSLATE_TO_EN=0|1`
- `MONGO_URI`
- `OUT_JSONL`
- `SCRAPE_SINCE`
- `SCRAPE_UNTIL`

Provider-specific settings:

- Ollama:
  - `AI_PROVIDER=ollama`
  - `OLLAMA_URL`
  - `OLLAMA_MODEL`
- Google GenAI:
  - `AI_PROVIDER=genai`
  - `GOOGLE_API_KEY`
  - `GENAI_MODEL`

Optional integrations:

- MongoDB persistence uses `MONGO_URI`, `MONGO_DB`, and `MONGO_COLLECTION`.
- Downstream webhook delivery uses `NEWSAPI_KEY`, `NEWS_API_BASE_URL`, `WEBHOOK_URL`, and `WEBHOOK_URL_THREAD_EVENTS`.
- Sentiment and topic classifiers can use `TRANSFORMERS_CACHE` to control the local Hugging Face cache path.

If webhook or external article lookup settings are unset, those integrations are skipped.

## Usage

Run with the values from `.env`:

```bash
poetry run telegram-scrape
```

Run with a time window override:

```bash
poetry run telegram-scrape --since 24h --until 2h
```

Run with explicit UTC timestamps:

```bash
poetry run telegram-scrape --since 2026-04-01T00:00:00Z --until 2026-04-02T00:00:00Z
```

## Output

Each processed message is normalized into a record shaped roughly like:

```json
{
  "title": "Example title",
  "url": "https://t.me/example_channel",
  "text": "English or normalized message text",
  "source": "example_channel",
  "scraped_at": "2026-04-10T12:00:00+00:00"
}
```

When MongoDB is configured, stored records can also include Telegram metadata, original text, language, sentiment, and topic fields.

## Development

Run tests:

```bash
poetry run pytest
```

Run a syntax check:

```bash
python3 -m py_compile $(rg --files src tests)
```

## Project Standards

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

## Security Notes

- Do not commit `.env` or Telegram session files.
- Treat API keys, Mongo URIs, and webhook secrets as credentials.
- Assume scraped content may contain sensitive or regulated data depending on the channels you configure.

## License

MIT. See `LICENSE`.
