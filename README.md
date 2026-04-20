# telegram-scraper

[![CI](https://github.com/christianfitaram/telegram-scraper-open/actions/workflows/ci.yml/badge.svg)](https://github.com/christianfitaram/telegram-scraper-open/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`telegram-scraper` is a Telegram channel ingestion tool for developers who want a reusable pipeline rather than a fixed data source list. It scrapes user-supplied Telegram channels, normalizes messages, can translate and title content with pluggable AI providers, stores records in MongoDB, and can forward inserted articles to downstream webhooks.

## Why This Exists

This repository is the public, reusable version of a production ingestion pipeline. It keeps the operational patterns that matter for reliability, including checkpointing, deduplication, optional integrations, and safe fallbacks, while leaving out proprietary data sources, business rules, and deployment details.

## Features

- Scrape one or more Telegram channels with checkpointed resume state.
- Filter by `--since` and `--until` using ISO timestamps or durations such as `24h` and `7d`.
- Generate titles with `heuristic`, `ollama`, or `genai`.
- Optionally translate content to English before enrichment.
- Store normalized records in MongoDB or write JSONL as a fallback.
- Optionally run sentiment and topic enrichment with Hugging Face models.
- Forward inserted records to optional downstream webhooks.

## Requirements

- Python `>=3.10,<3.15`
- A Telegram API ID and hash from https://my.telegram.org
- Poetry for dependency management

## Installation

Minimal install:

```bash
poetry install
cp .env.example .env
```

Fill in `.env` with your own values before running the scraper.

Install optional integrations as needed:

```bash
poetry install --extras mongo
poetry install --extras genai
poetry install --extras local-ai
poetry install --all-extras
```

Use `local-ai` only when enabling local sentiment/topic enrichment.

## Configuration

Required:

- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `TELEGRAM_CHANNELS`

`TELEGRAM_CHANNELS` accepts comma-separated or newline-separated Telegram URLs. The project does not provide default channels.

Common optional settings:

- `AI_PROVIDER=heuristic|ollama|genai`
- `TRANSLATE_TO_EN=0|1`
- `ENABLE_LOCAL_ENRICHMENT=0|1`
- `LOG_LEVEL=INFO|DEBUG|WARNING`
- `MONGO_URI`
- `OUT_JSONL`
- `SCRAPE_SINCE`
- `SCRAPE_UNTIL`

Provider-specific settings:

- Ollama:
  - `AI_PROVIDER=ollama`
  - `OLLAMA_URL`
  - `OLLAMA_MODEL`
  - `OLLAMA_FALLBACK_MODELS`
- Google GenAI:
  - `AI_PROVIDER=genai`
  - `GOOGLE_API_KEY`
  - `GENAI_MODEL`

Optional integrations:

- MongoDB persistence uses `MONGO_URI`, `MONGO_DB`, and `MONGO_COLLECTION`.
- Downstream webhook delivery uses `WEBHOOK_URLS` or the legacy single `WEBHOOK_URL`.
- Webhook payload signing uses `WEBHOOK_SIGNATURE` when set.
- Local sentiment/topic enrichment requires `ENABLE_LOCAL_ENRICHMENT=1` and the `local-ai` extra.
- Sentiment and topic classifiers can use `TRANSFORMERS_CACHE` to control the local Hugging Face cache path.

If webhook settings are unset, delivery is skipped.

## Architecture

The pipeline is intentionally small and explicit:

1. `providers.telegram` paginates Telegram messages and yields them oldest-to-newest.
2. `core.scrape` normalizes text, applies optional translation/title generation, and can enrich MongoDB records.
3. `repositories.articles_repository` persists records with unique indexes for channel/message deduplication.
4. `core.state` checkpoints the last processed message ID per channel.
5. `providers.call_to_webhook` optionally posts inserted article payloads to configured downstream webhooks.

## Reliability Model

- Resume state is written after each processed message.
- MongoDB writes use unique indexes for `telegram_channel + external_id`.
- JSONL output is available when MongoDB is not configured.
- Optional integrations are skipped when their configuration is absent.
- Local sentiment/topic enrichment is disabled by default so MongoDB users do not need heavy ML dependencies unless they opt in.
- Webhook delivery uses retry/backoff for transient HTTP failures.
- GenAI failures fall back to configured Ollama models for title and translation tasks before using deterministic heuristics.

## Provider Matrix

| Capability | Default | Optional Providers |
| --- | --- | --- |
| Telegram ingestion | Telethon | - |
| Title generation | Heuristic | Ollama, Google GenAI |
| Translation | Disabled | Ollama, Google GenAI |
| Persistence | JSONL | MongoDB |
| Sentiment/topic enrichment | Disabled | Local Hugging Face models via `ENABLE_LOCAL_ENRICHMENT=1` and `local-ai` |
| Webhook delivery | Disabled | Any HTTP endpoint accepting JSON |

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

When MongoDB is configured, stored records can also include Telegram metadata, original text, and language fields. Sentiment and topic fields are populated when local enrichment is enabled.

## Development

Run tests:

```bash
poetry run pytest
```

Run lint and syntax checks:

```bash
poetry run ruff check .
python -m compileall src tests
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
