from __future__ import annotations

from typing import Any, Dict

from telegram_intel_scraper.providers.call_to_webhook import send_to_all_webhooks
from telegram_intel_scraper.providers.sentiment import get_sentiment
from telegram_intel_scraper.providers.topic_classifier import get_topic
from telethon import TelegramClient

from telegram_intel_scraper.core.config import Settings
from telegram_intel_scraper.core.logging import get_logger
from telegram_intel_scraper.core.mongo import get_articles_collection
from telegram_intel_scraper.core.state import load_state, save_state
from telegram_intel_scraper.core.writer import write_jsonl
from telegram_intel_scraper.repositories.articles_repository import ArticlesRepository

from telegram_intel_scraper.providers.telegram import parse_username, iter_channel_messages
from telegram_intel_scraper.providers.text_translate_genai import detect_translate_and_title
from telegram_intel_scraper.providers.text_translate_ollama import detect_translate_and_title_ollama_with_fallback
from telegram_intel_scraper.providers.title_genai import generate_title_genai
from telegram_intel_scraper.providers.title_llm import generate_title_ollama_with_fallback
from telegram_intel_scraper.utils.text import normalize_whitespace, title_heuristic
from telethon.errors import UsernameInvalidError, UsernameNotOccupiedError

logger = get_logger(__name__)


def _resolve_ai_provider(settings: Settings) -> str:
    provider = (getattr(settings, "ai_provider", "") or "").strip().lower()
    if provider:
        return provider

    provider = (getattr(settings, "title_provider", "") or "").strip().lower()
    if provider:
        return provider

    return "ollama" if getattr(settings, "enable_llm_titles", False) else "heuristic"


def _resolve_title(settings: Settings, text: str) -> str:
    """
    Single source of truth for how titles are generated.
    Priority:
      1) settings.ai_provider if set to 'genai' | 'ollama' | 'heuristic'
      2) legacy fallback: settings.title_provider
      3) legacy fallback: enable_llm_titles => use 'ollama'
      4) default: heuristic
    """
    provider = _resolve_ai_provider(settings)

    if not text:
        return "Telegram message"

    if provider == "genai":
        try:
            return generate_title_genai(text, model=getattr(settings, "genai_model", "gemini-2.0-flash"))
        except Exception as exc:
            logger.warning("genai title generation failed; trying ollama fallbacks: %s", exc)
            try:
                return generate_title_ollama_with_fallback(text, settings.ollama_url)
            except Exception as fallback_exc:
                logger.warning("ollama title fallbacks failed; using heuristic: %s", fallback_exc)
                return title_heuristic(text)

    if provider == "ollama":
        try:
            return generate_title_ollama_with_fallback(
                text,
                settings.ollama_url,
                primary_model=settings.ollama_model,
                include_primary=True,
            )
        except Exception as exc:
            logger.warning("ollama title generation failed; using heuristic: %s", exc)
            return title_heuristic(text)

    # heuristic (default)
    return title_heuristic(text)


def _translate_and_title(settings: Settings, text: str) -> tuple[str, str, str]:
    provider = _resolve_ai_provider(settings)
    original_text = normalize_whitespace(text)

    if not original_text:
        return "unknown", "", "Short Message"

    if len(original_text) < 5:
        return "unknown", original_text, "Short Message"

    if not settings.translate_to_en:
        return "unknown", original_text, _resolve_title(settings, original_text)

    if provider == "genai":
        try:
            result = detect_translate_and_title(
                original_text,
                model=getattr(settings, "genai_model", "gemini-2.0-flash"),
            )
            return result.language, result.english_text, result.title
        except Exception as exc:
            logger.warning("genai translation failed; trying ollama fallbacks: %s", exc)
            try:
                result = detect_translate_and_title_ollama_with_fallback(original_text, settings.ollama_url)
                return result.language, result.english_text, result.title
            except Exception as fallback_exc:
                logger.warning("ollama translation fallbacks failed; using original text: %s", fallback_exc)
                return "unknown", original_text, title_heuristic(original_text)

    if provider == "ollama":
        try:
            result = detect_translate_and_title_ollama_with_fallback(
                original_text,
                settings.ollama_url,
                primary_model=settings.ollama_model,
                include_primary=True,
            )
            return result.language, result.english_text, result.title
        except Exception as exc:
            logger.warning("ollama translation failed; using original text: %s", exc)
            return "unknown", original_text, title_heuristic(original_text)

    return "unknown", original_text, _resolve_title(settings, original_text)


async def run_scrape(settings: Settings) -> None:
    state = load_state(settings.state_file)

    repo: ArticlesRepository | None = None
    if settings.mongo_uri:
        collection = get_articles_collection(
            settings.mongo_uri,
            settings.mongo_db,
            settings.mongo_collection,
        )
        repo = ArticlesRepository(collection)

    async with TelegramClient(
        settings.telegram_session,
        settings.telegram_api_id,
        settings.telegram_api_hash,
    ) as client:
        for url in settings.channels:
            username = parse_username(url)
            last_id = int(state.get(username, {}).get("last_id", 0))
            logger.info("[%s] resume after last_id=%s", username, last_id)
            try:
                async for msg in iter_channel_messages(
                    client,
                    username=username,
                    min_id_exclusive=last_id,
                    since=settings.scrape_since,
                    until=settings.scrape_until,
                ):
                    logger.debug("[%s] candidate message id=%s date=%s", username, msg.id, msg.date)
                    raw_text = (msg.message or "").strip()

                    if not raw_text and not settings.include_empty_text:
                        # Skip purely media posts without captions, etc.
                        continue

                    original_text = normalize_whitespace(raw_text)
                    language, text_en, title = _translate_and_title(settings, original_text)

                    record: Dict[str, Any] = {
                        "title": title,
                        "url": url,
                        "text": text_en,  # canonical text = English
                        "source": username,
                        "scraped_at": msg.date,
                    }

                    if repo is not None:
                        sentiment_result_to_insert = {"label": None, "score": None}
                        categorization_result_to_insert = None

                        if settings.enable_local_enrichment:
                            sentiment_result = get_sentiment(text_en)
                            sentiment_result_to_insert = {
                                "label": sentiment_result.label if sentiment_result else None,
                                "score": sentiment_result.score if sentiment_result else None,
                            }
                            categorization_result = None
                            try:
                                categorization_result = get_topic(text_en)
                            except Exception as e:
                                logger.warning("[%s] topic classification failed: %s", username, e)
                            categorization_result_to_insert = (
                                categorization_result.top_label if categorization_result else None
                            )
                        else:
                            logger.debug("[%s] local enrichment skipped", username)

                        logger.debug(
                            "[%s] sentiment=%s topic=%s",
                            username,
                            sentiment_result_to_insert,
                            categorization_result_to_insert,
                        )
                        article_doc = {
                            **record,
                            "text_original": original_text,
                            "text_en": text_en,
                            "language": language,
                            "external_id": msg.id,
                            "telegram_date": msg.date,
                            "telegram_channel": username,
                            "telegram_url": f"https://t.me/{username}/{msg.id}",
                            "main_source": "telegram",
                            "sentiment": sentiment_result_to_insert,
                            "topic": categorization_result_to_insert,
                        }
                        inserted_id = repo.upsert_article(article_doc)
                        if inserted_id:
                            logger.info("[%s] stored message id=%s", username, msg.id)
                            send_to_all_webhooks({"article_id": inserted_id, **article_doc})
                        else:
                            logger.info("[%s] skipped duplicate message id=%s", username, msg.id)
                    else:
                        # Optional JSONL fallback / audit log
                        write_jsonl(settings.out_jsonl, record)

                    # checkpoint
                    state[username] = {"last_id": msg.id}
                    save_state(settings.state_file, state)
            except (UsernameInvalidError, UsernameNotOccupiedError) as e:
                logger.warning("[%s] skipped invalid/unknown username: %s", username, e.__class__.__name__)
                continue

            logger.info("[%s] done", username)
