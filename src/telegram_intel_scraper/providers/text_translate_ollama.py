from __future__ import annotations

from typing import NamedTuple

from telegram_intel_scraper.providers import ollama_provider
from telegram_intel_scraper.utils.text import normalize_whitespace


class ScraperResult(NamedTuple):
    language: str
    english_text: str
    title: str


def detect_translate_and_title_ollama(
    text: str,
    ollama_url: str,
    ollama_model: str,
    timeout: int = 60,
) -> ScraperResult:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ScraperResult("unknown", "", "Empty Content")

    data = ollama_provider.detect_translate_and_title(
        cleaned,
        ollama_url,
        ollama_model,
        timeout=timeout,
    )
    return ScraperResult(
        language=data["language"],
        english_text=data["english_text"],
        title=data["title"],
    )


def detect_translate_and_title_ollama_with_fallback(
    text: str,
    ollama_url: str,
    primary_model: str | None = None,
    *,
    include_primary: bool = False,
    timeout: int = 60,
) -> ScraperResult:
    return ollama_provider.with_ollama_fallbacks(
        lambda model: detect_translate_and_title_ollama(text, ollama_url, model, timeout=timeout),
        label="translate/title",
        primary_model=primary_model,
        include_primary=include_primary,
    )
