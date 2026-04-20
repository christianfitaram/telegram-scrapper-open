from __future__ import annotations

from telegram_intel_scraper.providers import ollama_provider


def generate_title_ollama(text: str, ollama_url: str, ollama_model: str) -> str:
    return ollama_provider.generate_title(text, ollama_url, ollama_model)


def generate_title_ollama_with_fallback(
    text: str,
    ollama_url: str,
    primary_model: str | None = None,
    *,
    include_primary: bool = False,
) -> str:
    return ollama_provider.with_ollama_fallbacks(
        lambda model: generate_title_ollama(text, ollama_url, model),
        label="title",
        primary_model=primary_model,
        include_primary=include_primary,
    )
