from __future__ import annotations

import json
import os
from typing import Callable, List, TypeVar

import requests

from telegram_intel_scraper.core.logging import get_logger
from telegram_intel_scraper.utils.text import normalize_whitespace

logger = get_logger(__name__)

DEFAULT_FALLBACK_MODELS = ("gpt-oss:120b-cloud", "gemma4:31b-cloud", "gpt-oss:20b-cloud")
T = TypeVar("T")


def get_fallback_models(primary_model: str | None = None, *, include_primary: bool = False) -> List[str]:
    configured = os.getenv("OLLAMA_FALLBACK_MODELS", "").strip()
    models = (
        [model.strip() for model in configured.replace("\n", ",").split(",") if model.strip()]
        if configured
        else list(DEFAULT_FALLBACK_MODELS)
    )
    if include_primary and primary_model and primary_model not in models:
        models.insert(0, primary_model)
    return models


def with_ollama_fallbacks(
    operation: Callable[[str], T],
    *,
    label: str,
    primary_model: str | None = None,
    include_primary: bool = False,
) -> T:
    last_exc: Exception | None = None
    for model in get_fallback_models(primary_model, include_primary=include_primary):
        try:
            return operation(model)
        except Exception as exc:
            last_exc = exc
            logger.warning("%s failed with ollama model=%s: %s", label, model, exc)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No Ollama fallback models configured")


def generate(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    response_format: str | None = None,
    timeout: int = 60,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if response_format:
        payload["format"] = response_format

    response = requests.post(ollama_url, json=payload, timeout=timeout)
    response.raise_for_status()
    text = (response.json().get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def parse_json_payload(payload: str) -> dict:
    cleaned = payload.strip()
    if not cleaned:
        raise ValueError("Empty Ollama response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def generate_title(text: str, ollama_url: str, model: str, timeout: int = 60) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return "Telegram message"

    prompt = (
        "Generate a short news-style title (max 12 words) for this Telegram message. "
        "Return ONLY the title.\n\n"
        f"Message:\n{cleaned[:2500]}"
    )
    title = generate(ollama_url=ollama_url, model=model, prompt=prompt, timeout=timeout).strip()
    return title[:120] if title else "Telegram message"


def detect_translate_and_title(text: str, ollama_url: str, model: str, timeout: int = 60) -> dict:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return {"language": "unknown", "english_text": "", "title": "Empty Content"}

    prompt = (
        "You detect the source language, translate text to English, and create a concise title.\n"
        "Return ONLY valid JSON with this shape:\n"
        '{"language":"string","english_text":"string","title":"string"}\n'
        "If the input is already English, preserve the meaning in english_text.\n"
        "Keep the title under 12 words.\n\n"
        f"Text:\n{cleaned[:12000]}"
    )
    payload = generate(
        ollama_url=ollama_url,
        model=model,
        prompt=prompt,
        response_format="json",
        timeout=timeout,
    )
    data = parse_json_payload(payload)
    return {
        "language": str(data.get("language", "unknown")).strip() or "unknown",
        "english_text": normalize_whitespace(str(data.get("english_text", cleaned)).strip()) or cleaned,
        "title": normalize_whitespace(str(data.get("title", "Telegram message")).strip())[:120]
        or "Telegram message",
    }
