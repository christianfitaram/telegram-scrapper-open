import pytest

from telegram_intel_scraper.core.config import Settings


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_API_ID", "1")
    monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
    monkeypatch.setenv("TELEGRAM_CHANNELS", "https://t.me/one,https://t.me/two")


def test_settings_require_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_API_ID", "1")
    monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
    monkeypatch.delenv("TELEGRAM_CHANNELS", raising=False)

    with pytest.raises(RuntimeError, match="Missing TELEGRAM_CHANNELS"):
        Settings.from_env()


def test_ai_provider_prefers_explicit_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "genai")
    monkeypatch.setenv("TITLE_PROVIDER", "ollama")
    monkeypatch.setenv("ENABLE_LLM_TITLES", "1")

    settings = Settings.from_env()

    assert settings.ai_provider == "genai"
    assert settings.title_provider == "ollama"
    assert settings.channels == ["https://t.me/one", "https://t.me/two"]


def test_ai_provider_falls_back_to_legacy_title_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _base_env(monkeypatch)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.setenv("TITLE_PROVIDER", "ollama")

    settings = Settings.from_env()

    assert settings.ai_provider == "ollama"


def test_local_enrichment_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _base_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.enable_local_enrichment is False

    monkeypatch.setenv("ENABLE_LOCAL_ENRICHMENT", "1")

    settings = Settings.from_env()

    assert settings.enable_local_enrichment is True
