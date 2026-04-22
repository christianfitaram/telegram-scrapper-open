from telegram_intel_scraper.core import scrape
from telegram_intel_scraper.core.config import Settings
from telegram_intel_scraper.providers import ollama_provider
from telegram_intel_scraper.providers.title_llm import generate_title_ollama_with_fallback


def _settings() -> Settings:
    return Settings(
        telegram_api_id=1,
        telegram_api_hash="hash",
        telegram_session="session",
        out_jsonl="out.jsonl",
        state_file="state.json",
        enable_llm_titles=True,
        ollama_url="http://ollama.test/api/generate",
        ollama_model="llama3.1:8b",
        channels=["https://t.me/example"],
        include_empty_text=False,
        mongo_uri=None,
        mongo_db="agents",
        mongo_collection="articles",
        ai_provider="genai",
        title_provider="genai",
        genai_model="gemini-test",
        scrape_since=None,
        scrape_until=None,
        translate_to_en=False,
        enable_local_enrichment=False,
    )


def test_default_ollama_fallback_order(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_FALLBACK_MODELS", raising=False)

    assert ollama_provider.get_fallback_models() == [
        "gpt-oss:120b-cloud",
        "gemma4:31b-cloud",
        "gpt-oss:20b-cloud",
        "gemma2:9b",
    ]


def test_title_ollama_fallbacks_try_models_in_order(monkeypatch) -> None:
    calls = []

    def fake_generate(*, ollama_url: str, model: str, prompt: str, response_format=None, timeout: int = 60):
        calls.append(model)
        if model != "gemma4:31b-cloud":
            raise RuntimeError("quota exhausted")
        return "Fallback title"

    monkeypatch.delenv("OLLAMA_FALLBACK_MODELS", raising=False)
    monkeypatch.setattr(ollama_provider, "generate", fake_generate)

    title = generate_title_ollama_with_fallback("Some article text", "http://ollama.test/api/generate")

    assert title == "Fallback title"
    assert calls == ["gpt-oss:120b-cloud", "gemma4:31b-cloud"]


def test_genai_title_falls_back_to_ollama_sequence(monkeypatch) -> None:
    calls = []

    def fake_genai(text: str, model: str):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    def fake_generate(*, ollama_url: str, model: str, prompt: str, response_format=None, timeout: int = 60):
        calls.append(model)
        if model == "gpt-oss:120b-cloud":
            raise RuntimeError("quota exhausted")
        return "Ollama title"

    monkeypatch.delenv("OLLAMA_FALLBACK_MODELS", raising=False)
    monkeypatch.setattr(scrape, "generate_title_genai", fake_genai)
    monkeypatch.setattr(ollama_provider, "generate", fake_generate)

    assert scrape._resolve_title(_settings(), "Important market update") == "Ollama title"
    assert calls == ["gpt-oss:120b-cloud", "gemma4:31b-cloud"]
