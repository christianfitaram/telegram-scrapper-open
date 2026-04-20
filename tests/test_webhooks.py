from datetime import datetime, timezone

from telegram_intel_scraper.providers import call_to_webhook


def test_webhooks_skip_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("WEBHOOK_URLS", raising=False)
    monkeypatch.delenv("WEBHOOK_URL", raising=False)

    assert call_to_webhook.send_to_all_webhooks({"article_id": "1"}) == {}


def test_webhooks_post_payload_without_logging_secrets(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class FakeSession:
        def post(self, url, data, headers, timeout):
            calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
            return FakeResponse()

    monkeypatch.setenv("WEBHOOK_URLS", "https://example.test/webhook")
    monkeypatch.setenv("WEBHOOK_SIGNATURE", "secret")
    monkeypatch.setattr(call_to_webhook, "SESSION", FakeSession())

    payload = {
        "article_id": "abc123",
        "url": "https://t.me/example/1",
        "title": "Example",
        "text": "Body",
        "source": "example",
        "scraped_at": datetime(2026, 4, 20, tzinfo=timezone.utc),
    }

    result = call_to_webhook.send_to_all_webhooks(payload)

    assert result == {"https://example.test/webhook": {"ok": True}}
    assert calls[0]["url"] == "https://example.test/webhook"
    assert calls[0]["headers"]["Content-Type"] == "application/json"
    assert calls[0]["headers"]["X-Signature"].startswith("sha256=")
    assert b"2026-04-20T00:00:00+00:00" in calls[0]["data"]


def test_webhooks_skip_invalid_payload(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URLS", "https://example.test/webhook")

    assert call_to_webhook.send_to_all_webhooks({"article_id": "abc123"}) == {}
