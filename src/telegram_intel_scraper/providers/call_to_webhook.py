from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import date, datetime
from typing import Any, Dict, Iterable, Mapping, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from telegram_intel_scraper.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "30"))


def _build_session(total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = _build_session()


def _split_urls(value: str) -> list[str]:
    raw = [url.strip() for url in value.replace(",", "\n").splitlines()]
    return [url for url in raw if url]


def _configured_webhook_urls() -> list[str]:
    urls = _split_urls(os.getenv("WEBHOOK_URLS", ""))
    legacy_url = os.getenv("WEBHOOK_URL", "").strip()
    if legacy_url and legacy_url not in urls:
        urls.append(legacy_url)
    return urls


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _encode_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _build_headers(raw_body: bytes) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("WEBHOOK_SIGNATURE", "").strip()
    if secret:
        signature = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
        headers["X-Signature"] = f"sha256={signature}"
    return headers


def _validate_payload(payload: Mapping[str, Any], required_fields: Iterable[str]) -> Optional[str]:
    missing = [field for field in required_fields if payload.get(field) in (None, "")]
    if missing:
        return f"payload missing required fields: {', '.join(missing)}"
    return None


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    raw_body = _encode_payload(payload)
    headers = _build_headers(raw_body)

    try:
        response = SESSION.post(url, data=raw_body, headers=headers, timeout=timeout)
        response.raise_for_status()
        logger.info("webhook delivered url=%s status=%s", url, response.status_code)
        try:
            return response.json()
        except ValueError:
            return None
    except requests.RequestException as exc:
        logger.warning("webhook delivery failed url=%s error=%s", url, exc)
        return None


def send_to_all_webhooks(
    article_payload: Mapping[str, Any],
    webhook_urls: Iterable[str] | str | None = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    urls = (
        _split_urls(webhook_urls) if isinstance(webhook_urls, str)
        else list(webhook_urls) if webhook_urls is not None
        else _configured_webhook_urls()
    )
    if not urls:
        logger.debug("webhook delivery skipped: no WEBHOOK_URLS configured")
        return {}

    payload: Dict[str, Any] = dict(article_payload)
    validation_error = _validate_payload(payload, ["article_id", "url", "title", "text", "source", "scraped_at"])
    if validation_error:
        logger.warning("webhook delivery skipped: %s", validation_error)
        return {}

    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for url in urls:
        results[url] = _post_json(url, payload)
    return results
