from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional

from telegram_intel_scraper.core.logging import get_logger

logger = get_logger(__name__)
ASCENDING = 1

try:
    from pymongo.collection import Collection
    from pymongo.errors import DuplicateKeyError
except ImportError:
    Collection = Any  # type: ignore[misc, assignment]

    class DuplicateKeyError(Exception):
        pass


class ArticlesRepository:
    def __init__(self, collection: Collection):
        self._collection = collection
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # Deduplication index: same telegram channel + telegram message id
        self._collection.create_index(
            [("telegram_channel", ASCENDING), ("external_id", ASCENDING)],
            unique=True,
            name="uniq_channel_external_id",
        )

        # Legacy index on source + external_id (still useful if source diverges from telegram_channel)
        self._collection.create_index(
            [("source", ASCENDING), ("external_id", ASCENDING)],
            unique=True,
            name="uniq_source_external_id",
        )

        self._collection.create_index("scraped_at", name="scraped_at_idx")

    def upsert_article(self, doc: Dict[str, Any]) -> Optional[str]:
        """
        Insert the document if the telegram_channel + external_id combo is new.
        Returns the new `_id` as a string when inserted, or None when skipped.
        """
        now = datetime.utcnow()
        insert_doc = {
            **doc,
            "created_at": now,
            "updated_at": now,
        }

        try:
            result = self._collection.insert_one(insert_doc)
            inserted_id = result.inserted_id
            return str(inserted_id) if inserted_id is not None else None
        except DuplicateKeyError:
            logger.debug(
                "duplicate article skipped source=%s external_id=%s",
                doc.get("source"),
                doc.get("external_id"),
            )
            return None
