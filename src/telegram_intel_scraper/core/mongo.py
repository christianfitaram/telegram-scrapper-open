from __future__ import annotations

from typing import Any


def get_articles_collection(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
) -> Any:
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("MongoDB support requires the 'mongo' extra: poetry install --extras mongo") from exc

    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]
