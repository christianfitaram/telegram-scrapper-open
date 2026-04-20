import json

from telegram_intel_scraper.core.state import load_state, save_state
from telegram_intel_scraper.core.writer import write_jsonl


def test_state_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    expected = {"channel": {"last_id": 123}}

    save_state(str(state_path), expected)

    assert load_state(str(state_path)) == expected
    assert not (tmp_path / "state.json.tmp").exists()


def test_load_state_missing_file_returns_empty_dict(tmp_path) -> None:
    assert load_state(str(tmp_path / "missing.json")) == {}


def test_write_jsonl_appends_records(tmp_path) -> None:
    out_path = tmp_path / "out.jsonl"

    write_jsonl(str(out_path), {"id": 1, "text": "first"})
    write_jsonl(str(out_path), {"id": 2, "text": "second"})

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]
