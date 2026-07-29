from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.news_intelligence import collector  # noqa: E402


FEED = {"name": "Test feed", "url": "https://example.test/rss", "symbols": []}


class _Entry(dict):
    """feedparser entries are attribute-accessed dicts."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _entry(uid: str) -> _Entry:
    return _Entry(
        id=uid,
        link=f"https://example.test/{uid}",
        title=f"Story {uid}",
        summary="body",
        published_parsed=(2026, 7, 20, 12, 0, 0, 0, 201, 0),
    )


def test_a_feed_returning_only_known_articles_is_still_throttled(
    tmp_path, monkeypatch
):
    """Regression: the throttle used to be derived from stored records, so a
    healthy feed whose articles were all already known wrote nothing, its
    timestamp never advanced, and it was re-fetched on every request. That
    cost ~3s of latency on every page load."""
    store = tmp_path / "news_raw.json"
    calls: list[str] = []

    def fake_fetch(feed):
        calls.append(feed["url"])
        return [_entry("a"), _entry("b")], True

    monkeypatch.setattr(collector, "_fetch_feed", fake_fetch)

    first = collector.collect_feeds([FEED], store_path=store)
    assert first["feeds_ok"] == 1
    assert first["new"] == 2
    assert len(calls) == 1

    # Second run: the same two articles come back, so nothing is written.
    second = collector.collect_feeds([FEED], store_path=store)
    assert second["new"] == 0
    assert second["feeds_skipped_fresh"] == 1
    assert second["feeds_ok"] == 0
    assert len(calls) == 1, "feed was re-fetched despite being fresh"


def _age_everything(store: Path, minutes: int) -> None:
    """Push both the sidecar and the stored records into the past.

    Both must move: the sidecar is the authority, but records are still read
    for backwards compatibility with stores written before it existed, and
    the newer of the two wins.
    """
    stale = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).isoformat()
    collector.save_feed_state({FEED["url"]: stale}, store_path=store)
    records = collector.load_store(store)
    for record in records.values():
        record["fetched_utc"] = stale
    collector.save_store(records, store)


def test_the_throttle_expires(tmp_path, monkeypatch):
    store = tmp_path / "news_raw.json"
    calls: list[str] = []

    def fake_fetch(feed):
        calls.append(feed["url"])
        return [_entry("a")], True

    monkeypatch.setattr(collector, "_fetch_feed", fake_fetch)

    collector.collect_feeds([FEED], store_path=store)
    _age_everything(store, collector.FEED_STALE_MINUTES + 1)

    again = collector.collect_feeds([FEED], store_path=store)
    assert again["feeds_skipped_fresh"] == 0
    assert len(calls) == 2


def test_a_corrupt_sidecar_degrades_to_the_record_timestamps(
    tmp_path, monkeypatch
):
    """Losing the sidecar must cost latency at worst, never raise. The
    record-derived timestamps remain as the backwards-compatible fallback."""
    store = tmp_path / "news_raw.json"
    monkeypatch.setattr(
        collector, "_fetch_feed", lambda feed: ([_entry("a")], True)
    )
    collector.collect_feeds([FEED], store_path=store)
    collector._feed_state_path(store).write_text("{ not json", encoding="utf-8")

    assert collector.load_feed_state(store) == {}
    # Records are still fresh, so the fallback correctly keeps the throttle on.
    fresh = collector.collect_feeds([FEED], store_path=store)
    assert fresh["feeds_skipped_fresh"] == 1

    # Once the records age out too, it refetches rather than staying stuck.
    _age_everything(store, collector.FEED_STALE_MINUTES + 1)
    collector._feed_state_path(store).write_text("{ not json", encoding="utf-8")
    assert collector.collect_feeds([FEED], store_path=store)["feeds_ok"] == 1


def test_a_failed_feed_is_not_marked_fresh(tmp_path, monkeypatch):
    """A feed that errored was not successfully fetched — retry it."""
    store = tmp_path / "news_raw.json"
    monkeypatch.setattr(collector, "_fetch_feed", lambda feed: (None, False))

    first = collector.collect_feeds([FEED], store_path=store)
    assert first["feeds_failed"] == 1
    second = collector.collect_feeds([FEED], store_path=store)
    assert second["feeds_skipped_fresh"] == 0
    assert second["feeds_failed"] == 1
