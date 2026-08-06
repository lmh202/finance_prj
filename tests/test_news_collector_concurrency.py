"""The store is one file and the backend is one process serving many threads.

FastAPI runs sync endpoints on a threadpool, and the /react page asks for the
daily recommendation and the event list at once — so two collect() runs
overlap on data/news_raw.json routinely, not exceptionally. These cover what
that used to break.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.news_intelligence import collector  # noqa: E402
from src.news_intelligence import engine as news_engine  # noqa: E402


def _entry(uid: str):
    class _Entry(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    return _Entry(
        id=uid,
        link=f"https://example.test/{uid}",
        title=f"Story {uid}",
        summary="body",
        published_parsed=(2026, 7, 20, 12, 0, 0, 0, 201, 0),
    )


def _feed(name: str) -> dict:
    return {"name": name, "url": f"https://example.test/{name}.rss", "symbols": []}


def test_overlapping_collect_runs_serialise_and_keep_both_results(
    tmp_path, monkeypatch
):
    """Regression: two threads collecting at once raced on the store.

    On Windows the loser's os.replace() raised WinError 32 (PermissionError,
    which the recommendation router's fallback does not catch) and 500'd
    /recommendation/daily. Everywhere else the damage was quieter: the later
    writer had loaded the store before the earlier one saved, so it wrote a
    merged set missing the earlier run's new records.
    """
    store = tmp_path / "news_raw.json"
    guard = threading.Lock()
    in_flight = [0]
    peak = [0]

    def fake_fetch(feed):
        with guard:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.05)
        with guard:
            in_flight[0] -= 1
        return [_entry(feed["name"])], True

    monkeypatch.setattr(collector, "_fetch_feed", fake_fetch)

    errors: list[BaseException] = []

    def run(feed):
        try:
            collector.collect_feeds([feed], store_path=store)
        except BaseException as exc:                            # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(_feed("alpha"),)),
        threading.Thread(target=run, args=(_feed("beta"),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent collect raised {errors}"
    assert peak[0] == 1, "two collect runs were inside the store critical section"
    titles = {r["title"] for r in collector.load_store(store).values()}
    assert titles == {"Story alpha", "Story beta"}, "a run's records were lost"


def test_each_writer_gets_its_own_temp_file(tmp_path):
    """A single fixed `.tmp` was itself shared state: writers overwrote each
    other's bytes, then replaced the target out from under an open handle."""
    store = tmp_path / "news_raw.json"
    names: set[str] = set()

    def record():
        names.add(collector._tmp_path(store).name)

    threads = [threading.Thread(target=record) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(names) == 3
    assert all(n.endswith(".tmp") and n.startswith("news_raw.json.") for n in names)


def test_replace_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    """WinError 32 while an indexer/antivirus/second process holds the target
    open is a millisecond window, not a permanent failure."""
    monkeypatch.setattr(collector, "STORE_REPLACE_BACKOFF_SECONDS", 0)
    attempts = [0]
    real_replace = Path.replace

    def flaky_replace(self, target):
        attempts[0] += 1
        if attempts[0] < 3:
            raise PermissionError(32, "another process is using this file")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    store = tmp_path / "news_raw.json"
    collector.save_store({"a": {"id": "a", "published_utc": None}}, store)

    assert attempts[0] == 3
    assert collector.load_store(store) == {"a": {"id": "a", "published_utc": None}}


def test_a_store_write_that_keeps_failing_does_not_raise_or_mark_feeds_fresh(
    tmp_path, monkeypatch
):
    """Losing the write costs freshness. Throttling the feed afterwards would
    cost the articles too — the retry that recovers them gets skipped."""
    store = tmp_path / "news_raw.json"
    calls: list[str] = []

    def fake_fetch(feed):
        calls.append(feed["url"])
        return [_entry("a")], True

    monkeypatch.setattr(collector, "_fetch_feed", fake_fetch)
    monkeypatch.setattr(
        collector,
        "save_store",
        lambda records, path: (_ for _ in ()).throw(PermissionError(32, "locked")),
    )

    stats = collector.collect_feeds([_feed("alpha")], store_path=store)
    assert stats["store_write_failed"] == 1
    assert stats["feeds_ok"] == 1

    again = collector.collect_feeds([_feed("alpha")], store_path=store)
    assert again["feeds_skipped_fresh"] == 0
    assert len(calls) == 2


def test_a_corrupt_store_reads_as_empty_rather_than_raising(tmp_path):
    store = tmp_path / "news_raw.json"
    store.write_text("[{ truncated", encoding="utf-8")
    assert collector.load_store(store) == {}


def test_essential_news_survives_a_failed_collect(monkeypatch):
    """The collector refreshes the store; it is not the data source. A failed
    refresh must degrade to whatever the store already holds."""
    def boom(_symbols):
        raise PermissionError(32, "another process is using this file")

    monkeypatch.setattr(collector, "collect", boom)
    monkeypatch.setattr(news_engine, "_recent_records", lambda hours, now: [])

    assert news_engine.essential_news(["AAPL"], max_events=5) == []


def test_a_locked_out_run_skips_collection_instead_of_blocking(
    tmp_path, monkeypatch
):
    """A waiter that can't get the lock in time serves the store as-is: the
    holder is fetching the same feeds anyway."""
    monkeypatch.setattr(collector, "COLLECT_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        collector, "_fetch_feed", lambda feed: ([_entry("a")], True)
    )
    store = tmp_path / "news_raw.json"

    collector._STORE_LOCK.acquire()
    try:
        stats = collector.collect_feeds([_feed("alpha")], store_path=store)
    finally:
        collector._STORE_LOCK.release()

    assert stats["feeds_skipped_locked"] == 1
    assert stats["feeds_ok"] == 0
