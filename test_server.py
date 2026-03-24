"""Tests for Shared Log."""

import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from server import LogDB, RateLimiter, create_server


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as d:
        yield LogDB(f"{d}/test.db")


# ── Storage ──────────────────────────────────────────────────


class TestLog:
    def test_add(self, db):
        eid = db.add(task="forecast demand", approach="transformer", outcome="CRPS 0.42")
        assert eid == 1

    def test_add_with_all_fields(self, db):
        eid = db.add(
            task="forecast demand",
            approach="Mamba SSM blocks",
            outcome="CRPS improved 12%",
            agent="agent-alpha",
            score=0.88,
            built_on=None,
        )
        entry = db.get(eid)
        assert entry["agent"] == "agent-alpha"
        assert entry["score"] == 0.88

    def test_add_with_lineage(self, db):
        parent = db.add(task="t", approach="baseline", outcome="ok")
        child = db.add(task="t", approach="improved", outcome="better", built_on=parent)
        entry = db.get(child)
        assert entry["built_on"] == parent

    def test_get_nonexistent(self, db):
        assert db.get(999) is None

    def test_add_with_correction(self, db):
        original = db.add(task="t", approach="wrong", outcome="bad")
        fix = db.add(task="t", approach="right", outcome="good", corrects=original)
        entry = db.get(fix)
        assert entry["corrects"] == original


class TestRecent:
    def test_recent_order(self, db):
        db.add(task="t", approach="first", outcome="a")
        db.add(task="t", approach="second", outcome="b")
        db.add(task="t", approach="third", outcome="c")
        entries = db.recent(3)
        assert entries[0]["approach"] == "third"
        assert entries[2]["approach"] == "first"

    def test_recent_limit(self, db):
        for i in range(10):
            db.add(task="t", approach=f"attempt {i}", outcome="ok")
        assert len(db.recent(3)) == 3

    def test_recent_pagination(self, db):
        for i in range(10):
            db.add(task="t", approach=f"attempt {i}", outcome="ok")
        page1 = db.recent(3)
        assert len(page1) == 3
        # Page 2: entries before the smallest ID in page 1
        smallest_id = min(e["id"] for e in page1)
        page2 = db.recent(3, before_id=smallest_id)
        assert len(page2) == 3
        # No overlap between pages
        ids1 = {e["id"] for e in page1}
        ids2 = {e["id"] for e in page2}
        assert ids1.isdisjoint(ids2)
        # Page 2 entries all have smaller IDs
        assert all(eid < smallest_id for eid in ids2)

    def test_recent_pagination_exhaustion(self, db):
        for i in range(3):
            db.add(task="t", approach=f"attempt {i}", outcome="ok")
        page1 = db.recent(3)
        smallest_id = min(e["id"] for e in page1)
        page2 = db.recent(3, before_id=smallest_id)
        assert len(page2) == 0


class TestBest:
    def test_best_by_score(self, db):
        db.add(task="t", approach="bad", outcome="poor", score=0.2)
        db.add(task="t", approach="good", outcome="great", score=0.9)
        db.add(task="t", approach="mid", outcome="ok", score=0.5)
        best = db.best(2)
        assert best[0]["score"] == 0.9
        assert best[1]["score"] == 0.5

    def test_best_skips_null_scores(self, db):
        db.add(task="t", approach="no score", outcome="dunno")
        db.add(task="t", approach="has score", outcome="good", score=0.8)
        best = db.best(5)
        assert len(best) == 1

    def test_best_pagination(self, db):
        for i in range(10):
            db.add(task="t", approach=f"attempt {i}", outcome="ok", score=float(i))
        page1 = db.best(3)
        assert page1[0]["score"] == 9.0
        lowest = min(e["score"] for e in page1)
        page2 = db.best(3, before_score=lowest)
        assert len(page2) == 3
        # Page 2 scores are all below page 1's lowest
        assert all(e["score"] < lowest for e in page2)


class TestFailures:
    def test_failures_no_score(self, db):
        db.add(task="t", approach="worked", outcome="good", score=0.8)
        db.add(task="t", approach="crashed", outcome="OOM")
        db.add(task="t", approach="bad", outcome="terrible", score=-0.1)
        fails = db.failures(5)
        assert len(fails) == 2
        approaches = {f["approach"] for f in fails}
        assert "crashed" in approaches
        assert "bad" in approaches

    def test_failures_unscored_only(self, db):
        db.add(task="t", approach="no score", outcome="pending")
        db.add(task="t", approach="negative", outcome="bad", score=-0.5)
        db.add(task="t", approach="good", outcome="nice", score=0.9)
        unscored = db.failures(5, kind="unscored")
        assert len(unscored) == 1
        assert unscored[0]["approach"] == "no score"

    def test_failures_negative_only(self, db):
        db.add(task="t", approach="no score", outcome="pending")
        db.add(task="t", approach="negative", outcome="bad", score=-0.5)
        db.add(task="t", approach="good", outcome="nice", score=0.9)
        negative = db.failures(5, kind="negative")
        assert len(negative) == 1
        assert negative[0]["approach"] == "negative"

    def test_failures_pagination(self, db):
        for i in range(6):
            db.add(task="t", approach=f"fail {i}", outcome="bad")
        page1 = db.failures(3)
        smallest_id = min(e["id"] for e in page1)
        page2 = db.failures(3, before_id=smallest_id)
        assert len(page2) == 3
        assert all(e["id"] < smallest_id for e in page2)


class TestSearch:
    def test_search_by_approach(self, db):
        db.add(task="forecast", approach="transformer encoder", outcome="ok")
        db.add(task="forecast", approach="Mamba SSM", outcome="better")
        db.add(task="classify", approach="CNN", outcome="ok")
        results = db.search("Mamba")
        assert len(results) == 1
        assert "Mamba" in results[0]["approach"]

    def test_search_by_outcome(self, db):
        db.add(task="t", approach="a", outcome="OOM at 50M params")
        db.add(task="t", approach="b", outcome="converged nicely")
        results = db.search("OOM")
        assert len(results) == 1

    def test_search_by_task(self, db):
        db.add(task="time series forecasting", approach="a", outcome="ok")
        db.add(task="image classification", approach="b", outcome="ok")
        results = db.search("time series")
        assert len(results) == 1

    def test_search_no_results(self, db):
        db.add(task="t", approach="a", outcome="b")
        results = db.search("nonexistent")
        assert len(results) == 0


class TestTaskFiltering:
    def test_tasks_list(self, db):
        db.add(task="forecast demand", approach="a", outcome="ok")
        db.add(task="forecast demand", approach="b", outcome="ok")
        db.add(task="image classification", approach="c", outcome="ok")
        task_list = db.tasks()
        assert len(task_list) == 2
        # Ordered by count descending
        assert task_list[0]["task"] == "forecast demand"
        assert task_list[0]["entry_count"] == 2
        assert task_list[1]["task"] == "image classification"
        assert task_list[1]["entry_count"] == 1

    def test_tasks_empty(self, db):
        assert db.tasks() == []

    def test_tasks_ignores_empty_task(self, db):
        db.add(task="", approach="a", outcome="ok")
        db.add(task="real task", approach="b", outcome="ok")
        task_list = db.tasks()
        assert len(task_list) == 1
        assert task_list[0]["task"] == "real task"

    def test_recent_filter_by_task(self, db):
        db.add(task="forecast demand", approach="a", outcome="ok")
        db.add(task="image classification", approach="b", outcome="ok")
        db.add(task="forecast demand", approach="c", outcome="ok")
        results = db.recent(10, task="forecast")
        assert len(results) == 2
        assert all("forecast" in e["task"].lower() for e in results)

    def test_recent_filter_case_insensitive(self, db):
        db.add(task="Forecast Demand", approach="a", outcome="ok")
        db.add(task="forecast demand", approach="b", outcome="ok")
        results = db.recent(10, task="forecast demand")
        assert len(results) == 2

    def test_recent_no_task_filter_returns_all(self, db):
        db.add(task="task A", approach="a", outcome="ok")
        db.add(task="task B", approach="b", outcome="ok")
        results = db.recent(10)
        assert len(results) == 2

    def test_best_filter_by_task(self, db):
        db.add(task="forecast", approach="a", outcome="ok", score=0.9)
        db.add(task="classify", approach="b", outcome="ok", score=0.95)
        db.add(task="forecast", approach="c", outcome="ok", score=0.8)
        results = db.best(10, task="forecast")
        assert len(results) == 2
        assert results[0]["score"] == 0.9
        assert all("forecast" in e["task"].lower() for e in results)

    def test_failures_filter_by_task(self, db):
        db.add(task="forecast", approach="a", outcome="OOM")
        db.add(task="classify", approach="b", outcome="crash")
        db.add(task="forecast", approach="c", outcome="ok", score=0.8)
        results = db.failures(10, task="forecast")
        assert len(results) == 1
        assert results[0]["approach"] == "a"

    def test_top_rated_filter_by_task(self, db):
        e1 = db.add(task="forecast", approach="good", outcome="ok", agent="a")
        e2 = db.add(task="classify", approach="also good", outcome="ok", agent="a")
        db.rate(e1, "v1", 1)
        db.rate(e2, "v1", 1)
        results = db.top_rated(10, task="forecast")
        assert len(results) == 1
        assert results[0]["id"] == e1

    def test_pagination_with_task_filter(self, db):
        for i in range(6):
            db.add(task="forecast", approach=f"attempt {i}", outcome="ok")
        db.add(task="other", approach="unrelated", outcome="ok")
        page1 = db.recent(3, task="forecast")
        assert len(page1) == 3
        smallest_id = min(e["id"] for e in page1)
        page2 = db.recent(3, before_id=smallest_id, task="forecast")
        assert len(page2) == 3
        # All from forecast task
        assert all("forecast" in e["task"] for e in page2)
        # No overlap
        assert {e["id"] for e in page1}.isdisjoint({e["id"] for e in page2})


class TestLineage:
    def test_lineage_single_entry(self, db):
        eid = db.add(task="t", approach="solo", outcome="ok")
        chain = db.lineage(eid)
        assert len(chain) == 1
        assert chain[0]["id"] == eid

    def test_lineage_chain(self, db):
        a = db.add(task="t", approach="v1", outcome="baseline")
        b = db.add(task="t", approach="v2", outcome="better", built_on=a)
        c = db.add(task="t", approach="v3", outcome="best", built_on=b)
        chain = db.lineage(c)
        assert len(chain) == 3
        # Root-first order
        assert chain[0]["id"] == a
        assert chain[1]["id"] == b
        assert chain[2]["id"] == c

    def test_lineage_nonexistent(self, db):
        chain = db.lineage(999)
        assert len(chain) == 0

    def test_lineage_cycle_protection(self, db):
        """Verify lineage doesn't infinite-loop on malformed data."""
        a = db.add(task="t", approach="a", outcome="ok")
        b = db.add(task="t", approach="b", outcome="ok", built_on=a)
        # Manually create a cycle (shouldn't happen normally, but defensive)
        db._conn.execute("UPDATE entries SET built_on = ? WHERE id = ?", (b, a))
        db._conn.commit()
        chain = db.lineage(b)
        # Should terminate, not loop forever
        assert len(chain) <= 2


class TestCorrections:
    def test_corrections_for(self, db):
        original = db.add(task="t", approach="wrong", outcome="bad", score=0.1)
        fix = db.add(task="t", approach="right", outcome="good", score=0.9, corrects=original)
        corrections = db.corrections_for(original)
        assert len(corrections) == 1
        assert corrections[0]["id"] == fix

    def test_no_corrections(self, db):
        eid = db.add(task="t", approach="fine", outcome="ok")
        assert db.corrections_for(eid) == []

    def test_multiple_corrections(self, db):
        original = db.add(task="t", approach="wrong", outcome="bad")
        db.add(task="t", approach="fix1", outcome="better", corrects=original)
        db.add(task="t", approach="fix2", outcome="best", corrects=original)
        corrections = db.corrections_for(original)
        assert len(corrections) == 2


class TestRender:
    def test_render_empty(self, db):
        text = db._render([])
        assert "No entries" in text

    def test_render_includes_fields(self, db):
        db.add(
            task="forecast demand",
            approach="Mamba blocks",
            outcome="CRPS 0.38",
            agent="alpha",
            score=0.88,
            built_on=None,
        )
        entries = db.recent(1)
        text = db._render(entries)
        assert "forecast demand" in text
        assert "Mamba" in text
        assert "CRPS 0.38" in text
        assert "alpha" in text
        assert "0.88" in text

    def test_render_lineage(self, db):
        p = db.add(task="t", approach="parent", outcome="ok")
        db.add(task="t", approach="child", outcome="better", built_on=p)
        entries = db.recent(1)
        text = db._render(entries)
        assert f"#{p}" in text

    def test_render_correction(self, db):
        original = db.add(task="t", approach="wrong", outcome="bad")
        db.add(task="t", approach="right", outcome="good", corrects=original)
        entries = db.recent(1)
        text = db._render(entries)
        assert f"Corrects" in text
        assert f"#{original}" in text


# ── Rate Limiter ─────────────────────────────────────────────


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_per_minute=5)
        for _ in range(5):
            assert limiter.check("agent-a") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_per_minute=3)
        for _ in range(3):
            limiter.check("agent-a")
        assert limiter.check("agent-a") is False

    def test_separate_agents(self):
        limiter = RateLimiter(max_per_minute=2)
        limiter.check("agent-a")
        limiter.check("agent-a")
        assert limiter.check("agent-a") is False
        # Different agent should still be fine
        assert limiter.check("agent-b") is True

    def test_disabled_when_zero(self):
        limiter = RateLimiter(max_per_minute=0)
        assert not limiter.enabled
        # Should always allow
        for _ in range(1000):
            assert limiter.check("agent-a") is True


# ── Thread Safety ────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_writes(self):
        """Multiple threads writing simultaneously should not corrupt data."""
        with tempfile.TemporaryDirectory() as d:
            db = LogDB(f"{d}/test.db")
            errors = []

            def writer(agent_name: str, count: int):
                try:
                    for i in range(count):
                        db.add(
                            task="concurrent test",
                            approach=f"{agent_name} attempt {i}",
                            outcome="ok",
                            agent=agent_name,
                            score=float(i),
                        )
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=writer, args=(f"agent-{i}", 20))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"Concurrent writes produced errors: {errors}"
            # All 100 entries should be present
            all_entries = db.recent(200)
            assert len(all_entries) == 100

    def test_concurrent_read_write(self):
        """Reads during writes should not fail."""
        with tempfile.TemporaryDirectory() as d:
            db = LogDB(f"{d}/test.db")
            # Seed some data
            for i in range(10):
                db.add(task="seed", approach=f"seed {i}", outcome="ok", score=float(i))

            errors = []

            def writer():
                try:
                    for i in range(50):
                        db.add(task="t", approach=f"w {i}", outcome="ok")
                except Exception as e:
                    errors.append(("write", e))

            def reader():
                try:
                    for _ in range(50):
                        db.recent(5)
                        db.best(5)
                        db.search("seed")
                except Exception as e:
                    errors.append(("read", e))

            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader),
                threading.Thread(target=reader),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"Concurrent read/write errors: {errors}"


# ── MCP Server ───────────────────────────────────────────────


class TestMCPServer:
    def test_creates_server(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db")
            assert server is not None
            tools = server._tool_manager._tools
            names = set(tools.keys())
            assert names == {
                "log", "correct", "rate", "tasks", "recent", "best",
                "top_rated", "failures", "search", "lineage",
            }

    def test_require_agent_rejects_empty(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db", require_agent=True)
            log_tool = server._tool_manager._tools["log"]
            # Access the underlying function
            from server import LogDB
            db = LogDB(f"{d}/test2.db")
            # Test via create_server's closure — call the tool function
            # We test the check logic directly
            result = server._tool_manager._tools["log"].fn(
                task="t", approach="a", outcome="o", agent=""
            )
            assert "Error" in result

    def test_rate_limit_rejects_flood(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db", rate_limit=3)
            log_fn = server._tool_manager._tools["log"].fn
            for _ in range(3):
                result = log_fn(task="t", approach="a", outcome="o", agent="flood-bot")
                assert "Logged" in result
            result = log_fn(task="t", approach="a", outcome="o", agent="flood-bot")
            assert "rate limit" in result.lower()


# ── Multi-agent workflow ─────────────────────────────────────


class TestMultiAgent:
    def test_agents_see_each_others_work(self, db):
        """Core use case: agent B learns from agent A's experiments."""
        # Agent A tries something
        db.add(
            agent="agent-A",
            task="forecast electricity demand",
            approach="Standard transformer with patch embeddings, d_model=64",
            outcome="CRPS 0.45, training stable, 8 iter/s",
            score=0.45,
        )

        # Agent A tries an improvement
        a2 = db.add(
            agent="agent-A",
            task="forecast electricity demand",
            approach="Added flash attention to transformer",
            outcome="CRPS 0.42, training stable, 15 iter/s — big speedup",
            score=0.42,
            built_on=1,
        )

        # Agent B comes along, reads recent work
        recent = db.recent(5)
        assert len(recent) == 2
        assert any("flash attention" in e["approach"] for e in recent)

        # Agent B searches for what worked
        good = db.best(5)
        assert good[0]["score"] == 0.45  # higher is better in this test

        # Agent B builds on agent A's work
        db.add(
            agent="agent-B",
            task="forecast electricity demand",
            approach="Flash attention + Mamba SSM replacing self-attention layers",
            outcome="CRPS 0.38, training stable, 12 iter/s",
            score=0.38,
            built_on=a2,
        )

        # Agent C searches for Mamba specifically
        mamba_results = db.search("Mamba")
        assert len(mamba_results) == 1
        assert mamba_results[0]["agent"] == "agent-B"

        # The log now tells the whole story
        all_entries = db.recent(10)
        assert len(all_entries) == 3
        agents = {e["agent"] for e in all_entries}
        assert agents == {"agent-A", "agent-B"}

    def test_correction_workflow(self, db):
        """Agent realises a logged score was wrong and corrects it."""
        original = db.add(
            agent="agent-A",
            task="forecast demand",
            approach="transformer",
            outcome="CRPS 0.42",
            score=0.42,
        )
        # Oops, the score was actually evaluated on the wrong split
        fix = db.add(
            agent="agent-A",
            task="forecast demand",
            approach="transformer (re-evaluated on correct test split)",
            outcome="CRPS 0.51 — worse than reported",
            score=0.51,
            corrects=original,
        )
        # The correction is discoverable
        corrections = db.corrections_for(original)
        assert len(corrections) == 1
        assert corrections[0]["id"] == fix

    def test_lineage_workflow(self, db):
        """Trace the full evolution of an idea across agents."""
        a = db.add(agent="A", task="t", approach="v1", outcome="baseline", score=0.5)
        b = db.add(agent="A", task="t", approach="v2", outcome="better", score=0.6, built_on=a)
        c = db.add(agent="B", task="t", approach="v3", outcome="best", score=0.8, built_on=b)
        chain = db.lineage(c)
        assert len(chain) == 3
        assert [e["agent"] for e in chain] == ["A", "A", "B"]
        assert [e["score"] for e in chain] == [0.5, 0.6, 0.8]


# ── Ratings ───────────────────────────────────────────────────


class TestRatings:
    def test_rate_useful(self, db):
        eid = db.add(task="t", approach="a", outcome="good", agent="author")
        db.rate(eid, "voter-1", useful=1)
        r = db.get_ratings(eid)
        assert r["useful"] == 1
        assert r["not_useful"] == 0
        assert r["unhelpful"] == 0
        assert r["total"] == 1
        assert r["ratio"] == 1.0

    def test_rate_not_useful(self, db):
        eid = db.add(task="t", approach="a", outcome="bad", agent="author")
        db.rate(eid, "voter-1", useful=0)
        r = db.get_ratings(eid)
        assert r["useful"] == 0
        assert r["not_useful"] == 1
        assert r["unhelpful"] == 0
        assert r["ratio"] == 0.0

    def test_rate_unhelpful(self, db):
        eid = db.add(task="t", approach="a", outcome="misleading", agent="author")
        db.rate(eid, "voter-1", useful=-1)
        r = db.get_ratings(eid)
        assert r["useful"] == 0
        assert r["not_useful"] == 0
        assert r["unhelpful"] == 1
        assert r["ratio"] == 0.0

    def test_rate_with_note(self, db):
        eid = db.add(task="t", approach="a", outcome="ok", agent="author")
        db.rate(eid, "voter-1", useful=1, note="Saved me hours")
        r = db.get_ratings(eid)
        assert r["total"] == 1

    def test_rate_invalid_value(self, db):
        eid = db.add(task="t", approach="a", outcome="ok")
        with pytest.raises(ValueError):
            db.rate(eid, "voter", useful=2)

    def test_one_vote_per_agent(self, db):
        """Re-voting overwrites the previous vote."""
        eid = db.add(task="t", approach="a", outcome="ok", agent="author")
        db.rate(eid, "voter-1", useful=1)
        db.rate(eid, "voter-1", useful=-1)  # Changed mind
        r = db.get_ratings(eid)
        assert r["total"] == 1
        assert r["useful"] == 0
        assert r["unhelpful"] == 1

    def test_multiple_voters_ternary(self, db):
        eid = db.add(task="t", approach="a", outcome="ok", agent="author")
        db.rate(eid, "voter-1", useful=1)
        db.rate(eid, "voter-2", useful=0)
        db.rate(eid, "voter-3", useful=-1)
        r = db.get_ratings(eid)
        assert r["useful"] == 1
        assert r["not_useful"] == 1
        assert r["unhelpful"] == 1
        assert r["total"] == 3
        assert abs(r["ratio"] - 1 / 3) < 0.01

    def test_no_ratings(self, db):
        eid = db.add(task="t", approach="a", outcome="ok")
        r = db.get_ratings(eid)
        assert r["total"] == 0
        assert r["ratio"] is None

    def test_bulk_ratings(self, db):
        e1 = db.add(task="t", approach="a", outcome="ok")
        e2 = db.add(task="t", approach="b", outcome="ok")
        e3 = db.add(task="t", approach="c", outcome="ok")
        db.rate(e1, "v", useful=1)
        db.rate(e2, "v", useful=-1)
        # e3 has no ratings
        bulk = db.get_ratings_bulk([e1, e2, e3])
        assert bulk[e1]["useful"] == 1
        assert bulk[e2]["unhelpful"] == 1
        assert bulk[e3]["total"] == 0

    def test_render_includes_ternary_ratings(self, db):
        eid = db.add(task="t", approach="a", outcome="ok", agent="author")
        db.rate(eid, "voter-1", useful=1)
        db.rate(eid, "voter-2", useful=0)
        db.rate(eid, "voter-3", useful=-1)
        entries = db.recent(1)
        text = db._render(entries)
        assert "Community" in text
        assert "1 useful" in text
        assert "1 not useful" in text
        assert "1 unhelpful" in text
        assert "3 votes" in text
        assert "33% useful" in text

    def test_render_no_ratings_no_line(self, db):
        db.add(task="t", approach="a", outcome="ok")
        entries = db.recent(1)
        text = db._render(entries)
        assert "Community" not in text

    def test_render_hides_zero_buckets(self, db):
        """Only non-zero rating categories should appear in the breakdown."""
        eid = db.add(task="t", approach="a", outcome="ok", agent="author")
        db.rate(eid, "voter-1", useful=1)
        db.rate(eid, "voter-2", useful=1)
        entries = db.recent(1)
        text = db._render(entries)
        assert "2 useful" in text
        assert "not useful" not in text
        assert "unhelpful" not in text


class TestTopRated:
    def test_top_rated_order(self, db):
        e1 = db.add(task="t", approach="loved", outcome="ok", agent="a")
        e2 = db.add(task="t", approach="meh", outcome="ok", agent="a")
        e3 = db.add(task="t", approach="hated", outcome="ok", agent="a")
        # e1: 3/3 useful
        db.rate(e1, "v1", 1)
        db.rate(e1, "v2", 1)
        db.rate(e1, "v3", 1)
        # e2: 1/3 useful, 1 not useful, 1 unhelpful
        db.rate(e2, "v1", 1)
        db.rate(e2, "v2", 0)
        db.rate(e2, "v3", -1)
        # e3: 0/2 useful (all unhelpful)
        db.rate(e3, "v1", -1)
        db.rate(e3, "v2", -1)
        top = db.top_rated(10)
        assert len(top) == 3
        assert top[0]["id"] == e1
        assert top[1]["id"] == e2
        assert top[2]["id"] == e3

    def test_top_rated_min_votes(self, db):
        e1 = db.add(task="t", approach="popular", outcome="ok", agent="a")
        e2 = db.add(task="t", approach="one vote", outcome="ok", agent="a")
        db.rate(e1, "v1", 1)
        db.rate(e1, "v2", 1)
        db.rate(e1, "v3", 1)
        db.rate(e2, "v1", 1)
        # With min_votes=2, only e1 qualifies
        top = db.top_rated(10, min_votes=2)
        assert len(top) == 1
        assert top[0]["id"] == e1

    def test_top_rated_empty(self, db):
        db.add(task="t", approach="no votes", outcome="ok")
        top = db.top_rated(10)
        assert len(top) == 0

    def test_top_rated_unhelpful_vs_not_useful(self, db):
        """Both drag ratio down equally, but agents can see the distinction."""
        e1 = db.add(task="t", approach="neutral bad", outcome="ok", agent="a")
        e2 = db.add(task="t", approach="actively bad", outcome="ok", agent="a")
        # e1: 1 useful, 1 not useful
        db.rate(e1, "v1", 1)
        db.rate(e1, "v2", 0)
        # e2: 1 useful, 1 unhelpful
        db.rate(e2, "v1", 1)
        db.rate(e2, "v2", -1)
        # Same ratio (0.5), but agents reading the render can see the unhelpful flag
        top = db.top_rated(10)
        assert len(top) == 2
        ratios = [db.get_ratings(e["id"])["ratio"] for e in top]
        assert ratios[0] == ratios[1] == 0.5


class TestRateMCPTool:
    def test_rate_tool_blocks_self_rating(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db")
            log_fn = server._tool_manager._tools["log"].fn
            rate_fn = server._tool_manager._tools["rate"].fn
            log_fn(task="t", approach="a", outcome="o", agent="alice")
            result = rate_fn(entry_id=1, rating=1, agent="alice")
            assert "cannot rate your own" in result.lower()

    def test_rate_tool_allows_other_rating(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db")
            log_fn = server._tool_manager._tools["log"].fn
            rate_fn = server._tool_manager._tools["rate"].fn
            log_fn(task="t", approach="a", outcome="o", agent="alice")
            result = rate_fn(entry_id=1, rating=1, agent="bob")
            assert "rated" in result.lower()

    def test_rate_tool_nonexistent_entry(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db")
            rate_fn = server._tool_manager._tools["rate"].fn
            result = rate_fn(entry_id=999, rating=1, agent="bob")
            assert "does not exist" in result.lower()

    def test_rate_tool_invalid_rating(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db")
            log_fn = server._tool_manager._tools["log"].fn
            rate_fn = server._tool_manager._tools["rate"].fn
            log_fn(task="t", approach="a", outcome="o", agent="alice")
            result = rate_fn(entry_id=1, rating=5, agent="bob")
            assert "error" in result.lower()

    def test_rate_tool_unhelpful(self):
        with tempfile.TemporaryDirectory() as d:
            server = create_server(f"{d}/test.db")
            log_fn = server._tool_manager._tools["log"].fn
            rate_fn = server._tool_manager._tools["rate"].fn
            log_fn(task="t", approach="a", outcome="o", agent="alice")
            result = rate_fn(entry_id=1, rating=-1, agent="bob")
            assert "unhelpful" in result.lower()


# ── Schema Migration ─────────────────────────────────────────


class TestMigration:
    def test_migration_adds_corrects_column(self):
        """Opening an old DB without 'corrects' column should auto-migrate."""
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/old.db"
            # Create an old-style DB without corrects
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL DEFAULT '',
                    task TEXT NOT NULL DEFAULT '',
                    approach TEXT NOT NULL DEFAULT '',
                    outcome TEXT NOT NULL DEFAULT '',
                    score REAL,
                    built_on INTEGER,
                    timestamp REAL NOT NULL
                );
            """)
            conn.execute(
                "INSERT INTO entries (agent, task, approach, outcome, timestamp) "
                "VALUES ('a', 't', 'ap', 'out', 1.0)"
            )
            conn.commit()
            conn.close()

            # Open with LogDB — should migrate
            db = LogDB(path)
            entry = db.get(1)
            assert entry is not None
            assert entry["corrects"] is None
            # Should be able to add with corrects now
            fix = db.add(task="t", approach="fix", outcome="ok", corrects=1)
            assert db.get(fix)["corrects"] == 1
