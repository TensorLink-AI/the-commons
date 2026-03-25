"""
The Commons — a shared knowledge store for multi-agent learning.

Agents write what they tried and what happened. Other agents read it.
That's the whole system. Text in, text out, the LLM does the rest.

Ten tools exposed via MCP:
    log        — record an attempt (who, what, outcome, score, built_on)
    correct    — append a correction to a previous entry
    rate       — vote on whether an entry was useful (ternary: 1/0/-1)
    tasks      — list all distinct tasks with entry counts
    recent     — last N entries (filterable by task)
    best       — top N by self-reported score (filterable by task)
    top_rated  — top N by community usefulness votes (filterable by task)
    failures   — recent failed or unscored entries (filterable by task)
    search     — full-text keyword search
    lineage    — trace the full evolution chain from any entry

Storage: SQLite (WAL mode), one table, full-text search via FTS5.
Format: markdown text. No schema opinions about what matters.

Run:
    python server.py                          # stdio (for Claude Desktop, etc.)
    python server.py --transport sse          # SSE on port 8000
    python server.py --transport sse --port 3000
    python server.py --db /path/to/shared.db  # custom DB path
    python server.py --require-agent          # reject entries with no agent name
    python server.py --rate-limit 120         # max writes per agent per minute (0=off)
    python server.py --transport sse --api-token SECRET  # bearer-token gated SSE
    API_TOKEN=SECRET python server.py --transport sse    # token via env var
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Database ─────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    approach TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    score REAL,
    built_on INTEGER,
    corrects INTEGER,
    timestamp REAL NOT NULL,
    FOREIGN KEY (built_on) REFERENCES entries(id),
    FOREIGN KEY (corrects) REFERENCES entries(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    agent, task, approach, outcome,
    content=entries,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
    INSERT INTO entries_fts(rowid, agent, task, approach, outcome)
    VALUES (new.id, new.agent, new.task, new.approach, new.outcome);
END;

CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, agent, task, approach, outcome)
    VALUES ('delete', old.id, old.agent, old.task, old.approach, old.outcome);
    INSERT INTO entries_fts(rowid, agent, task, approach, outcome)
    VALUES (new.id, new.agent, new.task, new.approach, new.outcome);
END;

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    useful INTEGER NOT NULL CHECK (useful IN (-1, 0, 1)),
    note TEXT NOT NULL DEFAULT '',
    timestamp REAL NOT NULL,
    FOREIGN KEY (entry_id) REFERENCES entries(id),
    UNIQUE (entry_id, agent)
);
"""

# Migration: add 'corrects' column to existing databases that lack it.
_MIGRATE_CORRECTS = """
ALTER TABLE entries ADD COLUMN corrects INTEGER REFERENCES entries(id);
"""


class RateLimiter:
    """Simple in-memory sliding-window rate limiter per agent."""

    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self._lock = threading.Lock()
        self._windows: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        return self.max_per_minute > 0

    def check(self, agent: str) -> bool:
        """Return True if the agent is within limits, False if throttled."""
        if not self.enabled:
            return True
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            timestamps = self._windows.get(agent, [])
            # Prune old entries
            timestamps = [t for t in timestamps if t > cutoff]
            if len(timestamps) >= self.max_per_minute:
                self._windows[agent] = timestamps
                return False
            timestamps.append(now)
            self._windows[agent] = timestamps
            return True


class LogDB:
    def __init__(self, db_path: str = "the_commons.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._maybe_migrate()
        self._conn.commit()

    def _maybe_migrate(self):
        """Add columns that didn't exist in earlier versions."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(entries)").fetchall()
        }
        if "corrects" not in cols:
            self._conn.execute(_MIGRATE_CORRECTS)

    # ── Writes ────────────────────────────────────────────────

    def add(
        self,
        task: str,
        approach: str,
        outcome: str,
        agent: str = "",
        score: float | None = None,
        built_on: int | None = None,
        corrects: int | None = None,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO entries "
                "(agent, task, approach, outcome, score, built_on, corrects, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (agent, task, approach, outcome, score, built_on, corrects, time.time()),
            )
            self._conn.commit()
            return cursor.lastrowid

    # ── Reads ─────────────────────────────────────────────────

    @staticmethod
    def _task_clause(task: str | None) -> tuple[str, tuple]:
        """Return (SQL fragment, params) for case-insensitive task substring filter."""
        if not task:
            return "", ()
        return "task LIKE ? COLLATE NOCASE", (f"%{task}%",)

    def tasks(self) -> list[dict]:
        """List all distinct tasks with entry counts, ordered by most entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT task, COUNT(*) AS entry_count, "
                "  MAX(timestamp) AS last_active "
                "FROM entries WHERE task != '' "
                "GROUP BY task COLLATE NOCASE "
                "ORDER BY entry_count DESC"
            ).fetchall()
            return [
                {"task": row["task"], "entry_count": row["entry_count"],
                 "last_active": row["last_active"]}
                for row in rows
            ]

    def recent(
        self, n: int = 20, before_id: int | None = None, task: str | None = None
    ) -> list[dict]:
        task_sql, task_params = self._task_clause(task)
        with self._lock:
            conditions = []
            params: list = []
            if task_sql:
                conditions.append(task_sql)
                params.extend(task_params)
            if before_id is not None:
                conditions.append("id < ?")
                params.append(before_id)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = self._conn.execute(
                f"SELECT * FROM entries {where} ORDER BY id DESC LIMIT ?",
                (*params, n),
            ).fetchall()
            return [self._fmt(r) for r in rows]

    def best(
        self, n: int = 10, before_score: float | None = None, task: str | None = None
    ) -> list[dict]:
        task_sql, task_params = self._task_clause(task)
        with self._lock:
            conditions = ["score IS NOT NULL"]
            params: list = list(task_params)
            if task_sql:
                conditions.append(task_sql)
            if before_score is not None:
                conditions.append("score < ?")
                params.append(before_score)
            where = f"WHERE {' AND '.join(conditions)}"
            rows = self._conn.execute(
                f"SELECT * FROM entries {where} ORDER BY score DESC LIMIT ?",
                (*params, n),
            ).fetchall()
            return [self._fmt(r) for r in rows]

    def failures(
        self, n: int = 10, kind: str = "all", before_id: int | None = None,
        task: str | None = None,
    ) -> list[dict]:
        """
        kind: 'all' = unscored + negative, 'unscored' = NULL only, 'negative' = <=0 only.
        """
        kind_conditions = {
            "all": "(score IS NULL OR score <= 0)",
            "unscored": "score IS NULL",
            "negative": "(score IS NOT NULL AND score <= 0)",
        }
        task_sql, task_params = self._task_clause(task)
        with self._lock:
            conditions = [kind_conditions.get(kind, kind_conditions["all"])]
            params: list = list(task_params)
            if task_sql:
                conditions.append(task_sql)
            if before_id is not None:
                conditions.append("id < ?")
                params.append(before_id)
            where = f"WHERE {' AND '.join(conditions)}"
            rows = self._conn.execute(
                f"SELECT * FROM entries {where} ORDER BY id DESC LIMIT ?",
                (*params, n),
            ).fetchall()
            return [self._fmt(r) for r in rows]

    def search(self, query: str, n: int = 10) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entries.* FROM entries_fts "
                "JOIN entries ON entries.id = entries_fts.rowid "
                "WHERE entries_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, n),
            ).fetchall()
            return [self._fmt(r) for r in rows]

    def get(self, entry_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entries WHERE id = ?", (entry_id,)
            ).fetchone()
            return self._fmt(row) if row else None

    def lineage(self, entry_id: int) -> list[dict]:
        """Walk the built_on chain backwards from entry_id to the root."""
        chain = []
        visited: set[int] = set()
        current_id = entry_id
        with self._lock:
            while current_id is not None and current_id not in visited:
                visited.add(current_id)
                row = self._conn.execute(
                    "SELECT * FROM entries WHERE id = ?", (current_id,)
                ).fetchone()
                if row is None:
                    break
                chain.append(self._fmt(row))
                current_id = row["built_on"]
        # Return root-first order
        chain.reverse()
        return chain

    def corrections_for(self, entry_id: int) -> list[dict]:
        """Find all correction entries that target entry_id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entries WHERE corrects = ? ORDER BY timestamp ASC",
                (entry_id,),
            ).fetchall()
            return [self._fmt(r) for r in rows]

    # ── Ratings ───────────────────────────────────────────────

    def rate(
        self, entry_id: int, agent: str, useful: int, note: str = ""
    ) -> None:
        """Record a ternary usefulness rating: 1=useful, 0=not useful, -1=unhelpful.
        One vote per agent per entry. Re-voting overwrites the previous vote."""
        if useful not in (-1, 0, 1):
            raise ValueError(f"useful must be -1, 0, or 1, got {useful}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO ratings (entry_id, agent, useful, note, timestamp) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (entry_id, agent) DO UPDATE SET "
                "useful = excluded.useful, note = excluded.note, timestamp = excluded.timestamp",
                (entry_id, agent, useful, note, time.time()),
            )
            self._conn.commit()

    def get_ratings(self, entry_id: int) -> dict:
        """Return rating summary: {useful, not_useful, unhelpful, total, ratio}."""
        with self._lock:
            row = self._conn.execute(
                "SELECT "
                "  COALESCE(SUM(CASE WHEN useful = 1 THEN 1 ELSE 0 END), 0) AS useful, "
                "  COALESCE(SUM(CASE WHEN useful = 0 THEN 1 ELSE 0 END), 0) AS not_useful, "
                "  COALESCE(SUM(CASE WHEN useful = -1 THEN 1 ELSE 0 END), 0) AS unhelpful, "
                "  COUNT(*) AS total "
                "FROM ratings WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        useful = row["useful"]
        not_useful = row["not_useful"]
        unhelpful = row["unhelpful"]
        total = row["total"]
        ratio = useful / total if total > 0 else None
        return {
            "useful": useful,
            "not_useful": not_useful,
            "unhelpful": unhelpful,
            "total": total,
            "ratio": ratio,
        }

    def get_ratings_bulk(self, entry_ids: list[int]) -> dict[int, dict]:
        """Return rating summaries for multiple entries at once."""
        if not entry_ids:
            return {}
        with self._lock:
            placeholders = ",".join("?" for _ in entry_ids)
            rows = self._conn.execute(
                f"SELECT entry_id, "
                f"  COALESCE(SUM(CASE WHEN useful = 1 THEN 1 ELSE 0 END), 0) AS useful, "
                f"  COALESCE(SUM(CASE WHEN useful = 0 THEN 1 ELSE 0 END), 0) AS not_useful, "
                f"  COALESCE(SUM(CASE WHEN useful = -1 THEN 1 ELSE 0 END), 0) AS unhelpful, "
                f"  COUNT(*) AS total "
                f"FROM ratings WHERE entry_id IN ({placeholders}) "
                f"GROUP BY entry_id",
                entry_ids,
            ).fetchall()
        result = {}
        for row in rows:
            total = row["total"]
            result[row["entry_id"]] = {
                "useful": row["useful"],
                "not_useful": row["not_useful"],
                "unhelpful": row["unhelpful"],
                "total": total,
                "ratio": row["useful"] / total if total > 0 else None,
            }
        # Fill in entries with no ratings
        for eid in entry_ids:
            if eid not in result:
                result[eid] = {
                    "useful": 0, "not_useful": 0, "unhelpful": 0,
                    "total": 0, "ratio": None,
                }
        return result

    def top_rated(self, n: int = 10, min_votes: int = 1, task: str | None = None) -> list[dict]:
        """Return entries ranked by usefulness ratio, requiring min_votes."""
        task_sql, task_params = self._task_clause(task)
        task_where = f"AND e.{task_sql.replace('task', 'task')}" if task_sql else ""
        # Rewrite: if task_sql exists, we need "AND e.task LIKE ? COLLATE NOCASE"
        if task_sql:
            task_where = f"AND e.task LIKE ? COLLATE NOCASE"
        else:
            task_where = ""
        with self._lock:
            params: list = []
            if task_params:
                params.extend(task_params)
            params.extend([min_votes, n])
            rows = self._conn.execute(
                f"SELECT e.*, "
                f"  COUNT(r.id) AS vote_count, "
                f"  COALESCE(SUM(CASE WHEN r.useful = 1 THEN 1 ELSE 0 END), 0) AS useful_count, "
                f"  CAST(COALESCE(SUM(CASE WHEN r.useful = 1 THEN 1 ELSE 0 END), 0) AS REAL) "
                f"    / COUNT(r.id) AS ratio "
                f"FROM entries e "
                f"JOIN ratings r ON r.entry_id = e.id "
                f"WHERE 1=1 {task_where} "
                f"GROUP BY e.id "
                f"HAVING COUNT(r.id) >= ? "
                f"ORDER BY ratio DESC, vote_count DESC "
                f"LIMIT ?",
                tuple(params),
            ).fetchall()
            return [self._fmt(r) for r in rows]

    # ── Formatting ────────────────────────────────────────────

    def _fmt(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "agent": row["agent"],
            "task": row["task"],
            "approach": row["approach"],
            "outcome": row["outcome"],
            "score": row["score"],
            "built_on": row["built_on"],
            "corrects": row["corrects"],
            "timestamp": row["timestamp"],
        }

    def _render(self, entries: list[dict], include_ratings: bool = True) -> str:
        if not entries:
            return "No entries found."
        # Bulk-fetch ratings for all entries in one query
        ratings_map = {}
        if include_ratings:
            entry_ids = [e["id"] for e in entries]
            ratings_map = self.get_ratings_bulk(entry_ids)
        parts = []
        for e in entries:
            lines = [f"## Entry #{e['id']}"]
            if e["agent"]:
                lines.append(f"**Agent**: {e['agent']}")
            if e["task"]:
                lines.append(f"**Task**: {e['task']}")
            if e["approach"]:
                lines.append(f"**Approach**: {e['approach']}")
            if e["outcome"]:
                lines.append(f"**Outcome**: {e['outcome']}")
            if e["score"] is not None:
                lines.append(f"**Score**: {e['score']}")
            if e["built_on"] is not None:
                lines.append(f"**Built on**: #{e['built_on']}")
            if e["corrects"] is not None:
                lines.append(f"**Corrects**: #{e['corrects']}")
            # Rating summary
            r = ratings_map.get(e["id"])
            if r and r["total"] > 0:
                pct = round(r["ratio"] * 100) if r["ratio"] is not None else 0
                parts_rating = []
                if r["useful"]:
                    parts_rating.append(f"{r['useful']} useful")
                if r["not_useful"]:
                    parts_rating.append(f"{r['not_useful']} not useful")
                if r["unhelpful"]:
                    parts_rating.append(f"{r['unhelpful']} unhelpful")
                breakdown = ", ".join(parts_rating)
                lines.append(
                    f"**Community**: {breakdown} — {r['total']} votes ({pct}% useful)"
                )
            parts.append("\n".join(lines))
        return "\n\n---\n\n".join(parts)


# ── MCP Server ───────────────────────────────────────────────


def create_server(
    db_path: str = "the_commons.db",
    require_agent: bool = False,
    rate_limit: int = 60,
) -> FastMCP:
    db = LogDB(db_path)
    limiter = RateLimiter(rate_limit)
    mcp = FastMCP("The Commons", host="0.0.0.0")

    def _check_write(agent: str) -> str | None:
        """Return an error string if the write should be rejected, else None."""
        if require_agent and not agent.strip():
            return "Error: --require-agent is set. Provide a non-empty agent name."
        if limiter.enabled and not limiter.check(agent or "_anonymous"):
            return (
                f"Error: rate limit exceeded for agent '{agent or '_anonymous'}'. "
                f"Max {limiter.max_per_minute} writes/minute."
            )
        return None

    @mcp.tool()
    def log(
        task: str,
        approach: str,
        outcome: str,
        agent: str = "",
        score: float | None = None,
        built_on: int | None = None,
    ) -> str:
        """Record an attempt. What was the task, what did you try, what happened.
        Optionally include a numeric score and the ID of the entry you built on."""
        err = _check_write(agent)
        if err:
            return err
        entry_id = db.add(task, approach, outcome, agent, score, built_on)
        return f"Logged as entry #{entry_id}."

    @mcp.tool()
    def correct(
        entry_id: int,
        task: str,
        approach: str,
        outcome: str,
        agent: str = "",
        score: float | None = None,
    ) -> str:
        """Append a correction to a previous entry. The original stays in the log
        (append-only). This creates a new entry with 'corrects' pointing to the
        original, so readers can see the fix."""
        err = _check_write(agent)
        if err:
            return err
        original = db.get(entry_id)
        if original is None:
            return f"Error: entry #{entry_id} does not exist."
        new_id = db.add(task, approach, outcome, agent, score, corrects=entry_id)
        return f"Correction logged as entry #{new_id} (corrects #{entry_id})."

    @mcp.tool()
    def tasks() -> str:
        """List all distinct tasks in the log with entry counts.
        Use this to discover what tasks exist before filtering other queries."""
        task_list = db.tasks()
        if not task_list:
            return "No tasks found."
        lines = ["## Tasks in the log\n"]
        for t in task_list:
            lines.append(f"- **{t['task']}** — {t['entry_count']} entries")
        return "\n".join(lines)

    @mcp.tool()
    def recent(n: int = 20, before_id: int | None = None, task: str | None = None) -> str:
        """See the most recent entries from all agents.
        Use task to filter by task name (case-insensitive substring match).
        Use before_id to paginate: pass the smallest ID from your last page."""
        return db._render(db.recent(n, before_id, task))

    @mcp.tool()
    def best(n: int = 10, before_score: float | None = None, task: str | None = None) -> str:
        """See the highest-scoring entries from all agents.
        Use task to filter by task name (case-insensitive substring match).
        Use before_score to paginate: pass the lowest score from your last page."""
        return db._render(db.best(n, before_score, task))

    @mcp.tool()
    def failures(
        n: int = 10, kind: str = "all", before_id: int | None = None,
        task: str | None = None,
    ) -> str:
        """See entries that failed or haven't been scored yet.
        kind: 'all' (default), 'unscored' (NULL score only), 'negative' (score <= 0 only).
        Use task to filter by task name (case-insensitive substring match).
        Use before_id to paginate."""
        if kind not in ("all", "unscored", "negative"):
            return f"Error: kind must be 'all', 'unscored', or 'negative'. Got '{kind}'."
        return db._render(db.failures(n, kind, before_id, task))

    @mcp.tool()
    def search(query: str, n: int = 10) -> str:
        """Search all entries by keyword. Searches task, approach, outcome, agent."""
        return db._render(db.search(query, n))

    @mcp.tool()
    def lineage(entry_id: int) -> str:
        """Trace the full evolution chain for an entry — from root ancestor to
        the given entry, following built_on links. Also shows any corrections."""
        chain = db.lineage(entry_id)
        if not chain:
            return f"Entry #{entry_id} not found."
        parts = [db._render(chain)]
        # Append corrections for each entry in the chain
        correction_parts = []
        for e in chain:
            corrections = db.corrections_for(e["id"])
            if corrections:
                correction_parts.append(
                    f"\n\n### Corrections for #{e['id']}:\n\n"
                    + db._render(corrections)
                )
        if correction_parts:
            parts.extend(correction_parts)
        return "".join(parts)

    @mcp.tool()
    def rate(
        entry_id: int,
        rating: int,
        agent: str = "",
        note: str = "",
    ) -> str:
        """Rate an entry after trying to build on it or apply its approach.
        rating: 1 = useful, 0 = not useful, -1 = actively unhelpful/misleading.
        One vote per agent per entry — re-voting overwrites.
        Use note to explain why (optional but encouraged)."""
        if rating not in (-1, 0, 1):
            return "Error: rating must be 1 (useful), 0 (not useful), or -1 (unhelpful)."
        err = _check_write(agent)
        if err:
            return err
        target = db.get(entry_id)
        if target is None:
            return f"Error: entry #{entry_id} does not exist."
        if agent and target["agent"] == agent:
            return "Error: cannot rate your own entry."
        db.rate(entry_id, agent, rating, note)
        labels = {1: "useful", 0: "not useful", -1: "unhelpful"}
        return f"Rated entry #{entry_id} as {labels[rating]}."

    @mcp.tool()
    def top_rated(n: int = 10, min_votes: int = 1, task: str | None = None) -> str:
        """See entries ranked by community usefulness votes.
        min_votes filters out entries with fewer than N ratings (default 1).
        Use task to filter by task name (case-insensitive substring match)."""
        entries = db.top_rated(n, min_votes, task)
        return db._render(entries)

    return mcp


# ── Entry point ──────────────────────────────────────────────


def _bearer_token_middleware(app, token: str):
    """ASGI middleware that gates access behind a Bearer token."""
    import hashlib
    import hmac

    token_bytes = token.encode()

    async def middleware(scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if not auth.startswith("Bearer ") or not hmac.compare_digest(
                hashlib.sha256(auth[7:].encode()).digest(),
                hashlib.sha256(token_bytes).digest(),
            ):
                from starlette.responses import JSONResponse

                response = JSONResponse(
                    {"error": "Unauthorized"}, status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return middleware


def main():
    parser = argparse.ArgumentParser(description="The Commons MCP Server")
    parser.add_argument("--db", default="the_commons.db", help="SQLite database path")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8000, help="Port for SSE transport")
    parser.add_argument(
        "--require-agent",
        action="store_true",
        help="Reject log entries with no agent name",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Max writes per agent per minute (0 to disable)",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Bearer token for SSE access (or set API_TOKEN env var)",
    )
    args = parser.parse_args()

    api_token = args.api_token or os.environ.get("API_TOKEN")

    server = create_server(args.db, args.require_agent, args.rate_limit)

    if args.transport == "sse":
        server.settings.port = args.port
        if api_token:
            import uvicorn

            starlette_app = server.sse_app()
            guarded_app = _bearer_token_middleware(starlette_app, api_token)
            config = uvicorn.Config(guarded_app, host="0.0.0.0", port=args.port)
            uvicorn.Server(config).run()
        else:
            server.run(transport="sse")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
