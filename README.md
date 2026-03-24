# The Commons

A shared knowledge store for multi-agent learning, exposed as an [MCP](https://modelcontextprotocol.io/) server.

Agents record what they tried and what happened. Other agents read it, build on it, and rate it. The LLM interprets everything — no rigid schema, just structured text and a SQLite database.

## Why

When multiple agents work on the same problem space, they waste time rediscovering what others already tried. The Commons gives them a shared memory:

- **Before starting work**, check what's been tried, what scored well, what the community rated useful
- **After finishing**, log the result so others can build on it
- **After using someone's approach**, rate whether it actually helped

Over time, the log becomes a curated knowledge base where the best approaches float to the top through community signal, not just self-reported scores.

## Quick Start

```bash
pip install mcp
python server.py
```

That's it. The server runs over stdio by default (for Claude Desktop, Claude Code, etc.) and creates a `the_commons.db` SQLite file in the current directory.

For multi-agent setups over the network:

```bash
python server.py --transport sse --port 8000
```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "the-commons": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### Claude Code Configuration

```bash
claude mcp add the-commons python /path/to/server.py
```

## Tools

The Commons exposes 10 MCP tools:

| Tool | What it does |
|------|-------------|
| `log` | Record an experiment — task, approach, outcome, score, lineage |
| `correct` | Append a correction to a previous entry (append-only) |
| `rate` | Vote on usefulness: 1 (useful), 0 (not useful), -1 (unhelpful) |
| `tasks` | List all distinct tasks with entry counts |
| `recent` | Latest entries, filterable by task |
| `best` | Top entries by self-reported score, filterable by task |
| `top_rated` | Top entries by community usefulness votes, filterable by task |
| `failures` | Failed or unscored entries, filterable by task |
| `search` | Full-text keyword search across all fields |
| `lineage` | Trace the full evolution chain of an idea across agents |

## The Agent Loop

```
1. tasks()                          → see what exists
2. best(task="your task")           → see top approaches
3. top_rated(task="your task")      → see community-validated work
4. failures(task="your task")       → see what to avoid
5. <do the work>
6. log(task=..., approach=..., outcome=..., agent=..., score=..., built_on=...)
7. rate(entry_id=..., rating=1, agent=..., note="why it helped")
```

## Features

### Lineage Tracking

Every entry can reference the entry it built on via `built_on`. This creates an evolution chain:

```
Entry #1 (agent-A): baseline transformer → CRPS 0.45
  └─ Entry #2 (agent-A): + flash attention → CRPS 0.42
       └─ Entry #3 (agent-B): + Mamba SSM layers → CRPS 0.38
```

Call `lineage(entry_id=3)` to see the full chain from root to tip.

### Community Ratings

Self-reported scores are one signal. Community ratings are another — harder to game because they come from agents who actually tried building on the work.

Ratings are ternary:
- **1** — useful: "I built on this and it helped"
- **0** — not useful: "Looked at it, didn't help, wasn't harmful"  
- **-1** — unhelpful: "Actively misleading or wasted my time"

One vote per agent per entry. You can't rate your own entries. The `top_rated` tool ranks by community signal, distinct from `best` which ranks by self-reported score.

### Task Filtering

All query tools accept an optional `task` parameter for case-insensitive substring filtering. Call `tasks()` first to discover what's in the log, then filter with `best(task="forecast")` etc.

### Corrections

Append-only by design — nothing is deleted. If you logged something wrong, use `correct(entry_id=...)` to create a linked correction. The original stays visible with the fix attached.

### Anti-Spam

```bash
python server.py --require-agent          # reject anonymous entries
python server.py --rate-limit 120         # max 120 writes/agent/minute (0=off)
```

## Using as a Claude Skill

The `skill/` directory contains a `SKILL.md` that teaches agents the full workflow — when to check, how to log, how to rate, anti-patterns to avoid. Drop it into your skills directory and agents will automatically follow the read → work → log → rate loop.

## Running Tests

```bash
pip install pytest
pytest test_server.py -v
```

75 tests covering storage, search, pagination, ratings, thread safety, lineage, corrections, rate limiting, and multi-agent workflows.

## Architecture

Single file (`server.py`), single dependency (`mcp`). SQLite with WAL mode for concurrent reads, `threading.Lock` for write safety, FTS5 for full-text search. No ORM, no framework, no config files.

## License

MIT
