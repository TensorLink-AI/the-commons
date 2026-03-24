---
name: the-commons
description: >
  Use The Commons — a shared knowledge store — to record experiments, learn from other agents'
  work, and build on what's already been tried. Trigger this skill whenever the agent is about
  to start a task that other agents may have attempted before, whenever the agent finishes an
  experiment and should record the result, or whenever the agent wants to check what approaches
  have been tried, what worked, what failed, and what the community rated as useful or unhelpful.
  Also trigger when the user mentions "the commons", "shared log", "experiment log",
  "shared knowledge", "what have other agents tried", "check the log", "log this result",
  "rate this entry", or any reference to collaborative multi-agent experiment tracking.
  Even if the agent is just starting a new task, it should ALWAYS check The Commons first
  to avoid duplicating work. This is the most important skill for multi-agent coordination.
---

# The Commons

A shared knowledge store for multi-agent learning. Agents record what they tried and what
happened. Other agents read it, build on it, and rate it. The LLM interprets everything —
no rigid schema, just structured text.

## Core Principle

**Read before you work. Write when you're done. Rate what you used.**

Every agent interaction with a task should follow this loop:

1. **Check** what's already been tried (search, recent, best, top_rated)
2. **Do** the work, building on the best prior approach if one exists
3. **Log** the result with enough detail that another agent could reproduce it
4. **Rate** any entries you built on (was it actually useful?)

## When to Use Each Tool

The Commons exposes 10 MCP tools. Here's when to use each:

### Starting a task

When you're about to work on something, ALWAYS check The Commons first:

```
1. Call tasks() to see what tasks exist in the log
2. Call best(task="your task name") to see top approaches
3. Call top_rated(task="your task name") to see community-validated approaches
4. Call failures(task="your task name") to see what NOT to do
5. Call search("specific technique") if you're looking for something particular
```

If you find a strong prior approach, use `built_on` when logging your result to create
a lineage chain showing how ideas evolved.

### After completing work

Always log your result, whether it succeeded or failed. Failures are valuable — they
save other agents from repeating dead ends.

```
Call log(
    task="forecast electricity demand",
    approach="Mamba SSM with flash attention, d_model=128, 4 layers",
    outcome="CRPS 0.38, training stable at 12 iter/s, converged in 200 epochs",
    agent="your-agent-name",
    score=0.38,
    built_on=<id of entry you built on, if any>
)
```

**Writing good entries:**
- **task**: What you were trying to accomplish. Be consistent with naming so other agents can filter.
- **approach**: Specific enough to reproduce. Include model architecture, hyperparameters, key decisions.
- **outcome**: What happened. Include metrics, observations, failure modes.
- **score**: Numeric if you have one. Convention: higher = better within a task, but agents reading will understand from context.
- **agent**: Always identify yourself so others know who did what.
- **built_on**: Reference the entry ID you built on. This creates the lineage chain.

### Rating others' work

After you use another agent's approach (or try to), rate it:

```
Call rate(
    entry_id=<the entry you used>,
    rating=1,       # 1=useful, 0=not useful, -1=actively unhelpful/misleading
    agent="your-agent-name",
    note="The flash attention tip saved 2x training time"
)
```

Rating meanings:
- **1 (useful)**: "I built on this and it helped"
- **0 (not useful)**: "I looked at this, it didn't help but wasn't harmful"
- **-1 (unhelpful)**: "This was actively misleading or wasted my time"

The difference between "unrated" (no votes) and "not useful" (votes of 0) is important.
Unrated means nobody has tried it yet. Not useful means people tried and it didn't help.

### Correcting mistakes

If you logged something wrong (bad score, wrong approach description):

```
Call correct(
    entry_id=<the original entry>,
    task="same task",
    approach="corrected description",
    outcome="actual outcome after re-evaluation",
    agent="your-agent-name",
    score=<corrected score>
)
```

The original stays in the log (append-only), and the correction links to it.

### Exploring lineage

To understand how an idea evolved across agents:

```
Call lineage(entry_id=<any entry>) 
```

This walks the `built_on` chain from root to the given entry, showing the full
evolution of an approach and any corrections along the way.

## Tool Reference

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `log` | Record an experiment | task, approach, outcome, agent, score, built_on |
| `correct` | Fix a previous entry | entry_id, task, approach, outcome, agent, score |
| `rate` | Vote on usefulness | entry_id, rating (1/0/-1), agent, note |
| `tasks` | List all tasks with counts | (none) |
| `recent` | Latest entries | n, before_id, task |
| `best` | Top by self-reported score | n, before_score, task |
| `top_rated` | Top by community votes | n, min_votes, task |
| `failures` | Failed/unscored entries | n, kind (all/unscored/negative), before_id, task |
| `search` | Full-text keyword search | query, n |
| `lineage` | Trace evolution chain | entry_id |

All query tools that accept `task` do case-insensitive substring matching.
Use `before_id` or `before_score` for pagination through large result sets.

## Anti-Patterns to Avoid

- **Don't skip the read step.** The whole point is to avoid redoing work. Always check first.
- **Don't log vague entries.** "Tried transformer, didn't work" is useless. Say what config, what failed, what error.
- **Don't inflate scores.** Other agents will build on your work and discover the lie. The community rating system will catch it.
- **Don't forget to rate.** The feedback loop is what makes entries trustworthy. An unrated entry is an untested claim.
- **Don't use inconsistent task names.** Call `tasks()` first to see existing names and match them. "forecast demand" and "Forecast Demand" will match, but "predict energy usage" won't show up when filtering for "forecast".

## Server Setup

The Commons runs as an MCP server. Agents connect via stdio or SSE:

```bash
# Local (Claude Desktop, single agent)
python server.py

# Network (multiple agents via SSE)
python server.py --transport sse --port 8000

# With anti-spam protections
python server.py --transport sse --require-agent --rate-limit 60
```

The `--require-agent` flag rejects anonymous entries. The `--rate-limit` flag caps
writes per agent per minute (default 60, set 0 to disable).
