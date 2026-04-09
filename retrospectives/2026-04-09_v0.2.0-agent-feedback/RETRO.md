# Retrospective: v0.2.0 — Agent Feedback Improvements

**Date:** 2026-04-09
**Effort:** Implement three improvements identified by an agent that used greploom in a real extraction workflow
**Issues:** #25, #26, #27, treeloom#74
**Commits:** 938b327..993019c (v0.1.0 → v0.2.0)

## What We Set Out To Do

An agent provided structured feedback after using greploom for spec extraction on jsoup and dateutil codebases. Three improvements were identified:

1. **`--include-source`** — Raw source lines in query output (agent needed actual code, not just summaries)
2. **`--node` query mode** — Direct node ID lookup bypassing search (agent already knew the node, didn't want to re-search)
3. **Embedding model metadata** — Know which model built the index, warn on mismatches

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| `--include-source` pushed to treeloom as CPG enhancement | Good pivot | Source text belongs in the CPG, not greploom. Keeps "treeloom parses, greploom searches" principle intact. |
| `--node` JSON output is a bare list (no metadata envelope) | Good pivot | `--node` mode doesn't open the index store, so metadata isn't available. Cleaner than faking empty metadata. |
| README "enhanced tier" description corrected | Missed (pre-existing) | README said "docstrings" but the summarizer has never included docstrings. Wrong since v0.1.0, caught during release audit. |
| Review sub-agent prematurely wrote "Version 0.1.1" in changelog | Missed | Sub-agents shouldn't assign version numbers — that's a release decision. Caught and corrected before release. |

## What Went Well

- **Discussion before implementation.** Analyzing the source text question before coding prevented building the wrong thing. The CPG/search separation was preserved by pushing source spans upstream.
- **Parallel sub-agents.** Three implementation agents ran concurrently. The metadata agent landed changes cleanly on top of the `--node` agent's work with no conflicts in shared files (`query_cmd.py`, `server.py`, `test_integration.py`).
- **Multi-pass documentation audit.** First pass missed llms.txt, llms-full.txt, CLAUDE.md file tree, and the enhanced tier error. Second pass caught all of them. The release process's insistence on auditing docs caught real drift.
- **Clean release pipeline.** 194 tests green across Python 3.10/3.11/3.12, lint clean, secrets scan clean, PyPI publish succeeded on first attempt.
- **Real user feedback as input.** Starting from actual agent usage (not hypothetical improvements) meant every feature had a clear use case and could be validated against the stated need.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| No integration test for model mismatch warning | Follow-up | #28 |
| Review sub-agent wrote premature version number in changelog | Process | Note in retro patterns; adjust sub-agent prompts going forward |
| Issue #27 still open (source text) | Accept | Blocked on treeloom#74, correct upstream dependency |
| `llm` summary tier accepted by config but undocumented | Accept (pre-existing) | Not introduced by this effort; tracked as known debt |

## Action Items

- [x] File issue for mismatch warning test (#28)
- [ ] #28 — Add integration test for embedding model mismatch warning
- [ ] treeloom#74 — Implement source text spans in CPG (upstream)
- [ ] #27 — Surface source text in greploom once treeloom#74 lands

## Patterns

**Start:**
- When delegating to review sub-agents, explicitly instruct them not to write version numbers or changelog entries. Version assignment is a release-time decision.
- Run the docs audit twice on any release — the first pass consistently misses llms.txt and llms-full.txt.

**Stop:**
- Nothing identified to stop. The discussion-first, parallel-agent workflow worked well for this scope.

**Continue:**
- Starting from real user/agent feedback rather than speculative improvements.
- The discuss → plan → parallel implement → review → audit → release pipeline. Each phase caught issues the previous one missed.
- Keeping features scoped to the right layer (pushing source text to treeloom rather than bolting it onto greploom).
