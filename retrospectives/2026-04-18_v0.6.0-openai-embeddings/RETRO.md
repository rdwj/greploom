# Retrospective: v0.6.0 — OpenAI-compatible Embedding Endpoints

**Date:** 2026-04-18
**Effort:** Add `--embedding-url` flag for OpenAI `/v1/embeddings` protocol support
**Issues:** #31
**Commits:** 5fe0ceb..2908737 (v0.5.0 -> v0.6.0)

## What We Set Out To Do

Issue #31 requested support for OpenAI-compatible embedding endpoints (vLLM, OpenAI, etc.) alongside the existing Ollama support. The issue was well-specified: API diff documented, live endpoint verified, mutual exclusion with `--ollama-url` defined.

## What Changed

| Change | Type | Rationale |
|--------|------|-----------|
| Added `embedding_provider` config field + env var | Good pivot | Issue only mentioned CLI flag; a config field makes env-var-only workflows possible |
| Genericized error messages (removed "Is ollama running?") | Good pivot | Review sub-agent identified that error messages shouldn't assume server type |
| Added `RuntimeError` catch in CLI error handler | Good pivot | Review found HTTP errors would surface as tracebacks rather than clean messages |
| `--ollama-url` explicitly resets provider to "ollama" | Good pivot | Review found edge case: env `GREPLOOM_EMBEDDING_PROVIDER=openai` + CLI `--ollama-url` would use wrong protocol |

No scope deferrals or missed requirements relative to the issue.

## What Went Well

- **Well-written issue** with a clear API diff, verified endpoint, and motivation made planning trivial. Zero ambiguity in requirements.
- **Review sub-agent found real edge cases.** The provider-reset bug and RuntimeError catch were both legitimate issues that would have affected users. The parallel implement + review pattern continues to pay off.
- **Live validation** against the actual vLLM endpoint on OpenShift confirmed the implementation works end-to-end before release.
- **Clean release pipeline.** 246 tests green across Python 3.10/3.11/3.12, lint clean, secrets clean, PyPI publish succeeded first try.
- **Tight scope.** One feature, three commits (feat, docs, release). No scope creep.

## Gaps Identified

| Gap | Severity | Resolution |
|-----|----------|------------|
| Ruff line-length violations in initial implementation | Minor / Fixed | Caught during release pre-flight lint check |
| `greploom query` doesn't have `--embedding-url` (only index) | Follow-up | #32 |
| No `--embedding-dim` for non-768-dim models | Follow-up | #33 |

## Action Items

- [x] File issue for query `--embedding-url` symmetry (#32)
- [x] File issue for `--embedding-dim` support (#33)

## Patterns (across retros)

**Start:**
- Nothing new identified.

**Stop:**
- Nothing identified.

**Continue:**
- The discuss -> plan -> parallel implement -> review -> release pipeline. Review sub-agent caught four real issues this session (same pattern as v0.2.0 where review caught premature version numbering).
- Running the release docs audit thoroughly. This session caught stale llms-full.txt references, consistent with the v0.2.0 finding that the first pass misses llms files.
- Well-scoped issues with verified examples lead to clean, fast sessions.

**Recurring pattern (2 retros):**
- Docs audit consistently catches stale references in llms-full.txt. This file mirrors README content but doesn't get updated until release time. Consider whether llms-full.txt should be generated from README rather than maintained separately.
