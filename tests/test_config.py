from __future__ import annotations

import pytest

from greploom.config import GrepLoomConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_values():
    cfg = GrepLoomConfig()
    assert cfg.embedding_url == "http://localhost:11434"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.db_path == ".greploom/index.db"
    assert cfg.token_budget == 8192
    assert cfg.summary_tier == "enhanced"


# ---------------------------------------------------------------------------
# from_env — full override
# ---------------------------------------------------------------------------


def test_from_env_all_vars(monkeypatch):
    monkeypatch.setenv("GREPLOOM_EMBEDDING_URL", "http://remote:11434")
    monkeypatch.setenv("GREPLOOM_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("GREPLOOM_DB_PATH", "/tmp/test.db")
    monkeypatch.setenv("GREPLOOM_TOKEN_BUDGET", "4096")
    monkeypatch.setenv("GREPLOOM_SUMMARY_TIER", "fast")

    cfg = GrepLoomConfig.from_env()

    assert cfg.embedding_url == "http://remote:11434"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.db_path == "/tmp/test.db"
    assert cfg.token_budget == 4096
    assert cfg.summary_tier == "fast"


# ---------------------------------------------------------------------------
# from_env — partial overrides fall back to defaults
# ---------------------------------------------------------------------------


_ALL_ENV_KEYS = [
    "GREPLOOM_EMBEDDING_URL",
    "GREPLOOM_EMBEDDING_MODEL",
    "GREPLOOM_DB_PATH",
    "GREPLOOM_TOKEN_BUDGET",
    "GREPLOOM_SUMMARY_TIER",
]


@pytest.mark.parametrize(
    "env_key, env_val, attr, expected",
    [
        ("GREPLOOM_EMBEDDING_URL", "http://other:11434", "embedding_url", "http://other:11434"),
        ("GREPLOOM_EMBEDDING_MODEL", "custom-model", "embedding_model", "custom-model"),
        ("GREPLOOM_DB_PATH", "/data/index.db", "db_path", "/data/index.db"),
        ("GREPLOOM_TOKEN_BUDGET", "512", "token_budget", 512),
        ("GREPLOOM_SUMMARY_TIER", "llm", "summary_tier", "llm"),
    ],
)
def test_from_env_partial_override(monkeypatch, env_key, env_val, attr, expected):
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_key, env_val)
    cfg = GrepLoomConfig.from_env()
    assert getattr(cfg, attr) == expected


# ---------------------------------------------------------------------------
# Validation — summary_tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_tier", ["", "FAST", "medium", "llm-plus", "auto"])
def test_invalid_summary_tier_raises(bad_tier):
    with pytest.raises(ValueError, match="summary_tier"):
        GrepLoomConfig(summary_tier=bad_tier)


def test_invalid_summary_tier_from_env_raises(monkeypatch):
    monkeypatch.setenv("GREPLOOM_SUMMARY_TIER", "bogus")
    with pytest.raises(ValueError, match="summary_tier"):
        GrepLoomConfig.from_env()


# ---------------------------------------------------------------------------
# Validation — token_budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_budget", [0, -1, -1000])
def test_non_positive_token_budget_raises(bad_budget):
    with pytest.raises(ValueError, match="token_budget"):
        GrepLoomConfig(token_budget=bad_budget)


def test_non_positive_token_budget_from_env_raises(monkeypatch):
    monkeypatch.setenv("GREPLOOM_TOKEN_BUDGET", "0")
    with pytest.raises(ValueError, match="token_budget"):
        GrepLoomConfig.from_env()


def test_non_integer_token_budget_from_env_raises(monkeypatch):
    monkeypatch.setenv("GREPLOOM_TOKEN_BUDGET", "lots")
    with pytest.raises(ValueError, match="GREPLOOM_TOKEN_BUDGET"):
        GrepLoomConfig.from_env()


# ---------------------------------------------------------------------------
# Valid summary tiers
# ---------------------------------------------------------------------------


def test_empty_db_path_raises():
    with pytest.raises(ValueError, match="db_path"):
        GrepLoomConfig(db_path="")


@pytest.mark.parametrize("tier", ["fast", "enhanced", "llm"])
def test_valid_summary_tiers(tier):
    cfg = GrepLoomConfig(summary_tier=tier)
    assert cfg.summary_tier == tier
