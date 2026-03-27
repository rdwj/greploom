from __future__ import annotations

import os
from dataclasses import dataclass

_VALID_SUMMARY_TIERS = ("fast", "enhanced", "llm")


@dataclass
class GrepLoomConfig:
    embedding_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    db_path: str = ".greploom/index.db"
    token_budget: int = 8192
    summary_tier: str = "enhanced"

    def __post_init__(self) -> None:
        if self.summary_tier not in _VALID_SUMMARY_TIERS:
            raise ValueError(
                f"Invalid summary_tier {self.summary_tier!r}. "
                f"Must be one of: {', '.join(_VALID_SUMMARY_TIERS)}"
            )
        if self.token_budget <= 0:
            raise ValueError(
                f"token_budget must be a positive integer, got {self.token_budget}"
            )
        if not self.db_path:
            raise ValueError("db_path must not be empty")

    @classmethod
    def from_env(cls) -> GrepLoomConfig:
        """Build a config from environment variables, falling back to defaults."""
        defaults = cls.__dataclass_fields__  # type: ignore[attr-defined]

        token_budget_raw = os.environ.get("GREPLOOM_TOKEN_BUDGET")
        default_budget: int = defaults["token_budget"].default
        try:
            token_budget = (
                int(token_budget_raw) if token_budget_raw is not None else default_budget
            )
        except ValueError:
            raise ValueError(
                f"GREPLOOM_TOKEN_BUDGET must be an integer, got {token_budget_raw!r}"
            )

        return cls(
            embedding_url=os.environ.get(
                "GREPLOOM_EMBEDDING_URL", defaults["embedding_url"].default
            ),
            embedding_model=os.environ.get(
                "GREPLOOM_EMBEDDING_MODEL", defaults["embedding_model"].default
            ),
            db_path=os.environ.get("GREPLOOM_DB_PATH", defaults["db_path"].default),
            token_budget=token_budget,
            summary_tier=os.environ.get(
                "GREPLOOM_SUMMARY_TIER", defaults["summary_tier"].default
            ),
        )
