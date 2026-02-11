from __future__ import annotations

# Compatibilidade temporaria: mantenha importando de rna3d_local.bigdata.
from .bigdata import (  # noqa: F401
    DEFAULT_MAX_ROWS_IN_MEMORY,
    DEFAULT_MEMORY_BUDGET_MB,
    assert_memory_budget,
    assert_row_budget,
    current_rss_mb,
)

__all__ = [
    "DEFAULT_MAX_ROWS_IN_MEMORY",
    "DEFAULT_MEMORY_BUDGET_MB",
    "assert_memory_budget",
    "assert_row_budget",
    "current_rss_mb",
]
