"""Skip helpers for Python tests that cannot run in the portable CI environment."""

from __future__ import annotations

import importlib.util
import unittest


def require_cytools() -> None:
    """Skip the importing test module when CYTools is not installed (local-only tests)."""
    if importlib.util.find_spec("cytools") is None:
        raise unittest.SkipTest("requires CYTools; local-only test")
