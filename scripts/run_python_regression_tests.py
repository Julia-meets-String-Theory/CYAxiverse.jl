#!/usr/bin/env python3
"""Run the Python bridge and visible-sector regressions from the repository root.

The scientific-data modules are executable scripts by design.  Run their
unittest modules with ``scripts/`` as the import root so direct CLI imports
and package-level discovery cannot silently select different code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    scripts = Path(__file__).resolve().parent
    tests = (
        "test_build_orientifold_axion_database.py",
        "test_qed_divisor_assignment.py",
        "test_orientifold_population_preflight.py",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *tests],
        cwd=scripts,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
