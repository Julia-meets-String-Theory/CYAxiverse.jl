#!/usr/bin/env python3
"""Run the Python bridge and visible-sector regressions from the repository root.

The scientific-data modules are executable scripts by design.  Run their
unittest modules with ``scripts/`` as the import root and current directory so
direct CLI imports and package-level discovery cannot silently select different
code.

Each module is loaded with ``TestLoader.discover`` (one file per call), not
``python -m unittest <file>``: discovery reports an import-time
``unittest.SkipTest`` (for example, CYTools being absent) as a skipped test,
whereas loading by name lets it crash the whole run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_TESTS = (
    "test_build_orientifold_axion_database.py",
    "test_qed_divisor_assignment.py",
    "test_orientifold_population_preflight.py",
)

_CHILD = """\
import sys
import unittest

scripts, tests = sys.argv[1], sys.argv[2:]
loader = unittest.TestLoader()
suites = {
    name: loader.discover(start_dir=scripts, pattern=name, top_level_dir=scripts)
    for name in tests
}
missing = [name for name, suite in suites.items() if not suite.countTestCases()]
if missing:
    print("No tests found in: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
suite = unittest.TestSuite(suites.values())
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def main(scripts: Path | None = None, tests: tuple[str, ...] | None = None) -> int:
    """Run ``tests`` (default: the regression set) with ``scripts`` as import root.

    Both arguments exist so the runner can be tested against a temporary
    directory; the defaults are the repository's ``scripts/`` and test list.
    """
    scripts = Path(__file__).resolve().parent if scripts is None else Path(scripts).resolve()
    tests = DEFAULT_TESTS if tests is None else tuple(tests)
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, str(scripts), *tests],
        cwd=scripts,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scripts-dir", type=Path, default=None,
                        help="import root and cwd (default: this file's directory)")
    parser.add_argument("tests", nargs="*", help="test files (default: the regression set)")
    ns = parser.parse_args()
    raise SystemExit(main(ns.scripts_dir, tuple(ns.tests) or None))
