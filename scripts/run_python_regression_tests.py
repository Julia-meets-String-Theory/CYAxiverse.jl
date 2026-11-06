#!/usr/bin/env python3
"""Run the Python bridge and visible-sector regressions from the repository root.

The scientific-data modules are executable scripts by design.  Run their
unittest modules with ``scripts/`` as the import root and current directory so
direct CLI imports and package-level discovery cannot silently select different
code.

Each listed file is imported by its stem, and only that module is loaded.
Loading by name (``python -m unittest <file>``) lets an import-time
``unittest.SkipTest`` (for example, CYTools being absent) crash the whole run,
and ``TestLoader.discover`` would also import every package under ``scripts/``.
Here an import-time ``SkipTest`` becomes one skipped test, and any other import
error becomes one failing test that shows the original traceback.
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
import importlib
import os
import sys
import traceback
import unittest

scripts, tests = sys.argv[1], sys.argv[2:]
sys.path.insert(0, scripts)


def placeholder(kind, stem, body):
    # One synthetic test per module that could not be loaded; its name
    # identifies the module in the output.
    def run(self):
        body(self)

    run.__name__ = stem
    return type(kind, (unittest.TestCase,), {stem: run})(stem)


loader = unittest.TestLoader()
suite = unittest.TestSuite()
problems = []
for name in tests:
    path = os.path.join(scripts, name)
    stem = os.path.splitext(os.path.basename(name))[0]
    if not os.path.isfile(path):
        problems.append(f"Test file not found: {name}")
        continue
    try:
        module = importlib.import_module(stem)
    except unittest.SkipTest as exc:
        reason = str(exc)
        suite.addTest(placeholder(
            "ModuleSkipped", stem, lambda self, reason=reason: self.skipTest(reason)))
        continue
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        text = "".join(traceback.format_exception(exc))

        def fail(self, stem=stem, text=text):
            self.fail(f"Import of {stem} failed:\\n{text}")

        suite.addTest(placeholder("ModuleImportError", stem, fail))
        continue
    module_suite = loader.loadTestsFromModule(module)
    if not module_suite.countTestCases():
        problems.append(f"No tests found in: {name}")
        continue
    suite.addTest(module_suite)
for problem in problems:
    print(problem, file=sys.stderr)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and not problems else 1)
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
