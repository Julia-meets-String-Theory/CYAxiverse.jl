#!/usr/bin/env python3
"""Focused tests for scripts/run_python_regression_tests.py.

Runs the runner as a child process against a temporary scripts directory so
the repository's own regression list is never edited.  Does not invoke Julia,
CYTools, network access, or the scientific suite.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_python_regression_tests.py"
)

SKIPPING = """\
import unittest
raise unittest.SkipTest("optional dependency absent")
"""

PASSING = """\
import unittest


class PassingTests(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(True)
"""

BROKEN_IMPORT = """\
import a_module_that_does_not_exist
"""

FAILING = """\
import unittest


class FailingTests(unittest.TestCase):
    def test_fails(self):
        self.fail("deliberate failure")
"""


def _run_runner(modules, listed=None):
    """Write ``modules`` (path -> source) to a temp dir and run the runner on it.

    ``listed`` is the list of test files passed to the runner; by default the
    top-level files of ``modules``.
    """
    if listed is None:
        listed = [name for name in modules if "/" not in name]
    with tempfile.TemporaryDirectory(prefix="rprt_test_") as tmp:
        for name, source in modules.items():
            path = os.path.join(tmp, *name.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(textwrap.dedent(source))
        result = subprocess.run(
            [sys.executable, "-B", RUNNER, "--scripts-dir", tmp, *listed],
            capture_output=True, text=True, shell=False,
        )
    return result.returncode, result.stdout + result.stderr


class RunPythonRegressionTests(unittest.TestCase):
    def test_import_time_skip_is_reported_as_skip_and_run_passes(self):
        rc, out = _run_runner(
            {"test_skipping.py": SKIPPING, "test_passing.py": PASSING}
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("optional dependency absent", out)
        self.assertIn("skipped=1", out)
        self.assertIn("test_passes", out)
        self.assertNotIn("Traceback", out)

    def test_failing_module_gives_nonzero_exit(self):
        rc, out = _run_runner(
            {"test_skipping.py": SKIPPING, "test_failing.py": FAILING}
        )
        self.assertNotEqual(rc, 0, out)
        self.assertIn("deliberate failure", out)

    def test_missing_module_gives_nonzero_exit_and_names_the_file(self):
        rc, out = _run_runner({"test_passing.py": PASSING}, listed=["test_absent.py"])
        self.assertNotEqual(rc, 0, out)
        self.assertIn("test_absent.py", out)
        self.assertIn("not found", out)

    def test_non_skip_import_error_fails_and_names_the_error(self):
        rc, out = _run_runner(
            {"test_broken.py": BROKEN_IMPORT, "test_passing.py": PASSING}
        )
        self.assertNotEqual(rc, 0, out)
        self.assertIn("a_module_that_does_not_exist", out)
        self.assertIn("test_broken", out)

    def test_module_without_tests_fails_with_message(self):
        rc, out = _run_runner({"test_empty.py": "x = 1\n"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("test_empty.py", out)
        self.assertIn("No tests found", out)

    def test_packages_under_scripts_are_not_imported(self):
        rc, out = _run_runner(
            {
                "test_passing.py": PASSING,
                "pkg/__init__.py": "raise RuntimeError('package import must not happen')\n",
                "pkg/test_passing.py": PASSING,
            },
            listed=["test_passing.py"],
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("Ran 1 test in", out)
        self.assertNotIn("package import must not happen", out)
        self.assertNotIn("__pycache__", out)


if __name__ == "__main__":
    unittest.main()
