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

FAILING = """\
import unittest


class FailingTests(unittest.TestCase):
    def test_fails(self):
        self.fail("deliberate failure")
"""


def _run_runner(modules):
    """Write ``modules`` (name -> source) to a temp dir and run the runner on it."""
    with tempfile.TemporaryDirectory(prefix="rprt_test_") as tmp:
        for name, source in modules.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(textwrap.dedent(source))
        result = subprocess.run(
            [sys.executable, "-B", RUNNER, "--scripts-dir", tmp, *modules],
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

    def test_missing_module_gives_nonzero_exit(self):
        with tempfile.TemporaryDirectory(prefix="rprt_test_") as tmp:
            result = subprocess.run(
                [sys.executable, "-B", RUNNER, "--scripts-dir", tmp,
                 "test_absent.py"],
                capture_output=True, text=True, shell=False,
            )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
