#!/usr/bin/env python3
"""Focused tests for scripts/run_python_regression_tests.py.

Runs the runner as a child process against a temporary scripts directory so
the repository's own regression list is never edited.  Does not invoke Julia,
CYTools, network access, or the scientific suite.
"""

import os
import re
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


def _run_runner(modules, listed=None, inspect=None):
    """Write ``modules`` (path -> source) to a temp dir and run the runner on it.

    ``listed`` is the list of test files passed to the runner; by default the
    top-level files of ``modules``.  ``inspect``, if given, is called with the
    temporary directory after the run, before it is removed.
    """
    if listed is None:
        listed = [name for name in modules if "/" not in name]
    with tempfile.TemporaryDirectory(prefix="rprt_test_") as tmp:
        for name, source in modules.items():
            path = os.path.join(tmp, *name.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(textwrap.dedent(source))
        # The runner starts its own child interpreter, which does not inherit
        # -B; make sure the environment does not disable bytecode either, so
        # the presence or absence of __pycache__ is meaningful evidence.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
        result = subprocess.run(
            [sys.executable, "-B", RUNNER, "--scripts-dir", tmp, *listed],
            capture_output=True, text=True, shell=False, env=env,
        )
        if inspect is not None:
            inspect(tmp)
    return result.returncode, result.stdout + result.stderr


class RunPythonRegressionTests(unittest.TestCase):
    def test_import_time_skip_is_reported_as_skip_and_run_passes(self):
        rc, out = _run_runner(
            {"test_skipping.py": SKIPPING, "test_passing.py": PASSING}
        )
        self.assertEqual(rc, 0, out)
        # The skip is reported by a placeholder test named after the module.
        self.assertRegex(
            out,
            r"(?m)^test_skipping \(\S*ModuleSkipped\.test_skipping\) "
            r"\.\.\. skipped 'optional dependency absent'$",
        )
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
        # Reported as a failing placeholder test (self.fail), named after the
        # module, whose message names the module and the original error.
        self.assertRegex(
            out,
            r"(?m)^FAIL: test_broken \(\S*ModuleImportError\.test_broken\)$",
        )
        self.assertIn("Import of test_broken failed", out)
        self.assertIn(
            "ModuleNotFoundError: No module named 'a_module_that_does_not_exist'", out
        )
        self.assertIn("FAILED (failures=1)", out)

    def test_module_without_tests_fails_with_message(self):
        rc, out = _run_runner({"test_empty.py": "x = 1\n"})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("test_empty.py", out)
        self.assertIn("No tests found", out)

    def test_packages_under_scripts_are_not_imported(self):
        def check_filesystem(tmp):
            # The runner's child interpreter writes bytecode (it is not run
            # with -B).  The top-level module was imported, so __pycache__
            # must exist there: that shows bytecode writing is enabled.  A
            # package import would write pkg/__pycache__ even when its
            # __init__ raises, so its absence shows pkg was never imported.
            self.assertTrue(os.path.isdir(os.path.join(tmp, "__pycache__")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "pkg", "__pycache__")))

        rc, out = _run_runner(
            {
                "test_passing.py": PASSING,
                "pkg/__init__.py": "raise RuntimeError('package import must not happen')\n",
                "pkg/test_passing.py": PASSING,
            },
            listed=["test_passing.py"],
            inspect=check_filesystem,
        )
        self.assertEqual(rc, 0, out)
        self.assertIn("Ran 1 test in", out)

    def test_listed_name_with_directory_is_rejected_not_shadowed(self):
        # sub/test_pass.py exists but a different top-level test_pass.py would
        # be the module imported by stem; that must not give a false pass.
        top = PASSING.replace("test_passes", "test_top_level_ran")
        sub = FAILING.replace("test_fails", "test_sub_fails")
        rc, out = _run_runner(
            {"test_pass.py": top, "sub/test_pass.py": sub},
            listed=["sub/test_pass.py"],
        )
        self.assertNotEqual(rc, 0, out)
        self.assertIn("bare file name", out)
        self.assertIn("sub/test_pass.py", out)
        self.assertNotIn("test_top_level_ran", out)
        self.assertNotIn("test_sub_fails", out)

    def test_absolute_and_parent_names_are_rejected(self):
        rc, out = _run_runner(
            {"test_pass.py": PASSING}, listed=["../test_pass.py"]
        )
        self.assertNotEqual(rc, 0, out)
        self.assertIn("bare file name", out)
        self.assertIn("../test_pass.py", out)


    def test_module_imported_from_another_file_is_rejected(self):
        # The interpreter has already imported the stdlib ``os`` at startup, so
        # importing the listed ``os.py`` by stem returns that module, not the
        # listed file.  This must not be reported as a pass.
        top = PASSING.replace("test_passes", "test_shadowed_ran")
        rc, out = _run_runner({"os.py": top})
        self.assertNotEqual(rc, 0, out)
        self.assertIn("not from the listed file: os.py", out)


if __name__ == "__main__":
    unittest.main()
