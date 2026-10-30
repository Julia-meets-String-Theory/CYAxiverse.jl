"""Path gate that requires scientific qualification on pull requests to vmm."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = Path(__file__).with_name("ci_requires_qualification.py")
WORKFLOW = ROOT / ".github" / "workflows" / "CI.yml"


def _load_gate():
    spec = importlib.util.spec_from_file_location("ci_requires_qualification", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job_block(workflow: str, job: str) -> str:
    """Return the text of one job: its key line and the more-indented lines after it."""
    lines = workflow.splitlines(keepends=True)
    start = lines.index(f"  {job}:\n")
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith("   "):
            break
        block.append(line)
    return "".join(block)


class ClassifyPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = _load_gate()

    def test_source_and_certification_fixtures_require_qualification(self) -> None:
        self.assertTrue(self.gate.requires_qualification(["src/minimizer.jl"]))
        self.assertTrue(self.gate.requires_qualification(
            ["docs/src/api.md", "test/fixtures/scientific_certification/run_fixture.jl"]))

    def test_other_paths_do_not(self) -> None:
        for path in ("docs/src/api.md", "test/runtests.jl", "scripts/local_only.py",
                     "test/fixtures/other/data.h5", "srcs/x.jl", "test/fixtures/scientific"):
            with self.subTest(path=path):
                self.assertFalse(self.gate.requires_qualification([path]))
        self.assertFalse(self.gate.requires_qualification([]))


class GitDiffFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self._git("init")
        self._git("config", "user.name", "Fixture")
        self._git("config", "user.email", "fixture@example.invalid")
        (self.root / "README.md").write_text("base\n")
        self._git("add", ".")
        self._git("commit", "-m", "base")
        self.base = self._git("rev-parse", "HEAD").strip()

    def _git(self, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout

    def _commit(self, path: str) -> str:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")
        self._git("add", ".")
        self._git("commit", "-m", f"change {path}")
        return self._git("rev-parse", "HEAD").strip()

    def _run(self, head: str) -> tuple[subprocess.CompletedProcess, str]:
        output = self.root / "github_output"
        output.write_text("")
        environment = dict(os.environ, GITHUB_OUTPUT=str(output))
        result = subprocess.run(
            [sys.executable, str(GATE), "--base", self.base, "--head", head],
            cwd=self.root, capture_output=True, text=True, env=environment,
        )
        return result, output.read_text()

    def test_source_change_sets_output_true(self) -> None:
        head = self._commit("src/Fixture.jl")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=true\n")

    def test_documentation_change_sets_output_false(self) -> None:
        head = self._commit("docs/src/index.md")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=false\n")

    def test_unknown_revision_fails(self) -> None:
        result, output = self._run("0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output, "")


class WorkflowWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_changes_job_runs_the_gate_for_pull_requests_to_vmm(self) -> None:
        changes = _job_block(self.workflow, "changes")
        self.assertIn("github.event_name == 'pull_request' && github.base_ref == 'vmm'", changes)
        self.assertIn("fetch-depth: 0", changes)
        self.assertIn("scripts/ci_requires_qualification.py", changes)
        self.assertIn("qualification: ${{ steps.gate.outputs.qualification }}", changes)

    def test_qualification_runs_on_scientific_changes_without_the_label(self) -> None:
        qualification = _job_block(self.workflow, "qualification")
        self.assertIn("needs: changes", qualification)
        self.assertIn("needs.changes.outputs.qualification == 'true'", qualification)
        # A failed gate must not silently skip qualification.
        self.assertIn("needs.changes.result == 'failure'", qualification)
        self.assertIn("!cancelled()", qualification)
        self.assertIn("contains(github.event.pull_request.labels.*.name, 'full-tests')",
                      qualification)

    def test_python_test_requirements_trigger_pull_request_ci(self) -> None:
        # The lifecycle job installs requirements-test.txt, so a PR changing only
        # that file must still run the workflow.
        trigger = self.workflow[self.workflow.index("  pull_request:\n"):
                                self.workflow.index("  workflow_dispatch:")]
        self.assertIn("      - 'requirements-test.txt'\n", trigger)


if __name__ == "__main__":
    unittest.main()
