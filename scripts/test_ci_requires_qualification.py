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

    def test_qualification_inputs_and_gate_wiring_require_qualification(self) -> None:
        # The qualification job runs run_fixture.jl, which includes
        # test/coverage_support.jl and replays through
        # scripts/verify_scientific_certificate.jl in the root project environment;
        # the gate script and CI.yml decide whether the job runs at all.
        for path in ("Project.toml", "Manifest.toml", "scripts/verify_scientific_certificate.jl",
                     "test/coverage_support.jl", "scripts/ci_requires_qualification.py",
                     ".github/workflows/CI.yml", "LocalPreferences.toml", ".gitignore",
                     "JuliaLocalPreferences.toml", "JuliaProject.toml", "JuliaManifest.toml",
                     "Manifest-v1.12.toml", "JuliaManifest-v1.13.toml"):
            with self.subTest(path=path):
                self.assertTrue(self.gate.requires_qualification([path]))

    def test_other_paths_do_not(self) -> None:
        for path in ("docs/src/api.md", "test/runtests.jl", "scripts/local_only.py",
                     "test/fixtures/other/data.h5", "srcs/x.jl", "test/fixtures/scientific",
                     "docs/Project.toml", "notebooks/Manifest.toml", "Project.toml.orig",
                     "notebooks/LocalPreferences.toml", "LocalPreferences.toml.orig",
                     "docs/.gitignore", "docs/JuliaLocalPreferences.toml",
                     "docs/JuliaProject.toml", "docs/Manifest-v1.12.toml",
                     "Manifest-v1.toml", "Manifest-vX.Y.toml", "Manifest-v1.12.toml.orig",
                     ".github/workflows/Documentation.yml"):
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

    def _commit_to_base(self, path: str) -> None:
        """Commit path in the base, so that a later move has a file to rename."""
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Distinct content per path, so Git's rename detection (when on) pairs them.
        target.write_text(f"base content of {path}\n" * 5)
        self._git("add", ".")
        self._git("commit", "-m", f"base {path}")
        self.base = self._git("rev-parse", "HEAD").strip()

    def _move(self, old: str, new: str) -> str:
        """Move old to new with git mv in one commit and return that commit."""
        (self.root / new).parent.mkdir(parents=True, exist_ok=True)
        self._git("mv", old, new)
        self._git("commit", "-m", f"move {old} to {new}")
        return self._git("rev-parse", "HEAD").strip()

    def _assert_gate(self, head: str, expected: str) -> None:
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, f"qualification={expected}\n")

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

    def test_local_preferences_change_sets_output_true(self) -> None:
        head = self._commit("LocalPreferences.toml")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=true\n")

    def test_julia_local_preferences_change_sets_output_true(self) -> None:
        head = self._commit("JuliaLocalPreferences.toml")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=true\n")

    def test_versioned_manifest_change_sets_output_true(self) -> None:
        head = self._commit("Manifest-v1.12.toml")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=true\n")

    def test_root_gitignore_change_sets_output_true(self) -> None:
        head = self._commit(".gitignore")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=true\n")

    def test_nested_gitignore_change_sets_output_false(self) -> None:
        head = self._commit("docs/.gitignore")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=false\n")

    def test_documentation_change_sets_output_false(self) -> None:
        head = self._commit("docs/src/index.md")
        result, output = self._run(head)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, "qualification=false\n")

    # R-007: a rename lists the old path as deleted and the new path as added, so
    # moving an input to a path that is not an input (or the reverse) still counts.
    def test_moving_root_gitignore_away_sets_output_true(self) -> None:
        self._commit_to_base(".gitignore")
        self._assert_gate(self._move(".gitignore", "gitignore.old"), "true")

    def test_moving_source_file_out_of_src_sets_output_true(self) -> None:
        self._commit_to_base("src/Fixture.jl")
        self._assert_gate(self._move("src/Fixture.jl", "attic/Fixture.jl"), "true")

    def test_moving_a_file_onto_local_preferences_sets_output_true(self) -> None:
        self._assert_gate(self._move("README.md", "LocalPreferences.toml"), "true")

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

    def test_pull_requests_to_bases_other_than_vmm_run_qualification_when_the_workflow_runs(self) -> None:
        # The path gate runs only for PRs to vmm; a PR to main, a maintenance line
        # or a stacked branch skips it, so the job runs qualification directly
        # whenever the workflow runs (the pull_request paths filter decides that).
        qualification = _job_block(self.workflow, "qualification")
        self.assertIn("github.base_ref != 'vmm'", qualification)
        self.assertNotIn("github.base_ref == 'main'", qualification)

    def test_python_test_requirements_trigger_pull_request_ci(self) -> None:
        # The lifecycle job installs requirements-test.txt, so a PR changing only
        # that file must still run the workflow.
        trigger = self.workflow[self.workflow.index("  pull_request:\n"):
                                self.workflow.index("  workflow_dispatch:")]
        self.assertIn("      - 'requirements-test.txt'\n", trigger)

    def test_local_preferences_trigger_pull_request_ci(self) -> None:
        # Julia applies LocalPreferences.toml sections for packages the root
        # project lists, so a PR changing only that file must run the workflow.
        trigger = self.workflow[self.workflow.index("  pull_request:\n"):
                                self.workflow.index("  workflow_dispatch:")]
        self.assertIn("      - 'LocalPreferences.toml'\n", trigger)

    def test_julia_local_preferences_trigger_pull_request_ci(self) -> None:
        # When present, Julia reads the root JuliaLocalPreferences.toml in place of
        # LocalPreferences.toml (Base.preferences_names), so a PR changing only that
        # file must run the workflow.
        trigger = self.workflow[self.workflow.index("  pull_request:\n"):
                                self.workflow.index("  workflow_dispatch:")]
        self.assertIn("      - 'JuliaLocalPreferences.toml'\n", trigger)

    def test_alternative_project_and_manifest_names_trigger_pull_request_ci(self) -> None:
        # Julia 1.12 reads these root files in place of Project.toml and
        # Manifest.toml when present (Base.project_names, Base.manifest_names). In
        # GitHub's filter * does not match /, so these entries are root only.
        trigger = self.workflow[self.workflow.index("  pull_request:\n"):
                                self.workflow.index("  workflow_dispatch:")]
        for entry in ("JuliaProject.toml", "JuliaManifest.toml", "Manifest-v*.toml",
                      "JuliaManifest-v*.toml"):
            with self.subTest(entry=entry):
                self.assertIn(f"      - '{entry}'\n", trigger)

    def test_root_gitignore_triggers_pull_request_ci(self) -> None:
        # The qualification job requires a clean tree, and the root .gitignore
        # hides the root Manifest.toml that CI resolves, so a PR changing only
        # that file must run the workflow.
        trigger = self.workflow[self.workflow.index("  pull_request:\n"):
                                self.workflow.index("  workflow_dispatch:")]
        self.assertIn("      - '.gitignore'\n", trigger)

    def test_job_comment_states_when_qualification_runs(self) -> None:
        # Pushes and manual runs have no paths filter, so every push run and every
        # manual run runs qualification; the filter belongs to pull_request only.
        events = self.workflow[self.workflow.index("  push:\n"):
                               self.workflow.index("  pull_request:\n")]
        # "paths" also covers paths-ignore, which would filter pushes as well.
        self.assertNotIn("paths", events)
        # A PR to another base runs it only when the workflow runs, which the
        # pull_request paths filter decides; the comment must say so.
        qualification = _job_block(self.workflow, "qualification")
        self.assertIn("Pushes and manual runs always run", qualification)
        self.assertIn("pull_request paths filter", qualification)
        self.assertNotIn("Pushes and PRs to any other base", qualification)


if __name__ == "__main__":
    unittest.main()
