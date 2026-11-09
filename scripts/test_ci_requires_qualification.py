"""Path gate that requires scientific qualification on pull requests to vmm."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = Path(__file__).with_name("ci_requires_qualification.py")
WORKFLOW = ROOT / ".github" / "workflows" / "CI.yml"


def _load_gate():
    spec = importlib.util.spec_from_file_location("ci_requires_qualification", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Variables that Git reads for configuration or for the repository to use, so that an
# inherited value could change a fixture's result. GIT_CONFIG_GLOBAL and
# GIT_CONFIG_NOSYSTEM are replaced below; the others are removed. Basis: git-config(1)
# "ENVIRONMENT" (GIT_CONFIG, GIT_CONFIG_COUNT, GIT_CONFIG_KEY_<n>, GIT_CONFIG_VALUE_<n>)
# and git(1) "ENVIRONMENT VARIABLES" (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE,
# GIT_OBJECT_DIRECTORY, GIT_ALTERNATE_OBJECT_DIRECTORIES, GIT_NAMESPACE, GIT_COMMON_DIR).
# GIT_CONFIG_PARAMETERS is not documented; Git sets it for "git -c" and passes it to
# child processes (observed with Git 2.54).
_GIT_INHERITED_NAMES = ("GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS",
                        "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE",
                        "GIT_COMMON_DIR")
_GIT_INHERITED_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")


def _git_environment() -> dict[str, str]:
    """Return the environment for Git and the gate in a fixture, without user or system config.

    GIT_CONFIG_GLOBAL and GIT_CONFIG_NOSYSTEM are documented in git-config(1) (Git 2.32
    and later); without them a local setting such as diff.renames=false could change the
    result (CYAX-0222 G1). Configuration passed in the environment (GIT_CONFIG, and
    GIT_CONFIG_COUNT with its KEY and VALUE variables, GIT_CONFIG_PARAMETERS) and the
    variables that pick the repository or its object store are removed for the same
    reason.
    """
    environment = {name: value for name, value in os.environ.items()
                   if name not in _GIT_INHERITED_NAMES
                   and not name.startswith(_GIT_INHERITED_PREFIXES)}
    return dict(environment, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


def _glob_matches(pattern: str, path: str) -> bool:
    """Return whether a GitHub Actions filter pattern matches path.

    Basis: the "Filter pattern cheat sheet" in GitHub's workflow syntax reference
    (docs.github.com, "Workflow syntax for GitHub Actions"). There "*" matches zero or
    more characters but not "/", and "**" matches zero or more of any character,
    including "/"; "**/" before a name also matches that name in the root, as its
    rows "**/README.md" and "docs/**/*.md" show. It rejects "?", "[" and "+", which it does not model (the cheat
    sheet defines "?" and "+" by the preceding character, not as wildcards), so a
    filter that starts to use them fails loudly instead of being mis-matched.
    """
    expression = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if pattern.startswith("**/", index):
            # The cheat sheet's "**/README.md" matches README.md in the root, so "**/"
            # is zero or more directories, each with its trailing "/".
            expression.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            expression.append(".*")
            index += 2
            continue
        if character in "?[+":
            raise ValueError(f"filter pattern {pattern!r}: {character!r} is not modelled")
        expression.append({"*": "[^/]*"}.get(character, re.escape(character)))
        index += 1
    return re.fullmatch("".join(expression), path) is not None


def _pull_request_paths(workflow: str) -> list[str]:
    """Return the entries of the pull_request paths list, in order, "!" entries included.

    Every non-blank line from `paths:` to `workflow_dispatch:` must be exactly
    `      - '<entry>'` (CYAX-0222 R-011). A double-quoted entry, an entry with a
    trailing comment, or a comment line would otherwise be skipped, or would end the
    list early, and could hide a "**/" or "!" entry from the filter tests: YAML allows
    a comment line at any indentation inside a block sequence, with entries after
    it. So `paths` must stay the last key under `pull_request`; a later key makes
    this reader fail rather than guess where the list ends.
    """
    trigger = workflow[workflow.index("  pull_request:\n"):workflow.index("  workflow_dispatch:")]
    paths = trigger[trigger.index("    paths:\n") + len("    paths:\n"):]
    entries = []
    for line in paths.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"      - '([^']+)'", line)
        if match is None:
            raise ValueError(f"pull_request paths line {line!r} is not exactly: "
                             "      - '<entry>'")
        entries.append(match.group(1))
    return entries


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


def _job_condition(block: str) -> str:
    """Return the value of the job-level `if:` key, with its lines folded into one.

    The value is the rest of the `    if:` line and the lines indented by more than
    four spaces after it (a block scalar such as `>-`); comment lines inside the
    value are dropped. Comments above the key are not part of it, so a phrase that
    only the comment holds is not found in the condition.
    """
    lines = block.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("    if:"))
    value = [lines[start][len("    if:"):]]
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith("     "):
            break
        if not line.lstrip().startswith("#"):
            value.append(line)
    folded = re.sub(r"\s+", " ", " ".join(value)).strip()
    # Drop a YAML block scalar indicator (">-" folds, "|" keeps lines): not part of the text.
    return re.sub(r"^[>|][-+]?\s*", "", folded)


class GitEnvironment(unittest.TestCase):
    def test_inherited_git_configuration_and_repository_variables_are_removed(self) -> None:
        inherited = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "diff.renames",
                     "GIT_CONFIG_VALUE_0": "false", "GIT_CONFIG_KEY_7": "a.b",
                     "GIT_CONFIG_VALUE_7": "c", "GIT_CONFIG_PARAMETERS": "'diff.renames=false'",
                     "GIT_DIR": "/nonexistent", "GIT_WORK_TREE": "/nonexistent",
                     "GIT_INDEX_FILE": "/nonexistent", "GIT_CONFIG": "/nonexistent",
                     "GIT_OBJECT_DIRECTORY": "/nonexistent", "GIT_COMMON_DIR": "/nonexistent",
                     "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/nonexistent",
                     "GIT_NAMESPACE": "x", "UNRELATED_VARIABLE": "kept"}
        with unittest.mock.patch.dict(os.environ, inherited):
            environment = _git_environment()
        # Compare names only: a failure message must not print the environment, whose
        # values can include tokens, into a CI log.
        self.assertEqual(environment.get("UNRELATED_VARIABLE"), "kept")
        kept = sorted(name for name in inherited
                      if name != "UNRELATED_VARIABLE" and name in environment)
        self.assertEqual(kept, [])
        self.assertEqual(environment.get("GIT_CONFIG_GLOBAL"), os.devnull)
        self.assertEqual(environment.get("GIT_CONFIG_NOSYSTEM"), "1")


class GlobMatcher(unittest.TestCase):
    """The matcher that the filter tests use is itself tested, with hand-written cases."""

    def test_matches_and_non_matches(self) -> None:
        # (pattern, path, expected) from the cheat sheet's rules cited in _glob_matches.
        for pattern, path, expected in (
            ("src/**", "src/a.jl", True),
            ("src/**", "src/deep/er/a.jl", True),
            ("src/**", "docs/src/a.jl", False),
            ("**/.gitignore", "nested/dir/.gitignore", True),
            ("**/.gitignore", "nested/dir/gitignore", False),
            ("**", "any/path/at.all", True),
            (".gitignore", ".gitignore", True),
            (".gitignore", "nested/.gitignore", False),
            (".gitignore", "xgitignore", False),
            ("Manifest-v*.toml", "Manifest-v1.12.toml", True),
            ("Manifest-v*.toml", "nested/Manifest-v1.12.toml", False),
            ("Manifest-v*.toml", "Manifest-v1/2.toml", False),
            ("*.toml", "a.toml", True),
            ("*.toml", "dir/a.toml", False),
        ):
            with self.subTest(pattern=pattern, path=path):
                self.assertEqual(_glob_matches(pattern, path), expected)

    def test_double_star_slash_also_matches_when_the_directory_is_absent(self) -> None:
        # Each row is copied from the "Filter pattern cheat sheet" in GitHub's
        # workflow syntax reference (docs.github.com, "Workflow syntax for GitHub
        # Actions"), where "**/X" also matches a file X in the root.
        for pattern, path in (
            ("**/README.md", "README.md"),
            ("docs/**/*.md", "docs/README.md"),
            ("**/docs/**", "docs/hello.md"),
            ("**/*-post.md", "my-post.md"),
        ):
            with self.subTest(pattern=pattern, path=path):
                self.assertTrue(_glob_matches(pattern, path))

    def test_job_condition_reads_the_if_value_and_not_the_comment(self) -> None:
        block = ("  job:\n    needs: x\n    # !cancelled() is only in this comment\n"
                 "    if: >-\n      ${{ always() && (\n        # not this either !cancelled()\n"
                 "        a == 'b') }}\n    runs-on: y\n")
        condition = _job_condition(block)
        self.assertEqual(condition, "${{ always() && ( a == 'b') }}")
        self.assertNotIn("!cancelled()", condition)

    def test_unmodelled_syntax_is_rejected(self) -> None:
        for pattern in ("[ab].toml", "a+.toml", "a?.toml"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    _glob_matches(pattern, "a.toml")

    def test_pull_request_paths_rejects_a_line_it_cannot_read_exactly(self) -> None:
        # A double-quoted entry and an entry with a trailing comment are valid YAML
        # but not the form `      - '<entry>'`; skipping them silently could hide a
        # "!" entry from the R-011 test, so the reader must refuse them.
        head = "on:\n  pull_request:\n    paths:\n      - 'a/**'\n"
        for line in ('      - "!LocalPreferences.toml"\n',
                     "      - '!.gitignore'  # x\n",
                     "      - !.gitignore\n",
                     "      # note\n",
                     # YAML lets a comment sit at any indentation inside the list, and an
                     # entry may follow it; a reader that stopped at a less-indented
                     # comment would miss the "!" entry after it.
                     "# note\n      - '!.gitignore'\n",
                     "  # note\n      - '!.gitignore'\n",
                     "    # note\n      - '!.gitignore'\n",
                     "#      - 'JuliaProject.toml'\n"):
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    _pull_request_paths(head + line + "  workflow_dispatch:\n")

    def test_pull_request_paths_reads_the_list_in_order(self) -> None:
        workflow = ("on:\n  push:\n    branches:\n      - 'x'\n  pull_request:\n"
                    "    types: [opened]\n    paths:\n      - 'a/**'\n      - '!a/b'\n"
                    "  workflow_dispatch:\n")
        self.assertEqual(_pull_request_paths(workflow), ["a/**", "!a/b"])


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

    def test_versioned_manifest_names_use_ascii_digits_only(self) -> None:
        # Julia's names are Manifest-v<major>.<minor>.toml with the digits of the
        # version, which are ASCII; U+0661 and U+0662 are Arabic-Indic digits one and
        # two (Unicode category Nd), which a bare \d matches in Python 3 str patterns.
        for name in ("Manifest-v\u0661.\u0661\u0662.toml", "JuliaManifest-v1.\u0662.toml"):
            with self.subTest(name=name):
                self.assertFalse(self.gate.requires_qualification([name]))

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
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True,
                                env=_git_environment())
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
        environment = dict(_git_environment(), GITHUB_OUTPUT=str(output))
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

    def test_other_alternative_project_and_manifest_names_set_output_true(self) -> None:
        # R-009 names, through a real commit range like their siblings above. The base
        # moves to the previous commit each time, so each range holds one file only.
        for path in ("JuliaProject.toml", "JuliaManifest.toml", "JuliaManifest-v1.13.toml"):
            with self.subTest(path=path):
                self.base = self._git("rev-parse", "HEAD").strip()
                self._assert_gate(self._commit(path), "true")

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

    # R-010: Git quotes a path with a non-ASCII byte in plain --name-only output
    # (core.quotePath), so the gate reads NUL-separated paths. GitHub's src/** filter
    # matches src/café.jl, so the gate must too.
    def test_non_ascii_path_under_src_sets_output_true(self) -> None:
        self._assert_gate(self._commit("src/café.jl"), "true")

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
        # The conditions are read from the job's `if:` value, not from the comment
        # above it, which mentions !cancelled() and the failed gate in words.
        condition = _job_condition(qualification)
        self.assertIn("needs.changes.outputs.qualification == 'true'", condition)
        # A failed gate must not silently skip qualification.
        self.assertIn("needs.changes.result == 'failure'", condition)
        self.assertIn("!cancelled()", condition)
        self.assertIn("contains(github.event.pull_request.labels.*.name, 'full-tests')",
                      condition)

    def test_pull_requests_to_bases_other_than_vmm_run_qualification_when_the_workflow_runs(self) -> None:
        # The path gate runs only for PRs to vmm; a PR to main, a maintenance line
        # or a stacked branch skips it, so the job runs qualification directly
        # whenever the workflow runs (the pull_request paths filter decides that).
        qualification = _job_block(self.workflow, "qualification")
        self.assertIn("github.base_ref != 'vmm'", _job_condition(qualification))
        self.assertNotIn("github.base_ref == 'main'", qualification)

    def test_python_test_requirements_trigger_pull_request_ci(self) -> None:
        # The lifecycle job installs requirements-test.txt, so a PR changing only
        # that file must still run the workflow.
        entries = _pull_request_paths(self.workflow)
        self.assertIn("requirements-test.txt", entries)

    def test_local_preferences_trigger_pull_request_ci(self) -> None:
        # Julia applies LocalPreferences.toml sections for packages the root
        # project lists, so a PR changing only that file must run the workflow.
        entries = _pull_request_paths(self.workflow)
        self.assertIn("LocalPreferences.toml", entries)

    def test_julia_local_preferences_trigger_pull_request_ci(self) -> None:
        # When present, Julia reads the root JuliaLocalPreferences.toml in place of
        # LocalPreferences.toml (Base.preferences_names), so a PR changing only that
        # file must run the workflow.
        entries = _pull_request_paths(self.workflow)
        self.assertIn("JuliaLocalPreferences.toml", entries)

    def test_alternative_project_and_manifest_names_trigger_pull_request_ci(self) -> None:
        # Julia 1.12 reads these root files in place of Project.toml and
        # Manifest.toml when present (Base.project_names, Base.manifest_names). In
        # GitHub's filter * does not match /, so these entries are root only.
        entries = _pull_request_paths(self.workflow)
        for entry in ("JuliaProject.toml", "JuliaManifest.toml", "Manifest-v*.toml",
                      "JuliaManifest-v*.toml"):
            with self.subTest(entry=entry):
                self.assertIn(entry, entries)

    def test_root_gitignore_triggers_pull_request_ci(self) -> None:
        # The qualification job requires a clean tree, and the root .gitignore
        # hides the root Manifest.toml that CI resolves, so a PR changing only
        # that file must run the workflow.
        entries = _pull_request_paths(self.workflow)
        self.assertIn(".gitignore", entries)

    # R-011: the entry tests above show that each entry is present, not that no other
    # entry widens it. A "**/" form would also start the workflow for a nested file, and
    # a "!" entry (GitHub: it negates the positive patterns before it) could remove a
    # root file; neither may match these root inputs. Order is not modelled: any "!"
    # entry that matches the name fails the test.
    def test_filter_has_no_wider_or_negating_entry_for_root_inputs(self) -> None:
        entries = _pull_request_paths(self.workflow)
        positive = [entry for entry in entries if not entry.startswith("!")]
        negating = [entry[1:] for entry in entries if entry.startswith("!")]
        self.assertTrue(positive)
        for name in ("LocalPreferences.toml", ".gitignore", "JuliaLocalPreferences.toml",
                     "JuliaProject.toml", "JuliaManifest.toml", "Manifest-v1.12.toml",
                     "JuliaManifest-v1.12.toml"):
            with self.subTest(name=name):
                # One and two levels down: "*/<name>" matches only the first, "**/<name>"
                # both (GitHub cheat sheet: "*" does not match "/", "**" does).
                for nested in (f"nested/{name}", f"nested/dir/{name}"):
                    self.assertEqual([e for e in positive if _glob_matches(e, nested)], [],
                                     nested)
                self.assertEqual([e for e in negating if _glob_matches(e, name)], [])

    def test_job_comment_states_when_qualification_runs(self) -> None:
        # Pushes and manual runs have no paths filter, so every such run that starts
        # its jobs runs qualification; the filter belongs to pull_request only.
        events = self.workflow[self.workflow.index("  push:\n"):
                               self.workflow.index("  pull_request:\n")]
        # "paths" also covers paths-ignore, which would filter pushes as well.
        self.assertNotIn("paths", events)
        # A PR to another base runs it only when the workflow runs, which the
        # pull_request paths filter decides; the comment must say so. Only the comment
        # lines are read, with line breaks folded, so the wording cannot be matched
        # from the job condition and a phrase may wrap across lines.
        qualification = _job_block(self.workflow, "qualification")
        comment = " ".join(
            line.strip().lstrip("#").strip()
            for line in qualification.splitlines() if line.strip().startswith("#"))
        comment = re.sub(r"\s+", " ", comment)
        self.assertIn("no paths filter", comment)
        self.assertIn("whenever the workflow runs", comment)
        self.assertIn("pull_request paths filter", comment)
        # A pending run that the concurrency group replaces starts no job, so the
        # comment must not promise a run for every push (R-002).
        self.assertNotIn("always", comment.lower())


if __name__ == "__main__":
    unittest.main()
