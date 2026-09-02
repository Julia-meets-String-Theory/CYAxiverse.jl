#!/usr/bin/env python3
"""Focused tests for scripts/agent_verify.py.

Uses temporary Git repositories and harmless child commands only.
Does not invoke Julia, CYTools, network access, or the scientific suite.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_verify.py")


def _run_verify(args, cwd=None, env=None, script=None):
    """Run agent_verify.py with the given args; return (stdout, stderr, rc)."""
    target = script or SCRIPT
    result = subprocess.run(
        [sys.executable, target] + args,
        capture_output=True, text=True, cwd=cwd, env=env, shell=False,
    )
    return result.stdout, result.stderr, result.returncode


def _make_temp_repo():
    """Create a minimal temporary Git repository and return its path."""
    d = tempfile.mkdtemp(prefix="av_test_repo_")
    subprocess.run(["git", "init", d], capture_output=True, check=True)
    subprocess.run(["git", "-C", d, "config", "user.email", "test@test"],
                    capture_output=True, check=True)
    subprocess.run(["git", "-C", d, "config", "user.name", "Test"],
                    capture_output=True, check=True)
    readme = os.path.join(d, "README.md")
    with open(readme, "w") as f:
        f.write("test\n")
    subprocess.run(["git", "-C", d, "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", d, "commit", "-m", "init"],
                    capture_output=True, check=True)
    return d


class TestJSONOutputValidity(unittest.TestCase):
    def test_snapshot_emits_valid_json(self):
        out, err, rc = _run_verify(["snapshot"])
        self.assertEqual(rc, 0)
        obj = json.loads(out.strip())
        for key in ("schema_version", "command", "status", "exit_code",
                     "duration_seconds", "log_paths", "summary", "warnings",
                     "git_state"):
            self.assertIn(key, obj, f"missing key: {key}")

    def test_diff_check_emits_valid_json(self):
        out, err, rc = _run_verify(["diff-check"])
        obj = json.loads(out.strip())
        self.assertIn("status", obj)
        self.assertIn(obj["status"], ("passed", "failed", "unavailable"))


class TestSingleRecordContract(unittest.TestCase):
    def test_exactly_one_json_line(self):
        out, _, _ = _run_verify(["snapshot"])
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, "expected exactly one non-empty line on stdout")
        json.loads(lines[0])


class TestSuccessOutputSuppression(unittest.TestCase):
    def test_run_success_no_child_output_on_stdout(self):
        out, _, rc = _run_verify(["run", "--", sys.executable, "-c",
                                   "print('VISIBLE_MARKER')"])
        self.assertEqual(rc, 0)
        self.assertNotIn("VISIBLE_MARKER", out)
        obj = json.loads(out.strip())
        self.assertEqual(obj["status"], "passed")


class TestFailureTailTruncation(unittest.TestCase):
    def test_failure_includes_tail_and_log_path(self):
        script = textwrap.dedent("""\
            import sys
            for i in range(100):
                print(f"line_{i}")
            sys.exit(1)
        """)
        out, _, rc = _run_verify(["run", "--", sys.executable, "-c", script])
        self.assertNotEqual(rc, 0)
        obj = json.loads(out.strip())
        self.assertEqual(obj["status"], "failed")
        self.assertIn("line_99", obj["summary"])
        self.assertNotIn("line_0", obj["summary"])
        self.assertIn("Full logs:", obj["summary"])

    def test_tail_is_at_most_40_lines(self):
        script = textwrap.dedent("""\
            import sys
            for i in range(200):
                print(f"numbered_{i}")
            sys.exit(1)
        """)
        out, _, _ = _run_verify(["run", "--", sys.executable, "-c", script])
        obj = json.loads(out.strip())
        summary_lines = obj["summary"].splitlines()
        numbered = [l for l in summary_lines if l.startswith("numbered_")]
        self.assertLessEqual(len(numbered), 40)


class TestNonzeroExitPropagation(unittest.TestCase):
    def test_child_exit_code_propagated(self):
        _, _, rc = _run_verify(["run", "--", sys.executable, "-c",
                                 "import sys; sys.exit(42)"])
        self.assertEqual(rc, 42)

    def test_success_returns_zero(self):
        _, _, rc = _run_verify(["run", "--", sys.executable, "-c", "pass"])
        self.assertEqual(rc, 0)


class TestRepoRootResolution(unittest.TestCase):
    def test_resolves_from_scripts_dir(self):
        out, _, rc = _run_verify(["snapshot"])
        self.assertEqual(rc, 0)
        obj = json.loads(out.strip())
        self.assertIn("status", obj["git_state"])


class TestSnapshotContent(unittest.TestCase):
    def setUp(self):
        self.repo = _make_temp_repo()
        self.script_dir = os.path.join(self.repo, "scripts")
        os.makedirs(self.script_dir, exist_ok=True)
        import shutil
        shutil.copy(SCRIPT, os.path.join(self.script_dir, "agent_verify.py"))
        self.local_script = os.path.join(self.script_dir, "agent_verify.py")

    def test_modified_file_appears_in_state(self):
        readme = os.path.join(self.repo, "README.md")
        with open(readme, "a") as f:
            f.write("modified\n")
        out, _, rc = _run_verify(["snapshot"], script=self.local_script)
        self.assertEqual(rc, 0)
        obj = json.loads(out.strip())
        self.assertIn("README.md", obj["git_state"].get("diff_names", ""))

    def test_untracked_file_appears_in_state(self):
        with open(os.path.join(self.repo, "newfile.txt"), "w") as f:
            f.write("new\n")
        out, _, rc = _run_verify(["snapshot"], script=self.local_script)
        self.assertEqual(rc, 0)
        obj = json.loads(out.strip())
        self.assertIn("newfile.txt", obj["git_state"].get("untracked", ""))


class TestMissingExecutable(unittest.TestCase):
    def test_unavailable_when_executable_missing(self):
        out, _, rc = _run_verify(["run", "--",
                                   "nonexistent_binary_xyz_12345", "arg"])
        obj = json.loads(out.strip())
        self.assertEqual(obj["status"], "unavailable")
        self.assertEqual(obj["exit_code"], 127)
        self.assertNotEqual(rc, 0)


class TestNoRunCommand(unittest.TestCase):
    def test_run_without_argv_fails(self):
        out, _, rc = _run_verify(["run"])
        self.assertNotEqual(rc, 0)
        obj = json.loads(out.strip())
        self.assertEqual(obj["status"], "failed")


class TestNoStateChangingGitCommands(unittest.TestCase):
    def test_no_forbidden_git_subcommands_in_source(self):
        with open(SCRIPT) as f:
            source = f.read()
        tree = ast.parse(source)
        string_literals = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.add(node.value)
        import agent_verify
        forbidden = agent_verify._FORBIDDEN_GIT_SUBCOMMANDS
        for cmd in forbidden:
            for lit in string_literals:
                tokens = lit.split()
                if "git" in tokens:
                    idx = tokens.index("git")
                    if idx + 1 < len(tokens) and tokens[idx + 1] == cmd:
                        self.fail(
                            f"source contains 'git {cmd}' in a string literal"
                        )

    def test_shell_false_everywhere(self):
        with open(SCRIPT) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name == "run":
                    for kw in node.keywords:
                        if kw.arg == "shell":
                            if isinstance(kw.value, ast.Constant):
                                self.assertFalse(
                                    kw.value.value,
                                    "subprocess.run called with shell=True",
                                )


class TestSubcommandHelp(unittest.TestCase):
    def test_no_subcommand_prints_help(self):
        _, err, rc = _run_verify([])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(SCRIPT))
    unittest.main()
