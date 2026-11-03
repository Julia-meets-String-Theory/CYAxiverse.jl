"""Bounded JSON and action controls for specs/cyax-skills-2026-10-09/verify.py."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
VERIFY = REPOSITORY / "specs" / "cyax-skills-2026-10-09" / "verify.py"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPOSITORY, check=True,
                          capture_output=True, text=True).stdout.strip()


def _in_git_checkout() -> bool:
    try:
        return _git("rev-parse", "--is-inside-work-tree") == "true"
    except (OSError, subprocess.CalledProcessError):
        return False


@unittest.skipIf(sys.version_info < (3, 11), "verify.py needs tomllib (Python 3.11+)")
@unittest.skipUnless(VERIFY.is_file(), "skills checker is not in this checkout")
class SkillsCheckerControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("candidate_verify", VERIFY)
        cls.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.checker)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name)

    def _parse_result(self, path: Path) -> str:
        try:
            self.checker.parse_json(path)
        except self.checker.CheckFailure as exc:
            return exc.code
        except BaseException as exc:
            return "unhandled:" + type(exc).__name__
        return "ok"

    def test_json_depth_and_quoting_limits(self):
        deep = self.scratch / "depth-100000.json"
        deep.write_text("[" * 100000 + "0" + "]" * 100000, encoding="utf-8")
        self.assertEqual(self._parse_result(deep), "JSON_RESOURCE_LIMIT")
        depth12 = self.scratch / "depth-12.json"
        depth12.write_text("[" * 12 + "0" + "]" * 12, encoding="utf-8")
        self.assertEqual(self._parse_result(depth12), "ok")
        quoted = self.scratch / "quoted-brackets.json"
        quoted.write_text(json.dumps({"text": "[{[[ ]]} and an escaped quote: \\\""}),
                          encoding="utf-8")
        self.assertEqual(self._parse_result(quoted), "ok")

    @unittest.skipUnless(_in_git_checkout(), "needs a git checkout for candidate identity")
    def test_evidence_action_controls(self):
        checker = self.checker
        candidate = _git("rev-parse", "HEAD")
        tree = _git("rev-parse", "HEAD^{tree}")
        checker_hash = checker.sha(VERIFY.read_bytes())
        case_hash = checker.sha(
            (REPOSITORY / "specs/cyax-skills-2026-10-09/cases.json").read_bytes())
        procedure_hashes = {item: checker.sha((REPOSITORY / item).read_bytes())
                            for item in checker.PROC}
        artifact = self.scratch / "synthetic-control.txt"
        artifact_bytes = b"synthetic EVIDENCE_CANDIDATE_MISMATCH control; test fixture only\n"
        artifact.write_bytes(artifact_bytes)
        record = {"artifact_ref": "test-fixture:synthetic-control.txt",
                  "local_path": artifact.name,
                  "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                  "bytes": len(artifact_bytes)}
        cases = []
        for case_id in checker.CASE_IDS:
            commands = []
            if case_id == "SK-01":
                commands = [{"argv": ["synthetic-red"], "exit_code": 1,
                             "expected_exit": "nonzero", "sequence": 1},
                            {"argv": ["synthetic-green"], "exit_code": 0,
                             "expected_exit": "zero", "sequence": 2}]
            elif case_id == "SK-11":
                commands = [{"argv": ["synthetic-identity-control"], "exit_code": 1,
                             "expected_exit": "nonzero", "sequence": 1}]
            cases.append({"id": case_id, "evidence_status": "observed",
                          "criterion_status": "met",
                          "simulated_action_status":
                              "BLOCKED" if case_id == "SK-12" else "not_applicable",
                          "actor_ref": "test-fixture", "assessment_ref": "test-fixture",
                          "evidence": [record], "commands": commands})
        receipt = {"schema": "cyax-skills-smoke-1", "candidate_commit": candidate,
                   "candidate_tree": tree, "checker_sha256": checker_hash,
                   "case_pack_sha256": case_hash, "procedure_sha256": procedure_hashes,
                   "cases": cases}

        def validate(name, document):
            path = self.scratch / name
            path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            try:
                checker.validate_evidence(path, candidate, tree, checker_hash, case_hash,
                                          procedure_hashes)
            except checker.CheckFailure as exc:
                return exc.code
            return "ok"

        self.assertEqual(validate("synthetic-valid.json", receipt), "ok")
        mutant = copy.deepcopy(receipt)
        mutant["cases"][1]["simulated_action_status"] = "BLOCKED"
        self.assertEqual(validate("sk02-blocked.json", mutant), "EVIDENCE_INVALID")
        mutant = copy.deepcopy(receipt)
        mutant["cases"][-1]["simulated_action_status"] = "not_applicable"
        self.assertEqual(validate("sk12-not-applicable.json", mutant), "EVIDENCE_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
