#!/usr/bin/env python3
"""Focused tests for clean-rerun exact dispatch and verification machinery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DISPATCH = load("exact_dispatch", HERE / "exact_dispatch.py")
VERIFY = load("verify_rerun", HERE / "verify_rerun.py")
LAUNCH = load("launch_subject", HERE / "launch_subject.py")
BLIND = load("materialize_blind", HERE / "materialize_blind.py")
SCORER_INPUT = load("build_scorer_input", HERE / "build_scorer_input.py")
SCORECARDS = load("materialize_scorecards", HERE / "materialize_scorecards.py")
SCORER = load("launch_scorer", HERE / "launch_scorer.py")
IDENTITY = load("sanitize_identity", HERE / "sanitize_identity.py")


def valid_card(opaque: str) -> dict:
    scores = {f"K{i}": 1 for i in range(1, 13)}
    return {
        "opaque_id": opaque,
        "scores": scores,
        "raw_total": 12,
        "conflict_critical": 5,
        "automatic_failure": False,
        "automatic_failure_reasons": [],
        "unsupported_assertions": 0,
        "authority_errors": 0,
        "temporal_errors": 0,
        "supersession_errors": 0,
        "unjustified_inferences": 0,
        "correct_abstentions": 1,
        "incorrect_abstentions": 0,
        "next_action_correct": True,
        "source_reopenings": 0,
    }


class ExactDispatchTests(unittest.TestCase):
    def test_launcher_refuses_out_of_order_without_launching(self):
        manifest = LAUNCH.read_manifest()
        with self.assertRaises(RuntimeError):
            LAUNCH.preflight(manifest, "B1")
        if manifest["runs"][0]["status"] == "NOT_LAUNCHED":
            entry, data = LAUNCH.preflight(manifest, "A1")
            self.assertEqual(entry["status"], "NOT_LAUNCHED")
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["input_sha256"])
        else:
            with self.assertRaises(RuntimeError):
                LAUNCH.preflight(manifest, "A1")

    def test_launcher_captures_fake_codex_events_and_final_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            requests = runs / "requests"
            requests.mkdir(parents=True)
            source_manifest = json.loads(LAUNCH.RUN_MANIFEST.read_text())
            for item in source_manifest["runs"]:
                source = HERE / "runs" / item["input_artifact"]
                target = runs / item["input_artifact"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                item.clear()
                item.update({
                    "run_id": source.name.split(".")[0],
                    "condition": source.name.split(".")[0][0],
                    "input_artifact": f"requests/{source.name}",
                    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "event_artifact": f"events/{source.name}.jsonl",
                    "response_artifact": f"responses/{source.name}.response",
                    "status": "NOT_LAUNCHED",
                })
            manifest_path = runs / "manifest.json"
            manifest_path.write_text(json.dumps(source_manifest) + "\n")
            fake = root / "fake-codex"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "print(json.dumps({'type':'thread.started','thread_id':'fake-thread'}))\n"
                "print(json.dumps({'type':'turn.completed'}))\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
                "out.write_bytes(b'fake final response')\n"
            )
            fake.chmod(fake.stat().st_mode | os.stat(fake).st_mode | 0o111)
            with patch.object(LAUNCH, "RUNS_DIR", runs), patch.object(LAUNCH, "RUN_MANIFEST", manifest_path), patch.object(LAUNCH.CAPTURE, "RUN_MANIFEST", manifest_path), patch.object(LAUNCH.DISPATCH, "verify_artifacts"), patch.object(LAUNCH.DISPATCH, "verify_frozen_inputs", return_value={"A": b"", "B": b""}), patch.object(LAUNCH.DISPATCH, "verify_run_assignment"):
                result = LAUNCH.launch("A1", str(fake))
            self.assertEqual(result["event_audit"]["thread_id"], IDENTITY.pseudonym("fake-thread"))
            self.assertEqual(result["response"]["sha256"], hashlib.sha256(b"fake final response").hexdigest())
            updated = json.loads(manifest_path.read_text())
            self.assertEqual(updated["runs"][0]["status"], "CAPTURED")

    def test_launcher_is_directly_pinned_and_event_audit_is_strict(self):
        argv = LAUNCH.build_invocation()
        self.assertEqual(argv[0:4], ["codex", "exec", "--model", "gpt-5.6-sol"])
        self.assertIn("model_reasoning_effort=high", argv)
        for flag in ("--sandbox", "read-only", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--json", "-"):
            self.assertIn(flag, argv)
        clean = b'{"type":"thread.started","thread_id":"fresh"}\n{"type":"turn.completed"}\n'
        audit = LAUNCH.audit_events(clean)
        self.assertTrue(audit["valid_jsonl"])
        self.assertFalse(audit["tools_used"])
        self.assertEqual(audit["thread_id"], "fresh")
        dirty = b'{"type":"item.completed","item":{"type":"CommandExecution","command":["cat"]}}\n'
        dirty_audit = LAUNCH.audit_events(dirty)
        self.assertTrue(dirty_audit["forbidden_events"])
        self.assertTrue(dirty_audit["tools_used"])
        codex_style_dirty = b'{"type":"item.completed","item":{"type":"command_execution","command":["cat"]}}\n'
        codex_style_audit = LAUNCH.audit_events(codex_style_dirty)
        self.assertTrue(codex_style_audit["forbidden_events"])
        self.assertTrue(codex_style_audit["tools_used"])
        self.assertEqual(VERIFY.stable_identity({}, {"session_id": "session-fallback"}), "session-fallback")
        fresh = [({"thread_id": IDENTITY.pseudonym(f"thread-{i}")}, {}) for i in range(6)]
        self.assertEqual(len(VERIFY.require_unique_fresh_identities(fresh)), 6)
        with self.assertRaises(RuntimeError):
            VERIFY.require_unique_fresh_identities([({"thread_id": IDENTITY.pseudonym("same")}, {}) for _ in range(6)])

    def test_identity_sanitization_is_deterministic_and_idempotent(self):
        raw = b'{"type":"thread.started","thread_id":"local-thread-123"}\n{"type":"turn.completed"}\n'
        sanitized, metadata = IDENTITY.sanitize_event_stream(raw)
        self.assertNotIn(b"local-thread-123", sanitized)
        self.assertIn(IDENTITY.pseudonym("local-thread-123").encode(), sanitized)
        self.assertEqual(metadata["original_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(metadata["hashed_identity_count"], 1)
        again, again_metadata = IDENTITY.sanitize_event_stream(sanitized)
        self.assertEqual(again, sanitized)
        self.assertEqual(again_metadata["sanitized_sha256"], metadata["sanitized_sha256"])

    def test_frozen_inputs_and_pre_dispatch_verifier(self):
        phase = "captured" if all(item["status"] == "CAPTURED" for item in LAUNCH.read_manifest()["runs"]) else "pre-dispatch"
        subprocess.run(
            ["python3", str(HERE / "verify_rerun.py"), "--phase", phase],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_emit_is_exact_artifact_bytes(self):
        for run_id in ("A1", "B3"):
            if all(item["status"] == "NOT_LAUNCHED" for item in LAUNCH.read_manifest()["runs"]):
                emitted = subprocess.check_output(
                    ["python3", str(HERE / "exact_dispatch.py"), "emit", run_id], cwd=HERE
                )
            else:
                emitted = (HERE / "runs" / "requests" / f"{run_id}.input").read_bytes()
            self.assertEqual(emitted, (HERE / "runs" / "requests" / f"{run_id}.input").read_bytes())

    def test_response_hash_and_presence_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_dir = root / "responses"
            response_dir.mkdir()
            data = b"complete response bytes\x00\xff"
            path = response_dir / "A1.response"
            path.write_bytes(data)
            entries = [{
                "run_id": "A1",
                "status": "CAPTURED",
                "response_artifact": "responses/A1.response",
                "response": {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "completion_status": "COMPLETE",
                    "source_reopenings": 0,
                },
            }]
            with patch.object(VERIFY, "RUNS_DIR", root):
                VERIFY.verify_responses({}, entries)
                entries[0]["response"]["sha256"] = "0" * 64
                with self.assertRaises(RuntimeError):
                    VERIFY.verify_responses({}, entries)

    def test_blind_scorecard_completeness_and_metadata_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blind = root / "scoring" / "blind_responses"
            cards = root / "scoring" / "blind_scorecards"
            blind.mkdir(parents=True)
            cards.mkdir(parents=True)
            for opaque in VERIFY.OPAQUE:
                (blind / f"{opaque}.response").write_bytes(b"response")
                (cards / f"{opaque}.json").write_text(json.dumps(valid_card(opaque)) + "\n")
            with patch.object(VERIFY, "HERE", root), patch.object(VERIFY, "verify_blind_materialization"):
                VERIFY.verify_scoring({}, require_complete=True)
                bad = valid_card("S1")
                bad["condition"] = "A"
                (cards / "S1.json").write_text(json.dumps(bad) + "\n")
                with self.assertRaises(RuntimeError):
                    VERIFY.verify_scoring({}, require_complete=True)

    def test_blind_materializer_flags_without_changing_bytes(self):
        data = b"A1 graph relational Condition B B2"
        report = BLIND.leakage_record("S1", data)
        self.assertTrue(report["flagged"])
        self.assertEqual(report["counts"]["graph"], 1)
        self.assertEqual(report["counts"]["relational"], 1)
        self.assertEqual(report["counts"]["condition_label"], 2)
        self.assertIn("flag_only", report["action"])

    def test_blind_materialization_and_scorer_input_are_opaque(self):
        BLIND.verify()
        for opaque in BLIND.OPAQUE:
            self.assertEqual(
                (BLIND.OUTPUT_DIR / f"{opaque}.response").read_bytes(),
                (BLIND.HERE / "runs" / "responses" / f"{BLIND._mapping()[opaque]}.response").read_bytes(),
            )
        SCORER_INPUT.verify()
        scorer_bytes = SCORER_INPUT.INPUT.read_bytes()
        self.assertNotIn(b"condition_mapping.json", scorer_bytes)
        self.assertNotIn(b"runs/manifest.json", scorer_bytes)
        for opaque in SCORER_INPUT.OPAQUE:
            self.assertIn(b"--- BEGIN BLIND RESPONSE " + opaque.encode() + b" ---", scorer_bytes)

    def test_scorer_invocation_and_jsonl_scorecard_validation(self):
        argv = SCORER.build_invocation()
        self.assertEqual(argv[0:4], ["codex", "exec", "--model", "gpt-5.6-sol"])
        self.assertIn("model_reasoning_effort=high", argv)
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        cards = "\n".join(json.dumps(valid_card(opaque), sort_keys=True) for opaque in SCORECARDS.OPAQUE) + "\n"
        parsed = SCORECARDS.parse(cards.encode())
        self.assertEqual(tuple(parsed), SCORECARDS.OPAQUE)
        with self.assertRaises(RuntimeError):
            SCORECARDS.parse((cards + "{}\n").encode())

    def test_fake_scorer_capture_is_direct_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake-codex"
            cards = "\n".join(json.dumps(valid_card(opaque), sort_keys=True) for opaque in SCORECARDS.OPAQUE) + "\n"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "print(json.dumps({'type':'thread.started','thread_id':'scorer-thread'}))\n"
                "print(json.dumps({'type':'turn.completed'}))\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
                f"out.write_bytes({cards.encode()!r})\n"
            )
            fake.chmod(0o755)
            event_path = root / "events.jsonl"
            response_path = root / "scorer.response"
            record_path = root / "scorer_run.json"
            manifest = {"input": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
            with patch.object(SCORER, "preflight", return_value=(manifest, b"abc")), \
                 patch.object(SCORER, "EVENTS", event_path), \
                 patch.object(SCORER, "FINAL", response_path), \
                 patch.object(SCORER, "RUN_RECORD", record_path), \
                 patch.object(SCORER.materialize_scorecards, "parse", return_value={}), \
                 patch.object(SCORER.materialize_scorecards, "materialize", return_value={}):
                result = SCORER.launch(str(fake))
            self.assertEqual(result["result"]["event_audit"]["thread_id"], IDENTITY.pseudonym("scorer-thread"))
            self.assertEqual(result["result"]["completion_status"], "COMPLETE")
            self.assertEqual(response_path.read_bytes(), cards.encode())
            self.assertEqual(json.loads(record_path.read_text())["status"], "CAPTURED")

    def test_final_result_hash_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "final_result.json"
            result.write_bytes(b'{"status":"INCONCLUSIVE"}\n')
            (root / "final_result.sha256").write_text(
                hashlib.sha256(result.read_bytes()).hexdigest() + "\n"
            )
            with patch.object(VERIFY, "HERE", root):
                VERIFY.verify_final_immutability()
                result.write_bytes(b'{"status":"CHANGED"}\n')
                with self.assertRaises(RuntimeError):
                    VERIFY.verify_final_immutability()


if __name__ == "__main__":
    unittest.main()
