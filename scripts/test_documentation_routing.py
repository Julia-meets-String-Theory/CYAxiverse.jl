#!/usr/bin/env python3
"""Focused tests for the release-neutral documentation routing contract."""

from __future__ import annotations

import os
import copy
import importlib.util
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_route(**values: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"CYAX_DOCS_ROUTE_ONLY": "true", "DOCS_DEPLOY": "false"})
    env.update(values)
    return subprocess.run(
        ["julia", "--startup-file=no", "docs/make.jl"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@unittest.skipUnless(shutil.which("julia"), "Julia is required for route checks")
class DocumentationRoutingTests(unittest.TestCase):
    def test_release_ref_resolvers_reject_ambiguous_advertisements(self) -> None:
        module_spec = importlib.util.spec_from_file_location(
            "verify_release_context_ambiguous_refs",
            ROOT / "docs/verify_release_context.py",
        )
        verifier = importlib.util.module_from_spec(module_spec)
        assert module_spec.loader is not None
        module_spec.loader.exec_module(verifier)

        ref = "refs/tags/v0.3.0"
        conflicting = f"{'a' * 40}\t{ref}\n{'b' * 40}\t{ref}\n"
        malformed = f"{'a' * 40}\t{ref}\nnot-an-advertisement\n"
        for output in (conflicting, malformed):
            with self.subTest(output=output):
                with patch.object(verifier, "_command_output", return_value=output):
                    with self.assertRaisesRegex(
                        verifier.GitIdentityError,
                        "REMOTE_REF_ADVERTISEMENT_INVALID",
                    ):
                        verifier._remote_ref_commit(ref)
                with patch.object(
                    verifier.subprocess, "check_output", return_value=output
                ):
                    self.assertIsNone(verifier.resolve_tag_commit("v0.3.0"))

    def test_repository_evidence_resolves_refs_trees_and_versions_independently(self) -> None:
        module_spec = importlib.util.spec_from_file_location(
            "verify_release_context_live_fixture",
            ROOT / "docs/verify_release_context.py",
        )
        verifier = importlib.util.module_from_spec(module_spec)
        assert module_spec.loader is not None
        module_spec.loader.exec_module(verifier)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            remote = root / "origin.git"
            checkout = root / "checkout"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            subprocess.run(
                ["git", "init", "-q", "-b", "main", str(checkout)], check=True
            )
            for key, value in (
                ("user.name", "Docs Fixture"),
                ("user.email", "docs@example.invalid"),
            ):
                subprocess.run(
                    ["git", "-C", str(checkout), "config", key, value], check=True
                )
            subprocess.run(
                ["git", "-C", str(checkout), "remote", "add", "origin", str(remote)],
                check=True,
            )
            (checkout / "Project.toml").write_text(
                'name = "Fixture"\nversion = "0.3.0"\n', encoding="utf-8"
            )
            subprocess.run(
                ["git", "-C", str(checkout), "add", "Project.toml"], check=True
            )
            subprocess.run(
                ["git", "-C", str(checkout), "commit", "-q", "-m", "release"],
                check=True,
            )
            sha = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            tree = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True
            ).strip()
            subprocess.run(
                ["git", "-C", str(checkout), "branch", "candidates/0.3.0"], check=True
            )
            subprocess.run(
                ["git", "-C", str(checkout), "tag", "iterations/0.3.0"], check=True
            )
            subprocess.run(
                ["git", "-C", str(checkout), "tag", "v0.3.0"], check=True
            )
            subprocess.run(
                [
                    "git", "-C", str(checkout), "push", "-q", "origin",
                    "main", "candidates/0.3.0", "--tags",
                ],
                check=True,
            )
            event = {
                "public_tag": "v0.3.0",
                "anchor_ref": "refs/tags/iterations/0.3.0",
                "anchor_sha": sha,
                "candidate_ref": "refs/heads/candidates/0.3.0",
                "candidate_sha": sha,
                "certification_subject_sha": sha,
                "main_at_event_sha": sha,
            }
            verifier.ROOT = checkout
            evidence = verifier.resolve_repository_evidence(
                event,
                tag_ref="refs/tags/v0.3.0",
                tag_sha=sha,
                tag_tree=tree,
                tag_version="0.3.0",
                main_sha=sha,
                main_version="0.3.0",
            )
        self.assertTrue(all(item["tree"] == tree for item in evidence.values()))
        self.assertTrue(all(item["version"] == "0.3.0" for item in evidence.values()))

    def test_release_context_verifier_exports_only_verified_identities(self) -> None:
        sha = "a" * 40
        tree = "c" * 40
        module_spec = importlib.util.spec_from_file_location(
            "verify_release_context", ROOT / "docs/verify_release_context.py"
        )
        verifier = importlib.util.module_from_spec(module_spec)
        assert module_spec.loader is not None
        module_spec.loader.exec_module(verifier)
        actual_repository_resolver = verifier.resolve_repository_evidence

        def resolved_evidence(event, *, tag_sha, tag_tree, tag_version,
                              main_sha, main_version, **_):
            return {
                "tag": {"sha": tag_sha, "tree": tag_tree, "version": tag_version},
                "anchor": {
                    "sha": event["anchor_sha"], "tree": event["anchor_tree"],
                    "version": event["final_version"],
                },
                "candidate": {
                    "sha": event["candidate_sha"], "tree": event["candidate_tree"],
                    "version": event["final_version"],
                },
                "certified": {
                    "sha": event["certification_subject_sha"],
                    "tree": event["certification_subject_tree"],
                    "version": event["final_version"],
                },
                "main": {"sha": main_sha, "tree": tag_tree, "version": main_version},
                "event_main": {
                    "sha": event["main_at_event_sha"], "tree": tag_tree,
                    "version": event["main_at_event_version"],
                },
            }

        verifier.resolve_repository_evidence = resolved_evidence
        event = {
            "schema_version": 1,
            "event_type": "released",
            "event_id": "EVT-000000000003",
            "timestamp_utc": "2026-09-20T14:00:00Z",
            "transaction_id": "release-3",
            "static_iteration_snapshot": "b" * 64,
            "expected_event_head": "0" * 40,
            "closure_timestamp_utc": "2026-09-20T13:00:00Z",
            "public_tag": "v0.3.0",
            "release_line": "principal",
            "final_version": "0.3.0",
            "anchor_ref": "refs/tags/iterations/0.3.0",
            "anchor_sha": sha,
            "anchor_tree": tree,
            "candidate_ref": "refs/heads/candidates/0.3.0",
            "candidate_sha": sha,
            "candidate_tree": tree,
            "final_release_sha": sha,
            "final_release_tree": tree,
            "certification_binding": "tree-bound",
            "certification_subject_sha": sha,
            "certification_subject_tree": tree,
            "certification_policy_revision": "policy-2026-09",
            "certification_harness_revision": "harness-2026-09",
            "certification_environment": "julia-1.12-python-3.14",
            "certification_evidence_refs": ["evidence/certification.json"],
            "evidence_refs": ["evidence/released.json"],
            "main_at_event_sha": sha,
            "main_at_event_version": "0.3.0",
            "previous_main_sha": "e" * 40,
            "previous_main_version": "0.2.0",
        }
        with patch.object(
            verifier,
            "_remote_ref_commit",
            side_effect=lambda ref: "b" * 40
            if ref == event["candidate_ref"] else sha,
        ):
            with self.assertRaisesRegex(ValueError, "candidate ref"):
                actual_repository_resolver(
                    event,
                    tag_ref="refs/tags/v0.3.0",
                    tag_sha=sha,
                    tag_tree=tree,
                    tag_version="0.3.0",
                    main_sha=sha,
                    main_version="0.3.0",
                )
        values = type(
            "Args",
            (),
            {
                "event_stream": pathlib.Path(__file__),
                "tag": "v0.3.0",
                "tag_ref": "refs/tags/v0.3.0",
                "tag_sha": sha,
                "tag_tree": tree,
                "tag_version": "0.3.0",
                "main_sha": sha,
                "main_version": "0.3.0",
            },
        )()
        with tempfile.TemporaryDirectory() as temporary:
            values.github_env = pathlib.Path(temporary) / "github.env"
            with patch.object(verifier, "parse_stream", return_value=[event]), patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.3.0"]
            ), patch.object(verifier, "resolve_tag_commit", return_value=sha):
                self.assertEqual(verifier.verify(values), 0)
            exported = values.github_env.read_text(encoding="utf-8")
        self.assertIn("CYAX_DOCS_EVENT_STATUS=verified", exported)
        self.assertIn("CYAX_DOCS_RELEASE_LINE=principal", exported)
        self.assertIn("CYAX_DOCS_STABLE=true", exported)
        self.assertIn("CYAX_DOCS_STABLE_TAG=v0.3.0", exported)

        with tempfile.TemporaryDirectory() as temporary:
            values.github_env = pathlib.Path(temporary) / "nested-ref.env"
            values.tag_ref = "refs/tags/archive/v0.3.0"
            self.assertNotEqual(verifier.verify(values), 0)
        values.tag_ref = "refs/tags/v0.3.0"

        def unresolved_repository_evidence(*_args, **_kwargs):
            raise ValueError("fabricated event has no matching repository identities")

        with tempfile.TemporaryDirectory() as temporary:
            values.github_env = pathlib.Path(temporary) / "fabricated.env"
            verifier.resolve_repository_evidence = unresolved_repository_evidence
            with patch.object(verifier, "parse_stream", return_value=[event]):
                self.assertNotEqual(verifier.verify(values), 0)
            self.assertFalse(values.github_env.exists())
        verifier.resolve_repository_evidence = resolved_evidence

        stable_values = type(
            "Args",
            (),
            {
                "event_stream": pathlib.Path(__file__),
                "main_sha": sha,
                "main_version": "0.3.0",
            },
        )()
        with tempfile.TemporaryDirectory() as temporary:
            stable_values.github_env = pathlib.Path(temporary) / "github.env"
            with patch.object(verifier, "parse_stream", return_value=[event]), patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.3.0"]
            ), patch.object(verifier, "resolve_tag_commit", return_value=sha):
                self.assertEqual(verifier.verify_stable_context(stable_values), 0)
            stable_exported = stable_values.github_env.read_text(encoding="utf-8")
        self.assertIn("CYAX_DOCS_STABLE_TAG=v0.3.0", stable_exported)

        multi_ref = copy.deepcopy(event)
        multi_ref["certification_evidence_refs"] = [
            "evidence/certification.json",
            "evidence/certification-attestation.json",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            values.github_env = pathlib.Path(temporary) / "multi-ref.env"
            with patch.object(verifier, "parse_stream", return_value=[multi_ref]), patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.3.0"]
            ), patch.object(verifier, "resolve_tag_commit", return_value=sha):
                self.assertEqual(verifier.verify(values), 0)
            self.assertIn(
                "CYAX_DOCS_EVENT_STATUS=verified",
                values.github_env.read_text(encoding="utf-8"),
            )

        transferred = copy.deepcopy(event)
        transferred["candidate_sha"] = "b" * 40
        transferred["certification_subject_sha"] = "b" * 40
        transferred["certification_transfer_evidence"] = {
            "verified": True,
            "candidate_sha": "b" * 40,
            "final_release_sha": sha,
            "certified_tree": tree,
            "candidate_tree": tree,
            "final_release_tree": tree,
            "anchor_tree": tree,
            "evidence_ref": "evidence/transfer.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            values.github_env = pathlib.Path(temporary) / "transfer.env"
            with patch.object(verifier, "parse_stream", return_value=[transferred]), patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.3.0"]
            ), patch.object(verifier, "resolve_tag_commit", return_value=sha):
                self.assertEqual(verifier.verify(values), 0)
            self.assertIn(
                "CYAX_DOCS_EVENT_STATUS=verified",
                values.github_env.read_text(encoding="utf-8"),
            )

        empty_values = type(
            "Args",
            (),
            {
                "event_stream": pathlib.Path(__file__),
                "main_sha": sha,
                "main_version": "0.3.0",
            },
        )()
        with tempfile.TemporaryDirectory() as temporary:
            empty_values.github_env = pathlib.Path(temporary) / "github.env"
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=[]
            ), patch.object(verifier, "parse_stream", return_value=[]):
                self.assertEqual(verifier.verify_stable_context(empty_values), 0)
            empty_exported = empty_values.github_env.read_text(encoding="utf-8")
        self.assertIn("CYAX_DOCS_STABLE_TAG=\n", empty_exported)

        reservation_values = type(
            "Args",
            (),
            {
                "event_stream": pathlib.Path(__file__),
                "main_sha": sha,
                "main_version": "0.3.0",
            },
        )()
        with tempfile.TemporaryDirectory() as temporary:
            reservation_values.github_env = pathlib.Path(temporary) / "github.env"
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=[]
            ), patch.object(
                verifier,
                "parse_stream",
                return_value=[{"event_type": "development_reservation_prepared"}],
            ):
                self.assertEqual(
                    verifier.verify_stable_context(reservation_values), 0
                )
            reservation_exported = reservation_values.github_env.read_text(
                encoding="utf-8"
            )
        self.assertIn("CYAX_DOCS_STABLE_TAG=\n", reservation_exported)

        pending_values = type(
            "Args",
            (),
            {
                "event_stream": pathlib.Path(__file__),
                "main_sha": sha,
                "main_version": "0.3.0",
            },
        )()
        with tempfile.TemporaryDirectory() as temporary:
            pending_values.github_env = pathlib.Path(temporary) / "github.env"
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=[]
            ), patch.object(
                verifier,
                "parse_stream",
                return_value=[
                    {
                        "event_type": "release_intent_prepared",
                        "public_tag": "v0.3.0",
                    }
                ],
            ):
                self.assertEqual(
                    verifier.verify_stable_context(pending_values), 0
                )
            pending_exported = pending_values.github_env.read_text(
                encoding="utf-8"
            )
        self.assertIn("CYAX_DOCS_STABLE_TAG=\n", pending_exported)

        with tempfile.TemporaryDirectory() as temporary:
            blocked_values = type(
                "Args",
                (),
                {
                    "event_stream": pathlib.Path(__file__),
                    "main_sha": sha,
                    "main_version": "0.3.0",
                    "github_env": pathlib.Path(temporary) / "github.env",
                },
            )()
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=None
            ), patch.object(verifier, "parse_stream", return_value=[]):
                self.assertNotEqual(
                    verifier.verify_stable_context(blocked_values), 0
                )

        with tempfile.TemporaryDirectory() as temporary:
            stable_values.github_env = pathlib.Path(temporary) / "missing.env"
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.3.0"]
            ), patch.object(verifier, "parse_stream", return_value=[event]), patch.object(
                verifier, "resolve_tag_commit", return_value=None
            ):
                self.assertNotEqual(verifier.verify_stable_context(stable_values), 0)
            stable_values.github_env = pathlib.Path(temporary) / "mismatch.env"
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.3.0"]
            ), patch.object(verifier, "parse_stream", return_value=[event]), patch.object(
                verifier, "resolve_tag_commit", return_value="b" * 40
            ):
                self.assertNotEqual(verifier.verify_stable_context(stable_values), 0)

        # A maintenance tag keeps the main identity recorded when it was
        # released. Current main may have advanced by the time that tag is
        # redeployed; only the stable principal candidate uses current main.
        maintenance = copy.deepcopy(event)
        maintenance.update(
            {
                "event_id": "EVT-000000000004",
                "transaction_id": "release-4",
                "public_tag": "v0.2.1",
                "release_line": "maintenance/0.2",
                "final_version": "0.2.1",
                "anchor_ref": "refs/tags/iterations/0.2.1",
                "candidate_ref": "refs/heads/candidates/0.2.1",
                "main_at_event_sha": sha,
                "main_at_event_version": "0.3.0",
            }
        )
        maintenance.pop("previous_main_sha")
        maintenance.pop("previous_main_version")
        current_main_sha = "d" * 40
        current_main_tree = "e" * 40
        current = copy.deepcopy(event)
        current.update(
            {
                "event_id": "EVT-000000000005",
                "transaction_id": "release-5",
                "public_tag": "v0.4.0",
                "final_version": "0.4.0",
                "anchor_ref": "refs/tags/iterations/0.4.0",
                "candidate_ref": "refs/heads/candidates/0.4.0",
                "anchor_sha": current_main_sha,
                "anchor_tree": current_main_tree,
                "candidate_sha": current_main_sha,
                "candidate_tree": current_main_tree,
                "final_release_sha": current_main_sha,
                "final_release_tree": current_main_tree,
                "certification_subject_sha": current_main_sha,
                "certification_subject_tree": current_main_tree,
                "main_at_event_sha": current_main_sha,
                "main_at_event_version": "0.4.0",
                "previous_main_sha": sha,
                "previous_main_version": "0.3.0",
            }
        )
        maintenance_values = type(
            "Args",
            (),
            {
                "event_stream": pathlib.Path(__file__),
                "tag": "v0.2.1",
                "tag_ref": "refs/tags/v0.2.1",
                "tag_sha": sha,
                "tag_tree": tree,
                "tag_version": "0.2.1",
                "main_sha": current_main_sha,
                "main_version": "0.4.0",
            },
        )()
        with tempfile.TemporaryDirectory() as temporary:
            maintenance_values.github_env = pathlib.Path(temporary) / "github.env"
            with patch.object(
                verifier, "list_canonical_public_tags", return_value=["v0.4.0", "v0.2.1"]
            ), patch.object(
                verifier, "parse_stream", return_value=[maintenance, current]
            ), patch.object(verifier, "resolve_tag_commit", return_value=current_main_sha):
                self.assertEqual(verifier.verify(maintenance_values), 0)
            maintenance_exported = maintenance_values.github_env.read_text(
                encoding="utf-8"
            )
        self.assertIn("CYAX_DOCS_STABLE=false", maintenance_exported)
        self.assertIn("CYAX_DOCS_STABLE_TAG=v0.4.0", maintenance_exported)

    def test_workflow_and_source_are_release_neutral(self) -> None:
        make_source = (ROOT / "docs/make.jl").read_text(encoding="utf-8")
        workflow = (ROOT / ".github/workflows/Documentation.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("_deploy_versions(route)", make_source)
        self.assertIn("DOCS_ROUTE_UNVERIFIED", make_source)
        self.assertNotIn("Project.toml", make_source)
        self.assertIn("tags: 'v*.*.*'", workflow)
        self.assertIn("CYAX_DOCS_EVENT_STATUS", workflow)
        self.assertIn("docs/verify_release_context.py", workflow)
        self.assertIn("GITHUB_ENV", workflow)
        self.assertIn("docs/verify_release_context.py'", workflow)
        self.assertIn("--stable-only", workflow)
        self.assertIn("refs/remotes/origin/main^{commit}", workflow)
        self.assertIn("git ls-remote --exit-code origin refs/heads/release-events", workflow)
        self.assertIn("refs/tags/v*", workflow)
        self.assertIn('EVENT_STREAM="$RUNNER_TEMP/release-events.jsonl"', workflow)
        verifier_source = (ROOT / "docs/verify_release_context.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"git", "ls-remote", "origin"', verifier_source)
        self.assertIn("list_canonical_public_tags", verifier_source)
        self.assertIn('"stable" => stable_tag', make_source)
        self.assertNotIn('"stable" => "v^"', make_source)
        self.assertIn('push!(versions, "v#.#.#")', make_source)

    def test_development_and_preview_routes(self) -> None:
        self.assertEqual(
            run_route(CYAX_DOCS_REF="refs/heads/vmm").stdout.strip(), "development"
        )
        self.assertEqual(
            run_route(
                CYAX_DOCS_REF="refs/pull/125/merge",
                CYAX_DOCS_EVENT_STATUS="preview",
            ).stdout.strip(),
            "preview",
        )

    def test_verified_principal_tag_routes_to_versioned_channel(self) -> None:
        result = run_route(
            CYAX_DOCS_REF="refs/tags/v0.3.0",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="principal",
            CYAX_DOCS_RELEASE_VERSION="0.3.0",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "versioned")

        maintenance = run_route(
            CYAX_DOCS_REF="refs/tags/v0.2.1",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="maintenance/0.2",
            CYAX_DOCS_RELEASE_VERSION="0.2.1",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
        )
        self.assertEqual(maintenance.returncode, 0, maintenance.stderr)
        self.assertEqual(maintenance.stdout.strip(), "versioned")

    def test_deploy_selector_keeps_dev_and_all_tags(self) -> None:
        result = run_route(
            CYAX_DOCS_ROUTE_ONLY="versions",
            CYAX_DOCS_REF="refs/tags/v0.2.1",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="maintenance/0.2",
            CYAX_DOCS_RELEASE_VERSION="0.2.1",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
            CYAX_DOCS_STABLE_TAG="v0.2.0",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "stable:v0.2.0,v#.#.#,dev:dev",
        )
        missing = run_route(
            CYAX_DOCS_ROUTE_ONLY="versions",
            CYAX_DOCS_REF="refs/tags/v0.2.1",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="maintenance/0.2",
            CYAX_DOCS_RELEASE_VERSION="0.2.1",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("DOCS_STABLE_CONTEXT_MISSING", missing.stderr)
        development = run_route(
            CYAX_DOCS_ROUTE_ONLY="versions",
            CYAX_DOCS_REF="refs/heads/vmm",
            CYAX_DOCS_STABLE_TAG="",
        )
        self.assertEqual(development.returncode, 0, development.stderr)
        self.assertEqual(development.stdout.strip(), "v#.#.#,dev:dev")

    def test_stable_requires_principal_main_equality(self) -> None:
        result = run_route(
            CYAX_DOCS_REF="refs/tags/v0.3.0",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="principal",
            CYAX_DOCS_RELEASE_VERSION="0.3.0",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
            CYAX_DOCS_PRINCIPAL_MAIN_SHA="a" * 40,
            CYAX_DOCS_STABLE="true",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "stable")

        mismatch = run_route(
            CYAX_DOCS_REF="refs/tags/v0.3.0",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="principal",
            CYAX_DOCS_RELEASE_VERSION="0.3.0",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
            CYAX_DOCS_PRINCIPAL_MAIN_SHA="b" * 40,
            CYAX_DOCS_STABLE="true",
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("DOCS_STABLE_INVALID", mismatch.stderr)

    def test_maintenance_cannot_advance_stable(self) -> None:
        result = run_route(
            CYAX_DOCS_REF="refs/tags/v0.2.1",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="maintenance/0.2",
            CYAX_DOCS_RELEASE_VERSION="0.2.1",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
            CYAX_DOCS_PRINCIPAL_MAIN_SHA="a" * 40,
            CYAX_DOCS_STABLE="true",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DOCS_STABLE_INVALID", result.stderr)
        legacy_line = run_route(
            CYAX_DOCS_REF="refs/tags/v0.2.1",
            CYAX_DOCS_EVENT_STATUS="verified",
            CYAX_DOCS_RELEASE_LINE="maintenance",
            CYAX_DOCS_RELEASE_VERSION="0.2.1",
            CYAX_DOCS_RELEASE_SHA="a" * 40,
        )
        self.assertNotEqual(legacy_line.returncode, 0)
        self.assertIn("DOCS_ROUTE_INVALID", legacy_line.stderr)

    def test_tag_requires_verified_event_and_canonical_identity(self) -> None:
        for values in (
            {
                "CYAX_DOCS_REF": "refs/tags/v0.3.0",
                "CYAX_DOCS_EVENT_STATUS": "preview",
            },
            {
                "CYAX_DOCS_REF": "refs/tags/v-0.1",
                "CYAX_DOCS_EVENT_STATUS": "verified",
            },
            {
                "CYAX_DOCS_REF": "refs/tags/v01.2.3",
                "CYAX_DOCS_EVENT_STATUS": "verified",
            },
            {
                "CYAX_DOCS_REF": "refs/tags/archive/v0.3.0",
                "CYAX_DOCS_EVENT_STATUS": "verified",
            },
            {
                "CYAX_DOCS_REF": "refs/tags/v4294967296.0.0",
                "CYAX_DOCS_EVENT_STATUS": "verified",
            },
        ):
            result = run_route(**values)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
