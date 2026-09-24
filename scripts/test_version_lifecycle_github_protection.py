"""Adversarial tests for the authenticated GitHub protection adapter."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from version_lifecycle.github_protection import (  # noqa: E402
    ApiResponse,
    GitHubProtectionAdapter,
    GitHubProtectionError,
)


REPOSITORY = "Julia-meets-String-Theory/CYAxiverse.jl"
OBSERVATION_REF = "refs/heads/candidates/cyax-0125/gate-a-observation"
SHA_A = "a" * 40
SHA_B = "b" * 40
TIMESTAMP = "2026-09-24T15:00:00Z"


def rule(kind: str) -> dict[str, object]:
    return {"type": kind}


def ruleset(
    identifier: int,
    name: str,
    *,
    target: str = "branch",
    enforcement: str = "active",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    rules: list[dict[str, object]] | None = None,
    actors: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": identifier,
        "name": name,
        "target": target,
        "enforcement": enforcement,
        "conditions": {
            "ref_name": {
                "include": include or [],
                "exclude": exclude or [],
            }
        },
        "rules": rules or [],
        "bypass_actors": actors or [],
        "updated_at": TIMESTAMP,
    }


def candidate_rulesets() -> list[dict[str, object]]:
    pattern = "refs/heads/candidates/**/*"
    return [
        ruleset(
            23967983,
            "CYAx candidate refs immutable",
            include=[pattern],
            rules=[rule("deletion"), rule("non_fast_forward"), rule("update")],
        ),
        ruleset(
            23967991,
            "CYAx candidate refs creation",
            include=[pattern],
            rules=[rule("creation")],
            actors=[{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}],
        ),
    ]


def freeze_rulesets() -> list[dict[str, object]]:
    return [
        ruleset(
            23948123,
            "CYAx vmm lifecycle freeze",
            enforcement="disabled",
            include=["refs/heads/vmm"],
            rules=[rule("update"), rule("deletion"), rule("non_fast_forward")],
        ),
        ruleset(
            23948127,
            "CYAx main lifecycle freeze",
            enforcement="disabled",
            include=["refs/heads/main"],
            rules=[rule("update"), rule("deletion"), rule("non_fast_forward")],
        ),
    ]


class FakeGitHub:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows if rows is not None else candidate_rulesets() + freeze_rulesets()
        self.calls: list[tuple[str, str, object]] = []
        self.fail_next = False
        self.stale_detail = False
        self.change_main_after_write = False
        self.move_main_after_disable = False
        self.branch_reads = {"vmm": 0, "main": 0}
        self.branch_shas = {"vmm": "a" * 40, "main": "a" * 40}
        self.project_version = "0.2.0"
        self.evaluated_rules = [
            {"ruleset_id": 23967991, "type": "creation"},
            {"ruleset_id": 23967983, "type": "update"},
            {"ruleset_id": 23967983, "type": "deletion"},
            {"ruleset_id": 23967983, "type": "non_fast_forward"},
        ]

    def __call__(self, method: str, path: str, payload=None) -> ApiResponse:
        self.calls.append((method, path, payload))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Bearer private-token must not escape")
        if "/rulesets?" in path and method == "GET":
            summaries = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "target": row["target"],
                    "enforcement": row["enforcement"],
                    "updated_at": (
                        "2026-09-24T14:59:59Z"
                        if self.stale_detail and row["id"] == 23967983
                        else row["updated_at"]
                    ),
                }
                for row in reversed(self.rows)
            ]
            return self._response(summaries, {"ETag": '"ruleset-list"'})
        if "/rulesets/" in path and method == "GET":
            identifier = int(path.rsplit("/", 1)[1])
            row = next((row for row in self.rows if row["id"] == identifier), None)
            if row is None:
                return ApiResponse(404, b"{}")
            return self._response(row)
        if "/rulesets/" in path and method in {"PATCH", "PUT"}:
            identifier = int(path.rsplit("/", 1)[1])
            row = next(row for row in self.rows if row["id"] == identifier)
            row["enforcement"] = payload["enforcement"]
            row["updated_at"] = "2026-09-24T15:00:01.001Z"
            if (
                identifier == 23948127
                and payload["enforcement"] == "disabled"
                and self.move_main_after_disable
            ):
                self.branch_shas["main"] = SHA_B
            return self._response(row)
        if "/rules/branches/" in path and method == "GET":
            return self._response(self.evaluated_rules, {
                "ETag": '"effective-rules"',
                "X-GitHub-Request-Id": "REQ-0125",
            })
        if "/branches/" in path and method == "GET":
            branch = path.rsplit("/", 1)[1]
            self.branch_reads[branch] = self.branch_reads.get(branch, 0) + 1
            sha = self.branch_shas.get(branch, SHA_A)
            if branch == "main" and self.change_main_after_write and self.branch_reads[branch] > 1:
                sha = SHA_B
            return self._response({"name": branch, "commit": {"sha": sha}})
        if "/contents/Project.toml?" in path and method == "GET":
            content = base64.b64encode(
                f'version = "{self.project_version}"\n'.encode("ascii")
            ).decode("ascii")
            return self._response({"path": "Project.toml", "encoding": "base64", "content": content})
        return ApiResponse(404, b"{}")

    @staticmethod
    def _response(value, headers=None) -> ApiResponse:
        return ApiResponse(200, json.dumps(value, sort_keys=True).encode(), headers or {})


def adapter_for(fake: FakeGitHub, *, gate=None) -> GitHubProtectionAdapter:
    return GitHubProtectionAdapter(
        REPOSITORY,
        token="private-token",
        transport=fake,
        clock=lambda: datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc),
        authorization_gate=gate,
    )


class GitHubProtectionTests(unittest.TestCase):
    def test_snapshot_is_canonical_and_digested_from_exact_details(self) -> None:
        fake = FakeGitHub()
        adapter = adapter_for(fake)
        first = adapter.read_snapshot()
        second = adapter.read_snapshot()
        self.assertEqual(first.snapshot_sha256, second.snapshot_sha256)
        self.assertEqual(first.snapshot_sha256, __import__("hashlib").sha256(first.canonical_bytes).hexdigest())
        self.assertEqual([row.id for row in first.rulesets], sorted(row["id"] for row in fake.rows))
        self.assertIn(("GET", "/repos/Julia-meets-String-Theory/CYAxiverse.jl/rulesets/23967983", None), fake.calls)

    def test_candidate_pair_produces_exact_live_protection_evidence(self) -> None:
        adapter = adapter_for(FakeGitHub())
        evidence = adapter.verify_candidate_ref(OBSERVATION_REF)
        self.assertEqual(evidence.ruleset_ids, (23967983, 23967991))
        evidence.require(OBSERVATION_REF, creation=True)
        self.assertTrue(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))
        self.assertFalse(adapter.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository="other/repository"
        ))
        with self.assertRaisesRegex(Exception, "TARGET_MISMATCH"):
            evidence.require("refs/heads/candidates/other/candidate", creation=True)
        single = adapter.verify_candidate_ref("refs/heads/candidates/single")
        single.require("refs/heads/candidates/single", creation=True)

    def test_candidate_pair_fails_closed_on_missing_or_wrong_definitions(self) -> None:
        cases = []
        missing = candidate_rulesets()[:1]
        cases.append((missing, "CANDIDATE_PROTECTION_INCOMPLETE"))
        wrong_pattern = candidate_rulesets()
        wrong_pattern[0]["conditions"]["ref_name"]["include"] = ["refs/heads/candidates/*"]
        cases.append((wrong_pattern, "CANDIDATE_PROTECTION_IMMUTABLE_INVALID"))
        wrong_enforcement = candidate_rulesets()
        wrong_enforcement[1]["enforcement"] = "disabled"
        cases.append((wrong_enforcement, "CANDIDATE_PROTECTION_CREATION_INVALID"))
        wrong_immutable_bypass = candidate_rulesets()
        wrong_immutable_bypass[0]["bypass_actors"] = [
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        ]
        cases.append((wrong_immutable_bypass, "CANDIDATE_PROTECTION_IMMUTABLE_INVALID"))
        wrong_creation_bypass = candidate_rulesets()
        wrong_creation_bypass[1]["bypass_actors"] = []
        cases.append((wrong_creation_bypass, "CANDIDATE_PROTECTION_CREATION_INVALID"))
        missing_update = candidate_rulesets()
        missing_update[0]["rules"] = [rule("deletion"), rule("non_fast_forward")]
        cases.append((missing_update, "CANDIDATE_PROTECTION_IMMUTABLE_INVALID"))
        for rows, reason in cases:
            with self.subTest(reason=reason, rows=rows):
                with self.assertRaises(GitHubProtectionError) as caught:
                    adapter_for(FakeGitHub(rows)).verify_candidate_ref(OBSERVATION_REF)
                self.assertEqual(caught.exception.reason_code, reason)

    def test_candidate_ref_pattern_ambiguity_and_bad_ref_are_rejected(self) -> None:
        rows = candidate_rulesets() + [
            ruleset(991, "overlap", include=["refs/heads/candidates/cyax*"], rules=[rule("update")])
        ]
        with self.assertRaisesRegex(GitHubProtectionError, "OVERLAP_AMBIGUOUS"):
            adapter_for(FakeGitHub(rows)).verify_candidate_ref(OBSERVATION_REF)
        with self.assertRaisesRegex(GitHubProtectionError, "CANDIDATE_REF_INVALID"):
            adapter_for(FakeGitHub()).verify_candidate_ref("refs/heads/candidates")

    def test_stale_and_cross_repository_snapshots_are_rejected(self) -> None:
        fake = FakeGitHub()
        fake.stale_detail = True
        with self.assertRaisesRegex(GitHubProtectionError, "SNAPSHOT_CHANGED"):
            adapter_for(fake).read_snapshot()
        first = adapter_for(FakeGitHub())
        evidence = first.verify_candidate_ref(OBSERVATION_REF)
        other = adapter_for(FakeGitHub())
        self.assertFalse(other.is_live_protection_evidence(
            evidence, ref=OBSERVATION_REF, repository=REPOSITORY
        ))

    def test_updated_at_preserves_fractional_provider_precision(self) -> None:
        self.assertEqual(
            GitHubProtectionAdapter._updated_at("2026-09-24T15:00:00.123Z"),
            "2026-09-24T15:00:00.123000Z",
        )
        first = candidate_rulesets()
        second = candidate_rulesets()
        first[0]["updated_at"] = "2026-09-24T15:00:00.123Z"
        second[0]["updated_at"] = "2026-09-24T15:00:00.124Z"
        self.assertNotEqual(
            adapter_for(FakeGitHub(first)).read_snapshot().snapshot_sha256,
            adapter_for(FakeGitHub(second)).read_snapshot().snapshot_sha256,
        )

    def test_api_errors_are_reason_coded_without_credentials(self) -> None:
        fake = FakeGitHub()
        fake.fail_next = True
        with self.assertRaises(GitHubProtectionError) as caught:
            adapter_for(fake).read_snapshot()
        self.assertEqual(caught.exception.reason_code, "GITHUB_API_REQUEST_FAILED")
        self.assertNotIn("private-token", str(caught.exception))
        self.assertNotIn("private-token", repr(caught.exception))
        self.assertNotIn("private-token", "".join(traceback.format_exception(caught.exception)))

    def test_read_only_rule_evaluation_binds_provider_and_snapshot_identity(self) -> None:
        fake = FakeGitHub()
        observation = adapter_for(fake).observe_rules_for_ref(OBSERVATION_REF)
        self.assertEqual(observation.ruleset_ids, (23967983, 23967991))
        self.assertEqual(
            observation.rule_types,
            ("creation", "deletion", "non_fast_forward", "update"),
        )
        self.assertEqual(observation.provider_etag, '"effective-rules"')
        self.assertEqual(observation.provider_request_id, "REQ-0125")
        self.assertEqual(len(observation.response_sha256), 64)
        self.assertIn("/repos/Julia-meets-String-Theory/CYAxiverse.jl/rules/branches/candidates/cyax-0125/gate-a-observation", [call[1] for call in fake.calls])

    def test_rule_evaluation_mismatch_and_wrong_namespace_fail_closed(self) -> None:
        fake = FakeGitHub()
        fake.evaluated_rules[0]["ruleset_id"] = 800
        with self.assertRaisesRegex(GitHubProtectionError, "EVALUATION_CANDIDATE_MISMATCH"):
            adapter_for(fake).observe_rules_for_ref(OBSERVATION_REF)
        with self.assertRaisesRegex(GitHubProtectionError, "EVALUATION_REF_INVALID"):
            adapter_for(FakeGitHub()).observe_rules_for_ref("refs/tags/v0.3.0")

    def test_freeze_controls_require_authorization_and_verify_write_readback(self) -> None:
        fake = FakeGitHub()
        calls = []
        adapter = adapter_for(
            fake,
            gate=lambda intent, action, ref: calls.append((action, ref)) or True,
        )
        result = adapter.freeze_main({"transaction": "test", "repository": REPOSITORY})
        lease = result["token"]
        self.assertEqual(result["sha"], SHA_A)
        self.assertEqual(result["version"], "0.2.0")
        self.assertTrue(adapter.is_live_freeze_lease(lease, ref="refs/heads/main"))
        self.assertEqual(next(row for row in fake.rows if row["id"] == 23948127)["enforcement"], "active")
        put_calls = [call for call in fake.calls if call[0] == "PUT"]
        self.assertEqual(len(put_calls), 1)
        self.assertEqual(set(put_calls[0][2]), {"name", "target", "enforcement", "conditions", "rules", "bypass_actors"})
        adapter.unfreeze_main(lease)
        self.assertFalse(adapter.is_live_freeze_lease(lease, ref="refs/heads/main"))
        self.assertEqual(next(row for row in fake.rows if row["id"] == 23948127)["enforcement"], "disabled")
        self.assertEqual(len([call for call in fake.calls if call[0] == "PUT"]), 2)
        self.assertEqual(calls, [("activate", "refs/heads/main"), ("deactivate", "refs/heads/main")])

    def test_freeze_failures_leave_control_active_or_do_not_write(self) -> None:
        no_gate = adapter_for(FakeGitHub())
        with self.assertRaisesRegex(GitHubProtectionError, "AUTHORIZATION_UNVERIFIED"):
            no_gate.freeze_line({"repository": REPOSITORY})
        fake = FakeGitHub()
        fake.change_main_after_write = True
        adapter = adapter_for(fake, gate=lambda *_: True)
        with self.assertRaisesRegex(GitHubProtectionError, "TARGET_MOVED"):
            adapter.freeze_main({"repository": REPOSITORY})
        self.assertEqual(next(row for row in fake.rows if row["id"] == 23948127)["enforcement"], "active")

    def test_freeze_rejects_wrong_rule_or_stale_lease_without_releasing(self) -> None:
        wrong = FakeGitHub()
        wrong.rows[-1]["bypass_actors"] = [
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        ]
        adapter = adapter_for(wrong, gate=lambda *_: True)
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_RULESET_INVALID"):
            adapter.freeze_main({"repository": REPOSITORY})
        self.assertEqual(next(row for row in wrong.rows if row["id"] == 23948127)["enforcement"], "disabled")

        fake = FakeGitHub()
        adapter = adapter_for(fake, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        target = next(row for row in fake.rows if row["id"] == 23948127)
        target["updated_at"] = "2026-09-24T15:00:02Z"
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_RULESET_STALE"):
            adapter.unfreeze_main(lease)
        self.assertEqual(target["enforcement"], "active")

    def test_live_lease_check_reloads_enforcement_and_timestamp(self) -> None:
        fake = FakeGitHub()
        adapter = adapter_for(fake, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        self.assertTrue(adapter.verify_live_freeze_lease(lease, ref="refs/heads/main"))
        target = next(row for row in fake.rows if row["id"] == 23948127)
        target["enforcement"] = "disabled"
        target["updated_at"] = "2026-09-24T15:00:02Z"
        self.assertFalse(adapter.verify_live_freeze_lease(lease, ref="refs/heads/main"))

    def test_release_rejects_branch_move_before_write_and_restores_after_move(self) -> None:
        before_write = FakeGitHub()
        adapter = adapter_for(before_write, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        put_count = len([call for call in before_write.calls if call[0] == "PUT"])
        before_write.branch_shas["main"] = SHA_B
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_TARGET_MOVED"):
            adapter.unfreeze_main(lease)
        self.assertEqual(
            len([call for call in before_write.calls if call[0] == "PUT"]),
            put_count,
        )
        self.assertEqual(
            next(row for row in before_write.rows if row["id"] == 23948127)["enforcement"],
            "active",
        )

        after_write = FakeGitHub()
        adapter = adapter_for(after_write, gate=lambda *_: True)
        lease = adapter.freeze_main({"repository": REPOSITORY})["token"]
        after_write.move_main_after_disable = True
        with self.assertRaisesRegex(GitHubProtectionError, "FREEZE_TARGET_MOVED"):
            adapter.unfreeze_main(lease)
        target = next(row for row in after_write.rows if row["id"] == 23948127)
        self.assertEqual(target["enforcement"], "active")
        self.assertEqual(after_write.branch_shas["main"], SHA_B)
        self.assertEqual(
            len([call for call in after_write.calls if call[0] == "PUT"]), 3,
            "activation, attempted release, then verified restoration",
        )


if __name__ == "__main__":
    unittest.main()
