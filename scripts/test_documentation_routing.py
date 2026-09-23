#!/usr/bin/env python3
"""Focused release-neutral documentation routing checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_version_lifecycle_manifests as manifest_tests  # noqa: E402
from version_lifecycle.manifests import (  # noqa: E402
    canonical_manifest_bytes,
    lifecycle_ref_for_manifest,
    seal_manifest,
)
from version_lifecycle.release import TERMINAL_CONSISTENT, validate_release_consistency  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
MAIN_SHA = "c" * 40
TREE = "b" * 40
PUBLICATION_EVIDENCE = b"synthetic publication evidence\n"
PUBLICATION_EVIDENCE_DIGEST = hashlib.sha256(PUBLICATION_EVIDENCE).hexdigest()


def complete_fixture(
    *, maintenance: bool = False
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    chain = manifest_tests.ManifestTests().chain()
    if maintenance:
        prepared = dict(chain[0])
        prepared.pop("manifest_id")
        prepared["owner_line"] = "maintenance/1.2"
        prepared = seal_manifest(prepared)

        opened = dict(chain[1])
        opened.pop("manifest_id")
        opened["owner_line"] = "maintenance/1.2"
        opened["predecessor_refs"] = [lifecycle_ref_for_manifest(prepared)]
        opened = seal_manifest(opened)

        claim = dict(chain[2])
        claim.pop("manifest_id")
        claim["owner_line"] = "maintenance/1.2"
        claim["predecessor_refs"] = [lifecycle_ref_for_manifest(opened)]
        claim = seal_manifest(claim)

        consumed = dict(chain[3])
        consumed.pop("manifest_id")
        consumed["owner_line"] = "maintenance/1.2"
        consumed["predecessor_refs"] = [lifecycle_ref_for_manifest(opened)]
        consumed = seal_manifest(consumed)

        candidate = dict(chain[4])
        candidate.pop("manifest_id")
        candidate["predecessor_refs"] = [lifecycle_ref_for_manifest(claim)]
        candidate["release_line"] = "maintenance/1.2"
        candidate = seal_manifest(candidate)

        intent = dict(chain[5])
        intent.pop("manifest_id")
        intent["predecessor_refs"] = [lifecycle_ref_for_manifest(candidate)]
        intent["release_line"] = "maintenance/1.2"
        intent = seal_manifest(intent)

        release = dict(chain[6])
        release.pop("manifest_id")
        release["predecessor_refs"] = [lifecycle_ref_for_manifest(intent)]
        release["release_line"] = "maintenance/1.2"
        release.pop("previous_main_sha")
        release.pop("previous_main_version")
        release["main_at_release_sha"] = MAIN_SHA
        release["main_at_release_version"] = "2.0.0"
        release["main_at_candidate_sha"] = MAIN_SHA
        release["main_at_candidate_version"] = "2.0.0"
        release = seal_manifest(release)
        prefix = [prepared, opened, claim, consumed]
    else:
        release = chain[6]
        prefix = chain[:6]

    publication_manifest = dict(chain[7])
    publication_manifest.pop("manifest_id")
    publication_manifest.pop("publication_id")
    publication_manifest["predecessor_refs"] = [
        lifecycle_ref_for_manifest(release)
    ]
    publication_manifest["released_manifest_ref"] = lifecycle_ref_for_manifest(
        release
    )
    publication_manifest["released_manifest_id"] = release["manifest_id"]
    publication_manifest["released_manifest_digest"] = hashlib.sha256(
        canonical_manifest_bytes(release)
    ).hexdigest()
    publication_manifest["publication_evidence_digest"] = (
        PUBLICATION_EVIDENCE_DIGEST
    )
    publication_manifest = seal_manifest(publication_manifest)
    records = {
        lifecycle_ref_for_manifest(item): item
        for item in [*prefix, *([candidate, intent] if maintenance else []), release, publication_manifest]
    }
    observations: dict[str, object] = {
        "records": records,
        "canonical_tags": [{
            "ref": f"refs/tags/{publication_manifest['public_tag']}",
            "tag": publication_manifest["public_tag"],
            "commit": publication_manifest["tag_commit"],
            "tree": publication_manifest["tag_tree"],
        }],
        "github_releases": [{
            "id": publication_manifest["github_release_id"],
            "tag": publication_manifest["public_tag"],
            "url": publication_manifest["github_release_url"],
        }],
    }
    return release, publication_manifest, observations


def complete_validation_kwargs(
    release: dict[str, object],
    publication_manifest: dict[str, object],
    observations: dict[str, object],
) -> dict[str, object]:
    return {
        "public_tag": publication_manifest["public_tag"],
        "tag_commit": publication_manifest["tag_commit"],
        "tag_tree": publication_manifest["tag_tree"],
        "certified_tree": release["anchor_tree"],
        "github_release_id": publication_manifest["github_release_id"],
        "publication_evidence_digest": PUBLICATION_EVIDENCE_DIGEST,
        "lifecycle_records": observations["records"],
        "anchor_tag_object": release["anchor_sha"],
        "anchor_tree": release["anchor_tree"],
        "anchor_closure_timestamp_utc": release["closure_timestamp_utc"],
        "candidate_ref": release["candidate_ref"],
        "candidate_commit": release["candidate_sha"],
        "candidate_tree": release["candidate_tree"],
        "canonical_tag_observations": observations["canonical_tags"],
        "github_release_observations": observations["github_releases"],
        "require_complete_namespace": True,
    }


def write_complete_context(
    root: Path, *, maintenance: bool = False
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    release, publication_manifest, observations = complete_fixture(
        maintenance=maintenance
    )
    release_path = root / "released.json"
    publication_path = root / "publication.json"
    evidence_path = root / "publication-evidence.txt"
    lifecycle_path = root / "lifecycle.json"
    tags_path = root / "canonical-tags.json"
    github_releases_path = root / "github-releases.json"
    release_path.write_bytes(canonical_manifest_bytes(release))
    publication_path.write_bytes(canonical_manifest_bytes(publication_manifest))
    evidence_path.write_bytes(PUBLICATION_EVIDENCE)
    lifecycle_path.write_text(
        json.dumps(observations["records"], sort_keys=True), encoding="utf-8"
    )
    tags_path.write_text(
        json.dumps(observations["canonical_tags"], sort_keys=True), encoding="utf-8"
    )
    github_releases_path.write_text(
        json.dumps(observations["github_releases"], sort_keys=True),
        encoding="utf-8",
    )
    complete_args = [
        "--manifest", str(release_path),
        "--publication-manifest", str(publication_path),
        "--publication-evidence", str(evidence_path),
        "--require-complete-lifecycle",
        "--lifecycle-index", str(lifecycle_path),
        "--canonical-tags", str(tags_path),
        "--github-releases", str(github_releases_path),
        "--anchor-tag-object", str(release["anchor_sha"]),
        "--anchor-tree", str(release["anchor_tree"]),
        "--anchor-closure-timestamp-utc",
        str(release["closure_timestamp_utc"]),
        "--candidate-ref", str(release["candidate_ref"]),
        "--candidate-commit", str(release["candidate_sha"]),
        "--candidate-tree", str(release["candidate_tree"]),
    ]
    return release, publication_manifest, complete_args


class DocumentationRoutingTests(unittest.TestCase):
    def test_workflow_uses_only_immutable_manifest_authority(self) -> None:
        workflow = (ROOT / ".github/workflows/Documentation.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("release-events", workflow)
        self.assertIn("refs/heads/lifecycle/v1/*", workflow)
        self.assertIn("publication-evidence", workflow)
        self.assertIn("gh api", workflow)
        self.assertIn("releases/assets", workflow)
        self.assertIn("--require-complete-lifecycle", workflow)
        self.assertIn("canonical_manifest_bytes", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("publication_ref:", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)
        self.assertNotIn("tags: 'v*.*.*'", workflow)
        self.assertIn('value.startswith(prefix)', workflow)
        self.assertIn('parse_public_tag(parts[0])', workflow)
        self.assertLess(
            workflow.index('publication_ref is not an exact publication ref'),
            workflow.index('PUBLICATION_REF="refs/remotes/origin/'),
        )
        self.assertLess(
            workflow.index("triggering immutable publication manifest"),
            workflow.index("matching immutable released manifest"),
        )
        self.assertIn('git checkout --detach "$CYAX_DOCS_RELEASE_SHA"', workflow)
        self.assertIn('git fetch --no-tags -- origin "$ANCHOR_REF:$ANCHOR_REF"', workflow)
        self.assertIn("--anchor-closure-timestamp-utc", workflow)
        self.assertNotIn(
            'git show "$GITHUB_REF:$PUBLICATION_EVIDENCE_REF"', workflow
        )

    def test_manifest_pair_is_terminal_only_when_both_bind(self) -> None:
        release, publication_manifest, observations = complete_fixture()
        self.assertEqual(
            validate_release_consistency(
                release,
                publication_manifest,
                **complete_validation_kwargs(
                    release, publication_manifest, observations
                ),
            )["status"],
            TERMINAL_CONSISTENT,
        )
        publication_manifest["tag_tree"] = "c" * 40
        self.assertNotEqual(validate_release_consistency(release, publication_manifest)["status"], TERMINAL_CONSISTENT)

    def test_python_verifier_rejects_tag_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _release, _publication, complete_args = write_complete_context(root)
            env_path = root / "github.env"
            result = subprocess.run(
                [sys.executable, str(ROOT / "docs/verify_release_context.py"),
                 *complete_args,
                 "--tag-ref", "refs/tags/v1.2.4", "--tag", "v1.2.4",
                 "--tag-sha", SHA, "--tag-tree", TREE, "--tag-version", "1.2.4",
                 "--main-sha", SHA, "--main-version", "1.2.3", "--github-env", str(env_path)],
                cwd=ROOT, check=False, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("blocked", result.stderr)

    def test_python_verifier_exports_only_exact_verified_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication_manifest, complete_args = write_complete_context(
                root
            )
            env_path = root / "github.env"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    *complete_args,
                    "--tag-ref",
                    f"refs/tags/{publication_manifest['public_tag']}",
                    "--tag",
                    str(publication_manifest["public_tag"]),
                    "--tag-sha",
                    str(publication_manifest["tag_commit"]),
                    "--tag-tree",
                    str(publication_manifest["tag_tree"]),
                    "--tag-version",
                    str(release["final_version"]),
                    "--main-sha",
                    str(release["main_at_release_sha"]),
                    "--main-version",
                    str(release["main_at_release_version"]),
                    "--github-release-id",
                    str(publication_manifest["github_release_id"]),
                    "--github-env",
                    str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = env_path.read_text(encoding="utf-8")
            self.assertIn("CYAX_DOCS_MANIFEST_STATUS=verified", exported)
            self.assertIn(
                f"CYAX_DOCS_RELEASE_SHA={release['final_release_sha']}", exported
            )
            self.assertIn("CYAX_DOCS_STABLE_TAG=v1.2.3", exported)

    def test_principal_verifier_rejects_current_main_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication_manifest, complete_args = write_complete_context(root)
            env_path = root / "github.env"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    *complete_args,
                    "--tag-ref", f"refs/tags/{publication_manifest['public_tag']}",
                    "--tag", str(publication_manifest["public_tag"]),
                    "--tag-sha", str(publication_manifest["tag_commit"]),
                    "--tag-tree", str(publication_manifest["tag_tree"]),
                    "--tag-version", str(release["final_version"]),
                    "--main-sha", "d" * 40,
                    "--main-version", "2.1.0",
                    "--github-release-id", str(publication_manifest["github_release_id"]),
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("current principal main SHA", result.stderr)

    def test_maintenance_verifier_preserves_current_principal_stable_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication_manifest, complete_args = write_complete_context(
                root, maintenance=True
            )
            env_path = root / "github.env"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    *complete_args,
                    "--tag-ref", "refs/tags/v1.2.3",
                    "--tag", "v1.2.3",
                    "--tag-sha", str(publication_manifest["tag_commit"]),
                    "--tag-tree", str(publication_manifest["tag_tree"]),
                    "--tag-version", "1.2.3",
                    "--main-sha", "d" * 40,
                    "--main-version", "2.1.0",
                    "--github-release-id", str(publication_manifest["github_release_id"]),
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = env_path.read_text(encoding="utf-8")
            self.assertIn("CYAX_DOCS_STABLE=false", exported)
            self.assertIn("CYAX_DOCS_PRINCIPAL_MAIN_SHA=" + "d" * 40, exported)
            self.assertIn("CYAX_DOCS_STABLE_TAG=v2.1.0", exported)

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for route checks")
    def test_julia_route_requires_verified_manifest_status(self) -> None:
        base = {
            "CYAX_DOCS_ROUTE_ONLY": "true", "DOCS_DEPLOY": "false",
            "CYAX_DOCS_REF": "refs/tags/v1.2.3",
            "CYAX_DOCS_RELEASE_LINE": "principal",
            "CYAX_DOCS_RELEASE_VERSION": "1.2.3",
            "CYAX_DOCS_RELEASE_SHA": SHA,
            "CYAX_DOCS_PRINCIPAL_MAIN_SHA": SHA,
        }
        missing = os.environ.copy()
        missing.update(base)
        result = subprocess.run(["julia", "--startup-file=no", "docs/make.jl"], cwd=ROOT, env=missing, check=False, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        verified = dict(missing)
        verified["CYAX_DOCS_MANIFEST_STATUS"] = "verified"
        result = subprocess.run(["julia", "--startup-file=no", "docs/make.jl"], cwd=ROOT, env=verified, check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
