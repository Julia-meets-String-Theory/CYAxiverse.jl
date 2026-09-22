#!/usr/bin/env python3
"""Focused release-neutral documentation routing checks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_version_lifecycle_release import publication, released  # noqa: E402
from version_lifecycle.manifests import canonical_manifest_bytes  # noqa: E402
from version_lifecycle.release import TERMINAL_CONSISTENT, validate_release_consistency  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
MAIN_SHA = "c" * 40
TREE = "b" * 40
PUBLICATION_EVIDENCE = b"synthetic publication evidence\n"
PUBLICATION_EVIDENCE_DIGEST = hashlib.sha256(PUBLICATION_EVIDENCE).hexdigest()


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
        self.assertNotIn(
            'git show "$GITHUB_REF:$PUBLICATION_EVIDENCE_REF"', workflow
        )

    def test_manifest_pair_is_terminal_only_when_both_bind(self) -> None:
        release = released()
        publication_manifest = publication(
            release, publication_evidence_digest=PUBLICATION_EVIDENCE_DIGEST
        )
        self.assertEqual(
            validate_release_consistency(
                release,
                publication_manifest,
                public_tag="v1.2.3",
                tag_commit=SHA,
                tag_tree=TREE,
                certified_tree=TREE,
                github_release_id=9001,
                publication_evidence_digest=PUBLICATION_EVIDENCE_DIGEST,
            )["status"],
            TERMINAL_CONSISTENT,
        )
        publication_manifest["tag_tree"] = "c" * 40
        self.assertNotEqual(validate_release_consistency(release, publication_manifest)["status"], TERMINAL_CONSISTENT)

    def test_python_verifier_rejects_tag_manifest_mismatch(self) -> None:
        release = released()
        publication_manifest = publication(
            release, publication_evidence_digest=PUBLICATION_EVIDENCE_DIGEST
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release_path = root / "released.json"
            publication_path = root / "publication.json"
            evidence_path = root / "publication-evidence.txt"
            env_path = root / "github.env"
            release_path.write_bytes(canonical_manifest_bytes(release))
            publication_path.write_bytes(canonical_manifest_bytes(publication_manifest))
            evidence_path.write_bytes(PUBLICATION_EVIDENCE)
            result = subprocess.run(
                [sys.executable, str(ROOT / "docs/verify_release_context.py"),
                 "--manifest", str(release_path), "--publication-manifest", str(publication_path),
                 "--publication-evidence", str(evidence_path),
                 "--tag-ref", "refs/tags/v1.2.4", "--tag", "v1.2.4",
                 "--tag-sha", SHA, "--tag-tree", TREE, "--tag-version", "1.2.4",
                 "--main-sha", SHA, "--main-version", "1.2.3", "--github-env", str(env_path)],
                cwd=ROOT, check=False, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("blocked", result.stderr)

    def test_python_verifier_exports_only_exact_verified_context(self) -> None:
        release = released()
        publication_manifest = publication(
            release, publication_evidence_digest=PUBLICATION_EVIDENCE_DIGEST
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release_path = root / "released.json"
            publication_path = root / "publication.json"
            evidence_path = root / "publication-evidence.txt"
            env_path = root / "github.env"
            release_path.write_bytes(canonical_manifest_bytes(release))
            publication_path.write_bytes(canonical_manifest_bytes(publication_manifest))
            evidence_path.write_bytes(PUBLICATION_EVIDENCE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--manifest",
                    str(release_path),
                    "--publication-manifest",
                    str(publication_path),
                    "--publication-evidence",
                    str(evidence_path),
                    "--tag-ref",
                    "refs/tags/v1.2.3",
                    "--tag",
                    "v1.2.3",
                    "--tag-sha",
                    SHA,
                    "--tag-tree",
                    TREE,
                    "--tag-version",
                    "1.2.3",
                    "--main-sha",
                    SHA,
                    "--main-version",
                    "1.2.3",
                    "--github-release-id",
                    "9001",
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
            self.assertIn(f"CYAX_DOCS_RELEASE_SHA={SHA}", exported)
            self.assertIn("CYAX_DOCS_STABLE_TAG=v1.2.3", exported)

    def test_maintenance_verifier_preserves_current_principal_stable_tag(self) -> None:
        release = released(
            release_line="maintenance/1.2",
            previous_main_sha=None,
            previous_main_version=None,
            main_at_release_sha=MAIN_SHA,
            main_at_release_version="2.0.0",
        )
        publication_manifest = publication(
            release, publication_evidence_digest=PUBLICATION_EVIDENCE_DIGEST
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release_path = root / "released.json"
            publication_path = root / "publication.json"
            evidence_path = root / "publication-evidence.txt"
            env_path = root / "github.env"
            release_path.write_bytes(canonical_manifest_bytes(release))
            publication_path.write_bytes(canonical_manifest_bytes(publication_manifest))
            evidence_path.write_bytes(PUBLICATION_EVIDENCE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--manifest", str(release_path),
                    "--publication-manifest", str(publication_path),
                    "--publication-evidence", str(evidence_path),
                    "--tag-ref", "refs/tags/v1.2.3",
                    "--tag", "v1.2.3",
                    "--tag-sha", SHA,
                    "--tag-tree", TREE,
                    "--tag-version", "1.2.3",
                    "--main-sha", MAIN_SHA,
                    "--main-version", "2.0.0",
                    "--github-release-id", "9001",
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
            self.assertIn("CYAX_DOCS_STABLE_TAG=v2.0.0", exported)

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
