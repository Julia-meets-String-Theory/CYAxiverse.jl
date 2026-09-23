#!/usr/bin/env python3
"""Focused release-neutral documentation routing checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
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
from version_lifecycle.publication_evidence import (  # noqa: E402
    canonical_publication_evidence_bytes,
    publication_evidence_for_tag,
)
from version_lifecycle.release import TERMINAL_CONSISTENT, validate_release_consistency  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
MAIN_SHA = "c" * 40
TREE = "b" * 40

def publication_evidence_bytes(release: dict[str, object]) -> bytes:
    tag = {
        "name": release["public_tag"],
        "commit": release["final_release_sha"],
        "tree": release["final_release_tree"],
    }
    record = publication_evidence_for_tag(release, tag)
    return canonical_publication_evidence_bytes(
        record,
        release,
        public_tag=str(release["public_tag"]),
        tag_commit=str(release["final_release_sha"]),
        tag_tree=str(release["final_release_tree"]),
    )


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

    exact_publication_evidence = publication_evidence_bytes(release)
    exact_publication_evidence_digest = hashlib.sha256(
        exact_publication_evidence
    ).hexdigest()
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
        exact_publication_evidence_digest
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
        "publication_evidence_digest": hashlib.sha256(
            publication_evidence_bytes(release)
        ).hexdigest(),
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
    evidence_path.write_bytes(publication_evidence_bytes(release))
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


def write_stable_selector_context(
    root: Path,
    lifecycle_records: object,
    canonical_tags: object,
    github_releases: object,
) -> list[str]:
    lifecycle_path = root / "stable-lifecycle.json"
    tags_path = root / "stable-tags.json"
    releases_path = root / "stable-releases.json"
    lifecycle_path.write_text(json.dumps(lifecycle_records, sort_keys=True), encoding="utf-8")
    tags_path.write_text(json.dumps(canonical_tags, sort_keys=True), encoding="utf-8")
    releases_path.write_text(json.dumps(github_releases, sort_keys=True), encoding="utf-8")
    return [
        "--lifecycle-index", str(lifecycle_path),
        "--canonical-tags", str(tags_path),
        "--github-releases", str(releases_path),
    ]


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
        self.assertIn("stable-lifecycle-index.json", workflow)
        self.assertIn("stable-canonical-tags.json", workflow)
        self.assertIn("stable-github-releases.json", workflow)
        self.assertIn(
            '--lifecycle-index "$RUNNER_TEMP/stable-lifecycle-index.json"',
            workflow,
        )
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

    def test_development_verification_only_runs_for_vmm_push(self) -> None:
        workflow = (ROOT / ".github/workflows/Documentation.yml").read_text(
            encoding="utf-8"
        )
        step = re.search(
            r"(?ms)^      - name: Verify development documentation context\n"
            r"        if: ([^\n]+)\n",
            workflow,
        )
        self.assertIsNotNone(step)
        condition = step.group(1)
        self.assertEqual(
            condition,
            "github.event_name == 'push' && github.ref == 'refs/heads/vmm'",
        )
        comparisons = re.findall(
            r"github\.(event_name|ref) == '([^']+)'", condition
        )
        self.assertEqual(len(comparisons), 2)

        def condition_matches(event_name: str, ref: str) -> bool:
            context = {"event_name": event_name, "ref": ref}
            return all(context[field] == value for field, value in comparisons)

        self.assertFalse(condition_matches("workflow_dispatch", "refs/heads/vmm"))
        self.assertTrue(condition_matches("push", "refs/heads/vmm"))
        self.assertFalse(condition_matches("push", "refs/heads/main"))

    def test_stable_only_clears_inherited_tag_before_first_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_path = root / "github.env"
            authority_args = write_stable_selector_context(root, {}, [], [])
            process_env = os.environ.copy()
            process_env["CYAX_DOCS_STABLE_TAG"] = "v9.9.9"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--stable-only",
                    "--main-sha", SHA,
                    "--main-version", "2.1.0",
                    *authority_args,
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                env=process_env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            context = json.loads(verified.stdout)
            self.assertEqual(context["CYAX_DOCS_STABLE_TAG"], "")
            self.assertEqual(
                env_path.read_text(encoding="utf-8").splitlines()[-1],
                "CYAX_DOCS_STABLE_TAG=",
            )

    def test_stable_only_uses_terminal_release_matching_current_main(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication, complete_args = write_complete_context(root)
            authority_args = []
            for name in ("--lifecycle-index", "--canonical-tags", "--github-releases"):
                index = complete_args.index(name)
                authority_args.extend(complete_args[index:index + 2])
            env_path = root / "github.env"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--stable-only",
                    "--main-sha", str(release["final_release_sha"]),
                    "--main-version", str(release["final_version"]),
                    *authority_args,
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            context = json.loads(verified.stdout)
            self.assertEqual(context["CYAX_DOCS_STABLE_TAG"], publication["public_tag"])

    def test_stable_only_rejects_public_tag_without_terminal_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority_args = write_stable_selector_context(
                root,
                {},
                [{"ref": "refs/tags/v2.1.0", "tag": "v2.1.0", "commit": SHA, "tree": TREE}],
                [],
            )
            env_path = root / "github.env"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--stable-only",
                    "--main-sha", SHA,
                    "--main-version", "2.1.0",
                    *authority_args,
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 2)
            self.assertIn("complete public tag/release/lifecycle universe disagrees", verified.stderr)

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
            self.assertIn("CYAX_DOCS_STABLE=true", exported)
            self.assertIn("CYAX_DOCS_STABLE_TAG=v1.2.3", exported)

    def test_python_verifier_rejects_contradictory_publication_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication_manifest, complete_args = write_complete_context(root)
            evidence_path = root / "publication-evidence.txt"
            evidence = json.loads(evidence_path.read_bytes())
            evidence["tag_tree"] = "f" * 40
            evidence_raw = json.dumps(
                evidence, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            evidence_path.write_bytes(evidence_raw)

            publication_manifest["publication_evidence_digest"] = hashlib.sha256(
                evidence_raw
            ).hexdigest()
            publication_manifest.pop("manifest_id")
            publication_manifest.pop("publication_id")
            publication_manifest = seal_manifest(publication_manifest)
            publication_path = root / "publication.json"
            publication_path.write_bytes(canonical_manifest_bytes(publication_manifest))
            lifecycle_path = root / "lifecycle.json"
            records = json.loads(lifecycle_path.read_text(encoding="utf-8"))
            records[lifecycle_ref_for_manifest(publication_manifest)] = publication_manifest
            lifecycle_path.write_text(
                json.dumps(records, sort_keys=True), encoding="utf-8"
            )

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
                    "--main-sha", str(release["main_at_release_sha"]),
                    "--main-version", str(release["main_at_release_version"]),
                    "--github-release-id", str(publication_manifest["github_release_id"]),
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("publication evidence is invalid", result.stderr)

    def test_historical_principal_tag_remains_versioned_when_main_advances(self) -> None:
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
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = env_path.read_text(encoding="utf-8")
            self.assertIn("CYAX_DOCS_STABLE=false", exported)
            self.assertIn("CYAX_DOCS_STABLE_TAG=\n", exported)
            self.assertIn(
                f"CYAX_DOCS_RELEASE_VERSION={release['final_version']}",
                exported,
            )

    def test_principal_commit_without_current_version_does_not_advance_stable(self) -> None:
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
                    "--main-sha", str(release["final_release_sha"]),
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
            self.assertIn("CYAX_DOCS_STABLE_TAG=\n", exported)

    def test_maintenance_verifier_does_not_invent_principal_stable_tag(self) -> None:
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
            self.assertIn("CYAX_DOCS_STABLE_TAG=\n", exported)

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for Documenter route checks")
    def test_vmm_before_first_release_does_not_fabricate_stable_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_path = root / "github.env"
            env_path.write_text("CYAX_DOCS_STABLE_TAG=v9.9.9\n", encoding="utf-8")
            authority_args = write_stable_selector_context(root, {}, [], [])
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--stable-only",
                    "--main-sha", SHA,
                    "--main-version", "2.1.0",
                    *authority_args,
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            context = json.loads(verified.stdout)
            self.assertEqual(context["CYAX_DOCS_STABLE"], "false")
            self.assertEqual(context["CYAX_DOCS_STABLE_TAG"], "")
            self.assertEqual(
                env_path.read_text(encoding="utf-8").splitlines()[-1],
                "CYAX_DOCS_STABLE_TAG=",
            )

            environment = os.environ.copy()
            environment.pop("CYAX_DOCS_STABLE_TAG", None)
            environment.update(context)
            environment.update({
                "CYAX_DOCS_REF": "refs/heads/vmm",
                "CYAX_DOCS_ROUTE_ONLY": "true",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/vmm",
            })
            route = subprocess.run(
                ["julia", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(route.returncode, 0, route.stderr)
            self.assertEqual(route.stdout.strip(), "development")

            environment["CYAX_DOCS_ROUTE_ONLY"] = "versions"
            versions = subprocess.run(
                ["julia", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(versions.returncode, 0, versions.stderr)
            self.assertEqual(
                versions.stdout.strip(),
                "v#.#.#,dev:dev",
            )

            environment.update({
                "CYAX_DOCS_ROUTE_ONLY": "documenter",
                "GITHUB_ACTIONS": "true",
                "GITHUB_REPOSITORY": "Julia-meets-String-Theory/CYAxiverse.jl",
                "GITHUB_ACTOR": "fixture-owner",
                "GITHUB_TOKEN": "synthetic-token-for-routing-only",
            })
            selected = subprocess.run(
                ["julia", "--project=docs/", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertEqual(selected.stdout.strip().splitlines()[-1], "dev")

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for Documenter route checks")
    def test_vmm_after_current_principal_release_uses_verified_stable_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication, complete_args = write_complete_context(root)
            authority_args = [
                complete_args[complete_args.index("--lifecycle-index")],
                complete_args[complete_args.index("--lifecycle-index") + 1],
                complete_args[complete_args.index("--canonical-tags")],
                complete_args[complete_args.index("--canonical-tags") + 1],
                complete_args[complete_args.index("--github-releases")],
                complete_args[complete_args.index("--github-releases") + 1],
            ]
            env_path = root / "github.env"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    "--stable-only",
                    "--main-sha", str(release["final_release_sha"]),
                    "--main-version", str(release["final_version"]),
                    *authority_args,
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            context = json.loads(verified.stdout)
            self.assertEqual(context["CYAX_DOCS_STABLE"], "false")
            self.assertEqual(context["CYAX_DOCS_STABLE_TAG"], publication["public_tag"])

            environment = os.environ.copy()
            environment.update(context)
            environment.update({
                "CYAX_DOCS_REF": "refs/heads/vmm",
                "CYAX_DOCS_ROUTE_ONLY": "versions",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REF": "refs/heads/vmm",
            })
            versions = subprocess.run(
                ["julia", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(versions.returncode, 0, versions.stderr)
            self.assertEqual(
                versions.stdout.strip(),
                f"stable:{publication['public_tag']},v#.#.#,dev:dev",
            )

            environment.update({
                "CYAX_DOCS_ROUTE_ONLY": "documenter",
                "GITHUB_ACTIONS": "true",
                "GITHUB_REPOSITORY": "Julia-meets-String-Theory/CYAxiverse.jl",
                "GITHUB_ACTOR": "fixture-owner",
                "GITHUB_TOKEN": "synthetic-token-for-routing-only",
            })
            selected = subprocess.run(
                ["julia", "--project=docs/", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertEqual(selected.stdout.strip().splitlines()[-1], "dev")

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

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for Documenter route checks")
    def test_workflow_dispatch_uses_verified_tag_for_documenter_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication_manifest, complete_args = write_complete_context(root)
            env_path = root / "github.env"
            tag_ref = f"refs/tags/{publication_manifest['public_tag']}"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    *complete_args,
                    "--tag-ref", tag_ref,
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
            self.assertEqual(verified.returncode, 0, verified.stderr)
            doc_context = json.loads(verified.stdout)
            self.assertEqual(doc_context["CYAX_DOCS_STABLE"], "false")

            environment = os.environ.copy()
            environment.update(doc_context)
            environment.update({
                "CYAX_DOCS_ROUTE_ONLY": "documenter",
                "CYAX_DOCS_REF": tag_ref,
                "CYAX_DOCS_TAG_REF": tag_ref,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/vmm",
                "GITHUB_REPOSITORY": "Julia-meets-String-Theory/CYAxiverse.jl",
                "GITHUB_ACTOR": "fixture-owner",
                "GITHUB_TOKEN": "synthetic-token-for-routing-only",
            })
            selected = subprocess.run(
                ["julia", "--project=docs/", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertEqual(selected.stdout.strip().splitlines()[-1], "v1.2.3")
            self.assertNotIn("deploying devbranch build", selected.stderr)

            wrong_tag = dict(environment)
            wrong_tag["CYAX_DOCS_TAG_REF"] = "refs/tags/v9.9.9"
            rejected = subprocess.run(
                ["julia", "--project=docs/", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=wrong_tag,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("DOCS_DISPATCH_REF_MISMATCH", rejected.stderr)

    @unittest.skipUnless(shutil.which("julia"), "Julia is required for Documenter route checks")
    def test_current_principal_workflow_dispatch_uses_stable_route_and_exact_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release, publication_manifest, complete_args = write_complete_context(root)
            env_path = root / "github.env"
            tag_ref = f"refs/tags/{publication_manifest['public_tag']}"
            verified = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "docs/verify_release_context.py"),
                    *complete_args,
                    "--tag-ref", tag_ref,
                    "--tag", str(publication_manifest["public_tag"]),
                    "--tag-sha", str(publication_manifest["tag_commit"]),
                    "--tag-tree", str(publication_manifest["tag_tree"]),
                    "--tag-version", str(release["final_version"]),
                    "--main-sha", str(release["final_release_sha"]),
                    "--main-version", str(release["final_version"]),
                    "--github-release-id", str(publication_manifest["github_release_id"]),
                    "--github-env", str(env_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            doc_context = json.loads(verified.stdout)
            self.assertEqual(doc_context["CYAX_DOCS_STABLE"], "true")

            environment = os.environ.copy()
            environment.update(doc_context)
            environment.update({
                "CYAX_DOCS_ROUTE_ONLY": "true",
                "CYAX_DOCS_REF": tag_ref,
                "CYAX_DOCS_TAG_REF": tag_ref,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/vmm",
            })
            route = subprocess.run(
                ["julia", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(route.returncode, 0, route.stderr)
            self.assertEqual(route.stdout.strip(), "stable")

            environment.update({
                "CYAX_DOCS_ROUTE_ONLY": "documenter",
                "GITHUB_REPOSITORY": "Julia-meets-String-Theory/CYAxiverse.jl",
                "GITHUB_ACTOR": "fixture-owner",
                "GITHUB_TOKEN": "synthetic-token-for-routing-only",
            })
            selected = subprocess.run(
                ["julia", "--project=docs/", "--startup-file=no", "docs/make.jl"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertEqual(selected.stdout.strip().splitlines()[-1], "v1.2.3")
            self.assertNotIn("deploying devbranch build", selected.stderr)


if __name__ == "__main__":
    unittest.main()
