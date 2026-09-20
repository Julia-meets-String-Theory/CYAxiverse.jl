"""Exact Git identity and protected-ref operations for lifecycle transactions.

These helpers do not establish GitHub protection. Callers must supply verified
protection evidence before any remote mutation. Gate A tests use bare fixture
repositories; no production lifecycle operation is performed by this module.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


FULL_REF = re.compile(r"^refs/(?:heads|tags|candidates)/[A-Za-z0-9._/-]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
UTC = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
PUBLIC_TAG_REF = re.compile(
    r"^refs/tags/v(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
)
LEGACY_TAG_REF = "refs/tags/v-0.1"


class GitIdentityError(RuntimeError):
    """A Git identity, protection, or exact-ref expectation failed."""


def require_full_ref(value: str) -> str:
    """Reject Git-invalid names before they become durable ref identities."""

    if (
        not isinstance(value, str)
        or FULL_REF.fullmatch(value) is None
        or value.endswith("/")
        or ".." in value
        or "//" in value
        or any(
            component.startswith(".")
            or component.endswith(".")
            or component.endswith(".lock")
            for component in value.split("/")
        )
    ):
        raise GitIdentityError("invalid fully qualified ref")
    return value


def require_candidate_ref(value: str) -> str:
    """Require a valid durable candidate ref under its canonical namespace."""

    require_full_ref(value)
    if not value.startswith("refs/heads/candidates/"):
        raise GitIdentityError("invalid candidate ref")
    return value


@dataclass(frozen=True)
class ProtectionEvidence:
    """Evidence that an external repository rule protects a ref pattern."""

    rule_id: str
    pattern: str
    snapshot_sha256: str
    retrieved_at_utc: str
    creation_guarded: bool
    update_guarded: bool
    deletion_guarded: bool

    def _matches(self, ref: str) -> bool:
        # Only exact names and validated trailing-star prefixes are accepted
        # here; GitHub's broader ruleset pattern language needs a separate
        # live matcher before its result can be represented by this evidence.
        return self.pattern == ref or (
            self.pattern.endswith("*") and ref.startswith(self.pattern[:-1])
        )

    def require(self, ref: str, *, creation: bool = False) -> None:
        if not self.rule_id or not re.fullmatch(r"[0-9a-f]{64}", self.snapshot_sha256):
            raise GitIdentityError("PROTECTION_EVIDENCE_UNAVAILABLE")
        if not self.pattern or not UTC.fullmatch(self.retrieved_at_utc):
            raise GitIdentityError("PROTECTION_EVIDENCE_UNAVAILABLE")
        try:
            dt.datetime.strptime(self.retrieved_at_utc, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise GitIdentityError("PROTECTION_EVIDENCE_UNAVAILABLE") from exc
        if creation and not self.creation_guarded:
            raise GitIdentityError("PROTECTION_EVIDENCE_UNAVAILABLE")
        if not self.update_guarded or not self.deletion_guarded:
            raise GitIdentityError("PROTECTION_EVIDENCE_UNAVAILABLE")
        # GitHub's pattern syntax is not fnmatch-compatible in every case.
        # The caller must provide an exact or validated prefix pattern.
        if not self._matches(ref):
            raise GitIdentityError("PROTECTION_EVIDENCE_UNAVAILABLE")

    def require_public_tag(self, ref: str) -> None:
        """Require an applicable create/update/delete rule excluding legacy."""

        try:
            if PUBLIC_TAG_REF.fullmatch(ref) is None:
                raise GitIdentityError("noncanonical public tag")
            self.require(ref, creation=True)
            if self._matches(LEGACY_TAG_REF):
                raise GitIdentityError("ruleset would also change legacy tag protection")
        except GitIdentityError as exc:
            raise GitIdentityError("PUBLIC_TAG_RULESET_UNAVAILABLE") from exc


class GitRepository:
    def __init__(self, root: str | Path, remote: str = "origin") -> None:
        self.root = Path(root)
        self.remote = remote

    def git(self, *args: str, input_bytes: bytes | None = None) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            input=input_bytes,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise GitIdentityError(f"git {' '.join(args)} failed: {message}")
        return result.stdout

    def object_type(self, object_id: str) -> str:
        self._sha(object_id)
        return self.git("cat-file", "-t", object_id).decode().strip()

    def tree(self, commit: str) -> str:
        self._sha(commit)
        if self.object_type(commit) != "commit":
            raise GitIdentityError("expected commit object")
        value = self.git("rev-parse", f"{commit}^{{tree}}").decode().strip()
        return self._sha(value)

    def project_version(self, commit: str) -> str:
        self._sha(commit)
        raw = self.git("show", f"{commit}:Project.toml")
        try:
            version = tomllib.loads(raw.decode("utf-8"))["version"]
        except (UnicodeError, KeyError, tomllib.TOMLDecodeError) as exc:
            raise GitIdentityError("invalid Project.toml version") from exc
        if not isinstance(version, str):
            raise GitIdentityError("Project.toml version must be a string")
        return version

    def remote_ref(self, ref: str) -> str | None:
        self._ref(ref)
        output = self.git("ls-remote", "--refs", self.remote, ref).decode().strip()
        if not output:
            return None
        entries = output.splitlines()
        if len(entries) != 1:
            raise GitIdentityError("remote ref is ambiguous")
        sha, name = entries[0].split("\t", 1)
        if name != ref:
            raise GitIdentityError("remote ref name mismatch")
        return self._sha(sha)

    def verify_durable_ref(self, ref: str, expected_object: str) -> None:
        self._sha(expected_object)
        if self.remote_ref(ref) != expected_object:
            raise GitIdentityError("DURABLE_REF_MISMATCH")

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        self._sha(ancestor)
        self._sha(descendant)
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=self.root,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise GitIdentityError(result.stderr.decode(errors="replace").strip())

    def principal_interval(
        self, candidate_main: str, frozen_main: str, certified_tree: str
    ) -> dict[str, object]:
        """Enumerate and disposition every commit between candidate and freeze.

        This is the concrete Git proof consumed by a principal release port.
        A caller still must prove that ``frozen_main`` is the protected remote
        head under an effective freeze before promoting anything.
        """
        self._sha(candidate_main)
        self._sha(frozen_main)
        self._sha(certified_tree)
        if not self.is_ancestor(candidate_main, frozen_main):
            raise GitIdentityError("CANDIDATE_MAIN_NOT_ANCESTOR")
        candidate_tree = self.tree(candidate_main)
        freeze_tree = self.tree(frozen_main)
        output = self.git("rev-list", "--reverse", f"{candidate_main}..{frozen_main}")
        commits = [
            {"sha": self._sha(sha), "tree": self.tree(sha)}
            for sha in output.decode("ascii").splitlines()
        ]
        if candidate_main == frozen_main:
            disposition = "no_drift"
        elif freeze_tree == certified_tree:
            disposition = "tree_neutral_included"
        else:
            disposition = "content_drift_blocked"
        return {
            "verified": True,
            "candidate_main_sha": candidate_main,
            "freeze_main_sha": frozen_main,
            "candidate_main_tree": candidate_tree,
            "freeze_main_tree": freeze_tree,
            "intervening_commits": commits,
            "disposition": disposition,
        }

    def push_create_only(
        self,
        ref: str,
        object_id: str,
        protection: ProtectionEvidence,
    ) -> None:
        """Create and verify a ref; block unless update/delete are protected.

        An existing identical ref is idempotent. Protected update denial is
        essential: Git alone cannot atomically promise create-if-absent.
        """
        self._ref(ref)
        self._sha(object_id)
        if ref.startswith("refs/tags/v"):
            protection.require_public_tag(ref)
        else:
            protection.require(ref, creation=True)
        current = self.remote_ref(ref)
        if current == object_id:
            return
        if current is not None:
            raise GitIdentityError("REF_ALREADY_EXISTS")
        # An absence precheck is not atomic with the push advertisement.  An
        # empty expected value in this per-ref lease makes the server reject
        # a ref that appeared in between, even when our object would be a
        # fast-forward update of the concurrent creator's object.
        try:
            self.git(
                "push", "--porcelain", f"--force-with-lease={ref}:",
                self.remote, f"{object_id}:{ref}",
            )
        except GitIdentityError as exc:
            observed = self.remote_ref(ref)
            if observed == object_id:
                return
            if observed is not None:
                raise GitIdentityError("REF_ALREADY_EXISTS") from exc
            raise
        self.verify_durable_ref(ref, object_id)

    def make_annotated_iteration_tag(
        self,
        *,
        version: str,
        commit: str,
        closure_timestamp_utc: str,
        tagger_name: str,
        tagger_email: str,
    ) -> str:
        """Create an annotated tag object with an explicit canonical UTC time."""
        self._sha(commit)
        if self.object_type(commit) != "commit":
            raise GitIdentityError("iteration anchor must identify a commit")
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version):
            raise GitIdentityError("invalid final version")
        if not UTC.fullmatch(closure_timestamp_utc):
            raise GitIdentityError("invalid closure UTC timestamp")
        try:
            instant = dt.datetime.strptime(closure_timestamp_utc, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise GitIdentityError("invalid closure UTC timestamp") from exc
        if any(char in tagger_name + tagger_email for char in "<>\n\r\x00"):
            raise GitIdentityError("invalid tagger identity")
        seconds = int(instant.replace(tzinfo=dt.timezone.utc).timestamp())
        tag_name = f"iterations/{version}"
        raw = (
            f"object {commit}\n"
            "type commit\n"
            f"tag {tag_name}\n"
            f"tagger {tagger_name} <{tagger_email}> {seconds} +0000\n"
            "\n"
            f"closure_timestamp_utc={closure_timestamp_utc}\n"
        ).encode("utf-8")
        object_id = self.git("mktag", input_bytes=raw).decode().strip()
        return self._sha(object_id)

    @staticmethod
    def _sha(value: str) -> str:
        if not SHA.fullmatch(value):
            raise GitIdentityError("invalid Git object ID")
        return value

    @staticmethod
    def _ref(value: str) -> str:
        return require_full_ref(value)
