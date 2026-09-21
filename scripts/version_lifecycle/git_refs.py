"""Exact Git identity and protected-ref operations for lifecycle transactions.

These helpers do not establish GitHub protection. Callers must supply verified
protection evidence before any remote mutation. Gate A tests use bare fixture
repositories; no production lifecycle operation is performed by this module.
"""

from __future__ import annotations

import datetime as dt
from contextlib import ExitStack, contextmanager
import os
import re
import subprocess
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported runner is POSIX.
    fcntl = None  # type: ignore[assignment]


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


class _ExclusionState:
    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.owner: int | None = None
        self.depth = 0
        self.handle: Any = None


_EXCLUSION_STATES: dict[Path, _ExclusionState] = {}
_EXCLUSION_STATES_GUARD = threading.Lock()


def _exclusion_state(path: Path) -> _ExclusionState:
    with _EXCLUSION_STATES_GUARD:
        return _EXCLUSION_STATES.setdefault(path, _ExclusionState())


class StaticMutationLease:
    """A reentrant, process and host scoped lease for static ref mutation."""

    def __init__(self, exclusion: "StaticMutationExclusion") -> None:
        self._exclusion = exclusion
        self._active = False

    def __enter__(self) -> "StaticMutationLease":
        if not self._active:
            self._exclusion._acquire()
            self._active = True
        return self

    def release(self) -> None:
        if self._active:
            self._exclusion._release()
            self._active = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.release()
        return False


class StaticMutationExclusion:
    """Shared fail-closed exclusion for static and protected-ref mutation.

    The lock lives under the repository's Git directory, so all local writers
    for one repository use the same boundary without creating a tracked file.
    Acquisition is deliberately nonblocking: a racing writer receives a
    machine-visible failure and must retain its surrounding freeze.
    """

    LOCK_NAME = "cyaxiverse-static-mutation.lock"

    def __init__(self, lock_path: str | os.PathLike[str]) -> None:
        self.lock_path = Path(lock_path).resolve()
        self._state = _exclusion_state(self.lock_path)

    @classmethod
    def for_repository(cls, root: str | os.PathLike[str]) -> "StaticMutationExclusion":
        repository = Path(root)
        git_marker = repository / ".git"
        if git_marker.is_dir():
            git_directory = git_marker
        elif git_marker.is_file():
            marker = git_marker.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_UNAVAILABLE")
            git_directory = Path(marker.split(":", 1)[1].strip())
            if not git_directory.is_absolute():
                git_directory = repository / git_directory
        else:
            # Bare fixture repositories have no .git directory. Their root
            # is already outside a tracked working tree.
            git_directory = repository
        commondir = git_directory / "commondir"
        if commondir.is_file():
            common_path = Path(commondir.read_text(encoding="utf-8").strip())
            if not common_path.is_absolute():
                common_path = git_directory / common_path
            git_directory = common_path
        return cls(git_directory / cls.LOCK_NAME)

    def acquire(self) -> StaticMutationLease:
        lease = StaticMutationLease(self)
        lease.__enter__()
        return lease

    def held(self) -> bool:
        """Return whether the current thread owns this exclusion boundary."""

        with self._state.guard:
            return self._state.owner == threading.get_ident() and self._state.depth > 0

    def _acquire(self) -> None:
        owner = threading.get_ident()
        with self._state.guard:
            if self._state.owner == owner:
                self._state.depth += 1
                return
            if self._state.owner is not None:
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_UNAVAILABLE")
            if fcntl is None:
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_UNAVAILABLE")
            handle = None
            try:
                handle = self.lock_path.open("a+")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, ValueError) as exc:
                if handle is not None:
                    handle.close()
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_UNAVAILABLE") from exc
            self._state.owner = owner
            self._state.depth = 1
            self._state.handle = handle

    def _release(self) -> None:
        owner = threading.get_ident()
        with self._state.guard:
            if self._state.owner != owner or self._state.depth <= 0:
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_NOT_HELD")
            self._state.depth -= 1
            if self._state.depth:
                return
            handle = self._state.handle
            self._state.owner = None
            self._state.handle = None
            if handle is None or fcntl is None:
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_UNAVAILABLE")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            except OSError as exc:
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_UNAVAILABLE") from exc


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
    """Exact Git identity operations under local and live exclusion proofs.

    ``exclusion_lease`` must hold the externally governed allocation authority
    through the final remote mutation. ``exclusion_checker`` supplies a fresh
    proof under that lease. Both are required for remote static-ref writes.
    """

    def __init__(
        self,
        root: str | Path,
        remote: str = "origin",
        *,
        static_exclusion: StaticMutationExclusion | None = None,
        exclusion_checker: Callable[[], bool] | None = None,
        exclusion_lease: Callable[[], ContextManager[bool]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.remote = remote
        self.static_exclusion = static_exclusion or StaticMutationExclusion.for_repository(
            self.root
        )
        self.exclusion_checker = exclusion_checker
        self.exclusion_lease = exclusion_lease

    def acquire_static_mutation(self, intent: Any = None) -> StaticMutationLease:
        """Acquire the repository's shared static mutation boundary."""

        return self.static_exclusion.acquire()

    @staticmethod
    def release_static_mutation(lease: StaticMutationLease) -> None:
        """Release a lease acquired through :meth:`acquire_static_mutation`."""

        lease.release()

    def _require_live_exclusion(self) -> None:
        """Require a fresh externally governed exclusion proof before pushing."""

        if self.exclusion_checker is None:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")
        try:
            verified = self.exclusion_checker()
        except Exception as exc:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE") from exc
        if verified is not True:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")

    def _enter_external_exclusion(self, stack: ExitStack) -> None:
        if self.exclusion_lease is None:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")
        try:
            held = stack.enter_context(self.exclusion_lease())
        except Exception as exc:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE") from exc
        if held is not True:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")

    @contextmanager
    def _remote_mutation_boundary(self):
        """Hold both exclusion proofs through remote mutation and reconciliation."""

        try:
            with ExitStack() as stack:
                stack.enter_context(self.acquire_static_mutation())
                self._enter_external_exclusion(stack)
                self._require_live_exclusion()
                yield
        except GitIdentityError:
            raise
        except Exception as exc:
            # A lease release failure can occur after the remote accepted the
            # ref. Callers must reconcile rather than treating it as absence.
            raise GitIdentityError("STATIC_MUTATION_OUTCOME_UNCERTAIN") from exc

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
        with self._remote_mutation_boundary():
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
            # An absence precheck is not atomic with the push advertisement. An
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
        with self.acquire_static_mutation():
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
