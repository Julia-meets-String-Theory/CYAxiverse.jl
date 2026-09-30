"""Exact Git identity and protected-ref operations for lifecycle transactions.

These helpers do not establish GitHub protection. Callers must supply verified
protection evidence before any remote mutation. Gate A tests use bare fixture
repositories; no production lifecycle operation is performed by this module.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
import os
import re
import subprocess
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

from .versions import parse_package_version, parse_public_tag

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported runner is POSIX.
    fcntl = None  # type: ignore[assignment]


FULL_REF = re.compile(r"^refs/(?:heads|tags|candidates)/[A-Za-z0-9._/-]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
UTC = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
LEGACY_TAG_REF = "refs/tags/v-0.1"


class GitIdentityError(RuntimeError):
    """A Git identity, protection, or exact-ref expectation failed."""


class RemoteValueError(GitIdentityError, ValueError):
    """A remote argument is unsafe to pass to Git."""


def validate_remote(value: str) -> str:
    """Validate and return a remote argument before passing it to Git.

    Git accepts remote names, URLs, and local repository paths in the same
    argument position.  Keep those forms open, but reject values that Git
    could interpret as options or that contain shell/control delimiters.
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise RemoteValueError("REMOTE_VALUE_UNSAFE")
    if value.startswith("-"):
        raise RemoteValueError("REMOTE_VALUE_UNSAFE")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise RemoteValueError("REMOTE_VALUE_UNSAFE")
    return value


def parse_remote_ref_advertisement(
    output: bytes | str,
    ref: str,
    *,
    allow_peeled: bool = False,
) -> str | None:
    """Parse one exact ``ls-remote`` advertisement without ambiguity.

    Empty output means the ref is absent. Every advertised record must be
    well-formed, name only the requested ref (or its allowed peeled form), and
    occur exactly once. This prevents a valid-looking record from masking
    malformed, duplicate, conflicting, or unrelated transport output.
    """

    if not isinstance(ref, str) or FULL_REF.fullmatch(ref) is None:
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")
    if allow_peeled and not ref.startswith("refs/tags/"):
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")

    peeled_ref = f"{ref}^{{}}"
    allowed = {ref, peeled_ref} if allow_peeled else {ref}
    records = parse_remote_ref_advertisements(output, allow_peeled=allow_peeled)
    if any(name not in allowed for name in records):
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")

    if not records:
        return None
    if peeled_ref in records and ref not in records:
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")
    return records.get(peeled_ref, records.get(ref))


def parse_remote_ref_advertisements(
    output: bytes | str, *, allow_peeled: bool = False
) -> dict[str, str]:
    """Parse a complete ``ls-remote`` response with strict ASCII/LF framing."""

    try:
        text = (
            output.decode("ascii", errors="strict")
            if isinstance(output, bytes)
            else output
        )
        if isinstance(text, str):
            text.encode("ascii", errors="strict")
    except UnicodeError as error:
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID") from error
    if not isinstance(text, str):
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")
    if any(
        (ord(character) < 0x20 and character not in {"\n", "\t"})
        or ord(character) > 0x7E
        for character in text
    ):
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")
    if text and not text.endswith("\n"):
        raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")

    records: dict[str, str] = {}
    for line in text[:-1].split("\n") if text else []:
        fields = line.split("\t")
        advertised_ref = fields[1] if len(fields) == 2 else ""
        is_peeled = advertised_ref.endswith("^{}")
        structural_ref = (
            advertised_ref[:-3]
            if is_peeled
            else advertised_ref
        )
        if (
            len(fields) != 2
            or SHA.fullmatch(fields[0]) is None
            or (is_peeled and not allow_peeled)
            or not _valid_advertised_ref(structural_ref)
            or advertised_ref in records
        ):
            raise GitIdentityError("REMOTE_REF_ADVERTISEMENT_INVALID")
        records[advertised_ref] = fields[0]
    return records


def _valid_advertised_ref(ref: str) -> bool:
    """Return whether ``ref`` satisfies Git's structural ref-name rules."""

    if not ref.startswith("refs/") or ref.endswith(("/", ".")):
        return False
    if ".." in ref or "@{" in ref or "//" in ref:
        return False
    if any(character in " ~^:?*[\\" for character in ref):
        return False
    return all(
        component
        and not component.startswith(".")
        and not component.endswith(".lock")
        for component in ref.split("/")
    )


def canonical_remote_authority(repository: str | Path, remote: str) -> str:
    """Return a private, canonical identity for the actual Git endpoint.

    Public snapshot metadata can use a caller-supplied sanitized repository
    label, but allocation authority must bind to the transport endpoint that
    was actually queried.  Resolve configured remote names first, then
    normalize local paths so two checkouts of one fixture remote compare
    equal without exposing this identity in serialized evidence.
    """

    remote = validate_remote(remote)
    root = Path(repository).resolve()
    configured = subprocess.run(
        ["git", "-C", str(root), "config", "--get", f"remote.{remote}.url"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    endpoint = configured.stdout.strip() if configured.returncode == 0 else remote
    if endpoint.startswith("file://"):
        return f"file://{Path(endpoint[7:]).expanduser().resolve()}"
    if "://" not in endpoint and not re.match(r"^[^/@:]+@[^/:]+:", endpoint):
        candidate = Path(endpoint).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return f"file://{candidate.resolve()}"
    return endpoint.removesuffix("/").removesuffix(".git")


class _ExclusionState:
    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.owner: int | None = None
        self.depth = 0
        self.handle: Any = None
        self.governed_owner: int | None = None


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


class CompositeStaticMutationLease:
    """Hold the local static lock and the external allocation exclusion.

    A lifecycle controller keeps this lease for the complete static/lifecycle
    mutation sequence.  Nested Git ref helpers acquire only a reentrant local
    lease when this same exclusion is already governed by an outer composite
    lease; the external authority therefore remains held until the outer
    controller releases it.
    """

    def __init__(
        self,
        exclusion: "StaticMutationExclusion",
        external_factory: Callable[[], ContextManager[bool]] | None,
        checker: Callable[[], bool] | None,
    ) -> None:
        self._exclusion = exclusion
        self._external_factory = external_factory
        self._checker = checker
        self._local: StaticMutationLease | None = None
        self._external: ContextManager[bool] | None = None
        self._active = False

    def __enter__(self) -> "CompositeStaticMutationLease":
        if self._active:
            return self
        local = self._exclusion.acquire()
        external: ContextManager[bool] | None = None
        entered = False
        try:
            if self._external_factory is None:
                raise GitIdentityError("EXCLUSION_UNAVAILABLE")
            external = self._external_factory()
            held = external.__enter__()
            entered = True
            if held is not True:
                raise GitIdentityError("EXCLUSION_UNAVAILABLE")
            if self._checker is None or self._checker() is not True:
                raise GitIdentityError("EXCLUSION_UNAVAILABLE")
        except GitIdentityError:
            if entered and external is not None:
                try:
                    external.__exit__(None, None, None)
                except Exception:
                    pass
            local.release()
            raise
        except Exception as exc:
            if entered and external is not None:
                try:
                    external.__exit__(None, None, None)
                except Exception:
                    pass
            local.release()
            raise GitIdentityError("EXCLUSION_UNAVAILABLE") from exc
        self._local = local
        self._external = external
        self._active = True
        self._exclusion._mark_governed()
        return self

    def release(
        self,
        exc_type: Any = None,
        exc: Any = None,
        traceback: Any = None,
    ) -> None:
        if not self._active:
            return
        self._active = False
        external = self._external
        local = self._local
        self._external = None
        self._local = None
        release_error: BaseException | None = None
        try:
            if external is not None:
                external.__exit__(exc_type, exc, traceback)
        except BaseException as error:  # preserve the uncertain outcome
            release_error = error
        finally:
            if local is not None:
                try:
                    local.release()
                except BaseException as error:
                    if release_error is None:
                        release_error = error
            self._exclusion._unmark_governed()
        if release_error is not None:
            raise GitIdentityError("STATIC_MUTATION_OUTCOME_UNCERTAIN") from release_error

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.release(exc_type, exc, traceback)
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

    def governed_held(self) -> bool:
        """Return whether an outer composite lease governs this thread."""

        with self._state.guard:
            return self._state.governed_owner == threading.get_ident()

    def _mark_governed(self) -> None:
        with self._state.guard:
            if self._state.owner != threading.get_ident() or self._state.depth <= 0:
                raise GitIdentityError("STATIC_MUTATION_EXCLUSION_NOT_HELD")
            self._state.governed_owner = threading.get_ident()

    def _unmark_governed(self) -> None:
        with self._state.guard:
            if self._state.governed_owner == threading.get_ident():
                self._state.governed_owner = None

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
    canonical_public_tags_globally_guarded: bool = False

    def _matches(self, ref: str) -> bool:
        # The candidate namespace uses one specifically approved GitHub glob.
        # Its shape is narrow enough to validate locally after the adapter has
        # verified the exact live ruleset definition.
        if self.pattern == "refs/heads/candidates/**/*":
            suffix = ref.removeprefix("refs/heads/candidates/")
            return (
                ref.startswith("refs/heads/candidates/")
                and bool(suffix)
                and all(suffix.split("/"))
            )
        # Other GitHub patterns need a separate live matcher before their
        # result can be represented by this evidence.
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
            if not isinstance(ref, str) or not ref.startswith("refs/tags/"):
                raise GitIdentityError("noncanonical public tag")
            tag = ref.removeprefix("refs/tags/")
            try:
                parsed = parse_public_tag(tag)
            except (TypeError, ValueError) as exc:
                raise GitIdentityError("noncanonical public tag") from exc
            if ref != f"refs/tags/v{parsed.canonical}":
                raise GitIdentityError("noncanonical public tag")
            if (
                self.pattern != "refs/tags/v*.*.*"
                or self.canonical_public_tags_globally_guarded is not True
            ):
                raise GitIdentityError("ruleset does not cover every canonical public tag")
            self.require(self.pattern, creation=True)
            if self._matches(LEGACY_TAG_REF):
                raise GitIdentityError("ruleset would also change legacy tag protection")
        except GitIdentityError as exc:
            raise GitIdentityError("PUBLIC_TAG_RULESET_UNAVAILABLE") from exc


class GitRepository:
    """Exact Git identity operations under local and live exclusion proofs.

    ``exclusion_lease`` holds the externally governed allocation authority
    together with the local static lock for each governed mutation sequence.
    ``exclusion_checker`` supplies a fresh proof while that lease is held.
    Both are required for static-ref writes.
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
        self.remote = validate_remote(remote)
        self.static_exclusion = static_exclusion or StaticMutationExclusion.for_repository(
            self.root
        )
        self.exclusion_checker = exclusion_checker
        self.exclusion_lease = exclusion_lease

    def acquire_static_mutation(
        self, intent: Any = None
    ) -> StaticMutationLease | CompositeStaticMutationLease:
        """Acquire local and externally governed static mutation exclusion.

        When an outer composite lease already governs this thread, return a
        reentrant local lease.  This lets nested annotated-tag and protected
        ref helpers use the same repository APIs without reacquiring a
        non-reentrant external authority.
        """

        if self.static_exclusion.governed_held():
            return self.static_exclusion.acquire()
        lease = CompositeStaticMutationLease(
            self.static_exclusion, self.exclusion_lease, self.exclusion_checker
        )
        lease.__enter__()
        return lease

    @staticmethod
    def release_static_mutation(
        lease: StaticMutationLease | CompositeStaticMutationLease,
    ) -> None:
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

    @contextmanager
    def _remote_mutation_boundary(self):
        """Hold the composite exclusion through mutation and reconciliation."""

        try:
            with self.acquire_static_mutation():
                # Recheck immediately before the remote operation.  The
                # acquisition proof remains held for the entire body.
                self._require_live_exclusion()
                yield
        except GitIdentityError:
            raise
        except Exception as exc:
            # A lease release failure can occur after the remote accepted the
            # ref. Callers must reconcile rather than treating it as absence.
            raise GitIdentityError("STATIC_MUTATION_OUTCOME_UNCERTAIN") from exc

    def git(self, *args: str, input_bytes: bytes | None = None) -> bytes:
        # Validate on every Git invocation as well as construction.  The
        # attribute is mutable for fixture adapters, and a later invalid
        # replacement must not reach Git's option parser.
        validate_remote(self.remote)
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
        output = self.git("ls-remote", "--refs", self.remote, ref)
        return parse_remote_ref_advertisement(output, ref)

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
            try:
                parsed_version = parse_package_version(version)
            except (TypeError, ValueError) as exc:
                raise GitIdentityError("invalid final version") from exc
            if parsed_version.is_dev or parsed_version.canonical != version:
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
