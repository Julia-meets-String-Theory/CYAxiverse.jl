"""Git-backed append-only writer for the canonical release event ledger.

The writer uses Git's expected-old-value form of ``update-ref`` for local
compare-and-swap updates.  Remote appends build the descendant commit first
and use an expected-old compare-and-swap push.  An exception from a remote
push is treated as an uncertain outcome until the remote stream has been read
and reconciled by transaction ID and exact canonical event bytes.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, ContextManager, Mapping
import uuid

from .events import (
    EventTransitionError,
    UnsupportedCertificationBinding,
    append_only_prefix,
    canonical_event_bytes,
    parse_stream,
    validate_event,
    validate_transition,
)
from .git_refs import (
    GitIdentityError,
    StaticMutationExclusion,
    canonical_remote_authority,
    validate_remote,
)
from .codec import canonical_json


class WriterError(RuntimeError):
    reason_code = "WRITER_ERROR"


class BranchUnavailable(WriterError):
    reason_code = "EVENT_BRANCH_UNAVAILABLE"


class StaleHeadError(WriterError):
    reason_code = "EXPECTED_EVENT_HEAD_STALE"


class DuplicateTransactionError(WriterError):
    reason_code = "TRANSACTION_ID_PAYLOAD_MISMATCH"


class AppendOutcomeUncertain(WriterError):
    reason_code = "APPEND_OUTCOME_UNCERTAIN"


class ExclusionUnavailable(WriterError):
    reason_code = "EXCLUSION_UNAVAILABLE"


class StaticSnapshotStale(WriterError):
    reason_code = "STATIC_SNAPSHOT_STALE"


class StaticAuthorityUnavailable(WriterError):
    reason_code = "STATIC_AUTHORITY_SELECTOR_UNRESOLVED"


class PublicTagAbsenceUnavailable(WriterError):
    reason_code = "PUBLIC_TAG_ABSENCE_UNAVAILABLE"


class PublicTagExists(WriterError):
    reason_code = "PUBLIC_TAG_EXISTS"


class ReservationNonEntryUnavailable(WriterError):
    reason_code = "RESERVATION_NON_ENTRY_UNAVAILABLE"


class _ExternalLeaseReleaseError(WriterError):
    """The external exclusion lease could not be released cleanly."""

    reason_code = "APPEND_OUTCOME_UNCERTAIN"


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERIFIED_HEAD_TOKEN = object()
CANONICAL_EVENT_BRANCH = "release-events"
CANONICAL_EVENT_REF = f"refs/heads/{CANONICAL_EVENT_BRANCH}"
CANONICAL_EVENT_STREAM = "release-events.jsonl"


@dataclass(frozen=True)
class AppendResult:
    """The durable outcome observed by a caller."""

    status: str
    reason_code: str | None
    transaction_id: str
    event: dict[str, Any]
    head: str | None
    idempotent: bool = False
    frozen: bool = False


@dataclass(frozen=True)
class LedgerHead:
    commit: str
    raw: bytes
    events: tuple[dict[str, Any], ...]
    repository_identity: str = ""
    ref: str = ""
    stream_path: str = ""
    _authority_token: object | None = field(default=None, init=False, repr=False, compare=False)
    _authority_binding: tuple[str, str, str, str, bytes, tuple[bytes, ...]] | None = field(
        default=None, init=False, repr=False, compare=False
    )


def _verified_ledger_head(
    repository_identity: str,
    ref: str,
    stream_path: str,
    commit: str,
    raw: bytes,
    events: tuple[dict[str, Any], ...],
) -> LedgerHead:
    """Bind a head only after the writer has verified its Git-backed stream."""

    head = LedgerHead(
        commit=commit,
        raw=raw,
        events=events,
        repository_identity=repository_identity,
        ref=ref,
        stream_path=stream_path,
    )
    object.__setattr__(head, "_authority_token", _VERIFIED_HEAD_TOKEN)
    object.__setattr__(
        head,
        "_authority_binding",
        (
            repository_identity,
            ref,
            stream_path,
            commit,
            raw,
            tuple(canonical_json(dict(event)) for event in events),
        ),
    )
    return head


def _is_verified_ledger_head(value: Any) -> bool:
    """Return whether *value* retains the writer's verified head binding."""

    if not isinstance(value, LedgerHead) or value._authority_token is not _VERIFIED_HEAD_TOKEN:
        return False
    if not all(
        isinstance(identity, str) and identity
        for identity in (value.repository_identity, value.ref, value.stream_path)
    ):
        return False
    try:
        binding = (
            value.repository_identity,
            value.ref,
            value.stream_path,
            value.commit,
            value.raw,
            tuple(canonical_json(dict(event)) for event in value.events),
        )
    except (TypeError, ValueError):
        return False
    return value._authority_binding == binding


class _ExternalLeaseGuard:
    """Preserve a lease's body errors while classifying release failures."""

    def __init__(self, manager: ContextManager[bool]) -> None:
        self.manager = manager

    def __enter__(self) -> bool:
        return self.manager.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.manager.__exit__(exc_type, exc, traceback)
        except Exception as error:
            raise _ExternalLeaseReleaseError(
                "external exclusion lease could not be released"
            ) from error
        # External context managers cannot suppress a validation or mutation
        # error raised inside the protected boundary.
        return False


def _as_text(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class ReleaseEventWriter:
    """Append lifecycle events to one protected release-events branch.

    ``repo`` is a Git working tree or bare repository.  The writer never
    force-updates the branch and never edits a package-source branch.
    ``exclusion_checker``, ``static_snapshot_checker`` and abort-specific
    proof callbacks are injected by the allocation layer so this bounded
    module remains usable with fixture repos.  The tag checker is mandatory
    for ``release_intent_aborted``; the non-entry checker is mandatory for
    ``development_reservation_aborted``.  Local and remote operations also
    require an externally governed ``exclusion_lease`` and hold it with the
    local static exclusion through mutation.  Remote operations retain the
    lease through every observation, push and reconciliation step.  Both event
    proof callbacks run under exclusion immediately before append.
    """

    def __init__(
        self,
        repo: str | os.PathLike[str],
        *,
        branch: str = CANONICAL_EVENT_BRANCH,
        stream_path: str = CANONICAL_EVENT_STREAM,
        remote: str | None = None,
        exclusion_checker: Callable[[], bool] | None = None,
        static_snapshot_checker: Callable[[str], bool] | None = None,
        public_tag_absence_checker: Callable[[str], bool] | None = None,
        reservation_non_entry_checker: Callable[[Mapping[str, Any]], bool] | None = None,
        static_exclusion: StaticMutationExclusion | None = None,
        exclusion_lease: Callable[[], ContextManager[bool]] | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.branch = self._validate_ref_component(branch)
        self.stream_path = self._validate_stream_path(stream_path)
        self.remote = validate_remote(remote) if remote is not None else None
        self.exclusion_checker = exclusion_checker
        self.static_snapshot_checker = static_snapshot_checker
        self.public_tag_absence_checker = public_tag_absence_checker
        self.reservation_non_entry_checker = reservation_non_entry_checker
        self.static_exclusion = static_exclusion or StaticMutationExclusion.for_repository(
            self.repo
        )
        self.exclusion_lease = exclusion_lease
        if not (self.repo / ".git").exists() and not (self.repo / "HEAD").exists():
            raise BranchUnavailable(f"not a Git repository: {self.repo}")

    @staticmethod
    def _validate_ref_component(value: str) -> str:
        if not isinstance(value, str) or not value or value.startswith("-"):
            raise ValueError("invalid branch name")
        if ".." in value or value.endswith("/") or value.startswith("/"):
            raise ValueError("invalid branch name")
        return value

    @staticmethod
    def _validate_stream_path(value: str) -> str:
        if not isinstance(value, str) or not value or value.startswith("/"):
            raise ValueError("stream_path must be a relative path")
        parts = Path(value).parts
        if len(parts) != 1 or ".." in parts or any(part in {"", "."} for part in parts):
            raise ValueError("stream_path must be one file at the branch root")
        return value

    @property
    def ref(self) -> str:
        return f"refs/heads/{self.branch}"

    def _repository_authority(self) -> str:
        remote = self.remote
        if remote is None:
            configured = self._git(
                ["config", "--get", "remote.origin.url"], check=False
            ).stdout.decode("utf-8", errors="strict").strip()
            remote = "origin" if configured else str(self.repo.resolve())
        return canonical_remote_authority(self.repo, remote)

    def _git(
        self,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.repo,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=check,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                detail = exc.stderr.decode("utf-8", errors="replace").strip()
            else:
                detail = str(exc)
            raise WriterError(f"git {' '.join(args)} failed: {detail}") from exc
        return result

    def branch_exists(self) -> bool:
        result = self._git(["show-ref", "--verify", "--quiet", self.ref], check=False)
        return result.returncode == 0

    def current_head(self) -> str:
        if not self.branch_exists():
            raise BranchUnavailable(f"branch does not exist: {self.branch}")
        result = self._git(["rev-parse", self.ref])
        return _as_text(result.stdout).strip()

    def read_head(self) -> LedgerHead:
        commit = self.current_head()
        result = self._git(["show", f"{commit}:{self.stream_path}"])
        raw = bytes(result.stdout)
        return self._verified_head_from_stream(commit, raw)

    def _verified_head_from_stream(self, commit: str, raw: bytes) -> LedgerHead:
        """Verify one observed stream and return its authority-bound head."""

        self._verify_stream_topology(commit, raw)
        events = tuple(parse_stream(raw))
        return _verified_ledger_head(
            self._repository_authority(),
            self.ref,
            self.stream_path,
            commit,
            raw,
            events,
        )

    def bootstrap(
        self,
        *,
        expected_absent: bool = True,
        message: str = "Initialize release event ledger",
        protection_checker: Callable[[], bool] | None = None,
    ) -> str:
        """Create the minimal orphan event branch with an empty stream.

        Public bootstrap is a protected mutation.  The shared local
        exclusion and externally governed lease remain held while protection
        is checked and while the compare-and-swap runs.  The update is a
        compare-and-swap against the all-zero object ID, so a concurrent
        creator cannot be overwritten.  Existing branches are validated and
        returned when ``expected_absent`` is false.
        """

        try:
            with ExitStack() as stack:
                stack.enter_context(self.static_exclusion.acquire())
                self._enter_external_exclusion(stack)
                return self._bootstrap_under_exclusion(
                    expected_absent=expected_absent,
                    message=message,
                    protection_checker=protection_checker,
                )
        except _ExternalLeaseReleaseError as error:
            raise AppendOutcomeUncertain(
                "event bootstrap outcome cannot be classified"
            ) from error
        except GitIdentityError as error:
            raise ExclusionUnavailable(
                "serialized static/event exclusion is unavailable"
            ) from error

    def _bootstrap_under_exclusion(
        self,
        *,
        expected_absent: bool,
        message: str,
        protection_checker: Callable[[], bool] | None,
    ) -> str:
        """Check creation protection before entering the unchecked mutation."""

        try:
            protected = (
                protection_checker is not None and protection_checker() is True
            )
        except Exception as error:
            raise ExclusionUnavailable(
                "event branch creation protection could not be verified"
            ) from error
        if not protected:
            raise ExclusionUnavailable(
                "event branch creation protection is unavailable"
            )
        return self._bootstrap_unchecked(
            expected_absent=expected_absent,
            message=message,
        )

    def _bootstrap_unchecked(
        self, *, expected_absent: bool, message: str
    ) -> str:
        """Bootstrap after the caller has established all exclusion proofs.

        This helper is private because it does not acquire either exclusion
        layer.  It is used by remote bootstrap while that operation already
        holds the shared local and external leases.
        """
        if self.branch_exists():
            if expected_absent:
                raise StaleHeadError(f"branch already exists: {self.branch}")
            self.read_head()
            return self.current_head()
        commit = self._make_commit(b"", parent=None, message=message)
        result = self._git(
            ["update-ref", self.ref, commit, "0" * 40], check=False
        )
        if result.returncode != 0:
            raise StaleHeadError("event branch was created concurrently")
        return commit

    def _remote_observation(self, remote: str) -> tuple[str | None, bytes]:
        """Fetch one remote branch into a disposable local ref for reconciliation."""

        remote = validate_remote(remote)
        remote_ref = f"refs/heads/{self.branch}"
        listed = self._git(["ls-remote", "--refs", remote, remote_ref])
        lines = [line for line in _as_text(listed.stdout).splitlines() if line]
        if not lines:
            return None, b""
        if len(lines) != 1 or "\t" not in lines[0]:
            raise AppendOutcomeUncertain("remote event branch identity is ambiguous")
        remote_head, name = lines[0].split("\t", 1)
        if name != remote_ref or len(remote_head) != 40:
            raise AppendOutcomeUncertain("remote event branch identity is malformed")
        temp_ref = f"refs/codex/reconcile/{uuid.uuid4().hex}"
        try:
            self._git(["fetch", "--no-tags", remote, f"{remote_ref}:{temp_ref}"])
            fetched_head = _as_text(
                self._git(["rev-parse", f"{temp_ref}^{{commit}}"]).stdout
            ).strip()
            if fetched_head != remote_head:
                raise AppendOutcomeUncertain(
                    "remote event branch advanced during reconciliation"
                )
            raw = bytes(self._git(["show", f"{temp_ref}:{self.stream_path}"]).stdout)
            self._verify_stream_topology(fetched_head, raw)
            return remote_head, raw
        except AppendOutcomeUncertain:
            raise
        except Exception as error:
            raise AppendOutcomeUncertain(
                "remote event branch could not be fetched and verified"
            ) from error
        finally:
            self._git(["update-ref", "-d", temp_ref], check=False)

    def _verify_stream_topology(self, commit: str, raw: bytes) -> None:
        """Verify one observed commit has the canonical complete history.

        Callback reconciliation is accepted only when the advertised commit
        is also present in this Git object database and its complete ancestry
        proves the one-file tree, exact blob bytes, orphan bootstrap, and one
        canonical append per child commit. A callback cannot claim a remote
        head with an unverified or divergent topology.
        """

        if not isinstance(commit, str) or _GIT_SHA_RE.fullmatch(commit) is None:
            raise AppendOutcomeUncertain("observed event head is not a full Git SHA")
        if not isinstance(raw, bytes):
            raise AppendOutcomeUncertain("observed event stream is not bytes")
        try:
            current_commit = commit
            current_raw = raw
            visited: set[str] = set()
            while True:
                if current_commit in visited:
                    raise AppendOutcomeUncertain(
                        "observed event history contains a commit cycle"
                    )
                visited.add(current_commit)
                parents = self._verify_commit_stream(current_commit, current_raw)
                if len(parents) > 1:
                    raise AppendOutcomeUncertain(
                        "observed event history contains a merge commit"
                    )
                if not parents:
                    if current_raw != b"":
                        raise AppendOutcomeUncertain(
                            "observed event history does not end at an empty orphan bootstrap"
                        )
                    return

                parent = parents[0]
                parent_raw = bytes(
                    self._git(["show", f"{parent}:{self.stream_path}"]).stdout
                )
                if not current_raw.startswith(parent_raw):
                    raise AppendOutcomeUncertain(
                        "observed event stream is not an append-only extension"
                    )
                appended = current_raw[len(parent_raw):]
                try:
                    if not appended.endswith(b"\n") or b"\n" in appended[:-1]:
                        raise EventTransitionError(
                            "observed event commit did not append exactly one record"
                        )
                    appended_event = validate_event(appended[:-1])
                    if appended != canonical_event_bytes(appended_event) + b"\n":
                        raise EventTransitionError(
                            "observed event commit appended noncanonical bytes"
                        )
                except Exception as error:
                    raise AppendOutcomeUncertain(
                        "observed event commit appended a noncanonical stream"
                    ) from error
                if (
                    appended_event.get("expected_event_head") != parent
                ):
                    raise AppendOutcomeUncertain(
                        "observed event append does not bind its actual parent"
                    )
                current_commit = parent
                current_raw = parent_raw
        except AppendOutcomeUncertain:
            raise
        except Exception as error:
            raise AppendOutcomeUncertain(
                "observed event commit topology could not be verified"
            ) from error

    def _verify_commit_stream(self, commit: str, raw: bytes) -> list[str]:
        """Verify one commit's exact stream tree/blob and return its parents."""

        object_type = _as_text(self._git(["cat-file", "-t", commit]).stdout).strip()
        if object_type != "commit":
            raise AppendOutcomeUncertain("observed event history contains a non-commit")
        commit_text = _as_text(self._git(["cat-file", "-p", commit]).stdout)
        headers = commit_text.split("\n\n", 1)[0].splitlines()
        tree_values = [line.split(" ", 1)[1] for line in headers if line.startswith("tree ")]
        parents = [line.split(" ", 1)[1] for line in headers if line.startswith("parent ")]
        if len(tree_values) != 1 or any(_GIT_SHA_RE.fullmatch(value) is None for value in tree_values):
            raise AppendOutcomeUncertain("observed event commit has an invalid tree identity")
        if any(_GIT_SHA_RE.fullmatch(parent) is None for parent in parents):
            raise AppendOutcomeUncertain("observed event commit has an invalid parent identity")

        entries = _as_text(self._git(["ls-tree", commit]).stdout).splitlines()
        if len(entries) != 1 or "\t" not in entries[0]:
            raise AppendOutcomeUncertain(
                "observed event commit does not contain exactly one stream file"
            )
        metadata, path = entries[0].split("\t", 1)
        fields = metadata.split()
        if (
            len(fields) != 3
            or fields[0] != "100644"
            or fields[1] != "blob"
            or _GIT_SHA_RE.fullmatch(fields[2]) is None
            or path != self.stream_path
        ):
            raise AppendOutcomeUncertain(
                "observed event commit tree is not the canonical one-file tree"
            )
        if _as_text(self._git(["cat-file", "-t", fields[2]]).stdout).strip() != "blob":
            raise AppendOutcomeUncertain("observed event stream entry is not a blob")
        stored = bytes(self._git(["show", f"{commit}:{self.stream_path}"]).stdout)
        if stored != raw:
            raise AppendOutcomeUncertain(
                "observed event stream does not match its commit tree"
            )
        parse_stream(raw)
        return parents

    def _verified_remote_observation(
        self, observation: tuple[str | None, bytes]
    ) -> tuple[str | None, bytes]:
        """Verify callback observations before treating them as remote state."""

        if not isinstance(observation, tuple) or len(observation) != 2:
            raise AppendOutcomeUncertain("remote observation shape is invalid")
        head, raw = observation
        if head is None:
            if raw != b"":
                raise AppendOutcomeUncertain(
                    "missing remote event head must have an empty stream"
                )
            return None, b""
        self._verify_stream_topology(head, raw)
        return head, raw

    def _materialize_remote_cache(
        self,
        remote_head: str,
        remote_raw: bytes,
        *,
        fail_on_cache_update: bool = True,
    ) -> None:
        """Advance the local event cache to a verified remote authority.

        The remote branch is authoritative when bootstrap observes it.  The
        local branch is only a cache, so it may be created or advanced when
        the observed remote commit is an append-only descendant of the local
        commit.  A missing ancestry relation or byte-prefix mismatch is
        treated as an uncertain divergent cache; this prevents bootstrap from
        rewinding a local event authority.  Remote replay can set
        ``fail_on_cache_update`` to false because the verified remote event is
        authoritative even when the local cache compare-and-swap loses a
        race.
        """

        if _GIT_SHA_RE.fullmatch(remote_head) is None:
            raise AppendOutcomeUncertain("remote event head is not a full Git SHA")
        self._verify_stream_topology(remote_head, remote_raw)
        parse_stream(remote_raw)

        if not self.branch_exists():
            created = self._git(
                ["update-ref", self.ref, remote_head, "0" * 40], check=False
            )
            if created.returncode != 0:
                if fail_on_cache_update:
                    raise AppendOutcomeUncertain(
                        "local event cache was created concurrently"
                    )
            return

        try:
            local = self.read_head()
        except Exception as error:
            raise AppendOutcomeUncertain(
                "local event cache could not be verified"
            ) from error
        if local.commit == remote_head:
            if local.raw != remote_raw:
                raise AppendOutcomeUncertain(
                    "local event cache does not match the remote authority"
                )
            return
        if not remote_raw.startswith(local.raw):
            raise AppendOutcomeUncertain(
                "remote event stream is not an append-only extension of the local cache"
            )
        ancestry = self._git(
            ["merge-base", "--is-ancestor", local.commit, remote_head],
            check=False,
        )
        if ancestry.returncode != 0:
            raise AppendOutcomeUncertain(
                "remote event authority is divergent from the local cache"
            )
        advanced = self._git(
            ["update-ref", self.ref, remote_head, local.commit], check=False
        )
        if advanced.returncode != 0 and fail_on_cache_update:
            raise AppendOutcomeUncertain(
                "local event cache advanced concurrently"
            )

    def _enter_external_exclusion(self, stack: ExitStack) -> None:
        """Enter the externally governed exclusion held across mutations."""

        if self.static_exclusion.governed_held():
            # A transaction controller already holds the same repository's
            # governed lease. Its Git and event mutations must share that
            # boundary without reacquiring a nonreentrant external lease.
            return
        if self.exclusion_lease is None:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")
        try:
            manager = self.exclusion_lease()
            held = stack.enter_context(_ExternalLeaseGuard(manager))
        except Exception as error:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE") from error
        if held is not True:
            raise GitIdentityError("EXCLUSION_UNAVAILABLE")

    def bootstrap_remote(
        self,
        *,
        remote: str | None = None,
        push: Callable[[str, str, str], Any] | None = None,
        reconcile: Callable[[], tuple[str, bytes]] | None = None,
        protection_checker: Callable[[], bool] | None = None,
        message: str = "Initialize release event ledger",
    ) -> str:
        """Create the orphan event branch remotely with create-if-absent CAS.

        Production callers must supply protection evidence through
        ``protection_checker`` and an externally held ``exclusion_lease``.
        The default Git transport uses a normal create-only push and verifies
        the resulting remote object.
        """

        try:
            with ExitStack() as stack:
                stack.enter_context(self.static_exclusion.acquire())
                self._enter_external_exclusion(stack)
                return self._bootstrap_remote_under_exclusion(
                    remote=remote,
                    push=push,
                    reconcile=reconcile,
                    protection_checker=protection_checker,
                    message=message,
                )
        except _ExternalLeaseReleaseError as error:
            raise AppendOutcomeUncertain(
                "remote bootstrap outcome cannot be classified"
            ) from error
        except GitIdentityError as error:
            raise ExclusionUnavailable(
                "serialized static/event exclusion is unavailable"
            ) from error

    def _bootstrap_remote_under_exclusion(
        self,
        *,
        remote: str | None,
        push: Callable[[str, str, str], Any] | None,
        reconcile: Callable[[], tuple[str, bytes]] | None,
        protection_checker: Callable[[], bool] | None,
        message: str,
    ) -> str:
        remote_name = remote if remote is not None else self.remote
        if remote_name is not None:
            remote_name = validate_remote(remote_name)
        if not remote_name and push is None:
            raise ValueError("bootstrap_remote needs a remote name or push callback")
        if remote_name is None and reconcile is None:
            raise AppendOutcomeUncertain(
                "callback-only bootstrap has no remote observation; outcome is uncertain"
            )
        try:
            protected = protection_checker is not None and protection_checker() is True
        except Exception as error:
            raise ExclusionUnavailable(
                "event branch creation protection could not be verified"
            ) from error
        if not protected:
            raise ExclusionUnavailable("event branch creation protection is unavailable")
        remote_reader = reconcile
        if remote_reader is None and remote_name is not None:
            remote_reader = lambda: self._remote_observation(remote_name)
        existing = (
            self._verified_remote_observation(remote_reader())
            if remote_reader is not None
            else None
        )
        if existing is not None and existing[0] is not None:
            # A pre-existing branch is valid only after full stream validation;
            # materialize its verified authority into this repository's cache
            # before returning so a continuation append can read the head.
            self._materialize_remote_cache(existing[0], existing[1])
            return str(existing[0])
        # A local branch is only a cache for an already empty authority.  If
        # the remote branch is absent, never promote an existing nonempty
        # local ledger into the protected authority.
        if self.branch_exists():
            try:
                local = self.read_head()
                if local.raw:
                    raise AppendOutcomeUncertain(
                        "local event branch is nonempty while remote event branch is absent"
                    )
                parents = _as_text(
                    self._git(
                        ["rev-list", "--parents", "-n", "1", local.commit]
                    ).stdout
                ).split()
                if len(parents) != 1:
                    raise AppendOutcomeUncertain(
                        "local empty event branch is not an orphan while remote event branch is absent"
                    )
            except AppendOutcomeUncertain:
                raise
            except Exception as error:
                raise AppendOutcomeUncertain(
                    "local event branch could not be verified while remote event branch is absent"
                ) from error
            commit = local.commit
        else:
            commit = self._bootstrap_unchecked(expected_absent=False, message=message)
        try:
            if push is not None:
                push(commit, self.branch, "0" * 40)
            else:
                self._git(["push", remote_name, f"{commit}:refs/heads/{self.branch}"])
        except Exception as error:
            # A failed response is not evidence that create-if-absent failed.
            # Reconcile the protected remote before deciding whether the new
            # branch is durable; otherwise leave bootstrap unresolved.
            try:
                observed_head, observed_raw = self._verified_remote_observation(
                    remote_reader()  # type: ignore[misc]
                )
                parse_stream(observed_raw)
                if observed_head == commit and observed_raw == b"":
                    return commit
            except Exception:
                pass
            raise AppendOutcomeUncertain(
                "remote bootstrap outcome cannot be classified"
            ) from error
        try:
            observed_head, observed_raw = self._verified_remote_observation(
                remote_reader()  # type: ignore[misc]
            )
        except Exception as error:
            raise AppendOutcomeUncertain(
                "remote bootstrap outcome cannot be classified"
            ) from error
        if observed_head != commit or observed_raw != b"":
            raise AppendOutcomeUncertain(
                "remote bootstrap did not verify the expected empty stream"
            )
        return commit

    def _make_blob(self, raw: bytes) -> str:
        result = self._git(["hash-object", "-w", "--stdin"], input_bytes=raw)
        return _as_text(result.stdout).strip()

    def _make_commit(self, raw: bytes, *, parent: str | None, message: str) -> str:
        blob = self._make_blob(raw)
        # Gate A's branch contains one and only one mutable stream.  Nested
        # stream paths are intentionally not accepted by the constructor.
        tree_input = f"100644 blob {blob}\t{self.stream_path}\n".encode("ascii")
        tree = _as_text(self._git(["mktree"], input_bytes=tree_input).stdout).strip()
        args = ["commit-tree", tree]
        if parent:
            args.extend(["-p", parent])
        env = os.environ.copy()
        # Do not inherit a developer's ambient Git identity into the durable
        # authority.  These values are fixed for every ledger commit.
        env["GIT_AUTHOR_NAME"] = "CYAxiverse Lifecycle Writer"
        env["GIT_AUTHOR_EMAIL"] = "release-events@cyaxiverse.invalid"
        env["GIT_COMMITTER_NAME"] = "CYAxiverse Lifecycle Writer"
        env["GIT_COMMITTER_EMAIL"] = "release-events@cyaxiverse.invalid"
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.repo,
                input=(message.rstrip("\n") + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                env=env,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                exc.stderr.decode("utf-8", errors="replace").strip()
                if isinstance(exc, subprocess.CalledProcessError)
                else str(exc)
            )
            raise WriterError(f"git commit-tree failed: {detail}") from exc
        return _as_text(result.stdout).strip()

    @staticmethod
    def _transaction_event(
        events: list[Mapping[str, Any]], transaction_id: str
    ) -> Mapping[str, Any] | None:
        return next(
            (event for event in events if event.get("transaction_id") == transaction_id),
            None,
        )

    def _preflight(
        self,
        event: Mapping[str, Any],
        *,
        expected_head: str | None,
    ) -> tuple[dict[str, Any], bytes, LedgerHead]:
        try:
            excluded = self.exclusion_checker is not None and self.exclusion_checker() is True
        except Exception as error:
            raise ExclusionUnavailable(
                "serialized static/event exclusion could not be verified"
            ) from error
        if not excluded:
            raise ExclusionUnavailable("serialized static/event exclusion is unavailable")
        current = self.read_head()
        canonical = validate_event(event)
        encoded = canonical_event_bytes(canonical)
        transaction_id = canonical["transaction_id"]
        existing = self._transaction_event(list(current.events), transaction_id)
        if existing is not None:
            if canonical_event_bytes(existing) == encoded:
                return canonical, encoded, current
            raise DuplicateTransactionError(
                f"transaction {transaction_id} is already bound to another event payload"
            )
        proposed_head = expected_head or canonical["expected_event_head"]
        if proposed_head != current.commit:
            raise StaleHeadError(
                f"expected event head {proposed_head}, current head is {current.commit}"
            )
        if canonical["expected_event_head"] != current.commit:
            raise StaleHeadError("event expected_event_head does not match current branch head")
        try:
            static_current = (
                self.static_snapshot_checker is not None
                and self.static_snapshot_checker(canonical["static_iteration_snapshot"]) is True
            )
        except Exception as error:
            raise StaticAuthorityUnavailable(
                "static snapshot authority could not be verified"
            ) from error
        if not static_current:
            raise StaticSnapshotStale("static snapshot changed before append")
        if canonical["event_type"] == "development_reservation_aborted":
            checker = self.reservation_non_entry_checker
            if checker is None:
                raise ReservationNonEntryUnavailable(
                    "live reservation non-entry proof is unavailable"
                )
            try:
                non_entry_verified = checker(canonical) is True
            except Exception as error:
                raise ReservationNonEntryUnavailable(
                    "live reservation non-entry proof could not be verified"
                ) from error
            if not non_entry_verified:
                raise ReservationNonEntryUnavailable(
                    "matching DEV non-entry was not proved under exclusion"
                )
        if canonical["event_type"] == "release_intent_aborted":
            checker = self.public_tag_absence_checker
            if checker is None:
                raise PublicTagAbsenceUnavailable(
                    "live public-tag absence proof is unavailable"
                )
            try:
                tag_absent = checker(canonical["public_tag"])
            except Exception as error:
                raise PublicTagAbsenceUnavailable(
                    "live public-tag absence proof could not be read"
                ) from error
            if tag_absent is not True:
                raise PublicTagExists(
                    f"public tag already exists: {canonical['public_tag']}"
                )
        validate_transition(list(current.events), canonical)
        return canonical, encoded, current

    def append(
        self,
        event: Mapping[str, Any],
        *,
        expected_head: str | None = None,
        message: str | None = None,
    ) -> AppendResult:
        """Append one event with an expected-head compare-and-swap."""

        try:
            with ExitStack() as stack:
                stack.enter_context(self.static_exclusion.acquire())
                self._enter_external_exclusion(stack)
                return self._append_under_exclusion(
                    event, expected_head=expected_head, message=message
                )
        except _ExternalLeaseReleaseError:
            return self._remote_uncertain(event)
        except GitIdentityError:
            return self._exclusion_blocked(event)

    @staticmethod
    def _exclusion_blocked(event: Mapping[str, Any]) -> AppendResult:
        canonical = dict(event)
        return AppendResult(
            status="BLOCKED",
            reason_code="EXCLUSION_UNAVAILABLE",
            transaction_id=str(canonical.get("transaction_id", "")),
            event=canonical,
            head=None,
            frozen=True,
        )

    def _append_under_exclusion(
        self,
        event: Mapping[str, Any],
        *,
        expected_head: str | None,
        message: str | None,
    ) -> AppendResult:

        # Local refs are only a cache of the same allocation authority.  A
        # caller cannot turn this method into an unsafe production bypass by
        # omitting the live exclusion or static-snapshot proof callbacks.
        if self.exclusion_checker is None or self.static_snapshot_checker is None:
            try:
                canonical = validate_event(event)
            except UnsupportedCertificationBinding as error:
                canonical = dict(event)
                return AppendResult(
                    status="BLOCKED",
                    reason_code=error.reason_code,
                    transaction_id=str(canonical.get("transaction_id", "")),
                    event=canonical,
                    head=None,
                    frozen=True,
                )
            reason = (
                "EXCLUSION_UNAVAILABLE"
                if self.exclusion_checker is None
                else "STATIC_AUTHORITY_SELECTOR_UNRESOLVED"
            )
            return AppendResult(
                status="BLOCKED",
                reason_code=reason,
                transaction_id=canonical["transaction_id"],
                event=canonical,
                head=None,
                frozen=True,
            )

        try:
            canonical, encoded, current = self._preflight(event, expected_head=expected_head)
        except (
            ExclusionUnavailable,
            StaticAuthorityUnavailable,
            StaticSnapshotStale,
            PublicTagAbsenceUnavailable,
            PublicTagExists,
            ReservationNonEntryUnavailable,
            UnsupportedCertificationBinding,
        ) as error:
            # These are machine-visible fail-closed outcomes.  A valid event
            # is retained in the result so the caller can keep the line and
            # version frozen while the external authority is repaired.
            canonical = (
                dict(event)
                if isinstance(error, UnsupportedCertificationBinding)
                else validate_event(event)
            )
            return AppendResult(
                status="BLOCKED",
                reason_code=error.reason_code,
                transaction_id=str(canonical.get("transaction_id", "")),
                event=canonical,
                head=None,
                frozen=True,
            )
        transaction_id = canonical["transaction_id"]
        existing = self._transaction_event(list(current.events), transaction_id)
        if existing is not None:
            return AppendResult(
                status="IDEMPOTENT",
                reason_code=None,
                transaction_id=transaction_id,
                event=canonical,
                head=current.commit,
                idempotent=True,
            )
        new_raw = current.raw + encoded + b"\n"
        append_only_prefix(current.raw, new_raw)
        commit = self._make_commit(
            new_raw,
            parent=current.commit,
            message=message or f"release-events: {canonical['event_id']} {canonical['event_type']}",
        )
        result = self._git(["update-ref", self.ref, commit, current.commit], check=False)
        if result.returncode != 0:
            raise StaleHeadError("event branch advanced before the compare-and-swap update")
        return AppendResult(
            status="APPENDED",
            reason_code=None,
            transaction_id=transaction_id,
            event=canonical,
            head=commit,
        )

    def _prepare_descendant(
        self,
        event: Mapping[str, Any],
        *,
        expected_head: str | None,
        message: str | None,
    ) -> tuple[dict[str, Any], bytes, LedgerHead, str]:
        canonical, encoded, current = self._preflight(event, expected_head=expected_head)
        existing = self._transaction_event(list(current.events), canonical["transaction_id"])
        if existing is not None:
            return canonical, encoded, current, current.commit
        new_raw = current.raw + encoded + b"\n"
        append_only_prefix(current.raw, new_raw)
        commit = self._make_commit(
            new_raw,
            parent=current.commit,
            message=message or f"release-events: {canonical['event_id']} {canonical['event_type']}",
        )
        return canonical, encoded, current, commit

    def append_remote(
        self,
        event: Mapping[str, Any],
        *,
        expected_head: str | None = None,
        remote: str | None = None,
        push: Callable[[str, str, str], Any] | None = None,
        reconcile: Callable[[], tuple[str, bytes]] | None = None,
        message: str | None = None,
    ) -> AppendResult:
        """Append to a remote protected branch and reconcile uncertain pushes.

        ``push`` receives ``(commit, branch, expected_head)``.  ``reconcile``
        returns the observed remote ``(head, stream_bytes)``.  Both callbacks
        make failure injection deterministic in fixture tests and let the
        integration layer supply its authenticated Git transport.
        """

        try:
            with ExitStack() as stack:
                stack.enter_context(self.static_exclusion.acquire())
                self._enter_external_exclusion(stack)
                return self._append_remote_under_exclusion(
                    event,
                    expected_head=expected_head,
                    remote=remote,
                    push=push,
                    reconcile=reconcile,
                    message=message,
                )
        except _ExternalLeaseReleaseError:
            return self._remote_uncertain(event)
        except GitIdentityError:
            return self._exclusion_blocked(event)

    @staticmethod
    def _remote_uncertain(event: Mapping[str, Any]) -> AppendResult:
        canonical = dict(event)
        return AppendResult(
            status="BLOCKED",
            reason_code="APPEND_OUTCOME_UNCERTAIN",
            transaction_id=str(canonical.get("transaction_id", "")),
            event=canonical,
            head=None,
            frozen=True,
        )

    def _append_remote_under_exclusion(
        self,
        event: Mapping[str, Any],
        *,
        expected_head: str | None,
        remote: str | None,
        push: Callable[[str, str, str], Any] | None,
        reconcile: Callable[[], tuple[str, bytes]] | None,
        message: str | None,
    ) -> AppendResult:

        remote_name = remote if remote is not None else self.remote
        if remote_name is not None:
            remote_name = validate_remote(remote_name)
        if not remote_name and push is None:
            raise ValueError("append_remote needs a remote name or push callback")
        # Remote lifecycle mutation is never allowed to rely on a caller's
        # memory of protection or static authority.  Fixture callers provide
        # deterministic proof callbacks; production callers wire these to
        # freshly retrieved settings/snapshot evidence.
        if self.exclusion_checker is None or self.static_snapshot_checker is None:
            try:
                canonical = validate_event(event)
            except UnsupportedCertificationBinding as error:
                canonical = dict(event)
                return AppendResult(
                    status="BLOCKED",
                    reason_code=error.reason_code,
                    transaction_id=str(canonical.get("transaction_id", "")),
                    event=canonical,
                    head=None,
                    frozen=True,
                )
            reason = (
                "EXCLUSION_UNAVAILABLE"
                if self.exclusion_checker is None
                else "STATIC_AUTHORITY_SELECTOR_UNRESOLVED"
            )
            return AppendResult(
                status="BLOCKED",
                reason_code=reason,
                transaction_id=canonical["transaction_id"],
                event=canonical,
                head=None,
                frozen=True,
            )
        remote_reader = reconcile
        if remote_reader is None and remote_name is not None:
            remote_reader = lambda: self._remote_observation(remote_name)
        if remote_reader is None:
            try:
                canonical = validate_event(event)
            except UnsupportedCertificationBinding as error:
                canonical = dict(event)
                return AppendResult(
                    status="BLOCKED",
                    reason_code=error.reason_code,
                    transaction_id=str(canonical.get("transaction_id", "")),
                    event=canonical,
                    head=None,
                    frozen=True,
                )
            return AppendResult(
                status="BLOCKED",
                reason_code="APPEND_OUTCOME_UNCERTAIN",
                transaction_id=canonical["transaction_id"],
                event=canonical,
                head=None,
                frozen=True,
            )
        try:
            canonical, encoded, current, commit = self._prepare_descendant(
                event, expected_head=expected_head, message=message
            )
        except (
            ExclusionUnavailable,
            StaticAuthorityUnavailable,
            StaticSnapshotStale,
            PublicTagAbsenceUnavailable,
            PublicTagExists,
            ReservationNonEntryUnavailable,
            UnsupportedCertificationBinding,
        ) as error:
            canonical = (
                dict(event)
                if isinstance(error, UnsupportedCertificationBinding)
                else validate_event(event)
            )
            return AppendResult(
                status="BLOCKED",
                reason_code=error.reason_code,
                transaction_id=str(canonical.get("transaction_id", "")),
                event=canonical,
                head=None,
                frozen=True,
            )
        transaction_id = canonical["transaction_id"]
        existing = self._transaction_event(list(current.events), transaction_id)

        if remote_reader is not None:
            try:
                freshest_head, freshest_raw = self._verified_remote_observation(
                    remote_reader()
                )
                if freshest_head is None:
                    # Only bootstrap_remote may create the protected authority.
                    # An ordinary append must never recreate a missing branch.
                    return AppendResult(
                        status="BLOCKED",
                        reason_code="EVENT_BRANCH_UNAVAILABLE",
                        transaction_id=transaction_id,
                        event=canonical,
                        head=None,
                        frozen=True,
                    )
                freshest_events = parse_stream(freshest_raw)
                remote_existing = self._transaction_event(
                    freshest_events, transaction_id
                )
                if remote_existing is not None:
                    # A transaction match in either checkout is only
                    # idempotent after the remote authority proves the exact
                    # canonical event.  A stale local cache may be advanced
                    # to that verified append-only remote descendant.
                    if canonical_event_bytes(remote_existing) != encoded:
                        raise DuplicateTransactionError(
                            "remote transaction ID maps to a different canonical event"
                        )
                    if freshest_head == current.commit:
                        if freshest_raw != current.raw:
                            return AppendResult(
                                status="BLOCKED",
                                reason_code="APPEND_OUTCOME_UNCERTAIN",
                                transaction_id=transaction_id,
                                event=canonical,
                                head=freshest_head,
                                frozen=True,
                            )
                    else:
                        try:
                            self._materialize_remote_cache(
                                freshest_head,
                                freshest_raw,
                                fail_on_cache_update=False,
                            )
                        except AppendOutcomeUncertain:
                            return AppendResult(
                                status="BLOCKED",
                                reason_code="APPEND_OUTCOME_UNCERTAIN",
                                transaction_id=transaction_id,
                                event=canonical,
                                head=freshest_head,
                                frozen=True,
                            )
                    return AppendResult(
                        status="IDEMPOTENT",
                        reason_code=None,
                        transaction_id=transaction_id,
                        event=canonical,
                        head=freshest_head,
                        idempotent=True,
                    )
                if existing is not None:
                    # A local transaction match is only a cache result.  The
                    # remote authority must contain the exact same canonical
                    # event before a replay can be called idempotent.
                    return AppendResult(
                        status="BLOCKED",
                        reason_code="APPEND_OUTCOME_UNCERTAIN",
                        transaction_id=transaction_id,
                        event=canonical,
                        head=freshest_head,
                        frozen=True,
                    )
                if freshest_head is not None and freshest_head != current.commit:
                    # The local proposal was made against stale evidence.  A
                    # caller must refresh and recompute its event ID/payload.
                    # Do not push, retry, or consume an ID from this attempt.
                    parse_stream(freshest_raw)
                    return AppendResult(
                        status="BLOCKED",
                        reason_code="EXPECTED_EVENT_HEAD_STALE",
                        transaction_id=transaction_id,
                        event=canonical,
                        head=freshest_head,
                        frozen=False,
                    )
            except DuplicateTransactionError:
                raise
            except AppendOutcomeUncertain:
                return AppendResult(
                    status="BLOCKED",
                    reason_code="APPEND_OUTCOME_UNCERTAIN",
                    transaction_id=transaction_id,
                    event=canonical,
                    head=None,
                    frozen=True,
                )
            except Exception:
                return AppendResult(
                    status="BLOCKED",
                    reason_code="APPEND_OUTCOME_UNCERTAIN",
                    transaction_id=transaction_id,
                    event=canonical,
                    head=None,
                    frozen=True,
                )

        def do_push() -> None:
            if push is not None:
                push(commit, self.branch, current.commit)
                return
            self._git(
                [
                    "push",
                    f"--force-with-lease={self.ref}:{current.commit}",
                    remote_name,
                    f"{commit}:{self.ref}",
                ],
            )

        try:
            do_push()
        except Exception as first_error:
            observed: tuple[str, bytes] | None = None
            if remote_reader is not None:
                try:
                    observed = self._verified_remote_observation(remote_reader())
                except Exception:
                    observed = None
            if observed is not None:
                remote_head, remote_raw = observed
                try:
                    remote_events = parse_stream(remote_raw)
                    matching = self._transaction_event(remote_events, transaction_id)
                    if matching is not None:
                        if canonical_event_bytes(matching) != encoded:
                            raise DuplicateTransactionError(
                                "remote transaction ID maps to a different canonical event"
                            )
                        return AppendResult(
                            status="IDEMPOTENT",
                            reason_code=None,
                            transaction_id=transaction_id,
                            event=canonical,
                            head=remote_head,
                            idempotent=True,
                        )
                    if remote_head == current.commit:
                        do_push()
                    elif remote_head is None:
                        return AppendResult(
                            status="BLOCKED",
                            reason_code="APPEND_OUTCOME_UNCERTAIN",
                            transaction_id=transaction_id,
                            event=canonical,
                            head=None,
                            frozen=True,
                        )
                    else:
                        try:
                            # The remote advanced without this transaction.
                            # A valid byte-prefix history classifies the push
                            # as stale and asks the allocator to refresh; an
                            # unparseable/divergent stream stays frozen.
                            if not remote_raw.startswith(current.raw):
                                raise EventTransitionError("remote stream is not a byte prefix")
                            parse_stream(remote_raw)
                            return AppendResult(
                                status="BLOCKED",
                                reason_code="EXPECTED_EVENT_HEAD_STALE",
                                transaction_id=transaction_id,
                                event=canonical,
                                head=remote_head,
                                frozen=False,
                            )
                        except Exception:
                            return AppendResult(
                                status="BLOCKED",
                                reason_code="APPEND_OUTCOME_UNCERTAIN",
                                transaction_id=transaction_id,
                                event=canonical,
                                head=remote_head,
                                frozen=True,
                            )
                except DuplicateTransactionError:
                    raise
                except Exception:
                    return AppendResult(
                        status="BLOCKED",
                        reason_code="APPEND_OUTCOME_UNCERTAIN",
                        transaction_id=transaction_id,
                        event=canonical,
                        head=remote_head if observed is not None else None,
                        frozen=True,
                    )
            else:
                return AppendResult(
                    status="BLOCKED",
                    reason_code="APPEND_OUTCOME_UNCERTAIN",
                    transaction_id=transaction_id,
                    event=canonical,
                    head=None,
                    frozen=True,
                )

        try:
            observed_head, observed_raw = self._verified_remote_observation(
                remote_reader()
            )
            observed_events = parse_stream(observed_raw)
            matching = self._transaction_event(observed_events, transaction_id)
            expected_raw = current.raw + encoded + b"\n"
            if observed_head != commit or observed_raw != expected_raw:
                if matching is not None and canonical_event_bytes(matching) == encoded:
                    return AppendResult(
                        status="IDEMPOTENT",
                        reason_code=None,
                        transaction_id=transaction_id,
                        event=canonical,
                        head=observed_head,
                        idempotent=True,
                    )
                return AppendResult(
                    status="BLOCKED",
                    reason_code="APPEND_OUTCOME_UNCERTAIN",
                    transaction_id=transaction_id,
                    event=canonical,
                    head=observed_head,
                    frozen=True,
                )
        except Exception:
            return AppendResult(
                status="BLOCKED",
                reason_code="APPEND_OUTCOME_UNCERTAIN",
                transaction_id=transaction_id,
                event=canonical,
                head=None,
                frozen=True,
            )

        # A successful remote push may be reflected locally.  This local CAS
        # is only a cache update; failure means the remote event remains the
        # authority and is reported as a durable append with its known commit.
        local_result = self._git(
            ["update-ref", self.ref, commit, current.commit], check=False
        )
        if local_result.returncode != 0:
            return AppendResult(
                status="APPENDED",
                reason_code=None,
                transaction_id=transaction_id,
                event=canonical,
                head=commit,
            )
        return AppendResult(
            status="APPENDED",
            reason_code=None,
            transaction_id=transaction_id,
            event=canonical,
            head=commit,
        )


def bootstrap_release_events(
    repo: str | os.PathLike[str],
    *,
    branch: str = CANONICAL_EVENT_BRANCH,
    protection_checker: Callable[[], bool] | None = None,
    static_exclusion: StaticMutationExclusion | None = None,
    exclusion_lease: Callable[[], ContextManager[bool]] | None = None,
) -> str:
    """Bootstrap the event branch through the protected public boundary.

    Callers must provide both live protection evidence and the externally
    governed exclusion lease.  The wrapper intentionally has no unsafe
    defaults for either input.
    """

    return ReleaseEventWriter(
        repo,
        branch=branch,
        static_exclusion=static_exclusion,
        exclusion_lease=exclusion_lease,
    ).bootstrap(protection_checker=protection_checker)


__all__ = [
    "AppendOutcomeUncertain",
    "AppendResult",
    "BranchUnavailable",
    "DuplicateTransactionError",
    "ExclusionUnavailable",
    "LedgerHead",
    "PublicTagAbsenceUnavailable",
    "PublicTagExists",
    "ReleaseEventWriter",
    "StaleHeadError",
    "StaticSnapshotStale",
    "WriterError",
    "bootstrap_release_events",
]
