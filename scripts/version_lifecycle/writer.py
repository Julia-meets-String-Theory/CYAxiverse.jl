"""Git-backed append-only writer for the canonical release event ledger.

The writer uses Git's expected-old-value form of ``update-ref`` for local
compare-and-swap updates.  Remote appends build the descendant commit first
and use a normal non-force push.  An exception from a remote push is treated
as an uncertain outcome until the remote stream has been read and reconciled
by transaction ID and exact canonical event bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping
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


class PublicTagAbsenceUnavailable(WriterError):
    reason_code = "PUBLIC_TAG_ABSENCE_UNAVAILABLE"


class PublicTagExists(WriterError):
    reason_code = "PUBLIC_TAG_EXISTS"


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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


def _as_text(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class ReleaseEventWriter:
    """Append lifecycle events to one protected release-events branch.

    ``repo`` is a Git working tree or bare repository.  The writer never
    force-updates the branch and never edits a package-source branch.
    ``exclusion_checker``, ``static_snapshot_checker`` and the optional
    ``public_tag_absence_checker`` are injected by the allocation layer so
    this bounded module remains usable with fixture repos.  The tag checker
    is mandatory for a ``release_intent_aborted`` append and receives the
    exact canonical public tag from that event.
    """

    def __init__(
        self,
        repo: str | os.PathLike[str],
        *,
        branch: str = "release-events",
        stream_path: str = "release-events.jsonl",
        remote: str | None = None,
        exclusion_checker: Callable[[], bool] | None = None,
        static_snapshot_checker: Callable[[str], bool] | None = None,
        public_tag_absence_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.branch = self._validate_ref_component(branch)
        self.stream_path = self._validate_stream_path(stream_path)
        self.remote = remote
        self.exclusion_checker = exclusion_checker
        self.static_snapshot_checker = static_snapshot_checker
        self.public_tag_absence_checker = public_tag_absence_checker
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
        tree_entries = _as_text(
            self._git(["ls-tree", "--name-only", commit]).stdout
        ).splitlines()
        if tree_entries != [self.stream_path]:
            raise BranchUnavailable(
                "release-events branch must contain exactly release-events.jsonl"
            )
        result = self._git(["show", f"{commit}:{self.stream_path}"])
        raw = bytes(result.stdout)
        events = tuple(parse_stream(raw))
        return LedgerHead(commit=commit, raw=raw, events=events)

    def bootstrap(self, *, expected_absent: bool = True, message: str = "Initialize release event ledger") -> str:
        """Create the minimal orphan event branch with an empty stream.

        The update is a compare-and-swap against the all-zero object ID, so a
        concurrent creator cannot be overwritten.  Existing branches are
        validated and returned when ``expected_absent`` is false.
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
        """Verify one observed commit is exactly the event stream object.

        Callback reconciliation is accepted only when the advertised commit
        is also present in this Git object database and independently proves
        the one-file tree and exact blob bytes.  A callback cannot claim a
        remote head with an unverified or divergent topology.
        """

        if not isinstance(commit, str) or _GIT_SHA_RE.fullmatch(commit) is None:
            raise AppendOutcomeUncertain("observed event head is not a full Git SHA")
        if not isinstance(raw, bytes):
            raise AppendOutcomeUncertain("observed event stream is not bytes")
        try:
            object_type = _as_text(
                self._git(["cat-file", "-t", commit]).stdout
            ).strip()
            if object_type != "commit":
                raise AppendOutcomeUncertain("observed event head is not a commit")
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
                or path != self.stream_path
            ):
                raise AppendOutcomeUncertain(
                    "observed event commit tree is not the canonical one-file tree"
                )
            stored = bytes(self._git(["show", f"{commit}:{self.stream_path}"]).stdout)
            if stored != raw:
                raise AppendOutcomeUncertain(
                    "observed event stream does not match its commit tree"
                )
            parse_stream(raw)
        except AppendOutcomeUncertain:
            raise
        except Exception as error:
            raise AppendOutcomeUncertain(
                "observed event commit topology could not be verified"
            ) from error

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
        ``protection_checker``.  The default Git transport uses a normal
        create-only push and verifies the resulting remote object.
        """

        remote_name = remote or self.remote
        if not remote_name and push is None:
            raise ValueError("bootstrap_remote needs a remote name or push callback")
        if remote_name is None and reconcile is None:
            raise AppendOutcomeUncertain(
                "callback-only bootstrap has no remote observation; outcome is uncertain"
            )
        if protection_checker is None or not protection_checker():
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
            # a caller can then continue with normal expected-head appends.
            parse_stream(existing[1])
            return str(existing[0])
        commit = self.bootstrap(expected_absent=False, message=message)
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
        if self.exclusion_checker is not None and not self.exclusion_checker():
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
        if self.static_snapshot_checker is not None and not self.static_snapshot_checker(
            canonical["static_iteration_snapshot"]
        ):
            raise StaticSnapshotStale("static snapshot changed before append")
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
            StaticSnapshotStale,
            PublicTagAbsenceUnavailable,
            PublicTagExists,
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

        remote_name = remote or self.remote
        if not remote_name and push is None:
            raise ValueError("append_remote needs a remote name or push callback")
        # Remote lifecycle mutation is never allowed to rely on a caller's
        # memory of protection or static authority.  Fixture callers provide
        # deterministic proof callbacks; production callers wire these to
        # freshly retrieved settings/snapshot evidence.
        if self.exclusion_checker is None or self.static_snapshot_checker is None:
            canonical = validate_event(event)
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
            canonical = validate_event(event)
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
            StaticSnapshotStale,
            PublicTagAbsenceUnavailable,
            PublicTagExists,
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
        if existing is not None:
            return AppendResult(
                status="IDEMPOTENT",
                reason_code=None,
                transaction_id=transaction_id,
                event=canonical,
                head=current.commit,
                idempotent=True,
            )

        if remote_reader is not None:
            try:
                freshest_head, freshest_raw = self._verified_remote_observation(
                    remote_reader()
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
                ["push", remote_name, f"{commit}:refs/heads/{self.branch}"],
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
    repo: str | os.PathLike[str], *, branch: str = "release-events"
) -> str:
    """Convenience wrapper for the one-time orphan branch bootstrap."""

    return ReleaseEventWriter(repo, branch=branch).bootstrap()


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
