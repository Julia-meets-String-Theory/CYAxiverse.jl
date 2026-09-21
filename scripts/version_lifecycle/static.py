"""Static iteration authority and replayable allocation snapshots.

The mutable lifecycle ledger is implemented separately.  This module only
reads Git and ``iterations.toml`` and produces an immutable, digest-bound view
of the versions already occupied by static history.  It intentionally has no
production ref mutation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence
import tomllib
from urllib.parse import urlsplit, urlunsplit

from .codec import canonical_json, sha256_hex
from .versions import Version, final_version, parse_package_version, parse_public_tag


CANONICAL_STATIC_ITERATION_SOURCE = "refs/heads/vmm:iterations.toml"
canonical_static_iteration_source = CANONICAL_STATIC_ITERATION_SOURCE
SNAPSHOT_SCHEMA_VERSION = 1
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_PUBLIC_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PUBLIC_URL_PATH_RE = re.compile(r"^/(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)+[A-Za-z0-9][A-Za-z0-9_.-]*/?$")
_PUBLIC_SCP_RE = re.compile(r"^git@github\.com:(?P<path>[^/]+/[^/]+?)(?:\.git)?$")
_NUMERIC_HOST_LABEL_RE = re.compile(r"^(?:0[xX][0-9A-Fa-f]+|[0-9A-Fa-f]+)$")
_PUBLIC_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_NONPUBLIC_DNS_SUFFIXES = (
    ".arpa", ".corp", ".example.com", ".example.net", ".example.org",
    ".home", ".internal", ".intranet", ".lan",
    ".local", ".localdomain", ".localhost", ".private",
    ".test", ".example", ".invalid", ".onion",
)
_NONPUBLIC_DNS_NAMES = {"example.com", "example.net", "example.org"}
_UNSAFE_SOURCE_MARKERS = (
    "file:", "local:", "ssh:", "git:", "credential", "password", "token",
    "secret", "authorization", "bearer ", "api_key", "apikey",
)
_LOCAL_SOURCE_NAMES = {"file", "local", "localhost", "private", "users", "home", "tmp", "var"}
_LOCAL_SOURCE_NAMES.update({"intranet"})

# ``authority_verified`` is retained as a compatibility/status field, but it
# is not an authority capability.  The marker is installed only after the
# canonical remote source has been resolved and validated.  In particular, a
# caller cannot obtain allocation authority by constructing a snapshot with
# ``authority_verified=True``.
_STATIC_AUTHORITY_TOKEN = object()


class StaticValidationError(ValueError):
    """Raised when static registry data or a snapshot is not valid."""


class UnsafeSourceRepositoryError(StaticValidationError):
    """Raised when a source identity is not safe to persist as public metadata."""

    reason_code = "STATIC_SOURCE_REPOSITORY_UNSAFE"


class SnapshotStaleError(StaticValidationError):
    """Raised when a bound snapshot no longer describes the source refs."""


@dataclass(frozen=True, slots=True)
class BlockedResult:
    """A machine-visible fail-closed outcome."""

    status: str = "BLOCKED"
    reason_code: str = "STATIC_AUTHORITY_SELECTOR_UNRESOLVED"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class StaticSnapshot:
    """A four-digest static namespace snapshot.

    ``data`` is the exact canonical object used for ``snapshot_digest``.
    Properties expose the common fields without allowing callers to mutate the
    digest preimage.
    """

    data: Mapping[str, Any]
    source_bytes: bytes | None = None
    authority_verified: bool = False
    _authority_token: object | None = field(default=None, init=False, repr=False, compare=False)
    _authority_binding: tuple[str, str, str, str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def status(self) -> str:
        return "READY"

    @property
    def snapshot_digest(self) -> str:
        return str(self.data["snapshot_digest"])

    @property
    def occupied_versions(self) -> tuple[str, ...]:
        return tuple(self.data["occupied_versions"])

    @property
    def ref_bindings(self) -> tuple[Mapping[str, str], ...]:
        return tuple(self.data["iteration_ref_bindings"])

    @property
    def tag_bindings(self) -> tuple[Mapping[str, str], ...]:
        return tuple(self.data["public_tag_bindings"])

    @property
    def file_digest(self) -> str:
        return str(self.data["iterations_toml_sha256"])

    @property
    def ref_set_digest(self) -> str:
        return str(self.data["ref_set_digest"])

    @property
    def tag_set_digest(self) -> str:
        return str(self.data["tag_set_digest"])

    @property
    def source_commit(self) -> str:
        return str(self.data["source_commit"])

    @property
    def source_tree(self) -> str:
        return str(self.data["source_tree"])

    @property
    def source_repository(self) -> str:
        return str(self.data["source_repository"])

    @property
    def canonical_source(self) -> str:
        return str(self.data["canonical_static_iteration_source"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def _mark_authority_verified(snapshot: StaticSnapshot) -> StaticSnapshot:
    """Mark a snapshot as remote-authority verified using a private token."""

    object.__setattr__(snapshot, "_authority_token", _STATIC_AUTHORITY_TOKEN)
    source_bytes = snapshot.source_bytes
    if not isinstance(source_bytes, bytes):
        raise StaticValidationError("remote-authority snapshots require exact source bytes")
    object.__setattr__(
        snapshot,
        "_authority_binding",
        (
            snapshot.snapshot_digest,
            snapshot.source_commit,
            snapshot.source_tree,
            sha256_hex(source_bytes),
        ),
    )
    object.__setattr__(snapshot, "authority_verified", True)
    return snapshot


def _has_verified_authority(snapshot: StaticSnapshot) -> bool:
    """Return whether *snapshot* was verified by this module's authority path."""

    if snapshot._authority_token is not _STATIC_AUTHORITY_TOKEN:
        return False
    binding = snapshot._authority_binding
    if binding is None or not isinstance(snapshot.source_bytes, bytes):
        return False
    try:
        current = (
            snapshot.snapshot_digest,
            snapshot.source_commit,
            snapshot.source_tree,
            sha256_hex(snapshot.source_bytes),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return binding == current


def _git(repository: Path, *args: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise StaticValidationError(detail or f"git command failed: {' '.join(args)}")
    return result.stdout


def _resolve_commit_value(repository: Path, value: str) -> str:
    return _git(repository, "rev-parse", "--verify", f"{value}^{{commit}}").decode().strip()


def _resolve_tree(repository: Path, commit: str) -> str:
    return _git(repository, "rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()


def _show(repository: Path, treeish: str, path: str) -> bytes:
    return _git(repository, "show", f"{treeish}:{path}")


def _remote_refs(repository: Path, remote: str) -> dict[str, str]:
    """Resolve the complete remote ref namespace without trusting local refs."""

    output = _git(repository, "ls-remote", "--refs", remote)
    refs: dict[str, str] = {}
    for line in output.decode("ascii", errors="strict").splitlines():
        if not line.strip():
            continue
        try:
            object_id, ref = line.split("\t", 1)
        except ValueError as error:
            raise StaticValidationError("remote ref advertisement is malformed") from error
        if not _HEX40_RE.fullmatch(object_id) or not ref.startswith("refs/"):
            raise StaticValidationError("remote ref advertisement has invalid identity")
        if ref in refs and refs[ref] != object_id:
            raise StaticValidationError(f"remote ref {ref} was advertised twice with different objects")
        refs[ref] = object_id
    if "refs/heads/vmm" not in refs:
        raise StaticValidationError("remote does not advertise refs/heads/vmm")
    return refs


def _ensure_remote_objects(
    repository: Path,
    remote: str,
    advertised_refs: Mapping[str, str],
) -> dict[str, str]:
    """Fetch advertised objects into the local object store without refs.

    A remote advertisement can advance while its objects are absent from a
    shallow or stale checkout.  Fetching exact object IDs with no ref update
    makes the snapshot replayable while the second advertisement check keeps
    a race fail-closed.
    """

    relevant = [
        ref for ref in advertised_refs
        if ref == "refs/heads/vmm"
        or ref == "refs/heads/main"
        or ref.startswith(("refs/iterations/", "refs/heads/iterations/", "refs/tags/"))
    ]
    for object_id in sorted({advertised_refs[ref] for ref in relevant}):
        result = subprocess.run(
            [
                "git", "-C", str(repository), "fetch", "--no-tags",
                "--no-write-fetch-head", remote, object_id,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise StaticValidationError(detail or f"could not fetch remote object {object_id}")
    refreshed = _remote_refs(repository, remote)
    if dict(refreshed) != dict(advertised_refs):
        raise StaticValidationError("remote authority advanced during static snapshot resolution")
    return refreshed


def _selector(selector: str) -> tuple[str, str]:
    if not isinstance(selector, str) or selector.count(":") != 1:
        raise StaticValidationError("static selector must have exactly one ref:path separator")
    ref, path = selector.split(":", 1)
    if ref != "refs/heads/vmm" or path != "iterations.toml":
        raise StaticValidationError(
            f"selector must be {CANONICAL_STATIC_ITERATION_SOURCE!r}, got {selector!r}"
        )
    return ref, path


def sanitize_source_repository(value: str) -> str:
    """Return a canonical public repository identity or reject it.

    A source identity is metadata included in a durable snapshot.  Accept
    either a simple public slug (``owner/repository`` or ``repository``), a
    GitHub SCP identity, or an HTTP(S) URL without credentials.  Local
    locators, private hosts, credentials, and secret-like values are rejected
    without echoing the supplied value in the exception.
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise UnsafeSourceRepositoryError(
            "source_repository must be a sanitized public repository identity"
        )
    if any(ord(char) < 0x20 or ord(char) > 0x7E for char in value):
        raise UnsafeSourceRepositoryError(
            "source_repository must contain printable ASCII"
        )
    if value.startswith(("/", "~", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value):
        raise UnsafeSourceRepositoryError(
            "source_repository cannot contain a local locator"
        )
    lowered = value.lower()
    if any(marker in lowered for marker in _UNSAFE_SOURCE_MARKERS):
        raise UnsafeSourceRepositoryError(
            "source_repository cannot contain a local or secret-like locator"
        )

    scp = _PUBLIC_SCP_RE.fullmatch(value)
    if scp is not None:
        owner, repository = scp.group("path").split("/", 1)
        if _PUBLIC_SLUG_RE.fullmatch(owner) and _PUBLIC_SLUG_RE.fullmatch(repository):
            return f"https://github.com/{owner}/{repository}"
        raise UnsafeSourceRepositoryError(
            "source_repository must identify a public repository"
        )

    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise UnsafeSourceRepositoryError(
            "source_repository must be a well-formed public repository identity"
        ) from error
    if parsed.scheme:
        if parsed.scheme not in {"http", "https"} or parsed.username is not None or parsed.password is not None:
            raise UnsafeSourceRepositoryError(
                "source_repository must be an HTTP(S) URL without credentials"
            )
        if (
            "?" in value or "#" in value or not host
            or port is not None or "%" in parsed.netloc
        ):
            raise UnsafeSourceRepositoryError(
                "source_repository URL has an unsafe authority or suffix"
            )
        # DNS permits an absolute name with a trailing root label.  Strip it
        # before classifying the host so ``127.0.0.1.`` cannot bypass the IP
        # policy or the local-host suffix checks.
        host = host.lower().rstrip(".")
        if not host:
            raise UnsafeSourceRepositoryError(
                "source_repository URL must use a public host"
            )
        if (
            host in {"localhost", "localhost.localdomain", "intranet"}
            or host in _NONPUBLIC_DNS_NAMES
            or host.endswith(_NONPUBLIC_DNS_SUFFIXES)
        ):
            raise UnsafeSourceRepositoryError(
                "source_repository URL must use a public host"
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # Some URL clients interpret abbreviated or nondecimal dotted
            # numeric hosts as IPv4. Reject these aliases while retaining
            # canonical globally routable IP literals.
            if all(_NUMERIC_HOST_LABEL_RE.fullmatch(label) for label in host.split(".")):
                raise UnsafeSourceRepositoryError(
                    "source_repository URL must use a public host"
                )
            address = None
        if address is not None and not address.is_global:
            raise UnsafeSourceRepositoryError(
                "source_repository URL must use a public host"
            )
        if address is None and (
            len(host) > 253
            or "." not in host
            or not all(_PUBLIC_DNS_LABEL_RE.fullmatch(label) for label in host.split("."))
        ):
            raise UnsafeSourceRepositoryError(
                "source_repository URL must use a public host"
            )
        path = parsed.path
        if not _PUBLIC_URL_PATH_RE.fullmatch(path):
            raise UnsafeSourceRepositoryError(
                "source_repository URL must identify a public repository"
            )
        normalized_path = path.rstrip("/")
        if normalized_path.endswith(".git"):
            normalized_path = normalized_path[:-4]
        public_host = f"[{host}]" if address is not None and address.version == 6 else host
        return urlunsplit((parsed.scheme, public_host, normalized_path, "", ""))

    if "/" in value:
        parts = value.split("/")
        if len(parts) != 2 or any(part in _LOCAL_SOURCE_NAMES for part in (part.lower() for part in parts)):
            raise UnsafeSourceRepositoryError(
                "source_repository must identify a public repository"
            )
        if not all(_PUBLIC_SLUG_RE.fullmatch(part) for part in parts):
            raise UnsafeSourceRepositoryError(
                "source_repository must identify a public repository"
            )
        owner, repository = parts
    else:
        if value.lower() in _LOCAL_SOURCE_NAMES or not _PUBLIC_SLUG_RE.fullmatch(value):
            raise UnsafeSourceRepositoryError(
                "source_repository must identify a public repository"
            )
        repository = value
        owner = None
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not repository or not _PUBLIC_SLUG_RE.fullmatch(repository):
        raise UnsafeSourceRepositoryError(
            "source_repository must identify a public repository"
        )
    return f"{owner}/{repository}" if owner is not None else repository


def _repository_identity(repository: Path, remote_name: str = "origin") -> str:
    remote = _git(repository, "config", "--get", f"remote.{remote_name}.url", check=False).decode().strip()
    if not remote:
        raise UnsafeSourceRepositoryError(
            "source_repository requires an explicitly supplied public identity when no remote is configured"
        )
    try:
        return sanitize_source_repository(remote)
    except UnsafeSourceRepositoryError as error:
        raise UnsafeSourceRepositoryError(
            "configured remote does not provide a sanitized public source identity"
        ) from error


def _read_toml(raw: bytes, source: str) -> Mapping[str, Any]:
    try:
        value = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise StaticValidationError(f"invalid TOML in {source}: {error}") from error
    if not isinstance(value, dict):
        raise StaticValidationError(f"registry {source} must be a TOML table")
    return value


def _as_str(mapping: Mapping[str, Any], key: str, source: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise StaticValidationError(f"{source} requires nonempty string field {key!r}")
    return value


def _entry_lists(registry: Mapping[str, Any], source: str) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    schema_version = registry.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SNAPSHOT_SCHEMA_VERSION
    ):
        raise StaticValidationError(f"{source} requires schema_version = 1")
    # ``iterations`` is convenient for fixtures; the named arrays make the
    # prospective/retrospective boundary explicit in the checked-in registry.
    for key in (
        "iterations", "prospective", "retrospective",
        "prospective_iterations", "retrospective_iterations",
    ):
        value = registry.get(key, [])
        if not isinstance(value, list):
            raise StaticValidationError(f"{source} field {key!r} must be an array")
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise StaticValidationError(f"{source} {key}[{index}] must be a table")
            if key in {"prospective", "prospective_iterations"} and "kind" not in item:
                item = {**item, "kind": "prospective"}
            elif key in {"retrospective", "retrospective_iterations"} and "kind" not in item:
                item = {**item, "kind": "retrospective"}
            entries.append(item)
    return entries


def _entry_version(entry: Mapping[str, Any], source: str) -> Version:
    for key in ("final_version", "declared_version", "version"):
        if key in entry:
            try:
                return final_version(parse_package_version(_as_str(entry, key, source)))
            except ValueError as error:
                raise StaticValidationError(f"{source} invalid {key}: {error}") from error
    raise StaticValidationError(f"{source} lacks a final version field")


def _final_entry_version(entry: Mapping[str, Any], key: str, source: str) -> Version:
    """Parse one explicitly recorded final version field."""

    try:
        return final_version(parse_package_version(_as_str(entry, key, source)))
    except (TypeError, ValueError) as error:
        raise StaticValidationError(f"{source} invalid {key}: {error}") from error


def _historical_alias(
    entry: Mapping[str, Any],
    keys: tuple[str, ...],
    source: str,
) -> tuple[bool, Any]:
    """Return one historical field value and reject contradictory aliases."""

    present = [(key, entry[key]) for key in keys if key in entry]
    if not present:
        return False, None
    value = present[0][1]
    if any(candidate != value for _, candidate in present[1:]):
        raise StaticValidationError(
            f"{source} has contradictory historical fields: {[key for key, _ in present]}"
        )
    return True, value


def _validate_entry(entry: Mapping[str, Any], source: str, *, anchor: str | None = None) -> Version:
    kind = entry.get("kind")
    if kind not in {"prospective", "retrospective"}:
        raise StaticValidationError(f"{source} kind must be prospective or retrospective")
    _as_str(entry, "iteration_id", source)

    if kind == "prospective":
        version = _entry_version(entry, source)
        release_line = _as_str(entry, "release_line", source)
        if release_line != "principal":
            match = re.fullmatch(
                r"maintenance/(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)",
                release_line,
            )
            if match is None:
                raise StaticValidationError(
                    f"{source} release_line must be principal or maintenance/X.Y"
                )
            if (int(match.group(1)), int(match.group(2))) != (version.major, version.minor):
                raise StaticValidationError(
                    f"{source} release_line does not match final_version lineage"
                )
        if not isinstance(entry.get("aggregate_impact"), dict):
            raise StaticValidationError(f"{source} requires aggregate_impact table")
        anchor_ref = _as_str(entry, "anchor_ref", source)
        if anchor_ref != f"iterations/{version.canonical}":
            raise StaticValidationError(f"{source} anchor_ref does not match final_version")
        contributions = entry.get("contributing_identities", entry.get("contributions"))
        if not isinstance(contributions, list) or not contributions:
            raise StaticValidationError(f"{source} requires contributing_identities")
        for index, contribution in enumerate(contributions):
            if not isinstance(contribution, dict):
                raise StaticValidationError(f"{source} contributing_identities[{index}] must be a table")
            _as_str(contribution, "role", f"{source}.contributing_identities[{index}]")
            _as_str(contribution, "identity", f"{source}.contributing_identities[{index}]")
        forbidden = {
            "self_sha", "self_tree", "iteration_sha", "iteration_tree",
            "anchor_sha", "anchor_tree", "closure_sha", "closure_tree",
            "release_state", "release_status", "future_release_identity",
            "closure_timestamp_utc", "guessed_closure_time", "github_release_identity",
        }
        present = forbidden.intersection(entry)
        if present:
            raise StaticValidationError(f"{source} has forbidden self/mutable fields: {sorted(present)}")
    else:
        # Retrospective records are Gate B inputs.  Keep declared, actual and
        # release facts distinct so a historical mismatch cannot be hidden.
        for key in ("historical_release_status", "anchor_sha", "anchor_tree"):
            _as_str(entry, key, source)
        # ``declared_version`` is the historical truth source.  Do not let a
        # legacy ``final_version``/``version`` field silently override it.
        if "declared_version" not in entry:
            raise StaticValidationError(f"{source} requires declared_version")
        version = _final_entry_version(entry, "declared_version", source)
        for key in ("final_version", "version"):
            if key in entry and _final_entry_version(entry, key, source) != version:
                raise StaticValidationError(
                    f"{source} {key} contradicts declared_version"
                )

        carried_present, carried = _historical_alias(
            entry,
            ("project_toml_carried", "project_toml_carried_version", "actual_version_present"),
            source,
        )
        if not carried_present:
            raise StaticValidationError(f"{source} requires project_toml_carried")
        if not isinstance(carried, bool):
            raise StaticValidationError(f"{source} project_toml_carried must be boolean")

        actual_present, actual = _historical_alias(
            entry,
            ("actual_project_version", "project_toml_version", "actual_version"),
            source,
        )
        if carried is True and (not actual_present or actual is None):
            raise StaticValidationError(f"{source} marks Project.toml carried but omits actual version")
        if carried is False and actual_present:
            raise StaticValidationError(f"{source} marks Project.toml absent but supplies actual version")
        if actual_present:
            try:
                parse_package_version(actual)
            except (TypeError, ValueError) as error:
                raise StaticValidationError(f"{source} invalid actual Project.toml version: {error}") from error
    if anchor is not None and entry.get("anchor_ref") not in {anchor, f"refs/{anchor}"}:
        raise StaticValidationError(f"{source} does not bind anchor {anchor}")
    return version


def _validate_registry(registry: Mapping[str, Any], source: str) -> set[str]:
    occupied: set[str] = set()
    for index, entry in enumerate(_entry_lists(registry, source)):
        if source == CANONICAL_STATIC_ITERATION_SOURCE and str(entry.get("release_line", "")).startswith("maintenance/"):
            raise StaticValidationError(
                "maintenance metadata must remain in its protected iteration anchor tree"
            )
        version = _validate_entry(entry, f"{source} entry {index}")
        if version.canonical in occupied:
            raise StaticValidationError(f"duplicate static version {version.canonical}")
        occupied.add(version.canonical)
        actual = entry.get(
            "actual_project_version",
            entry.get("project_toml_version", entry.get("actual_version")),
        )
        if actual is not None:
            actual_version = parse_package_version(actual).final
            if actual_version.canonical in occupied and actual_version.canonical != version.canonical:
                raise StaticValidationError(f"duplicate static version {actual_version.canonical}")
            occupied.add(actual_version.canonical)
    return occupied


def _anchor_bindings(
    repository: Path,
    remote_refs: Mapping[str, str],
) -> list[dict[str, str]]:
    # Iteration anchors are protected annotated tags.  A branch or an
    # unqualified custom namespace is mutable and cannot establish the static
    # identity required by R-019.
    invalid_refs = [
        ref for ref in remote_refs
        if ref.startswith(("refs/iterations/", "refs/heads/iterations/"))
    ]
    if invalid_refs:
        raise StaticValidationError(
            "protected iteration anchors must use annotated refs/tags/iterations/* tags"
        )
    full_refs = [ref for ref in remote_refs if ref.startswith("refs/tags/iterations/")]
    bindings: list[dict[str, str]] = []
    seen: set[str] = set()
    for full_ref in sorted(set(full_refs)):
        ref = full_ref.removeprefix("refs/tags/")
        if ref in seen:
            raise StaticValidationError(f"duplicate protected iteration identity {ref}")
        seen.add(ref)
        suffix = ref.removeprefix("iterations/")
        try:
            version = final_version(parse_package_version(suffix))
        except (TypeError, ValueError) as error:
            raise StaticValidationError(f"invalid protected iteration ref {ref!r}: {error}") from error
        object_id = remote_refs[full_ref]
        object_type = _git(repository, "cat-file", "-t", object_id).decode().strip()
        if object_type != "tag":
            raise StaticValidationError(
                f"protected iteration anchor {ref} must be an annotated tag"
            )
        commit = _resolve_commit_value(repository, object_id)
        tree = _resolve_tree(repository, commit)
        anchor_raw: bytes | None = None
        for path in ("iterations.toml", ".cyaxiverse/iteration.toml"):
            try:
                anchor_raw = _show(repository, tree, path)
                break
            except StaticValidationError:
                continue
        if anchor_raw is None:
            raise StaticValidationError(f"anchor {ref} has no static iteration registry")
        registry = _read_toml(anchor_raw, f"{ref}:{path}")
        entries = _entry_lists(registry, f"{ref}:{path}")
        matches = []
        for index, entry in enumerate(entries):
            entry_version = _validate_entry(entry, f"{ref}:{path} entry {index}", anchor=ref)
            if entry_version == version:
                matches.append(entry)
        if len(matches) != 1:
            raise StaticValidationError(f"anchor {ref} must contain exactly one matching static entry")
        bindings.append({"ref": ref, "commit": commit, "tree": tree})
    return sorted(bindings, key=lambda item: item["ref"])


def _tag_bindings(
    repository: Path,
    remote_refs: Mapping[str, str],
) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    seen: set[str] = set()
    tags = [ref.removeprefix("refs/tags/") for ref in remote_refs if ref.startswith("refs/tags/")]
    for tag in sorted(tags):
        if tag in seen or tag == "v-0.1":
            continue
        try:
            parse_public_tag(tag)
        except (TypeError, ValueError):
            # Existing noncanonical tags remain historical evidence and are
            # excluded from future canonical allocation.
            continue
        seen.add(tag)
        commit = _resolve_commit_value(repository, remote_refs[f"refs/tags/{tag}"])
        bindings.append({"tag": tag, "commit": commit, "tree": _resolve_tree(repository, commit)})
    return sorted(bindings, key=lambda item: item["tag"])


def _source_versions(
    repository: Path,
    source_commit: str,
    source_ref: str,
    remote_refs: Mapping[str, str],
) -> set[str]:
    occupied: set[str] = set()
    refs = [(source_ref, source_commit)]
    if "refs/heads/main" in remote_refs:
        refs.append(("refs/heads/main", _resolve_commit_value(repository, remote_refs["refs/heads/main"])))
    for ref, commit in refs:
        try:
            raw = _show(repository, commit, "Project.toml")
        except StaticValidationError:
            continue
        try:
            project = tomllib.loads(raw.decode("utf-8"))
            value = project.get("version")
            if isinstance(value, str):
                # A DEV package identity still reserves its final namespace
                # member; the mutable event head records its owner state.
                version = parse_package_version(value).final
                occupied.add(version.canonical)
        except (UnicodeDecodeError, tomllib.TOMLDecodeError, TypeError, ValueError) as error:
            raise StaticValidationError(f"invalid package version at {ref}: {error}") from error
    return occupied


def _bind_digest(bindings: Sequence[Mapping[str, str]], keys: tuple[str, str, str]) -> str:
    normalized = [
        {keys[0]: str(item[keys[0]]), keys[1]: str(item[keys[1]]), keys[2]: str(item[keys[2]])}
        for item in bindings
    ]
    return sha256_hex(canonical_json(normalized))


def _snapshot_preimage(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: data[key]
        for key in (
            "snapshot_schema_version",
            "canonical_static_iteration_source",
            "source_repository",
            "source_ref",
            "source_path",
            "source_commit",
            "source_tree",
            "iterations_toml_sha256",
            "iteration_ref_bindings",
            "ref_set_digest",
            "public_tag_bindings",
            "tag_set_digest",
            "occupied_versions",
        )
    }


def recompute_snapshot_digests(snapshot: StaticSnapshot | Mapping[str, Any]) -> dict[str, str]:
    """Recompute all snapshot digests from their exact preimages.

    This function does not trust any digest field supplied by the caller.  It
    also validates sortedness and duplicate identities before hashing.
    """

    data = snapshot.data if isinstance(snapshot, StaticSnapshot) else snapshot
    fields = (
        "snapshot_schema_version", "canonical_static_iteration_source", "source_repository",
        "source_ref", "source_path", "source_commit", "source_tree", "iterations_toml_sha256",
        "iteration_ref_bindings", "ref_set_digest", "public_tag_bindings", "tag_set_digest",
        "occupied_versions",
    )
    allowed_fields = set(fields) | {"snapshot_digest"}
    unexpected = sorted(set(data) - allowed_fields)
    if unexpected:
        raise StaticValidationError(f"snapshot has undeclared fields: {unexpected}")
    missing = [key for key in fields if key not in data]
    if missing:
        raise StaticValidationError(f"snapshot missing fields: {missing}")
    if (
        isinstance(data["snapshot_schema_version"], bool)
        or not isinstance(data["snapshot_schema_version"], int)
        or data["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION
    ):
        raise StaticValidationError("unsupported snapshot_schema_version")
    if data["canonical_static_iteration_source"] != CANONICAL_STATIC_ITERATION_SOURCE:
        raise StaticValidationError(
            "canonical_static_iteration_source must be the exact canonical selector"
        )
    if data["source_ref"] != "refs/heads/vmm" or data["source_path"] != "iterations.toml":
        raise StaticValidationError(
            "snapshot source_ref/source_path must identify the canonical selector"
        )
    for field in ("source_repository", "source_ref", "source_path", "source_commit", "source_tree"):
        if not isinstance(data[field], str) or not data[field]:
            raise StaticValidationError(f"snapshot field {field} must be a nonempty string")
    normalized_source_repository = sanitize_source_repository(data["source_repository"])
    if normalized_source_repository != data["source_repository"]:
        raise StaticValidationError("snapshot source_repository is not canonical")
    if not _HEX64_RE.fullmatch(str(data["iterations_toml_sha256"])):
        raise StaticValidationError("iterations_toml_sha256 must be lowercase SHA-256 hex")
    if not _HEX40_RE.fullmatch(str(data["source_commit"])) or not _HEX40_RE.fullmatch(str(data["source_tree"])):
        raise StaticValidationError("source commit/tree must be lowercase Git object IDs")
    refs = data["iteration_ref_bindings"]
    tags = data["public_tag_bindings"]
    occupied = data["occupied_versions"]
    if not isinstance(refs, list) or not isinstance(tags, list) or not isinstance(occupied, list):
        raise StaticValidationError("snapshot binding and occupied fields must be arrays")
    for index, version in enumerate(occupied):
        if not isinstance(version, str):
            raise StaticValidationError(
                f"occupied_versions[{index}] must be a canonical final version string"
            )
        try:
            parsed = final_version(parse_package_version(version))
        except (TypeError, ValueError) as error:
            raise StaticValidationError(
                f"occupied_versions[{index}] is not a canonical final version"
            ) from error
        if parsed.canonical != version:
            raise StaticValidationError(
                f"occupied_versions[{index}] is not a canonical final version"
            )
    ref_names = [item.get("ref") for item in refs if isinstance(item, dict)]
    tag_names = [item.get("tag") for item in tags if isinstance(item, dict)]
    if len(ref_names) != len(refs) or len(tag_names) != len(tags):
        raise StaticValidationError("snapshot binding entries must be objects")
    if ref_names != sorted(ref_names) or tag_names != sorted(tag_names) or occupied != sorted(occupied):
        raise StaticValidationError("snapshot arrays must use canonical ASCII order")
    if len(set(ref_names)) != len(ref_names) or len(set(tag_names)) != len(tag_names) or len(set(occupied)) != len(occupied):
        raise StaticValidationError("snapshot arrays contain duplicate identities")
    for item in refs:
        if set(item) != {"ref", "commit", "tree"}:
            raise StaticValidationError("iteration ref binding has unexpected fields")
        if not isinstance(item["ref"], str) or not item["ref"].startswith("iterations/"):
            raise StaticValidationError("iteration ref binding has invalid ref")
        if not _HEX40_RE.fullmatch(str(item["commit"])) or not _HEX40_RE.fullmatch(str(item["tree"])):
            raise StaticValidationError("iteration ref binding has invalid commit/tree")
    for item in tags:
        if set(item) != {"tag", "commit", "tree"}:
            raise StaticValidationError("public tag binding has unexpected fields")
        try:
            parse_public_tag(item["tag"])
        except (TypeError, ValueError) as error:
            raise StaticValidationError(f"invalid canonical public tag binding: {error}") from error
        if not _HEX40_RE.fullmatch(str(item["commit"])) or not _HEX40_RE.fullmatch(str(item["tree"])):
            raise StaticValidationError("public tag binding has invalid commit/tree")
    ref_digest = _bind_digest(refs, ("ref", "commit", "tree"))
    tag_digest = _bind_digest(tags, ("tag", "commit", "tree"))
    raw_digest = str(data["iterations_toml_sha256"])
    preimage = _snapshot_preimage(data)
    preimage["ref_set_digest"] = ref_digest
    preimage["tag_set_digest"] = tag_digest
    return {
        "iterations_toml_sha256": raw_digest,
        "ref_set_digest": ref_digest,
        "tag_set_digest": tag_digest,
        "snapshot_digest": sha256_hex(canonical_json(preimage)),
    }


recompute_snapshot = recompute_snapshot_digests


def validate_static_snapshot(
    snapshot: StaticSnapshot | Mapping[str, Any],
    *,
    repository: str | Path | None = None,
    source_bytes: bytes | None = None,
    remote: str = "origin",
    structural_only: bool = False,
) -> StaticSnapshot:
    """Validate all four digests and optionally re-read the remote source.

    A structural-only check validates the serialized shape and nested digest
    preimages.  Allocation callers must provide the exact source bytes (the
    builder stores them in an in-memory snapshot) or a repository/remote from
    which those bytes can be freshly resolved.
    """

    if source_bytes is None and isinstance(snapshot, StaticSnapshot):
        source_bytes = snapshot.source_bytes
    data = dict(snapshot.data if isinstance(snapshot, StaticSnapshot) else snapshot)
    if "snapshot_digest" not in data:
        raise StaticValidationError("snapshot_digest is required")
    recomputed = recompute_snapshot_digests(data)
    for key, value in recomputed.items():
        if data.get(key) != value:
            raise StaticValidationError(f"snapshot {key} does not match its canonical preimage")
    if source_bytes is None and repository is None and not structural_only:
        raise StaticValidationError("exact source bytes or repository authority are required")
    if source_bytes is not None:
        if not isinstance(source_bytes, bytes):
            raise TypeError("source_bytes must be exact bytes")
        if sha256_hex(source_bytes) != data["iterations_toml_sha256"]:
            raise StaticValidationError("iterations.toml raw digest does not match source bytes")
    authority_verified = isinstance(snapshot, StaticSnapshot) and snapshot.authority_verified
    authority_token = isinstance(snapshot, StaticSnapshot) and _has_verified_authority(snapshot)
    if repository is not None:
        fresh = build_static_snapshot(
            repository=repository,
            source_repository=data["source_repository"],
            remote=remote,
        )
        if isinstance(fresh, BlockedResult):
            raise SnapshotStaleError(fresh.detail)
        if fresh.to_dict() != data:
            raise SnapshotStaleError("bound static snapshot differs from current source refs")
        source_bytes = fresh.source_bytes
        authority_verified = True
        authority_token = True
    validated = StaticSnapshot(
        data,
        source_bytes=source_bytes,
        authority_verified=authority_verified,
    )
    if authority_token:
        _mark_authority_verified(validated)
    return validated


def build_static_snapshot(
    repository: str | Path = ".",
    *,
    source_repository: str | None = None,
    selector: str = CANONICAL_STATIC_ITERATION_SOURCE,
    repo: str | Path | None = None,
    repository_path: str | Path | None = None,
    remote: str = "origin",
) -> StaticSnapshot | BlockedResult:
    """Resolve and validate the canonical source and create its snapshot.

    Selector resolution and source validation fail closed with the exact
    ``STATIC_AUTHORITY_SELECTOR_UNRESOLVED`` result required by the spec.
    Other callers can use :func:`validate_static_snapshot` to reject tampered
    serialized snapshots before relying on them.
    """

    if repo is not None and repository_path is not None:
        raise TypeError("pass only one of repo and repository_path")
    if repo is not None:
        repository = repo
    if repository_path is not None:
        repository = repository_path
    repository_path = Path(repository).resolve()
    try:
        source_ref, source_path = _selector(selector)
        resolved_source_repository = (
            sanitize_source_repository(source_repository)
            if source_repository is not None
            else None
        )
        advertised_refs = _remote_refs(repository_path, remote)
        advertised_refs = _ensure_remote_objects(repository_path, remote, advertised_refs)
        source_commit = _resolve_commit_value(repository_path, advertised_refs[source_ref])
        source_tree = _resolve_tree(repository_path, source_commit)
        raw = _show(repository_path, source_commit, source_path)
        registry = _read_toml(raw, selector)
        target_iteration = registry.get("target_iteration")
        if not isinstance(target_iteration, str) or not target_iteration.strip():
            raise StaticValidationError("canonical registry requires target_iteration")
        occupied = _validate_registry(registry, selector)
        anchors = _anchor_bindings(repository_path, advertised_refs)
        tags = _tag_bindings(repository_path, advertised_refs)
        for anchor in anchors:
            occupied.add(anchor["ref"].removeprefix("iterations/"))
        for tag in tags:
            occupied.add(parse_public_tag(tag["tag"]).canonical)
        occupied.update(_source_versions(repository_path, source_commit, source_ref, advertised_refs))
        occupied_versions = sorted(occupied)
        ref_digest = _bind_digest(anchors, ("ref", "commit", "tree"))
        tag_digest = _bind_digest(tags, ("tag", "commit", "tree"))
        data: dict[str, Any] = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "canonical_static_iteration_source": selector,
            "source_repository": resolved_source_repository or _repository_identity(repository_path, remote),
            "source_ref": source_ref,
            "source_path": source_path,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "iterations_toml_sha256": sha256_hex(raw),
            "iteration_ref_bindings": anchors,
            "ref_set_digest": ref_digest,
            "public_tag_bindings": tags,
            "tag_set_digest": tag_digest,
            "occupied_versions": occupied_versions,
        }
        data["snapshot_digest"] = sha256_hex(canonical_json(_snapshot_preimage(data)))
        validated = validate_static_snapshot(StaticSnapshot(data, source_bytes=raw), source_bytes=raw)
        return _mark_authority_verified(StaticSnapshot(
            validated.data,
            source_bytes=raw,
            authority_verified=True,
        ))
    except UnsafeSourceRepositoryError as error:
        return BlockedResult(reason_code=error.reason_code, detail=str(error))
    except (OSError, subprocess.SubprocessError, StaticValidationError, UnicodeError, ValueError) as error:
        return BlockedResult(detail=str(error))


def static_snapshot(*args: Any, **kwargs: Any) -> StaticSnapshot | BlockedResult:
    """Compatibility alias for :func:`build_static_snapshot`."""

    return build_static_snapshot(*args, **kwargs)


# The specification names this artifact ``static_iteration_snapshot``.  Keep
# the descriptive aliases alongside the implementation name so callers can
# use either terminology without creating a second implementation.
static_iteration_snapshot = build_static_snapshot
compute_static_snapshot = build_static_snapshot


def snapshot_is_stale(snapshot: StaticSnapshot | Mapping[str, Any], repository: str | Path = ".") -> bool:
    """Return whether a fresh canonical source snapshot differs from *snapshot*."""

    current = build_static_snapshot(repository=repository, source_repository=(
        snapshot.source_repository if isinstance(snapshot, StaticSnapshot) else snapshot["source_repository"]
    ))
    if isinstance(current, BlockedResult):
        return True
    try:
        validate_static_snapshot(snapshot)
    except StaticValidationError:
        return True
    return current.to_dict() != (snapshot.data if isinstance(snapshot, StaticSnapshot) else dict(snapshot))


__all__ = [
    "BlockedResult",
    "CANONICAL_STATIC_ITERATION_SOURCE",
    "canonical_static_iteration_source",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotStaleError",
    "StaticSnapshot",
    "StaticValidationError",
    "UnsafeSourceRepositoryError",
    "build_static_snapshot",
    "compute_static_snapshot",
    "recompute_snapshot_digests",
    "recompute_snapshot",
    "snapshot_is_stale",
    "sanitize_source_repository",
    "static_iteration_snapshot",
    "static_snapshot",
    "validate_static_snapshot",
]
