"""Pure release identity and publication consistency validators.

Gate A separates the immutable lifecycle event from the later GitHub Release
publication.  These functions validate the durable identities at each side of
that boundary and return explicit pending states while forward reconciliation
is still safe.  They do not call GitHub, mutate Git refs, or write evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import re
from typing import Any

try:  # Namespace packages work when this module is run from ``scripts``.
    from .certification import (
        PASS as CERTIFICATION_PASS,
        SUPPORTED_BINDINGS,
        is_safe_public_value,
        validate_certification_transfer,
        validate_certification_identity,
    )
    from .events import (
        EventError,
        validate_event,
    )
    from .versions import maintenance_line, parse_package_version, parse_public_tag
except ImportError:  # pragma: no cover - direct script import fallback
    from certification import (  # type: ignore
        PASS as CERTIFICATION_PASS,
        SUPPORTED_BINDINGS,
        is_safe_public_value,
        validate_certification_transfer,
        validate_certification_identity,
    )
    from events import (  # type: ignore
        EventError,
        validate_event,
    )
    from versions import maintenance_line, parse_package_version, parse_public_tag  # type: ignore


PASS = "PASS"
INVALID = "INVALID"
BLOCKED = "BLOCKED"
TERMINAL_CONSISTENT = "terminal_consistent"
TAG_RECONCILIATION_PENDING = "tag_reconciliation_pending"
PUBLICATION_RECONCILIATION_PENDING = "publication_reconciliation_pending"
LEGACY_EXCLUDED = "LEGACY_EXCLUDED"
LEGACY_PUBLIC_TAG = "v-0.1"

_EVENT_ID = re.compile(r"^EVT-[0-9]{12}$")
_FULL_REF = re.compile(r"^refs/(?:heads|tags|candidates)/[A-Za-z0-9._/-]+$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_HEX = re.compile(r"^[0-9a-f]{7,64}$")
PROJECT_VERSION_KEYS = frozenset(("closure", "candidate", "anchor", "final_release", "certified", "main"))


def _value(record: Mapping[str, Any] | None, *names: str) -> Any:
    if not isinstance(record, Mapping):
        return None
    for name in names:
        if name in record:
            return record[name]
    return None


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "\x00" not in value


def _result(
    status: str,
    reason_code: str | None = None,
    errors: list[str] | None = None,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if reason_code is not None:
        result["reason_code"] = reason_code
    if errors:
        result["errors"] = list(errors)
    result.update(details)
    return result


def _missing(errors: list[str], record: Mapping[str, Any], name: str, *aliases: str) -> Any:
    value = _value(record, name, *aliases)
    if value is None:
        errors.append(f"missing_{name}")
    return value


def _valid_version(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return parse_package_version(value).is_final
    except (TypeError, ValueError):
        return False


def _valid_event_id(value: Any) -> bool:
    return isinstance(value, str) and _EVENT_ID.fullmatch(value) is not None and int(value[4:]) > 0


def _valid_utc(value: Any) -> bool:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def is_canonical_public_tag(value: Any) -> bool:
    """Return whether ``value`` is a prospective canonical public tag."""

    if not isinstance(value, str) or value == LEGACY_PUBLIC_TAG:
        return False
    try:
        parse_public_tag(value)
    except (TypeError, ValueError):
        return False
    return True


def tag_version(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_public_tag(value).canonical
    except (TypeError, ValueError):
        return None


def _safe_public_value(value: Any) -> bool:
    """Compatibility wrapper for the shared certification boundary gate."""

    return is_safe_public_value(value)


def _validate_sha(value: Any) -> bool:
    # Gate A records Git's full SHA-1 identities.  Abbreviated provider output
    # cannot prove exact-tree equality and is therefore rejected.
    return isinstance(value, str) and len(value) == 40 and _HEX.fullmatch(value) is not None


def _validate_ref(value: Any, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and _FULL_REF.fullmatch(value) is not None
        and value.startswith(prefix)
        and ".." not in value
        and "//" not in value
        and "@{" not in value
    )


def _validate_ref_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_text(item) for item in value)
        and len(set(value)) == len(value)
    )


def _validate_released_event_shape(event: Mapping[str, Any]) -> list[str]:
    """Run the canonical event schema check before release identity checks."""

    try:
        validate_event(event)
    except EventError as error:
        message = str(error)
        if message.startswith("schema_version"):
            return ["schema_version_must_be_one"]
        if message.startswith("undeclared event fields:"):
            return ["unknown_event_fields:" + message.split(": ", 1)[1]]
        return [message]
    except (TypeError, ValueError) as error:
        return [f"event_schema_error:{error}"]
    return []


def _normalise_certification_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map event-schema certification aliases to the cert validator contract."""

    normalized = dict(record)
    aliases = {
        "binding": ("certification_binding",),
        "package_commit": ("package_sha", "subject_sha", "certification_subject_sha"),
        "package_tree": ("subject_tree", "certified_tree", "certification_subject_tree"),
        "policy_revision": ("policy_sha", "certification_policy_revision"),
        "harness_revision": ("harness_sha", "certification_harness_revision"),
        "environment": ("environment_id", "environment_ref", "certification_environment"),
    }
    for canonical, alternate_names in aliases.items():
        present = [
            record[name] for name in (canonical, *alternate_names) if name in record
        ]
        if present:
            if canonical == "binding":
                equivalent = {"tree_bound": "tree-bound", "commit_bound": "commit-bound"}
                compared = [
                    equivalent.get(value, value) if isinstance(value, str) else value
                    for value in present
                ]
            else:
                compared = present
            if any(value != compared[0] for value in compared[1:]):
                raise ValueError(f"conflicting certification aliases for {canonical}")
            normalized[canonical] = present[0]
    evidence_values = tuple(
        record[name]
        for name in ("evidence", "evidence_ref", "evidence_digest", "evidence_sha")
        if name in record
    )
    if (
        "evidence_refs" in record
        and "certification_evidence_refs" in record
        and record["evidence_refs"] != record["certification_evidence_refs"]
    ):
        raise ValueError("conflicting certification evidence reference aliases")
    refs = record.get("evidence_refs", record.get("certification_evidence_refs"))
    if isinstance(refs, (list, tuple)):
        normalized["evidence_refs"] = list(refs)
        if "evidence" not in normalized and refs:
            normalized["evidence_ref"] = refs[0]
    elif evidence_values and "evidence" not in normalized:
        normalized["evidence"] = evidence_values[0]
    transfer_values = tuple(
        record[name]
        for name in (
            "transfer_evidence",
            "transfer_proof",
            "certification_transfer_evidence",
            "certification_transfer",
        )
        if name in record
    )
    if transfer_values:
        if any(value != transfer_values[0] for value in transfer_values[1:]):
            raise ValueError("conflicting certification transfer evidence aliases")
        normalized["transfer_evidence"] = transfer_values[0]
    return normalized


def _certification_ref_list(record: Mapping[str, Any]) -> list[Any] | None:
    refs = record.get("evidence_refs", record.get("certification_evidence_refs"))
    if isinstance(refs, (list, tuple)):
        return list(refs)
    # The released-event field is a list of evidence references.  A digest is
    # an evidence identity, but it is not an additional reference when a
    # record already carries its durable path.  Preserve the validator's
    # precedence (evidence, ref, digest, SHA) while projecting one fallback
    # reference for the event comparison.
    for name in ("evidence", "evidence_ref", "evidence_sha", "evidence_digest"):
        if name in record:
            return [record[name]]
    return None


def _transfer_evidence_projection(value: Any) -> dict[str, Any] | None:
    """Project transfer proof aliases to the canonical durable identity."""

    if not isinstance(value, Mapping) or value.get("verified") is not True:
        return None
    result: dict[str, Any] = {"verified": True}
    for field in (
        "candidate_sha",
        "final_release_sha",
        "candidate_tree",
        "final_release_tree",
        "anchor_tree",
    ):
        item = value.get(field)
        if not _validate_sha(item):
            return None
        result[field] = item
    if "certified_tree" in value:
        if not _validate_sha(value["certified_tree"]):
            return None
        result["certified_tree"] = value["certified_tree"]
    evidence = [
        (name, value[name])
        for name in ("evidence_ref", "evidence_digest", "digest")
        if name in value
    ]
    if len(evidence) != 1 or not _text(evidence[0][1]):
        return None
    result["evidence"] = evidence[0][1]
    return result


def _validate_event_transfer_evidence(
    proof: Any,
    *,
    candidate_sha: Any,
    final_sha: Any,
    subject_tree: Any,
    candidate_tree: Any,
    final_tree: Any,
    anchor_tree: Any,
) -> bool:
    projected = _transfer_evidence_projection(proof)
    if projected is None:
        return False
    if not all(
        projected[field] == expected
        for field, expected in (
            ("candidate_sha", candidate_sha),
            ("final_release_sha", final_sha),
            ("candidate_tree", candidate_tree),
            ("final_release_tree", final_tree),
            ("anchor_tree", anchor_tree),
        )
    ):
        return False
    return projected.get("certified_tree", subject_tree) == subject_tree


def _project_version_evidence(
    project_versions: Mapping[str, str] | None,
    *,
    final_version: str,
    line: str,
    main_at_event_version: str | None,
) -> dict[str, Any] | None:
    """Validate the six prospective Project.toml observations.

    ``main`` is the promoted principal tree for a principal release and the
    contemporaneous unchanged principal tree for a maintenance release.  The
    other five observations must carry the released package version.
    """

    if not isinstance(project_versions, Mapping):
        return _result(BLOCKED, "PROJECT_VERSION_EVIDENCE_UNAVAILABLE")
    aliases = {"certified_anchor": "certified", "release": "final_release"}
    normalized: dict[str, Any] = {}
    for raw_key, value in project_versions.items():
        key = aliases.get(raw_key, raw_key) if isinstance(raw_key, str) else raw_key
        if key not in PROJECT_VERSION_KEYS or key in normalized:
            return _result(BLOCKED, "PROJECT_VERSION_EVIDENCE_INVALID", [str(raw_key)])
        normalized[key] = value
    if set(normalized) != PROJECT_VERSION_KEYS:
        return _result(
            BLOCKED,
            "PROJECT_VERSION_EVIDENCE_INCOMPLETE",
            sorted(PROJECT_VERSION_KEYS - set(normalized)),
        )
    for key in PROJECT_VERSION_KEYS:
        if not _valid_version(normalized[key]):
            return _result(BLOCKED, "PROJECT_VERSION_EVIDENCE_INVALID", [key])
    for key in ("closure", "candidate", "anchor", "final_release", "certified"):
        if normalized[key] != final_version:
            return _result(INVALID, "PROJECT_VERSION_MISMATCH", [key])
    expected_main = final_version if line == "principal" else main_at_event_version
    if expected_main is None or normalized["main"] != expected_main:
        return _result(INVALID, "PROJECT_MAIN_VERSION_MISMATCH")
    return None


def validate_publication_evidence(
    evidence: Mapping[str, Any],
    *,
    event: Mapping[str, Any] | None = None,
    tag: Mapping[str, Any] | str | None = None,
    github_release: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable artifact recorded after GitHub Release publish."""

    if not isinstance(evidence, Mapping):
        return _result(INVALID, "PUBLICATION_EVIDENCE_INVALID")
    if not _safe_public_value(evidence):
        return _result(INVALID, "UNSAFE_PUBLIC_EVIDENCE")
    errors: list[str] = []
    event_id = _missing(errors, evidence, "event_id")
    public_tag = _missing(errors, evidence, "public_tag", "tag_name")
    release_id = _missing(errors, evidence, "github_release_id", "release_id")
    published_at = _missing(errors, evidence, "published_at", "publication_timestamp_utc")
    artifact_ref = _missing(errors, evidence, "evidence_ref", "artifact_ref")
    artifact_digest = _missing(errors, evidence, "evidence_digest", "artifact_digest")

    if event_id is not None and not _valid_event_id(event_id):
        errors.append("invalid_event_id")
    if public_tag is not None and not is_canonical_public_tag(public_tag):
        errors.append("invalid_public_tag")
    if release_id is not None and not (
        not isinstance(release_id, bool)
        and isinstance(release_id, (str, int))
        and str(release_id) != ""
    ):
        errors.append("invalid_github_release_id")
    if published_at is not None and not _valid_utc(published_at):
        errors.append("invalid_publication_timestamp")
    if artifact_ref is not None and not _text(artifact_ref):
        errors.append("invalid_evidence_ref")
    if artifact_digest is not None and not (
        isinstance(artifact_digest, str) and re.fullmatch(r"[0-9a-f]{64}", artifact_digest)
    ):
        errors.append("invalid_evidence_digest")

    tag_name = _value(tag, "name", "tag_name") if isinstance(tag, Mapping) else tag
    if tag_name is not None and public_tag != tag_name:
        errors.append("publication_tag_mismatch")
    event_event_id = _value(event, "event_id")
    event_tag = _value(event, "public_tag", "tag_name")
    if event_event_id is not None and event_id != event_event_id:
        errors.append("publication_event_mismatch")
    if event_tag is not None and public_tag != event_tag:
        errors.append("publication_event_tag_mismatch")
    release_tag = _value(github_release, "tag_name", "public_tag")
    release_id_from_api = _value(github_release, "id", "github_release_id", "release_id")
    release_published_at = _value(github_release, "published_at", "publication_timestamp_utc")
    if release_tag is not None and public_tag != release_tag:
        errors.append("github_release_tag_mismatch")
    if release_id_from_api is not None and str(release_id) != str(release_id_from_api):
        errors.append("github_release_id_mismatch")
    if release_published_at is not None and published_at != release_published_at:
        errors.append("github_release_timestamp_mismatch")
    if event is not None and _valid_utc(published_at) and _valid_utc(_value(event, "timestamp_utc")):
        event_time = datetime.strptime(_value(event, "timestamp_utc"), "%Y-%m-%dT%H:%M:%SZ")
        publication_time = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
        if publication_time < event_time:
            errors.append("publication_precedes_released_event")

    if errors:
        return _result(INVALID, "PUBLICATION_EVIDENCE_INVALID", errors)
    return _result(
        PASS,
        event_id=event_id,
        public_tag=public_tag,
        github_release_id=str(release_id),
        published_at=published_at,
        evidence_ref=artifact_ref,
        evidence_digest=artifact_digest,
    )


def validate_released_event(
    event: Mapping[str, Any],
    *,
    certification: Mapping[str, Any] | None = None,
    project_versions: Mapping[str, str] | None = None,
    principal_main: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the complete line-specific identity in a ``released`` event."""

    if not isinstance(event, Mapping):
        return _result(INVALID, "RELEASED_EVENT_INVALID")
    if not _safe_public_value(event):
        return _result(INVALID, "UNSAFE_PUBLIC_EVIDENCE")
    errors: list[str] = []
    event_id = _missing(errors, event, "event_id")
    event_type = _missing(errors, event, "event_type")
    timestamp = _missing(errors, event, "timestamp_utc")
    final_version = _missing(errors, event, "final_version")
    line = _missing(errors, event, "release_line")
    anchor_ref = _missing(errors, event, "anchor_ref")
    anchor_sha = _missing(errors, event, "anchor_sha")
    anchor_tree = _missing(errors, event, "anchor_tree")
    candidate_ref = _missing(errors, event, "candidate_ref")
    candidate_sha = _missing(errors, event, "candidate_sha")
    candidate_tree = _missing(errors, event, "candidate_tree")
    final_sha = _missing(errors, event, "final_release_sha")
    final_tree = _missing(errors, event, "final_release_tree")
    binding = _missing(errors, event, "certification_binding")
    subject_sha = _missing(errors, event, "certification_subject_sha")
    subject_tree = _missing(errors, event, "certification_subject_tree")
    policy_revision = _missing(errors, event, "certification_policy_revision")
    harness_revision = _missing(errors, event, "certification_harness_revision")
    environment = _missing(errors, event, "certification_environment")
    certification_refs = _missing(errors, event, "certification_evidence_refs")
    closure_time = _missing(errors, event, "closure_timestamp_utc")
    public_tag = _missing(errors, event, "public_tag")
    evidence_refs = _missing(errors, event, "evidence_refs")
    event_transfer_evidence = event.get("certification_transfer_evidence")

    if errors:
        return _result(INVALID, "RELEASED_EVENT_INVALID", errors)
    errors.extend(_validate_released_event_shape(event))
    if event_type != "released":
        errors.append("event_type_must_be_released")
    if not _valid_event_id(event_id):
        errors.append("invalid_event_id")
    if not _valid_utc(timestamp) or not _valid_utc(closure_time):
        errors.append("invalid_utc_timestamp")
    try:
        maintenance_parts = maintenance_line(line)
    except (TypeError, ValueError):
        maintenance_parts = None
    if not _valid_version(final_version) or not is_canonical_public_tag(public_tag):
        errors.append("invalid_final_version_or_tag")
    if is_canonical_public_tag(public_tag) and tag_version(public_tag) != final_version:
        errors.append("tag_version_mismatch")
    if maintenance_parts is not None and _valid_version(final_version):
        if tuple(parse_package_version(final_version).tuple[:2]) != maintenance_parts:
            errors.append("maintenance_line_version_mismatch")
    if line != "principal" and maintenance_parts is None:
        return _result(INVALID, "INVALID_RELEASE_LINE")
    if binding not in SUPPORTED_BINDINGS:
        return _result(BLOCKED, "UNSUPPORTED_CERTIFICATION_BINDING")
    for name, value in (
        ("anchor_ref", anchor_ref),
        ("anchor_sha", anchor_sha),
        ("anchor_tree", anchor_tree),
        ("candidate_ref", candidate_ref),
        ("candidate_sha", candidate_sha),
        ("candidate_tree", candidate_tree),
        ("final_release_sha", final_sha),
        ("final_release_tree", final_tree),
        ("certification_subject_sha", subject_sha),
        ("certification_subject_tree", subject_tree),
        ("certification_policy_revision", policy_revision),
        ("certification_harness_revision", harness_revision),
        ("certification_environment", environment),
    ):
        if name in {"anchor_sha", "anchor_tree", "candidate_sha", "candidate_tree", "final_release_sha", "final_release_tree", "certification_subject_sha", "certification_subject_tree"}:
            valid = _validate_sha(value)
        elif name == "anchor_ref":
            valid = _validate_ref(value, "refs/tags/iterations/")
        elif name == "candidate_ref":
            valid = _validate_ref(value, "refs/heads/candidates/")
        else:
            valid = _text(value)
        if not valid:
            errors.append(f"invalid_{name}")
    if not _validate_ref_list(certification_refs):
        errors.append("invalid_certification_evidence_refs")
    if not _validate_ref_list(evidence_refs):
        errors.append("invalid_evidence_refs")

    transfer_required = binding == "tree-bound" and candidate_sha != final_sha
    if transfer_required and event_transfer_evidence is None:
        errors.append("missing_certification_transfer_evidence")
    if event_transfer_evidence is not None:
        if binding != "tree-bound":
            errors.append("unexpected_certification_transfer_evidence")
        elif not _validate_event_transfer_evidence(
            event_transfer_evidence,
            candidate_sha=candidate_sha,
            final_sha=final_sha,
            subject_tree=subject_tree,
            candidate_tree=candidate_tree,
            final_tree=final_tree,
            anchor_tree=anchor_tree,
        ):
            errors.append("invalid_certification_transfer_evidence")

    if len({anchor_tree, candidate_tree, final_tree, subject_tree}) != 1:
        errors.append("release_tree_mismatch")
    if isinstance(line, str) and line == "principal" and anchor_ref != f"refs/tags/iterations/{final_version}":
        errors.append("anchor_ref_version_mismatch")
    if isinstance(line, str) and maintenance_parts is not None and anchor_ref != f"refs/tags/iterations/{final_version}":
        errors.append("anchor_ref_version_mismatch")
    if line == "principal":
        previous_sha = _missing(errors, event, "previous_main_sha")
        previous_version = _missing(errors, event, "previous_main_version")
        main_sha = _missing(errors, event, "main_at_event_sha")
        main_version = _missing(errors, event, "main_at_event_version")
        if not _validate_sha(previous_sha) or not _validate_sha(main_sha):
            errors.append("invalid_main_commit_identity")
        if not _valid_version(previous_version) or not _valid_version(main_version):
            errors.append("invalid_main_version")
        if _valid_version(previous_version) and _valid_version(final_version):
            if parse_package_version(previous_version).tuple >= parse_package_version(final_version).tuple:
                errors.append("principal_version_not_monotonic")
        if main_version != final_version:
            errors.append("principal_main_version_mismatch")
        if main_sha != final_sha:
            errors.append("principal_main_release_sha_mismatch")
    else:
        if any(name in event for name in ("previous_main_sha", "previous_main_version")):
            errors.append("maintenance_previous_main_forbidden")
        main_sha = _missing(errors, event, "main_at_event_sha")
        main_version = _missing(errors, event, "main_at_event_version")
        if not _validate_sha(main_sha) or not _valid_version(main_version):
            errors.append("invalid_main_at_event_identity")
        if isinstance(principal_main, Mapping):
            expected_sha = principal_main.get("sha")
            expected_version = principal_main.get("version")
            if not _validate_sha(expected_sha) or not _valid_version(expected_version):
                errors.append("invalid_principal_main_identity")
            elif main_sha != expected_sha:
                errors.append("maintenance_main_changed")
            if _valid_version(expected_version) and main_version != expected_version:
                errors.append("maintenance_main_version_changed")
        elif not errors:
            return _result(BLOCKED, "PRINCIPAL_MAIN_IDENTITY_UNAVAILABLE")

    if not errors:
        project_result = _project_version_evidence(
            project_versions,
            final_version=final_version,
            line=line,
            main_at_event_version=main_version,
        )
        if project_result is not None:
            return project_result

    if isinstance(certification, Mapping):
        try:
            normalized_certification = _normalise_certification_record(certification)
        except ValueError as error:
            return _result(INVALID, "CERTIFICATION_IDENTITY_AMBIGUOUS", [str(error)])
        cert_result = validate_certification_identity(normalized_certification)
        if cert_result["status"] != CERTIFICATION_PASS:
            return cert_result
        cert_identity = cert_result["identity"]
        if cert_identity["binding"] != binding:
            errors.append("certification_binding_mismatch")
        if cert_identity["package_commit"] != subject_sha:
            errors.append("certification_subject_commit_mismatch")
        if cert_identity["package_tree"] != subject_tree:
            errors.append("certification_subject_tree_mismatch")
        if cert_identity["policy_revision"] != policy_revision:
            errors.append("certification_policy_revision_mismatch")
        if cert_identity["harness_revision"] != harness_revision:
            errors.append("certification_harness_revision_mismatch")
        if cert_identity["environment"] != environment:
            errors.append("certification_environment_mismatch")
        certification_identity_refs = _certification_ref_list(normalized_certification)
        if not _validate_ref_list(certification_identity_refs):
            errors.append("certification_identity_refs_invalid")
        else:
            if cert_identity["evidence"] != certification_identity_refs[0]:
                errors.append("certification_evidence_identity_mismatch")
            if certification_identity_refs != certification_refs:
                errors.append("certification_evidence_refs_mismatch")
        certification_transfer = normalized_certification.get("transfer_evidence")
        if transfer_required:
            if not _validate_event_transfer_evidence(
                certification_transfer,
                candidate_sha=candidate_sha,
                final_sha=final_sha,
                subject_tree=subject_tree,
                candidate_tree=candidate_tree,
                final_tree=final_tree,
                anchor_tree=anchor_tree,
            ):
                errors.append("certification_transfer_evidence_missing_or_invalid")
            else:
                certification_projection = _transfer_evidence_projection(
                    certification_transfer
                )
                event_projection = _transfer_evidence_projection(
                    event_transfer_evidence
                )
                if certification_projection is None or event_projection is None:
                    errors.append("certification_transfer_evidence_mismatch")
                else:
                    mismatched_transfer_fields = (
                        "candidate_sha",
                        "final_release_sha",
                        "candidate_tree",
                        "final_release_tree",
                        "anchor_tree",
                    )
                    if any(
                        certification_projection[field]
                        != event_projection[field]
                        for field in mismatched_transfer_fields
                    ) or (
                        "certified_tree" in event_projection
                        and event_projection["certified_tree"]
                        != certification_projection.get("certified_tree", subject_tree)
                    ):
                        errors.append("certification_transfer_evidence_mismatch")
        if not errors:
            transfer_result = validate_certification_transfer(
                normalized_certification,
                candidate_commit=candidate_sha,
                candidate_tree=candidate_tree,
                final_release_commit=final_sha,
                final_release_tree=final_tree,
                anchor_tree=anchor_tree,
            )
            if transfer_result["status"] != CERTIFICATION_PASS:
                return transfer_result
    elif not errors:
        # R-024 requires pinned package, policy, harness, environment and
        # evidence identities even when the certified commit equals release.
        return _result(BLOCKED, "CERTIFICATION_EVIDENCE_UNAVAILABLE")

    if errors:
        return _result(INVALID, "RELEASED_EVENT_INVALID", errors)
    return _result(
        PASS,
        event_id=event_id,
        release_line=line,
        final_version=final_version,
        public_tag=public_tag,
        final_release_sha=final_sha,
        final_release_tree=final_tree,
    )


def _tag_identity(tag: Mapping[str, Any] | str | None) -> dict[str, Any] | None:
    if isinstance(tag, str):
        return {"name": tag}
    if isinstance(tag, Mapping):
        return {
            "name": _value(tag, "name", "tag_name", "public_tag"),
            "sha": _value(tag, "sha", "target_sha", "commit_sha", "target_commit"),
            "tree": _value(tag, "tree", "target_tree"),
            "version": _value(tag, "version"),
        }
    return None


def _release_identity(release: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(release, Mapping):
        return None
    return {
        "id": _value(release, "id", "github_release_id", "release_id"),
        "tag": _value(release, "tag_name", "public_tag"),
        "sha": _value(release, "target_sha", "target_commit", "commit_sha"),
        "tree": _value(release, "target_tree", "tree"),
        "published_at": _value(release, "published_at", "publication_timestamp_utc"),
    }


def _matches_intent(tag: Mapping[str, Any] | str, intent: Mapping[str, Any]) -> tuple[bool, list[str]]:
    tag_identity = _tag_identity(tag) or {}
    errors: list[str] = []
    tag_name = tag_identity.get("name")
    expected_tag = _value(intent, "public_tag", "tag_name", "proposed_tag")
    if tag_name != expected_tag:
        errors.append("intent_tag_mismatch")
    if not _validate_sha(tag_identity.get("sha")):
        errors.append("tag_commit_identity_missing")
    if not _validate_sha(tag_identity.get("tree")):
        errors.append("tag_tree_identity_missing")
    for actual, aliases, label in (
        (tag_identity.get("sha"), ("final_release_sha", "release_sha", "commit_sha"), "commit"),
        (tag_identity.get("tree"), ("final_release_tree", "release_tree"), "tree"),
    ):
        expected = _value(intent, *aliases)
        if expected is None:
            errors.append(f"intent_{label}_missing")
        elif actual != expected:
            errors.append(f"intent_{label}_mismatch")
    expected_version = _value(intent, "final_version", "version")
    if not _valid_version(expected_version):
        errors.append("intent_version_missing")
    elif tag_version(tag_name or "") != expected_version:
        errors.append("intent_version_mismatch")
    return not errors, errors


def _active_release_intent(intent: Mapping[str, Any] | None) -> bool:
    if not isinstance(intent, Mapping):
        return False
    event_type = intent.get("event_type")
    status = intent.get("status")
    return event_type in (None, "release_intent_prepared") and status not in {
        "aborted", "withdrawn", "release_intent_aborted"
    }


def validate_release_consistency(
    event: Mapping[str, Any] | None = None,
    tag: Mapping[str, Any] | str | None = None,
    github_release: Mapping[str, Any] | None = None,
    publication_evidence: Mapping[str, Any] | None = None,
    release_intent: Mapping[str, Any] | None = None,
    *,
    certification: Mapping[str, Any] | None = None,
    project_versions: Mapping[str, str] | None = None,
    principal_main: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check tag, released event, GitHub Release and publication evidence.

    A canonical tag plus matching intent without a released event is the
    explicit ``tag_reconciliation_pending`` state.  A valid released event
    whose later GitHub publication evidence is incomplete is
    ``publication_reconciliation_pending``.  Any identity disagreement is
    ``INVALID`` and never authorises a replacement tag.
    """

    for name, value in (
        ("released_event", event),
        ("tag", tag),
        ("github_release", github_release),
        ("publication_evidence", publication_evidence),
        ("release_intent", release_intent),
        ("certification", certification),
        ("project_versions", project_versions),
        ("principal_main", principal_main),
    ):
        if value is not None and not _safe_public_value(value):
            return _result(INVALID, "UNSAFE_PUBLIC_EVIDENCE", [name])

    tag_identity = _tag_identity(tag)
    release_identity = _release_identity(github_release)
    names = [
        tag_identity.get("name") if tag_identity else None,
        release_identity.get("tag") if release_identity else None,
        _value(event, "public_tag", "tag_name"),
        _value(release_intent, "public_tag", "tag_name", "proposed_tag"),
    ]
    legacy_names = [name for name in names if name == LEGACY_PUBLIC_TAG]
    nonlegacy_names = [name for name in names if name is not None and name != LEGACY_PUBLIC_TAG]
    if legacy_names and nonlegacy_names:
        return _result(INVALID, "MIXED_LEGACY_CANONICAL_IDENTITIES")
    if legacy_names:
        return _result(LEGACY_EXCLUDED, "LEGACY_PUBLIC_TAG_EXCLUDED", public_tag=LEGACY_PUBLIC_TAG)

    if event is None:
        if tag_identity is None:
            return _result(BLOCKED, "RELEASED_EVENT_UNAVAILABLE")
        if release_intent is None:
            return _result(INVALID, "TAG_WITHOUT_RELEASE_INTENT")
        if not _active_release_intent(release_intent):
            return _result(INVALID, "RELEASE_INTENT_NOT_ACTIVE")
        matches, errors = _matches_intent(tag, release_intent)
        if not matches:
            return _result(INVALID, "TAG_INTENT_MISMATCH", errors)
        if github_release is not None or publication_evidence is not None:
            return _result(INVALID, "PUBLICATION_WITHOUT_RELEASED_EVENT")
        return _result(TAG_RECONCILIATION_PENDING, intent_event_required=True)

    event_result = validate_released_event(
        event,
        certification=certification,
        project_versions=project_versions,
        principal_main=principal_main,
    )
    if event_result["status"] != PASS:
        return event_result
    if tag_identity is None:
        return _result(INVALID, "RELEASED_EVENT_TAG_MISSING")
    tag_name = tag_identity.get("name")
    if not is_canonical_public_tag(tag_name):
        return _result(INVALID, "INVALID_CANONICAL_PUBLIC_TAG")
    if not _validate_sha(tag_identity.get("sha")) or not _validate_sha(tag_identity.get("tree")):
        return _result(INVALID, "TAG_IDENTITY_INCOMPLETE")
    if tag_name != event_result["public_tag"]:
        return _result(INVALID, "TAG_EVENT_MISMATCH")
    if tag_identity.get("version") is None:
        return _result(INVALID, "TAG_VERSION_MISSING")
    if tag_identity["version"] != event_result["final_version"]:
        return _result(INVALID, "TAG_VERSION_MISMATCH")
    for key, expected in (("sha", event_result["final_release_sha"]), ("tree", event_result["final_release_tree"])):
        actual = tag_identity.get(key)
        if actual is not None and actual != expected:
            return _result(INVALID, f"TAG_{key.upper()}_MISMATCH")
    if not _active_release_intent(release_intent):
        return _result(INVALID, "RELEASE_INTENT_NOT_ACTIVE")
    intent_match, intent_errors = _matches_intent(tag, release_intent)
    if not intent_match:
        return _result(INVALID, "TAG_INTENT_MISMATCH", intent_errors)

    # A publication artifact may still be pending, but a GitHub Release that
    # already exists must match the event and tag before pending is reported.
    if release_identity is not None:
        if release_identity.get("id") is None or release_identity.get("published_at") is None:
            return _result(INVALID, "GITHUB_RELEASE_IDENTITY_INCOMPLETE")
        if release_identity.get("tag") != tag_name:
            return _result(INVALID, "GITHUB_RELEASE_TAG_MISMATCH")
        if not _validate_sha(release_identity.get("sha")):
            return _result(INVALID, "GITHUB_RELEASE_SHA_MISSING")
        if release_identity["sha"] != event_result["final_release_sha"]:
            return _result(INVALID, "GITHUB_RELEASE_SHA_MISMATCH")
        if not _validate_sha(release_identity.get("tree")):
            return _result(INVALID, "GITHUB_RELEASE_TREE_MISSING")
        if release_identity["tree"] != event_result["final_release_tree"]:
            return _result(INVALID, "GITHUB_RELEASE_TREE_MISMATCH")
    if release_identity is None or publication_evidence is None:
        return _result(
            PUBLICATION_RECONCILIATION_PENDING,
            event_id=event_result["event_id"],
            public_tag=tag_name,
        )
    publication_result = validate_publication_evidence(
        publication_evidence,
        event=event,
        tag=tag_identity,
        github_release=github_release,
    )
    if publication_result["status"] != PASS:
        return publication_result
    return _result(
        TERMINAL_CONSISTENT,
        event_id=event_result["event_id"],
        public_tag=tag_name,
        final_version=event_result["final_version"],
        final_release_sha=event_result["final_release_sha"],
        final_release_tree=event_result["final_release_tree"],
        github_release_id=publication_result["github_release_id"],
    )


def _records(value: Any) -> list[Any] | None:
    """Normalize a collection or one record without accepting strings."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        # A single lifecycle record has an identity key.  Otherwise accept a
        # mapping keyed by an external identifier and validate its values.
        identity_keys = {
            "event_id", "event_type", "name", "tag_name", "public_tag",
            "id", "github_release_id", "release_id", "evidence_ref",
        }
        if identity_keys.intersection(value):
            return [value]
        return list(value.values())
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        return None
    return list(value)


def _unique_index(
    records: list[Any],
    key: Any,
    reason: str,
) -> tuple[dict[Any, Mapping[str, Any]] | None, dict[str, Any] | None]:
    index: dict[Any, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return None, _result(INVALID, reason, ["record_is_not_an_object"])
        identity = key(record)
        if identity is None:
            return None, _result(INVALID, reason, ["record_identity_missing"])
        try:
            duplicate = identity in index
        except TypeError:
            return None, _result(INVALID, reason, ["record_identity_unhashable"])
        if duplicate:
            return None, _result(INVALID, reason, [f"duplicate:{identity}"])
        try:
            index[identity] = record
        except TypeError:
            return None, _result(INVALID, reason, ["record_identity_unhashable"])
    return index, None


def _event_auxiliary(value: Any, event_id: str) -> Mapping[str, Any] | None:
    """Resolve an event-keyed auxiliary record or a single record."""

    if isinstance(value, Mapping):
        if event_id in value and isinstance(value[event_id], Mapping):
            return value[event_id]
        marker_keys = {
            "binding", "package_commit", "package_tree", "closure", "candidate",
            "anchor", "final_release", "certified", "main", "sha", "version",
        }
        if marker_keys.intersection(value):
            return value
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for record in value:
            if isinstance(record, Mapping) and record.get("event_id") == event_id:
                return record
        return None
    return None


def _legacy_partition(
    records: list[Any],
    name_keys: tuple[str, ...],
    reason: str,
) -> tuple[list[Any] | None, dict[str, Any] | None]:
    """Drop standalone grandfathered records and reject mixed identities."""

    retained: list[Any] = []
    for record in records:
        if isinstance(record, str):
            names = [record]
        elif isinstance(record, Mapping):
            names = [record[key] for key in name_keys if key in record]
        else:
            return None, _result(INVALID, reason, ["record_is_not_an_object"])
        names = [name for name in names if name is not None]
        try:
            distinct = set(names)
        except TypeError:
            return None, _result(INVALID, reason, ["record_name_unhashable"])
        if LEGACY_PUBLIC_TAG in distinct and any(name != LEGACY_PUBLIC_TAG for name in distinct):
            return None, _result(INVALID, "MIXED_LEGACY_CANONICAL_IDENTITIES", [reason])
        if distinct == {LEGACY_PUBLIC_TAG}:
            continue
        retained.append(record)
    return retained, None


def validate_release_catalog(
    events: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    tags: Iterable[Mapping[str, Any] | str] | Mapping[str, Any] | None,
    github_releases: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None,
    publication_evidence: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    release_intents: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    certifications: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    project_versions: Mapping[str, Any] | None = None,
    principal_main: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all canonical release identities in one complete catalog.

    The collection check establishes the cardinality contract around
    :func:`validate_release_consistency`: one canonical tag, one released
    event, one GitHub Release and one publication-evidence identity per
    release.  Missing event/tag/publication records return the explicit
    forward-recovery states.  Duplicate or cross-linked identities are
    invalid and cannot be repaired by creating a replacement tag.
    """

    event_records = _records(events)
    tag_records = _records(tags)
    release_records = _records(github_releases)
    evidence_records = _records(publication_evidence)
    intent_records = _records(release_intents)
    if any(records is None for records in (event_records, tag_records, release_records, evidence_records, intent_records)):
        return _result(INVALID, "RELEASE_CATALOG_INVALID", ["collection_is_not_iterable"])
    for name, value in (
        ("certifications", certifications),
        ("project_versions", project_versions),
        ("principal_main", principal_main),
    ):
        if value is not None and not _safe_public_value(value):
            return _result(INVALID, "UNSAFE_PUBLIC_EVIDENCE", [name])
    assert event_records is not None
    assert tag_records is not None
    assert release_records is not None
    assert evidence_records is not None
    assert intent_records is not None

    for name, records in (
        ("events", event_records),
        ("tags", tag_records),
        ("github_releases", release_records),
        ("publication_evidence", evidence_records),
        ("release_intents", intent_records),
    ):
        if any(not _safe_public_value(record) for record in records):
            return _result(INVALID, "UNSAFE_PUBLIC_EVIDENCE", [name])

    original_event_records = event_records
    original_tag_records = tag_records
    original_release_records = release_records
    original_evidence_records = evidence_records
    event_records, error = _legacy_partition(
        event_records, ("public_tag", "tag_name"), "INVALID_RELEASED_EVENT_RECORD"
    )
    if error is not None:
        return error
    tag_records, error = _legacy_partition(
        tag_records, ("name", "tag_name", "public_tag"), "INVALID_CANONICAL_TAG_RECORD"
    )
    if error is not None:
        return error
    release_records, error = _legacy_partition(
        release_records, ("tag_name", "public_tag"), "INVALID_GITHUB_RELEASE_RECORD"
    )
    if error is not None:
        return error
    evidence_records, error = _legacy_partition(
        evidence_records, ("public_tag", "tag_name"), "INVALID_PUBLICATION_EVIDENCE_RECORD"
    )
    if error is not None:
        return error
    intent_records, error = _legacy_partition(
        intent_records, ("public_tag", "tag_name", "proposed_tag"), "INVALID_RELEASE_INTENT_RECORD"
    )
    if error is not None:
        return error
    assert event_records is not None
    assert tag_records is not None
    assert release_records is not None
    assert evidence_records is not None
    assert intent_records is not None

    released_records = [
        record for record in event_records
        if isinstance(record, Mapping) and record.get("event_type") == "released"
    ]
    if not (released_records or tag_records or release_records or evidence_records):
        if any(
            len(original) > len(filtered)
            for original, filtered in (
                (original_event_records, event_records),
                (original_tag_records, tag_records),
                (original_release_records, release_records),
                (original_evidence_records, evidence_records),
            )
        ):
            return _result(LEGACY_EXCLUDED, "LEGACY_PUBLIC_TAG_EXCLUDED", public_tag=LEGACY_PUBLIC_TAG)
        return _result(BLOCKED, "RELEASE_CATALOG_EMPTY")

    event_by_id, error = _unique_index(
        released_records,
        lambda record: record.get("event_id") if _valid_event_id(record.get("event_id")) else None,
        "DUPLICATE_RELEASED_EVENT_ID",
    )
    if error is not None:
        return error
    assert event_by_id is not None
    event_by_tag, error = _unique_index(
        released_records,
        lambda record: record.get("public_tag") if is_canonical_public_tag(record.get("public_tag")) else None,
        "DUPLICATE_RELEASED_EVENT_TAG",
    )
    if error is not None:
        return error
    assert event_by_tag is not None

    canonical_tags: list[Mapping[str, Any]] = []
    for record in tag_records:
        if isinstance(record, str):
            name = record
            record = {"name": record}
        elif isinstance(record, Mapping):
            name = _value(record, "name", "tag_name", "public_tag")
        else:
            return _result(INVALID, "INVALID_CANONICAL_TAG_RECORD")
        if not is_canonical_public_tag(name):
            return _result(INVALID, "INVALID_CANONICAL_TAG_RECORD")
        canonical_tags.append(record)
    tag_by_name, error = _unique_index(
        canonical_tags,
        lambda record: _value(record, "name", "tag_name", "public_tag"),
        "DUPLICATE_CANONICAL_TAG",
    )
    if error is not None:
        return error
    assert tag_by_name is not None

    release_by_id, error = _unique_index(
        release_records,
        lambda record: str(_value(record, "id", "github_release_id", "release_id"))
        if _value(record, "id", "github_release_id", "release_id") is not None else None,
        "DUPLICATE_GITHUB_RELEASE_ID",
    )
    if error is not None:
        return error
    assert release_by_id is not None
    release_by_tag, error = _unique_index(
        release_records,
        lambda record: _value(record, "tag_name", "public_tag"),
        "DUPLICATE_GITHUB_RELEASE_TAG",
    )
    if error is not None:
        return error
    assert release_by_tag is not None

    evidence_by_event, error = _unique_index(
        evidence_records,
        lambda record: record.get("event_id") if _valid_event_id(record.get("event_id")) else None,
        "DUPLICATE_PUBLICATION_EVENT_ID",
    )
    if error is not None:
        return error
    assert evidence_by_event is not None
    evidence_by_tag, error = _unique_index(
        evidence_records,
        lambda record: _value(record, "public_tag", "tag_name"),
        "DUPLICATE_PUBLICATION_TAG",
    )
    if error is not None:
        return error
    assert evidence_by_tag is not None

    intent_by_tag, error = _unique_index(
        intent_records,
        lambda record: _value(record, "public_tag", "tag_name", "proposed_tag"),
        "DUPLICATE_RELEASE_INTENT_TAG",
    )
    if error is not None:
        return error
    assert intent_by_tag is not None

    tag_names = set(tag_by_name)
    event_names = set(event_by_tag)
    release_names = set(release_by_tag)
    evidence_names = set(evidence_by_tag)
    if not event_names <= tag_names:
        return _result(INVALID, "RELEASED_EVENT_TAG_MISSING", sorted(event_names - tag_names))
    if not release_names <= tag_names:
        return _result(INVALID, "GITHUB_RELEASE_TAG_MISSING", sorted(release_names - tag_names))
    if not evidence_names <= tag_names or not set(evidence_by_event) <= set(event_by_id):
        return _result(INVALID, "PUBLICATION_EVIDENCE_ORPHANED")
    if evidence_names - release_names:
        return _result(INVALID, "PUBLICATION_EVIDENCE_RELEASE_MISSING", sorted(evidence_names - release_names))
    if release_names - event_names:
        return _result(INVALID, "GITHUB_RELEASE_EVENT_MISSING", sorted(release_names - event_names))

    outcomes: list[dict[str, Any]] = []
    for tag_name in sorted(tag_names):
        tag_record = tag_by_name[tag_name]
        event_record = event_by_tag.get(tag_name)
        intent_record = intent_by_tag.get(tag_name)
        if event_record is None:
            outcome = validate_release_consistency(
                None,
                tag_record,
                release_by_tag.get(tag_name),
                evidence_by_tag.get(tag_name),
                intent_record,
            )
        else:
            event_id = str(event_record["event_id"])
            outcome = validate_release_consistency(
                event_record,
                tag_record,
                release_by_tag.get(tag_name),
                evidence_by_event.get(event_id),
                intent_record,
                certification=_event_auxiliary(certifications, event_id),
                project_versions=_event_auxiliary(project_versions, event_id),
                principal_main=_event_auxiliary(principal_main, event_id),
            )
        outcomes.append(outcome)
        if outcome.get("status") in {INVALID, BLOCKED}:
            return _result(
                outcome["status"],
                outcome.get("reason_code", "RELEASE_CATALOG_INVALID"),
                outcome.get("errors"),
                outcomes=outcomes,
            )

    statuses = {outcome.get("status") for outcome in outcomes}
    if TAG_RECONCILIATION_PENDING in statuses:
        status = TAG_RECONCILIATION_PENDING
    elif PUBLICATION_RECONCILIATION_PENDING in statuses:
        status = PUBLICATION_RECONCILIATION_PENDING
    else:
        status = TERMINAL_CONSISTENT
    return _result(status, outcomes=outcomes, release_count=len(tag_names))


def validate_bidirectional_consistency(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Descriptive alias for callers using the R-043 terminology."""

    return validate_release_consistency(*args, **kwargs)


def validate_release_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the lifecycle release evidence validator."""

    return validate_release_consistency(*args, **kwargs)


__all__ = [
    "BLOCKED",
    "INVALID",
    "LEGACY_EXCLUDED",
    "LEGACY_PUBLIC_TAG",
    "PASS",
    "PROJECT_VERSION_KEYS",
    "PUBLICATION_RECONCILIATION_PENDING",
    "TAG_RECONCILIATION_PENDING",
    "TERMINAL_CONSISTENT",
    "is_canonical_public_tag",
    "tag_version",
    "validate_bidirectional_consistency",
    "validate_publication_evidence",
    "validate_release_consistency",
    "validate_release_catalog",
    "validate_release_evidence",
    "validate_released_event",
]
