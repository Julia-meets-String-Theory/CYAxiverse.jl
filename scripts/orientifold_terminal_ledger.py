"""Stream and validate the inherited-orientifold terminal ledger.

Keep one machine-readable terminal row for each matrix validation attempt and
each enumerated candidate.  Store rows as JSONL so a reproduction summary can
retain class-level accounting without embedding the candidate payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LEDGER_SCHEMA_VERSION = "cyaxiverse-orientifold-terminal-ledger-1.0"
LEDGER_TERMINAL_STATUSES = (
    "matrix_validation_passed",
    "numerical_geometry_failure",
    "polytope_not_preserved",
    "frst_not_preserved",
    "prime_divisor_set_not_preserved",
    "nonintegral_h2_action",
    "h2_action_not_involution",
    "torus_shift_not_involution",
    "orientifold_h11_minus_filter_rejection",
    "torus_shift_search_exhausted",
    "fixed_point_set_non_smooth",
    "smoothness_verification_unavailable",
    "accepted_verified_orientifold",
)


class TerminalLedgerError(ValueError):
    """Raise when a terminal-ledger record or artifact is invalid."""


def _jsonable(value: Any) -> Any:
    """Convert common numerical containers to JSON-compatible values."""
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _required_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate one terminal row."""
    row = _jsonable(dict(record))
    row.setdefault("ledger_schema_version", LEDGER_SCHEMA_VERSION)
    row.setdefault("record_kind", "candidate")
    row.setdefault("candidate_id", None)
    row.setdefault("lambda_f", None)
    row.setdefault("torus_shift", None)
    row.setdefault("accepted_witness", None)
    if "terminal_reason_code" not in row:
        row["terminal_reason_code"] = None
    required = (
        "polytope_id",
        "frst_hash",
        "matrix_id",
        "candidate_id",
        "lambda_f",
        "torus_shift",
        "terminal_status",
        "terminal_reason_code",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise TerminalLedgerError(
            "terminal row is missing required fields: " + ", ".join(missing)
        )
    parity = row.get("h11_parity")
    if parity is None:
        if "h11_plus" not in row or "h11_minus" not in row:
            raise TerminalLedgerError(
                "terminal row is missing required fields: h11_parity"
            )
        parity = {
            "h11_plus": row.pop("h11_plus"),
            "h11_minus": row.pop("h11_minus"),
        }
    else:
        row.pop("h11_plus", None)
        row.pop("h11_minus", None)
    if not isinstance(parity, dict):
        raise TerminalLedgerError("h11_parity must be an object")
    if "h11_plus" in parity or "h11_minus" in parity:
        if parity.get("h11_plus") is None or parity.get("h11_minus") is None:
            raise TerminalLedgerError(
                "h11_parity must not contain null h11_plus or h11_minus"
            )
    elif not parity.get("status") or not parity.get("reason"):
        raise TerminalLedgerError(
            "h11_parity requires non-null h11_plus/h11_minus or an explicit "
            "status and reason"
        )
    row["h11_parity"] = parity

    fixed_evidence = row.get("fixed_component_evidence")
    if fixed_evidence is None:
        legacy_fields = (
            "fixed_point_components",
            "fixed_point_set",
            "smoothness",
        )
        if not all(field in row for field in legacy_fields):
            raise TerminalLedgerError(
                "terminal row is missing required fields: fixed_component_evidence"
            )
        fixed_evidence = {
            "fixed_point_components": row.pop("fixed_point_components"),
            "fixed_point_set": row.pop("fixed_point_set"),
            "smoothness": row.pop("smoothness"),
            "fixed_surface_n_s_evidence": row.pop(
                "fixed_surface_n_s_evidence", None
            ),
        }
    else:
        row.pop("fixed_point_components", None)
        row.pop("fixed_point_set", None)
        row.pop("smoothness", None)
        row.pop("fixed_surface_n_s_evidence", None)
    if not isinstance(fixed_evidence, dict):
        raise TerminalLedgerError("fixed_component_evidence must be an object")
    if {
        "fixed_point_components",
        "fixed_point_set",
        "smoothness",
    }.issubset(fixed_evidence):
        if any(
            fixed_evidence.get(field) is None
            for field in ("fixed_point_components", "fixed_point_set", "smoothness")
        ):
            raise TerminalLedgerError(
                "fixed_component_evidence must not contain null required evidence"
            )
    elif not fixed_evidence.get("status") or not fixed_evidence.get("reason"):
        raise TerminalLedgerError(
            "fixed_component_evidence requires non-null component, set, and "
            "smoothness evidence or an explicit status and reason"
        )
    row["fixed_component_evidence"] = fixed_evidence
    if row["terminal_status"] not in LEDGER_TERMINAL_STATUSES:
        raise TerminalLedgerError(
            f"unsupported terminal status {row['terminal_status']!r}"
        )
    if row["record_kind"] == "candidate" and row["candidate_id"] is None:
        raise TerminalLedgerError("candidate rows require candidate_id")
    if row["terminal_reason_code"] is None:
        row["terminal_reason_code"] = str(row["terminal_status"])
    if row["terminal_status"] == "accepted_verified_orientifold":
        if row["lambda_f"] not in (0, 1):
            raise TerminalLedgerError(
                "accepted_verified_orientifold requires lambda_f=0 or lambda_f=1"
            )
        if row["lambda_f"] == 0:
            return row
        row["accepted_witness"] = {
            "candidate_id": row["candidate_id"],
            "matrix_id": row["matrix_id"],
            "lambda_f": row["lambda_f"],
            "torus_shift": row["torus_shift"],
            "fixed_component_evidence": row["fixed_component_evidence"],
        }
    return row


def _stable_class_key(row: dict[str, Any]) -> tuple[Any, Any]:
    """Return the explicit polytope and FRST class identity."""
    if "polytope_index" not in row or "frst_class_index" not in row:
        raise TerminalLedgerError(
            "terminal row requires polytope_index and frst_class_index"
        )
    return int(row["polytope_index"]), int(row["frst_class_index"])


def _digest_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_provenance(provenance: dict[str, Any]) -> None:
    """Require a clean commit or cryptographic identity for dirty sources."""
    if provenance.get("git_dirty") is True:
        identity = provenance.get("working_tree_identity")
        if not isinstance(identity, dict) or not (
            identity.get("diff_sha256") or identity.get("tree_sha256")
        ):
            raise TerminalLedgerError(
                "dirty source provenance requires diff_sha256 or tree_sha256"
            )
    if not provenance.get("source_commit"):
        raise TerminalLedgerError("source_commit is required")
    if "runtime_versions" not in provenance:
        raise TerminalLedgerError("runtime_versions are required")
    manifest = provenance.get("input_partition_manifest")
    if not isinstance(manifest, dict) or not manifest.get("status"):
        raise TerminalLedgerError("input_partition_manifest is required")


class TerminalLedgerWriter:
    """Write terminal rows to a bounded-memory JSONL sidecar."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        provenance: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validate_source_provenance(provenance)
        self.path = Path(path).expanduser().resolve()
        self.summary_path = Path(f"{self.path}.summary.json")
        if self.path.exists() or self.summary_path.exists():
            raise FileExistsError(
                f"refusing to overwrite terminal ledger artifact: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self.path.with_name(
            f".{self.path.name}.tmp-{os.getpid()}"
        )
        if self._temporary_path.exists():
            raise FileExistsError(f"temporary ledger path already exists: {self._temporary_path}")
        self._stream = self._temporary_path.open("x", encoding="utf-8")
        self._provenance = _jsonable(provenance)
        self._metadata = _jsonable(metadata or {})
        self._status_counts: Counter[str] = Counter()
        self._record_kind_counts: Counter[str] = Counter()
        self._class_records: dict[tuple[int, int], dict[str, Any]] = {}
        self._record_count = 0
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        """Validate and append one terminal row without retaining its payload."""
        if self._closed:
            raise TerminalLedgerError("cannot write to a closed terminal ledger")
        row = _required_record(record)
        class_key = _stable_class_key(row)
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.write(encoded)
        self._record_count += 1
        self._status_counts[row["terminal_status"]] += 1
        self._record_kind_counts[row["record_kind"]] += 1
        class_record = self._class_records.setdefault(
            class_key,
            {
                "polytope_index": class_key[0],
                "frst_class_index": class_key[1],
                "polytope_id": row["polytope_id"],
                "frst_hash": row["frst_hash"],
                "matrix_attempt_count": 0,
                "candidate_attempt_count": 0,
                "status_counts": Counter(),
                "accepted_witness": None,
            },
        )
        if row["record_kind"] == "candidate":
            class_record["candidate_attempt_count"] += 1
        elif row["record_kind"] == "matrix_validation":
            class_record["matrix_attempt_count"] += 1
        class_record["status_counts"][row["terminal_status"]] += 1
        if row.get("accepted_witness") is not None and class_record["accepted_witness"] is None:
            class_record["accepted_witness"] = row["accepted_witness"]

    def close(self) -> dict[str, Any]:
        """Atomically publish the JSONL sidecar and its aggregate summary."""
        if self._closed:
            raise TerminalLedgerError("terminal ledger is already closed")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._link_without_overwrite(self._temporary_path, self.path)
        class_funnel = []
        for key in sorted(self._class_records):
            record = dict(self._class_records[key])
            record["status_counts"] = dict(sorted(record["status_counts"].items()))
            record["accepted_for_table_1"] = record["accepted_witness"] is not None
            class_funnel.append(record)
        summary = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "provenance": self._provenance,
            "metadata": self._metadata,
            "sidecar_path": str(self.path),
            "sidecar_sha256": _digest_file(self.path),
            "record_count": self._record_count,
            "record_kind_counts": dict(sorted(self._record_kind_counts.items())),
            "terminal_status_counts": {
                status: self._status_counts.get(status, 0)
                for status in LEDGER_TERMINAL_STATUSES
            },
            "class_count": len(class_funnel),
            "class_funnel": class_funnel,
            "attempt_accounting": {
                "matrix_validation_attempts": self._record_kind_counts.get(
                    "matrix_validation", 0
                ),
                "candidate_attempts": self._record_kind_counts.get("candidate", 0),
                "terminal_records": self._record_count,
            },
        }
        temporary_summary = self.summary_path.with_name(
            f".{self.summary_path.name}.tmp-{os.getpid()}"
        )
        with temporary_summary.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._link_without_overwrite(temporary_summary, self.summary_path)
        self._closed = True
        return summary

    def abort(self) -> None:
        """Close and remove an unpublished temporary sidecar."""
        if self._closed:
            return
        self._stream.close()
        self._temporary_path.unlink(missing_ok=True)
        self._closed = True

    @staticmethod
    def _link_without_overwrite(source: Path, destination: Path) -> None:
        """Publish a same-filesystem temporary file without replacing output."""
        try:
            os.link(source, destination)
        except FileExistsError:
            source.unlink(missing_ok=True)
            raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")
        source.unlink()


def iter_terminal_ledger(path: str | os.PathLike[str]) -> Iterable[dict[str, Any]]:
    """Yield and validate terminal rows from a JSONL sidecar."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TerminalLedgerError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if row.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
                raise TerminalLedgerError(
                    f"unsupported ledger schema at {path}:{line_number}"
                )
            yield row


def read_terminal_ledger(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a complete terminal ledger for bounded fixture validation."""
    return list(iter_terminal_ledger(path))


def verify_terminal_ledger(path: str | os.PathLike[str], expected_sha256: str) -> None:
    """Verify a JSONL sidecar digest and row schema."""
    actual = _digest_file(Path(path))
    if actual != expected_sha256:
        raise TerminalLedgerError(
            f"terminal ledger SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    list(iter_terminal_ledger(path))
