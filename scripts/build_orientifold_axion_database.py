#!/usr/bin/env python3
"""Build the orientifolded axion database `cyax.h5` bridge (Phase 1, h11=2).

This is the reusable orientifold -> ``cyax.h5`` bridge described in
``PLAN_e2e_orientifold_database_20260821.md``.  For every accepted
``h11_minus=0, h21_plus=0`` (the "trilayer") orientifold class of a given
physical ``h11``, drawn from the preserved terminal-ledger population
artifact, it re-instantiates the class in CYTools, verifies it against the
ledger's own identity hashes, dilates the canonical stretched-cone tip to the
established QCD-divisor-volume-40 convention
(``scripts/qed_divisor_assignment.py`` ``homogeneous-qcd-volume-40-v1``), and
writes one ``cyax.h5`` per viable QCD-divisor assignment in the layout
``h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5`` that ``src/read.jl`` reads.

Population selection (Step A)
------------------------------
The source of truth for the accepted ``h11_minus=0`` set is
``terminal_ledger.class_funnel`` inside the preserved, compressed ledger
summary for the requested ``h11`` (``accepted_for_table_1 == true`` entries).
This script reloads the same KS Parquet mirror partition used to build that
ledger, rebuilds each polytope's two-face-inequivalent FRST classes with
``scripts/reproduce_fuzzy_axions_h11_4.py``'s own enumeration
(``_frst_classes``), and for every class evaluates the complete exact
source-derived trilayer action manifest. The manifest reconstructs every
authorized ``(L,t,lambda_f)`` from the primal/dual polytope and then checks
the chosen fan, GLSM ``H^2`` action, parity, fixed components, smoothness,
fixed-locus Euler characteristic, and eq. (4.51). Missing evidence is a
terminal rejection. The preserved ledger action is provenance only and is
not used to construct or select a trilayer action.

Any accepted-here class absent from the ledger's accepted set, or present
with a different ``frst_hash``, is a hard-gate failure: the script stops and
reports the discrepancy rather than silently proceeding (see ``--stage``).

Known schema-version note
--------------------------
The ledger under ``data/orientifold_h11_2_3_population_20260820/`` was built
by a worktree at ``inherited_orientifold_candidates`` candidate schema 2.5
(``polytope_normal_form_id`` normal-form geometry keying).  This package
checkout uses candidate schema 3.0: it ports the corrected schema-2.5
general-action dependency chain and adds the exact Euler evidence contract.
``compute_polytope_id`` and ``compute_triangulation_hash`` are unchanged, so
the ``frst_hash`` cross-check remains schema-independent and is the hard gate
this script enforces. The summary's accepted witness is retained only as an
optional provenance cross-reference; each live action record is canonical for
its own exact fixed-locus Euler and eq. (4.51) evaluation.

QCD-divisor / visible-sector evaluation (Steps B/C)
----------------------------------------------------
For each verified class, this script tries every zero-based prime-toric-
divisor index as a candidate QCD divisor and calls the package's own
``generate_geometric_data_multitriangulation.generate_and_save_geometry``
with ``moduli_policy="canonical_qcd"``, ``qcd_volume_target=40.0``, and
``visible_sector_policy="intersecting_d7"`` -- the same validated dilation,
orientifold-H2, and QCD/QED assignment machinery the rest of the package's
generators use.  Every index that succeeds becomes one ``cy_*`` entry;
indices rejected by the existing domain checks (no invariant QED partner, no
achievable QCD volume, ...) are skipped and recorded.  ``cy_*`` indices are
assigned in ascending QCD-divisor-index order within a class's ``np_*``
directory.

That writer's current on-disk schema stores geometric data as
reconstruction references only (``storage_schema="reconstruct_on_demand"``)
and does not materialize the dense ``cytools/geometric/{divisor_volumes,Kinv,
prime_divisor_volumes}`` or ``cytools/potential/{Q,L}`` datasets that
``src/read.jl`` reads directly.  This script therefore reopens each written
file and materializes those datasets itself, using the package's own
reconstruction formulas (``_reconstruct_intersection_geometry`` from the
stored ``kappa``/``tip``; ``_geometry_potential_terms`` for the direct
effective-cone charges, their pairwise differences, and the sign/log10
instanton-scale convention) rather than recomputing them independently, and
appends the ``orientifold/`` provenance group the plan requires.  No dataset
already written by ``generate_and_save_geometry`` is modified; only new
datasets/groups are added, and finalized geometry files are never
overwritten (`generate_and_save_geometry` already refuses to overwrite an
existing artifact).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_geometric_data_multitriangulation as mg
import inherited_orientifold_candidates as ioc
import orientifold_general_l_geometry as general_l
import reproduce_fuzzy_axions_h11_4 as repro
import toric_fixed_component_euler as toric_euler
from orientifold_population_preflight import run_population_preflight
from trilayer_involutions import reconstruct_trilayer_actions
from glimmers_raw_frst import compute_polytope_id, compute_triangulation_hash, stable_hash
from qed_divisor_assignment import (
    NORMALIZATION_MAP_VERSION,
    QCD_VOLUME_TARGET,
    record_potential_match,
)

BRIDGE_SCHEMA_VERSION = "cyaxiverse-phase1-orientifold-axiverse-bridge-1.2"
ORIENTIFOLD_PROVENANCE_SCHEMA_VERSION = "cyaxiverse-phase1-orientifold-provenance-1.2"
REPORT_SCHEMA_VERSION = "cyaxiverse-orientifold-bridge-report-1.0"
EXACT_ACTION_H21_PLUS_STATUS = "not_validated"
EXACT_ACTION_H21_PLUS_REASON = (
    "analytic exact-action kernels pass, but full permitted h11=2 and h11=3 "
    "population verification is not yet complete; production release remains blocked"
)


def require_exact_action_h21_plus_validation():
    """Fail closed until the canonical action has an exact h21+ kernel."""
    if EXACT_ACTION_H21_PLUS_STATUS != "validated":
        raise RuntimeError(
            "population release is blocked: "
            f"exact_action_h21_plus_status={EXACT_ACTION_H21_PLUS_STATUS}; "
            f"{EXACT_ACTION_H21_PLUS_REASON}"
        )

PAPER_TRILAYER_TARGETS = repro.PAPER_TARGETS_BY_H11

EXPECTED_REJECTIONS = (
    mg.PrefactorCriterionNotMet,
    mg.NoPhysicalKaehlerPoint,
    mg.NoQcdDivisorVolume,
    mg.NoVisibleSectorAssignment,
)
try:
    from qed_divisor_assignment import QEDAssignmentFailure

    EXPECTED_REJECTIONS = EXPECTED_REJECTIONS + (QEDAssignmentFailure,)
except ImportError:  # pragma: no cover - defensive only
    pass


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_path(path: str) -> str:
    """Hash one file or a directory's sorted relative file stream."""
    root = Path(path).resolve()
    if root.is_file():
        return sha256_of_file(str(root))
    if not root.is_dir():
        raise FileNotFoundError(f"input path does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not result:
        raise RuntimeError(f"Git repository has no known HEAD commit: {repo_root}")
    return result


def git_tree_digest(repo_root: str) -> str:
    """Digest the committed tree and current diff for replay attribution."""
    committed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{tree}"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip().encode("utf-8")
    if not committed:
        raise RuntimeError(f"Git repository has no known tree: {repo_root}")
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=repo_root,
        check=True, capture_output=True,
    ).stdout
    return hashlib.sha256(committed + b"\0" + diff).hexdigest()


def require_clean_git_source(repo_root: str):
    """Require a replayable source tree before any artifact is written."""
    commit = git_commit(repo_root)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "refusing generation from a dirty Git worktree; commit or stash "
            f"all changes first ({repo_root})"
        )
    tree_digest = git_tree_digest(repo_root)
    return commit, tree_digest


def _tool_version(command, *, required=False):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        if required:
            raise RuntimeError(f"required provenance tool is unavailable: {command[0]}")
        return "unavailable"


def manifest_configuration(args):
    """Return all output-affecting CLI settings in a JSON-stable form."""
    excluded = {"db_root", "manifest", "report"}
    values = {}
    for name, value in vars(args).items():
        if name in excluded:
            continue
        values[name] = str(value) if isinstance(value, Path) else value
    return values


def default_run_manifest_path(args):
    """Choose a collision-resistant default manifest name for one run config."""
    encoded = json.dumps(
        manifest_configuration(args), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return Path(args.db_root) / f"run_manifest_h11-{args.h11:03d}_{args.stage}-{digest}.json.zst"


def build_run_manifest(*, args, package_root, ledger_path, ledger_sha256, manifest_path):
    """Build a deterministic, replayable bridge-run manifest."""
    source_commit, source_tree_digest = require_clean_git_source(str(package_root))
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("generate_geometric_data_multitriangulation.py"),
        Path(__file__).with_name("qed_divisor_assignment.py"),
    ]
    source_hashes = {}
    for path in source_paths:
        source_hashes[str(path.resolve())] = sha256_of_file(str(path))
    payload = {
        "schema_version": "cyaxiverse-orientifold-run-manifest-1.1",
        "source_commit": source_commit,
        "source_tree_digest": source_tree_digest,
        "dirty_state": subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(package_root), check=True, capture_output=True, text=True,
        ).stdout.splitlines(),
        "command_line": [str(item) for item in sys.argv],
        "python": {"version": sys.version, "executable": sys.executable},
        "python_packages": {
            name: _package_version(name) for name in ("cytools", "h5py", "numpy")
        },
        "julia": _tool_version(["julia", "--version"]),
        "external_tools": {
            "git": _tool_version(["git", "--version"], required=True),
            "zstd": _tool_version(["zstd", "--version"]),
        },
        "julia_project_sha256": sha256_of_file(str(package_root / "Project.toml")),
        "julia_manifest_sha256": (
            sha256_of_file(str(package_root / "Manifest.toml"))
            if (package_root / "Manifest.toml").is_file() else "absent"
        ),
        "source_hashes": source_hashes,
        "ledger": {"path": str(Path(ledger_path).resolve()), "sha256": ledger_sha256},
        "input_mirror": {
            "path": str(Path(args.parquet_dir).resolve()),
            "sha256": sha256_of_path(args.parquet_dir),
        },
        "manifest_path": str(Path(manifest_path).resolve()),
        "configuration": manifest_configuration(args),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def write_run_manifest(path, manifest):
    """Write the manifest with maximum zstd compression, refusing collisions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    compressed = subprocess.run(
        ["zstd", "-q", "-19", "-c"], input=encoded, check=True,
        capture_output=True,
    ).stdout
    manifest_file_sha256 = hashlib.sha256(compressed).hexdigest()
    if path.exists():
        if path.read_bytes() == compressed:
            return {
                "path": path,
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "manifest_file_sha256": manifest_file_sha256,
            }
        raise FileExistsError(f"refusing to overwrite an existing run manifest: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        temporary.write_bytes(compressed)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform-specific directory fsync
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "path": path,
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "manifest_file_sha256": manifest_file_sha256,
    }


def write_json_report(path, report):
    """Write one compressed report atomically and refuse implicit overwrite."""
    path = Path(path)
    if path.suffixes[-2:] != [".json", ".zst"]:
        raise ValueError("report output must use the .json.zst contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload.setdefault("schema_version", REPORT_SCHEMA_VERSION)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    compressed = subprocess.run(
        ["zstd", "-q", "-19", "-c"], input=encoded, check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256(compressed).hexdigest()
    if path.exists():
        if path.read_bytes() == compressed:
            return {"path": path, "file_sha256": digest}
        raise FileExistsError(f"refusing to overwrite an existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_bytes(compressed)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform-specific directory fsync
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"path": path, "file_sha256": digest}


def load_mpcp_certificates(path):
    """Load and validate the complete bounded-certificate input before scanning."""
    if path is None:
        raise RuntimeError(
            "bounded MPCP certificate input is required; pass --mpcp-certificates PATH"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"bounded MPCP certificate input is missing: {path}")
    if path.name.endswith(".zst"):
        completed = subprocess.run(
            ["zstd", "-dc", str(path)], check=False, capture_output=True
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"could not read bounded MPCP certificates: {detail}")
        encoded = completed.stdout
    else:
        encoded = path.read_bytes()
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bounded MPCP certificate input is not valid JSON: {path}") from exc
    certificates = payload.get("certificates") if isinstance(payload, dict) else payload
    if not isinstance(certificates, list) or not certificates:
        raise RuntimeError("bounded MPCP certificate input must contain a non-empty certificates list")
    from mpcp_bounded_analysis import validate_replay_certificate

    seen = set()
    validated = []
    for position, certificate in enumerate(certificates):
        check = validate_replay_certificate(certificate)
        if check.get("status") != "valid":
            reasons = "; ".join(check.get("reasons", []))
            raise RuntimeError(
                f"bounded MPCP certificate {position} is invalid: {reasons}"
            )
        source = certificate.get("source", {})
        frst = certificate.get("frst", {})
        key = (source.get("polytope_id"), frst.get("frst_hash"))
        if not all(isinstance(value, str) and value for value in key):
            raise RuntimeError(
                f"bounded MPCP certificate {position} is missing polytope_id/frst_hash identity"
            )
        if key in seen:
            raise RuntimeError(f"duplicate bounded MPCP certificate identity: {key[0]}::{key[1]}")
        seen.add(key)
        validated.append(dict(certificate))
    return {
        "certificates": validated,
        "count": len(validated),
        "identity_digest": hashlib.sha256(
            json.dumps(sorted(f"{polytope}::{frst}" for polytope, frst in seen),
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "path": str(path.resolve()),
        "file_sha256": sha256_of_file(str(path)),
    }


def cytools_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("cytools")
    except Exception:  # pragma: no cover - best effort provenance only
        return "unknown"


def _package_version(name: str) -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:  # pragma: no cover - best effort provenance only
        return "unknown"


def canonical_prime_divisor_volumes(glsm, divisor_volumes, prime_labels):
    """Compute prime-divisor volumes from the canonical GLSM matrix.

    ``basis_matrix`` is a basis selector and is intentionally not accepted by
    this boundary.  CYTools orders GLSM columns and prime-divisor labels
    together, so the labels are used for validation while the matrix provides
    the charge rows used by the volume formula.
    """
    matrix = np.asarray(glsm, dtype=np.int64)
    volumes = np.asarray(divisor_volumes, dtype=float).reshape(-1)
    labels = np.asarray(prime_labels, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != volumes.size:
        raise ValueError(
            f"GLSM matrix must have shape (h11, n_prime), got {matrix.shape}; "
            f"divisor volumes have length {volumes.size}"
        )
    if matrix.shape[1] != labels.size:
        raise ValueError(
            "GLSM columns and prime-divisor labels must have equal length: "
            f"{matrix.shape[1]} != {labels.size}"
        )
    if labels.size == 0 or np.unique(labels).size != labels.size:
        raise ValueError("prime-divisor labels must be non-empty and unique")
    result = np.asarray(matrix.T @ volumes, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("prime-divisor volumes are non-finite")
    return result


def load_ledger_accepted_classes(ledger_zst_path: str, sha256sums_path: str):
    """Decompress the preserved ledger summary and verify its checksum.

    Returns the list of ``class_funnel`` entries with
    ``accepted_for_table_1 == true`` and the raw decoded ledger dict.

    Accepts two preserved shapes, both carrying the identical per-class
    schema (``accepted_for_table_1``, ``accepted_witness``, ``frst_hash``,
    ``polytope_id``, ``polytope_index``, ``polytope_normal_form_id``, ...):
    a single-shard summary with ``class_funnel`` at the top level (h11=2/3),
    and a merged, sharded artifact with the same list nested at
    ``terminal_ledger.class_funnel`` (h11=4's ``h4.merged.json.zst``, which
    also carries per-shard provenance under ``shards``).
    """
    ledger_zst_path = os.path.abspath(ledger_zst_path)
    basename = os.path.basename(ledger_zst_path)
    expected_sha = None
    with open(sha256sums_path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(maxsplit=1)
            if name.strip() == basename:
                expected_sha = digest.strip()
                break
    if expected_sha is None:
        raise RuntimeError(
            f"{basename} is not listed in {sha256sums_path}; refusing to trust "
            "an unverifiable ledger artifact"
        )
    actual_sha = sha256_of_file(ledger_zst_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA256 mismatch for {ledger_zst_path}: expected {expected_sha}, "
            f"got {actual_sha}"
        )
    decoded = subprocess.run(
        ["zstd", "-d", "-c", ledger_zst_path],
        check=True,
        capture_output=True,
    ).stdout
    ledger = json.loads(decoded)
    if "class_funnel" in ledger:
        class_funnel = ledger["class_funnel"]
    elif "terminal_ledger" in ledger and "class_funnel" in ledger["terminal_ledger"]:
        class_funnel = ledger["terminal_ledger"]["class_funnel"]
    else:
        raise RuntimeError(
            f"{basename} has neither a top-level 'class_funnel' nor a "
            "'terminal_ledger.class_funnel'; unrecognized ledger shape"
        )
    accepted = [
        entry for entry in class_funnel if entry.get("accepted_for_table_1")
    ]
    return accepted, ledger, actual_sha


def enrich_accepted_witness_matrices(
    accepted, matrix_catalog_zst_path: str, terminal_ledger_jsonl_path: str
):
    """Join immutable ``matrix_id -> L`` evidence without importing old acceptance.

    The schema-2.5 summaries intentionally remain authoritative for ``t``,
    ``lambda_f``, component evidence, and acceptance.  The raw terminal ledger
    supplies only the omitted lattice matrix.  The catalog is explicitly
    class-level and joins on ``(polytope_id, frst_hash, frst_class_index,
    matrix_id)``; it must never be treated as a full action witness.  Every
    key, source hash, matrix digest, and recomputed stable ID is fail-closed.
    """
    catalog_path = Path(matrix_catalog_zst_path).resolve()
    raw_path = Path(terminal_ledger_jsonl_path).resolve()
    if not catalog_path.is_file() or not raw_path.is_file():
        raise RuntimeError("matrix catalog and immutable terminal ledger are required")
    catalog_sha = sha256_of_file(str(catalog_path))
    decoded = subprocess.run(
        ["zstd", "-d", "-c", str(catalog_path)], check=True, capture_output=True
    ).stdout
    catalog = json.loads(decoded)
    if catalog.get("schema_version") != "cyaxiverse-accepted-matrix-catalog-1.1":
        raise RuntimeError("unsupported accepted-matrix catalog schema")
    if catalog.get("record_role") != "class_level_matrix_catalog":
        raise RuntimeError(
            "accepted matrix catalog must be explicitly labelled as a class-level catalog"
        )
    raw_sha = sha256_of_file(str(raw_path))
    if raw_sha != catalog.get("source_sha256"):
        raise RuntimeError(
            "terminal-ledger SHA256 does not match the matrix catalog source: "
            f"{raw_sha} != {catalog.get('source_sha256')}"
        )
    if catalog.get("missing") or catalog.get("ambiguous") or catalog.get(
        "stable_id_recompute_failures"
    ):
        raise RuntimeError("matrix catalog records a coverage, conflict, or stable-ID failure")
    index = {}
    for row in catalog.get("catalog", []):
        required_row_fields = (
            "polytope_id", "frst_hash", "frst_class_index", "matrix_id",
            "lattice_matrix", "matrix_digest",
        )
        missing_row_fields = [name for name in required_row_fields if name not in row]
        if missing_row_fields:
            raise RuntimeError(
                "class-level matrix catalog row is missing "
                + ", ".join(missing_row_fields)
            )
        if {
            "torus_shift", "lambda_f", "candidate_id", "action_digest",
            "action_witness_digest",
        }.intersection(row):
            raise RuntimeError(
                "class-level matrix catalog row contains full action-witness fields"
            )
        key = (
            row["polytope_id"], row["frst_hash"],
            int(row["frst_class_index"]), row["matrix_id"],
        )
        if key in index:
            raise RuntimeError(f"duplicate matrix catalog key: {key}")
        matrix = np.asarray(row["lattice_matrix"], dtype=np.int64)
        if matrix.shape != (4, 4):
            raise RuntimeError(f"matrix catalog key has non-4x4 L: {key}")
        if not np.array_equal(matrix @ matrix, np.eye(4, dtype=np.int64)):
            raise RuntimeError(f"matrix catalog key has non-involutive L: {key}")
        if general_l._exact_determinant(matrix) not in (-1, 1):
            raise RuntimeError(f"matrix catalog key has non-unimodular L: {key}")
        recomputed = stable_hash(
            [key[0], key[1], tuple(int(value) for value in matrix.flat)]
        )
        if recomputed != key[3]:
            raise RuntimeError(f"matrix catalog stable-ID mismatch: {key}")
        if row["matrix_digest"] != lattice_matrix_digest(matrix):
            raise RuntimeError(f"matrix catalog matrix digest mismatch: {key}")
        index[key] = row
    enriched = []
    for entry in accepted:
        witness = entry.get("accepted_witness") or {}
        key = (
            entry["polytope_id"], entry["frst_hash"],
            int(entry["frst_class_index"]), witness.get("matrix_id"),
        )
        row = index.get(key)
        if row is None:
            raise RuntimeError(f"accepted witness has no unique matrix catalog row: {key}")
        existing = witness.get("lattice_matrix")
        if existing is not None and existing != row["lattice_matrix"]:
            raise RuntimeError(f"accepted witness conflicts with matrix catalog: {key}")
        copy = dict(entry)
        copy_witness = dict(witness)
        copy_witness["lattice_matrix"] = row["lattice_matrix"]
        copy_witness["matrix_digest"] = row["matrix_digest"]
        copy_witness["lattice_matrix_provenance"] = {
            "record_role": "class_level_matrix_catalog",
            "join_key": "polytope_id::frst_hash::frst_class_index::matrix_id",
            "polytope_id": entry["polytope_id"],
            "frst_hash": entry["frst_hash"],
            "frst_class_index": int(entry["frst_class_index"]),
            "matrix_id": key[3],
            "matrix_digest": row["matrix_digest"],
            "terminal_ledger_sha256": raw_sha,
            "matrix_catalog_file_sha256": catalog_sha,
            "source_line_numbers": row.get("source_line_numbers", []),
            "use_boundary": "lattice_matrix_only; never an action witness",
        }
        copy_witness["action_witness_digest"] = action_witness_digest(
            copy_witness,
            polytope_id=entry["polytope_id"],
            frst_hash=entry["frst_hash"],
            frst_class_index=int(entry["frst_class_index"]),
        )
        copy["accepted_witness"] = copy_witness
        enriched.append(copy)
    if len(enriched) != int(catalog.get("accepted_count", -1)):
        raise RuntimeError("matrix catalog accepted-count coverage mismatch")
    return enriched, {
        "matrix_catalog_file_sha256": catalog_sha,
        "terminal_ledger_sha256": raw_sha,
        "enriched_witness_count": len(enriched),
        "join_identity": "polytope_id::frst_hash::frst_class_index::matrix_id",
        "record_role": "class_level_matrix_catalog_joined_to_full_action_witness",
    }


def _lookup_mpcp_certificate(certificates, *, polytope_id, frst_hash):
    """Find a bounded certificate by immutable geometry and FRST identity."""

    if certificates is None:
        return None
    if isinstance(certificates, dict) and "certificates" in certificates:
        certificates = certificates["certificates"]
    if isinstance(certificates, dict) and "certificate_digest" not in certificates:
        direct = certificates.get((polytope_id, frst_hash))
        if direct is None:
            direct = certificates.get(f"{polytope_id}::{frst_hash}")
        if direct is not None:
            return direct
        certificates = list(certificates.values())
    if isinstance(certificates, dict):
        certificates = [certificates]
    if not isinstance(certificates, (list, tuple)):
        return None
    for certificate in certificates:
        if not isinstance(certificate, dict):
            continue
        source = certificate.get("source", {})
        frst = certificate.get("frst", {})
        if source.get("polytope_id") == polytope_id and frst.get("frst_hash") == frst_hash:
            return certificate
    return None


def select_and_verify_trilayer_population(
    h11, parquet_dir, ledger_accepted, *, mpcp_certificates=None
):
    """Verify all live accepted actions in each ledger-member FRST class.

    Returns ``(selected, mismatches, total_h21_plus_zero_count, live_population)`` where
    ``selected`` is a list of dicts (one per verified accepted class) each
    holding the live CYTools ``poly``/``triangulation`` objects plus identity
    and exact-action evidence. The source-derived trilayer candidate manifest
    is the acceptance gate; the preserved ledger action is only an optional
    provenance cross-reference.
    """
    require_exact_action_h21_plus_validation()
    if mpcp_certificates is None and h11 in (4, 5):
        raise RuntimeError(
            "bounded MPCP certificate input is required for population selection"
        )
    ledger_by_key = {
        (entry["polytope_id"], entry["frst_class_index"]): entry
        for entry in ledger_accepted
    }

    records = mg.load_mirror_polytopes(parquet_dir, h11=h11, limit=10**9, favorable=True)
    print(f"loaded {len(records)} favorable h11={h11} polytopes from the KS mirror", flush=True)

    selected = []
    mismatches = []
    live_population = []
    total_h21_plus_zero = 0
    for poly_index, (poly, provenance) in enumerate(records):
        raw, classes = repro._frst_classes(poly)
        points = np.asarray(poly.points(), dtype=int)
        polytope_id = compute_polytope_id(points)
        for class_index, triangulation in enumerate(classes):
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            frst_hash = compute_triangulation_hash(simplices)
            key = (polytope_id, class_index)
            ledger_entry = ledger_by_key.get(key)
            record = {
                "poly_index": poly_index,
                "class_index": class_index,
                "polytope_id": polytope_id,
                "frst_hash": frst_hash,
                "provenance": provenance,
            }
            if ledger_entry is None or ledger_entry["frst_hash"] != frst_hash:
                record["ledger_entry"] = ledger_entry
                mismatches.append(record)
                continue
            bounded_certificate = _lookup_mpcp_certificate(
                mpcp_certificates,
                polytope_id=polytope_id,
                frst_hash=frst_hash,
            )
            if bounded_certificate is None and h11 in (4, 5):
                raise RuntimeError(
                    "missing bounded MPCP certificate for required identity "
                    f"{polytope_id}::{frst_hash}"
                )
            reconstruction_kwargs = {"return_all_evidence": True}
            if bounded_certificate is not None:
                reconstruction_kwargs.update(
                    mpcp_certificate=bounded_certificate,
                    source_record={
                        "source": {
                            "polytope_id": polytope_id,
                            "global_points": points.tolist(),
                        }
                    },
                )
            witnesses, live_action_evidence = find_exact_trilayer_witnesses(
                poly, triangulation, ledger_entry,
                frst_class_index=class_index, **reconstruction_kwargs,
            )
            for witness in witnesses:
                witness["polytope_id"] = polytope_id
                witness["frst_hash"] = frst_hash
                witness["frst_class_index"] = int(class_index)
                witness["matrix_digest"] = lattice_matrix_digest(
                    witness["lattice_matrix"]
                )
                witness["action_witness_digest"] = action_witness_digest(witness)
            if not witnesses:
                record["ledger_entry"] = ledger_entry
                record["exact_action_h21_evidence"] = {
                    "status": "unavailable",
                    "reason": "no source-reconstructed trilayer action has validated h21_plus=0",
                    "live_action_evidence": live_action_evidence,
                }
                mismatches.append(record)
                continue
            exact = exact_action_h21_diagnostic(poly, triangulation, witnesses[0])
            record["exact_action_h21_evidence"] = exact
            record["orientifold_action_digest"] = exact.get("action_digest")
            record["action_witness_digest"] = witnesses[0].get(
                "action_witness_digest"
            )
            if exact.get("status") != "validated" or exact.get("h21_plus") != 0:
                record["ledger_entry"] = ledger_entry
                mismatches.append(record)
                continue
            total_h21_plus_zero += 1
            selected_witness = witnesses[0]
            live_population.append(
                {
                    "poly_index": poly_index,
                    "class_index": class_index,
                    "polytope_id": polytope_id,
                    "frst_hash": frst_hash,
                    "exact_action_h21_plus": exact["h21_plus"],
                    "matrix_id": selected_witness.get(
                        "matrix_id", selected_witness.get("matrix_candidate_id")
                    ),
                    "matrix_digest": selected_witness["matrix_digest"],
                    "torus_shift": selected_witness["torus_shift"],
                    "lambda_f": int(selected_witness["lambda_f"]),
                    "candidate_id": selected_witness["candidate_id"],
                    "action_digest": exact["action_digest"],
                    "orientifold_action_digest": exact["action_digest"],
                    "action_witness_digest": selected_witness[
                        "action_witness_digest"
                    ],
                }
            )
            record["poly"] = poly
            record["triangulation"] = triangulation
            record["p0"] = witnesses[0]["source_trilayer_candidate"]["p0"]
            record["h21_plus_special_shift_diagnostic"] = exact["h21_plus"]
            record["ledger_entry"] = ledger_entry
            record["witnesses"] = witnesses
            selected.append(record)
        print(
            f"  polytope {poly_index + 1}/{len(records)}: trilayer, "
            f"{len(classes)} FRST class(es), running h21_plus_zero total="
            f"{total_h21_plus_zero}",
            flush=True,
        )
    return selected, mismatches, total_h21_plus_zero, live_population


def population_set_audit(live_population, ledger_accepted):
    """Compare live and ledger populations in both directions.

    Use the stable ``(polytope_id, frst_hash)`` identity rather than class
    indices, which are enumeration metadata and do not certify population
    equality by themselves.
    """
    def key(record):
        return f"{record['polytope_id']}::{record['frst_hash']}"

    live = {key(record) for record in live_population}
    ledger = {key(record) for record in ledger_accepted}
    live_entries = [key(record) for record in live_population]
    ledger_entries = [key(record) for record in ledger_accepted]
    live_duplicates = sorted(item for item, count in Counter(live_entries).items() if count > 1)
    ledger_duplicates = sorted(item for item, count in Counter(ledger_entries).items() if count > 1)
    live_sorted = sorted(live)
    ledger_sorted = sorted(ledger)
    digest = lambda values: hashlib.sha256(
        json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "stable_key_definition": "polytope_id::frst_hash",
        "live_count": len(live),
        "ledger_count": len(ledger),
        "live_entry_count": len(live_entries),
        "ledger_entry_count": len(ledger_entries),
        "live_duplicate_keys": live_duplicates,
        "ledger_duplicate_keys": ledger_duplicates,
        "live_minus_ledger": sorted(live - ledger),
        "ledger_minus_live": sorted(ledger - live),
        "live_manifest_sha256": digest(live_sorted),
        "ledger_manifest_sha256": digest(ledger_sorted),
        "input_mirror_digest_required": True,
        "equal": live == ledger and not live_duplicates and not ledger_duplicates,
    }


def orientifold_action_digest(witness):
    """Hash the canonical orientifold action ``(L, t, lambda_f)``."""
    return mg.stable_hash(
        {
            "lattice_matrix": _jsonable(witness["lattice_matrix"]),
            "torus_shift": _jsonable(witness["torus_shift"]),
            "lambda_f": int(witness["lambda_f"]),
        }
    )


def lattice_matrix_digest(lattice_matrix):
    """Hash one exact lattice matrix independently of any action shift."""
    raw = np.asarray(lattice_matrix)
    if raw.shape != (4, 4):
        raise ValueError("lattice matrix digest requires a 4 by 4 matrix")
    try:
        integer_values = [[int(value) for value in row] for row in raw.tolist()]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lattice matrix digest requires exact integer entries") from exc
    if any(
        raw[i, j] != integer_values[i][j]
        for i in range(raw.shape[0])
        for j in range(raw.shape[1])
    ):
        raise ValueError("lattice matrix digest requires exact integer entries")
    try:
        matrix = np.asarray(integer_values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lattice matrix digest entries exceed int64") from exc
    return mg.stable_hash({"lattice_matrix": matrix.tolist()})


def action_witness_digest(
    witness, *, polytope_id=None, frst_hash=None, frst_class_index=None
):
    """Hash the complete geometry/class/action witness identity.

    Keep this distinct from ``orientifold_action_digest``: a class-level matrix
    catalog identifies only ``(polytope, FRST class, L)``.  A full action
    witness additionally carries ``t``, ``lambda_f``, candidate identity, and
    the action digest itself.
    """
    polytope_id = witness.get("polytope_id", polytope_id)
    frst_hash = witness.get("frst_hash", frst_hash)
    frst_class_index = witness.get("frst_class_index", frst_class_index)
    required = (
        "lattice_matrix", "torus_shift", "lambda_f", "candidate_id",
    )
    missing = [name for name in required if name not in witness]
    if polytope_id is None or frst_hash is None or frst_class_index is None:
        missing.append("polytope_id/frst_hash/frst_class_index")
    if missing:
        raise ValueError("complete action witness is missing " + ", ".join(missing))
    matrix_digest = lattice_matrix_digest(witness["lattice_matrix"])
    supplied_matrix_digest = witness.get("matrix_digest")
    if supplied_matrix_digest is not None and supplied_matrix_digest != matrix_digest:
        raise ValueError("action witness matrix digest does not match L")
    action_digest = orientifold_action_digest(witness)
    payload = {
        "identity_schema": "cyaxiverse-full-action-witness-1.0",
        "polytope_id": str(polytope_id),
        "frst_hash": str(frst_hash),
        "frst_class_index": int(frst_class_index),
        "matrix_id": witness.get("matrix_id", witness.get("matrix_candidate_id")),
        "matrix_digest": matrix_digest,
        "lattice_matrix": _jsonable(witness["lattice_matrix"]),
        "torus_shift": _jsonable(witness["torus_shift"]),
        "lambda_f": int(witness["lambda_f"]),
        "candidate_id": str(witness["candidate_id"]),
        "action_digest": action_digest,
    }
    if payload["matrix_id"] is None:
        raise ValueError("complete action witness is missing matrix_id")
    return mg.stable_hash(payload)


def _decode_rational_vector(encoded, *, dimension=4):
    """Decode the canonical JSON rational-vector representation exactly."""
    from fractions import Fraction

    if not isinstance(encoded, dict):
        raise ValueError("rational vector must be an object")
    numerator = encoded.get("numerator")
    denominator = encoded.get("denominator")
    if not isinstance(numerator, list) or len(numerator) != dimension:
        raise ValueError(f"rational vector must have {dimension} numerators")
    if not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("rational vector denominator must be a positive integer")
    return tuple(Fraction(int(value), denominator) for value in numerator)


def exact_hodge_split_from_euler(*, h11, h21, h11_minus, chi_fixed_locus, chi_x):
    """Apply Moritz arXiv:2305.06363v1 eq. (4.51) with integer checks.

    ``h21_minus = h11_minus + (chi(F_I)-chi(X))/4 - 1``.  This helper is
    independent of fixed-locus enumeration so analytic fixtures can test the
    source sign and the nonidentity ``h11_minus`` term separately.
    """
    values = (h11, h21, h11_minus, chi_fixed_locus, chi_x)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError("Hodge and Euler inputs must be exact integers")
    h11, h21, h11_minus, chi_fixed_locus, chi_x = map(int, values)
    delta = chi_fixed_locus - chi_x
    if delta % 4:
        raise ValueError("chi(F_I)-chi(X) is not divisible by four")
    h21_minus = h11_minus + delta // 4 - 1
    h11_plus = h11 - h11_minus
    h21_plus = h21 - h21_minus
    if min(h11_plus, h11_minus, h21_plus, h21_minus) < 0:
        raise ValueError("eq. (4.51) produced a negative Hodge eigenspace dimension")
    return {
        "h11_plus": h11_plus,
        "h11_minus": h11_minus,
        "h21_plus": h21_plus,
        "h21_minus": h21_minus,
        "chi_fixed_locus": chi_fixed_locus,
        "chi_x": chi_x,
    }


def exact_action_h21_diagnostic(poly, triangulation, witness):
    """Evaluate the canonical ``(L,t,lambda_f)`` action, failing unavailable.

    Moritz arXiv:2305.06363v1 eqs. (4.34)--(4.46) determine the fixed
    components and their hypersurface parity; eq. (4.51) determines the Hodge
    split. Components are reconstructed from the exact canonical action and
    original fan. Euler characteristics use smooth complete quotient-star
    fans only. Identity actions also run the independent legacy calculation.
    """
    required = ("lattice_matrix", "torus_shift", "lambda_f", "h11_minus")
    missing = [key for key in required if key not in witness]
    digest = orientifold_action_digest(witness) if not missing else None
    base = {
        "schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0",
        "status": "unavailable",
        "action_digest": digest,
        "source": "Moritz arXiv:2305.06363v1 eqs. 4.34-4.51",
        "derived_transverse_euler_source": (
            "Khovanskii 1978 Section 3 Theorem 2; Danilov-Khovanskii 1987 "
            "Proposition 1.6 and Sections 3.2,3.6; validation formula ledger"
        ),
    }
    if missing:
        return {**base, "reason": f"canonical witness is missing {missing}"}
    try:
        matrix = np.asarray(witness["lattice_matrix"], dtype=np.int64)
        t = _decode_rational_vector(witness["torus_shift"])
    except (TypeError, ValueError) as exc:
        return {**base, "reason": f"invalid exact action: {exc}"}
    if matrix.shape != (4, 4) or not np.array_equal(matrix @ matrix, np.eye(4, dtype=np.int64)):
        return {**base, "reason": "L is not a rank-four involution"}
    two_t = tuple(2 * value for value in t)
    if any(value.denominator != 1 for value in two_t):
        return {**base, "reason": "Moritz involution condition 2t in N failed"}
    if int(witness["lambda_f"]) != 1:
        return {**base, "reason": "population contract requires the O3/O7 lambda_f=1 branch"}
    smoothness = witness.get("smoothness")
    if not isinstance(smoothness, dict) or smoothness.get("status") != "smooth":
        return {**base, "reason": "canonical action lacks a smoothness certificate"}
    is_identity = np.array_equal(matrix, np.eye(4, dtype=np.int64))
    if is_identity and int(witness["h11_minus"]) != 0:
        return {**base, "reason": "identity-L action must have h11_minus=0"}
    identity_diagnostic = None
    try:
        triangulation_cones = ioc._triangulation_cones(poly, triangulation)
        # The exact Eq. (4.42) support kernel needs the complete dual lattice
        # point set.  Keep this legacy diagnostic on the same CYTools
        # built-in-first path as candidate enumeration: ``points()`` is the
        # exact lattice-point API, with ``vertices()`` retained only as the
        # explicit compatibility fallback.  Omitting these inputs would
        # make an otherwise valid action fail closed as
        # ``missing_invariant_restricted_support``.
        dual_points = ioc._extract_dual_lattice_points(poly)
        ambient_rays = sorted({
            tuple(int(value) for value in ray)
            for cone in triangulation_cones
            for ray in cone
        })
        auxiliary_fan = general_l.build_auxiliary_fan(triangulation_cones, matrix)
        fixed_cone_keys = general_l._pointwise_invariant_cone_keys(
            triangulation_cones, matrix
        )
        components = general_l._fixed_component_records(
            auxiliary_fan,
            matrix,
            t,
            int(witness["lambda_f"]),
            fixed_cone_keys=fixed_cone_keys,
            dual_points=dual_points,
            ambient_rays=ambient_rays,
            fan_cones=triangulation_cones,
        )
        fixed = toric_euler.exact_fixed_locus_euler(
            auxiliary_fan,
            matrix,
            components,
            fixed_surface_n_s_evidence=witness.get(
                "fixed_surface_n_s_evidence", {}
            ),
        )
        fixed_locus_method = (
            "exact_componentwise_smooth_chern_or_ordinary_euler_orbit"
        )
        if is_identity:
            # Retain the published canonical-p0 implementation as a
            # non-gating diagnostic only. It uses CYTools Float64 intersection
            # tensors, so it must never be rounded into exact acceptance.
            p0 = np.asarray([int(value) for value in two_t], dtype=np.int64)
            identity_diagnostic = repro._fixed_locus_euler_characteristic(
                poly, triangulation, p0
            )
    except (ArithmeticError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return {**base, "reason": f"exact component construction failed: {exc}"}
    if fixed.get("status") != "computed":
        return {
            **base,
            "reason": "exact action fixed-locus Euler evidence is unavailable: "
            + str(fixed.get("reason")),
            "reason_code": fixed.get("reason_code"),
            "components": fixed.get("components", []),
        }
    chi_fixed = fixed["chi_F_I"]
    if isinstance(chi_fixed, bool) or not isinstance(chi_fixed, (int, np.integer)):
        return {**base, "reason": "fixed-locus Euler characteristic is not integral"}
    chi_fixed = int(chi_fixed)
    identity_cross_check = None
    if identity_diagnostic is not None:
        legacy_chi = identity_diagnostic.get("chi_F_I")
        tolerance = 1.0e-8
        residual = None
        agrees = False
        if isinstance(legacy_chi, (int, float, np.integer, np.floating)) and math.isfinite(
            float(legacy_chi)
        ):
            residual = float(legacy_chi) - chi_fixed
            agrees = abs(residual) <= tolerance
        identity_cross_check = {
            "role": "non_gating_float64_diagnostic_only",
            "status": identity_diagnostic.get("status"),
            "legacy_chi_F_I": legacy_chi,
            "exact_chi_F_I": chi_fixed,
            "residual_legacy_minus_exact": residual,
            "absolute_tolerance": tolerance,
            "agrees_within_tolerance": agrees,
            "legacy_reasons": identity_diagnostic.get("reasons", []),
        }
    cy = triangulation.get_cy()
    try:
        split = exact_hodge_split_from_euler(
            h11=int(cy.h11()), h21=int(cy.h21()),
            h11_minus=int(witness["h11_minus"]),
            chi_fixed_locus=chi_fixed, chi_x=int(cy.chi()),
        )
    except ValueError as exc:
        return {**base, "reason": str(exc), "components": fixed.get("components", [])}
    return {
        **base,
        **split,
        "status": "validated",
        "reason": None,
        "fixed_locus_method": fixed_locus_method,
        "identity_canonical_p0_float64_cross_check": identity_cross_check,
        "components": fixed["components"],
    }


def _witness_matches_ledger(record, ledger_witness):
    """Compare a live action with the summary's optional action cross-reference."""
    if not ledger_witness or any(
        key not in ledger_witness for key in ("lattice_matrix", "torus_shift", "lambda_f")
    ):
        return False
    matrix_id = ledger_witness.get("matrix_id")
    if matrix_id is not None and record.get("matrix_candidate_id") != matrix_id:
        return False
    if _jsonable(record.get("lattice_matrix")) != _jsonable(
        ledger_witness["lattice_matrix"]
    ):
        return False
    if int(record.get("lambda_f", -1)) != int(ledger_witness.get("lambda_f", -2)):
        return False
    if record.get("torus_shift") != ledger_witness.get("torus_shift"):
        return False
    return orientifold_action_digest(record) == orientifold_action_digest(ledger_witness)


def _exact_trilayer_topology(poly, triangulation):
    """Build the topology evidence required by the exact trilayer evaluator."""
    try:
        cy = triangulation.get_cy()
        topology = dict(mg.extract_topology(cy, triangulation))
        topology["compute_general_fixed_surface_n_s"] = True
        triangulation_cones = ioc._triangulation_cones(poly, triangulation)
        topology["fixed_surface_n_s"] = ioc.identity_fixed_surface_n_s_table(
            triangulation_cones, triangulation
        )
        topology["non_smooth_facet_dual_vertices"] = ioc.facets_with_non_smooth_cones(
            poly, triangulation
        )
        return topology
    except (ArithmeticError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "_exact_trilayer_topology_unavailable": str(exc),
            "fixed_surface_n_s": {},
            "non_smooth_facet_dual_vertices": set(),
        }


def find_exact_trilayer_witnesses(
    poly, triangulation, ledger_entry=None, *, frst_class_index,
    return_all_evidence=False, mpcp_certificate=None, source_record=None
):
    """Return witnesses from the complete source-derived trilayer manifest.

    This is the population-facing constructor.  It enumerates every primal
    vertex authorized by Moritz's trilayer criterion and evaluates each
    resulting ``(L,t,lambda_f)`` action on the chosen FRST.  The broad matrix
    catalog in :func:`find_accepted_o3o7_witness` remains available for legacy
    diagnostics, but is deliberately not used to select Sheridan trilayers.
    """
    polytope_id = compute_polytope_id(np.asarray(poly.points(), dtype=int))
    frst_hash = compute_triangulation_hash(
        np.asarray(triangulation.simplices(), dtype=int)
    )
    topology = _exact_trilayer_topology(poly, triangulation)
    reconstruction = reconstruct_trilayer_actions(
        poly,
        triangulation,
        topology,
        mpcp_certificate=mpcp_certificate,
        source_record=source_record,
    )
    qualifying = []
    evaluated = []
    for candidate in reconstruction["candidates"]:
        action = candidate.get("action") or {}
        action_digest_value = candidate.get("action_digest")
        matrix = candidate.get("lattice_matrix", action.get("lattice_matrix"))
        matrix_digest = None
        if matrix is not None:
            matrix_digest = lattice_matrix_digest(matrix)
        candidate_id = candidate.get("candidate_id")
        evaluated_record = {
            "candidate_id": candidate_id,
            "matrix_id": (
                f"source-trilayer-L-sha256:{matrix_digest}"
                if matrix_digest is not None else None
            ),
            "action_digest": action_digest_value,
            "matrix_digest": matrix_digest,
            "terminal_status": candidate.get("terminal_status"),
            "h11_minus": (candidate.get("h2_action") or {}).get("h11_minus"),
            "lambda_f": candidate.get("lambda_f", action.get("lambda_f")),
            "source_trilayer_candidate": candidate,
            "polytope_id": polytope_id,
            "frst_hash": frst_hash,
            "frst_class_index": int(frst_class_index),
            "mpcp_certificate_status": (
                candidate.get("mpcp_certificate_verification", {}).get("status")
                if isinstance(candidate.get("mpcp_certificate_verification"), dict)
                else None
            ),
        }
        if candidate.get("terminal_status") != "accepted_exact_trilayer_action":
            evaluated.append(evaluated_record)
            continue
        h2_action = candidate.get("h2_action") or {}
        hodge = candidate.get("hodge_split") or {}
        fixed_euler = candidate.get("fixed_locus_euler") or {}
        required = (
            candidate_id,
            matrix,
            candidate.get("torus_shift", action.get("torus_shift")),
            h2_action.get("h11_minus"),
            hodge.get("h21_plus"),
            fixed_euler.get("chi_F_I"),
        )
        if any(value is None for value in required):
            evaluated_record["terminal_status"] = "topology_evidence_unavailable"
            evaluated_record["reason"] = (
                "accepted exact reconstruction omitted a required witness field"
            )
            evaluated.append(evaluated_record)
            continue
        matrix_id = f"source-trilayer-L-sha256:{matrix_digest}"
        witness = {
            "candidate_id": str(candidate_id),
            "matrix_id": matrix_id,
            "matrix_candidate_id": matrix_id,
            "lattice_matrix": matrix,
            "h2_involution_matrix": h2_action["matrix"],
            "h2_action_proof": h2_action["proof"],
            "torus_shift": candidate.get("torus_shift", action["torus_shift"]),
            "lambda_f": int(candidate.get("lambda_f", action["lambda_f"])),
            "involution_type": "O3/O7",
            "h11_plus": int(hodge["h11_plus"]),
            "h11_minus": int(h2_action["h11_minus"]),
            "smoothness": candidate.get("smoothness"),
            "fixed_surface_n_s_evidence": topology.get("fixed_surface_n_s", {}),
            "source_trilayer_candidate": candidate,
            "reconstruction_provenance": reconstruction["provenance"],
            "polytope_id": polytope_id,
            "frst_hash": frst_hash,
            "frst_class_index": int(frst_class_index),
            "mpcp_certificate": candidate.get("mpcp_certificate"),
            "mpcp_certificate_verification": candidate.get(
                "mpcp_certificate_verification"
            ),
        }
        if ledger_entry is not None:
            ledger_witness = ledger_entry.get("accepted_witness") or {}
            witness["ledger_summary_action_cross_reference"] = {
                "matrix_id": ledger_witness.get("matrix_id"),
                "candidate_id": ledger_witness.get("candidate_id"),
                "same_action": _witness_matches_ledger(witness, ledger_witness)
                if "lattice_matrix" in ledger_witness else False,
                "role": "optional_cross_reference_not_selection_gate",
            }
        witness["exact_action_h21_evidence"] = {
            "schema_version": "cyaxiverse-exact-action-hodge-evidence-3.0",
            "status": "validated",
            "reason": None,
            "action_digest": orientifold_action_digest(witness),
            "source": "Moritz arXiv:2305.06363v1 eqs. 4.34-4.51",
            "h11_plus": int(hodge["h11_plus"]),
            "h11_minus": int(hodge["h11_minus"]),
            "h21_plus": int(hodge["h21_plus"]),
            "h21_minus": int(hodge["h21_minus"]),
            "chi_fixed_locus": int(fixed_euler["chi_F_I"]),
            "chi_x": int(hodge["chi_x"]),
            "fixed_locus_method": "exact_source_trilayer_reconstruction",
            "components": (
                candidate.get("fixed_components", {}).get("components", [])
                if isinstance(candidate.get("fixed_components"), dict)
                else candidate.get("fixed_components", [])
            ),
            "reconstruction_provenance": reconstruction["provenance"],
        }
        witness["matrix_digest"] = matrix_digest
        witness["orientifold_action_digest"] = witness["exact_action_h21_evidence"][
            "action_digest"
        ]
        evaluated_record.update(
            {
                "candidate_id": witness["candidate_id"],
                "matrix_id": matrix_id,
                "action_digest": witness["orientifold_action_digest"],
                "h11_minus": witness["h11_minus"],
                "lambda_f": witness["lambda_f"],
                "exact_action_h21_evidence": witness["exact_action_h21_evidence"],
            }
        )
        witness["action_witness_digest"] = action_witness_digest(witness)
        evaluated_record["action_witness_digest"] = witness["action_witness_digest"]
        qualifying.append(witness)
        evaluated.append(evaluated_record)
    qualifying.sort(key=lambda record: record["orientifold_action_digest"])
    evaluated.sort(key=lambda record: (str(record.get("action_digest")), str(record.get("candidate_id"))))
    return (qualifying, evaluated) if return_all_evidence else qualifying


def find_accepted_o3o7_witness(
    poly, triangulation, ledger_entry=None, *, return_all_evidence=False
):
    """Re-derive an exact-action O3/O7 witness with ``h21_plus=0``.

    The preserved class summary's ``accepted_witness`` is only an arbitrary
    accepted action and must not hide another live action with ``h21_plus=0``.
    Every live accepted action is its own exact witness. Qualifying actions are
    sorted by action digest for a deterministic class representative.
    """
    cy = triangulation.get_cy()
    topology = dict(mg.extract_topology(cy, triangulation))
    topology["compute_general_fixed_surface_n_s"] = True
    triangulation_cones = ioc._triangulation_cones(poly, triangulation)
    topology["fixed_surface_n_s"] = ioc.identity_fixed_surface_n_s_table(
        triangulation_cones, triangulation
    )
    topology["non_smooth_facet_dual_vertices"] = ioc.facets_with_non_smooth_cones(
        poly, triangulation
    )
    candidate_records = ioc.enumerate_orientifold_candidates(poly, triangulation, topology)
    accepted = [
        record
        for record in candidate_records
        if record.get("terminal_status") == "accepted_verified_orientifold"
        and record.get("h11_minus") == 0
        and record.get("lambda_f") == 1
    ]
    qualifying = []
    evaluated = []
    for record in accepted:
        evidence = exact_action_h21_diagnostic(poly, triangulation, record)
        evaluated_record = {
            "candidate_id": record.get("candidate_id"),
            "matrix_id": record.get("matrix_candidate_id"),
            "action_digest": evidence.get("action_digest")
            or orientifold_action_digest(record),
            "matrix_digest": lattice_matrix_digest(record["lattice_matrix"]),
            "terminal_status": record.get("terminal_status"),
            "h11_minus": record.get("h11_minus"),
            "lambda_f": record.get("lambda_f"),
            "exact_action_h21_evidence": evidence,
        }
        evaluated.append(evaluated_record)
        if evidence.get("status") != "validated" or evidence.get("h21_plus") != 0:
            continue
        witness = dict(record)
        witness["exact_action_h21_evidence"] = evidence
        witness["orientifold_action_digest"] = evidence["action_digest"]
        if ledger_entry is not None:
            ledger_witness = ledger_entry.get("accepted_witness") or {}
            witness["ledger_summary_action_cross_reference"] = {
                "matrix_id": ledger_witness.get("matrix_id"),
                "candidate_id": ledger_witness.get("candidate_id"),
                "same_action": _witness_matches_ledger(witness, ledger_witness)
                if "lattice_matrix" in ledger_witness else False,
                "role": "optional_cross_reference_not_selection_gate",
            }
        qualifying.append(witness)
    qualifying.sort(key=lambda record: record["orientifold_action_digest"])
    evaluated.sort(key=lambda record: record["action_digest"])
    return (qualifying, evaluated) if return_all_evidence else qualifying


def materialize_potential_and_kinetic_data(h5_path):
    """Append the dense datasets src/read.jl reads directly to a written cyax.h5.

    ``generate_and_save_geometry``'s current schema stores geometric and
    potential data as reconstruction references only.  Materialize the dense
    ``cytools/geometric/{divisor_volumes,Kinv,prime_divisor_volumes}`` and
    ``cytools/potential/{Q,L}`` datasets using the package's own
    reconstruction formulas, without altering anything already written.

    Also patches four ``cytools/geometric/visible_sector`` fields
    (``qed_unsorted_potential_index``, ``qed_post_sort_source_position``,
    ``qed_potential_scale``, ``qed_log10_lambda4``) that
    ``generate_and_save_geometry``'s current non-EFT path leaves
    ``"deferred_to_eft_row_reconstruction"`` (it defers their computation to
    a later EFT-row stage that only runs under ``--eft``), but that
    ``src/read.jl``'s ``visible_sector`` reader reads unconditionally.  Uses
    the same ``qed_divisor_assignment.record_potential_match`` formula the
    EFT-row reconstruction path itself uses (``glimmers_eft_row_schema.py``),
    applied to the Q/L just materialized above, so the patched values are
    computed the established way rather than re-derived ad hoc.
    """
    with h5py.File(h5_path, "r+") as file:
        geometric = file["cytools/geometric"]
        kappa = np.asarray(geometric["kappa"][()], dtype=float)
        tip = np.asarray(geometric["tip"][()], dtype=float)
        basis_matrix = np.asarray(geometric["basis_matrix"][()], dtype=np.int64)
        # `basis_matrix` selects a divisor basis; it is not the GLSM charge
        # matrix.  Prime-divisor volumes must use the canonical CYTools GLSM
        # rows stored by the geometry generator.
        glsm = np.asarray(geometric["glsm"][()], dtype=np.int64)
        prime_labels = np.asarray(geometric["prime_toric_divisors"][()], dtype=np.int64).reshape(-1)
        effective_cone = np.asarray(geometric["effective_cone"][()], dtype=np.int64)
        h11 = int(geometric["h11"][()])
        stored_cy_volume = float(geometric["CY_volume"][()])

        reconstructed = mg._reconstruct_intersection_geometry(kappa, tip)
        divisor_volumes = np.asarray(reconstructed["divisor_volumes"], dtype=float)
        kinv = np.asarray(reconstructed["inverse_metric"], dtype=float)
        cy_volume = float(reconstructed["cy_volume"])
        if not math.isclose(cy_volume, stored_cy_volume, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                "reconstructed CY volume disagrees with the value "
                f"generate_and_save_geometry stored: reconstructed={cy_volume!r} "
                f"stored={stored_cy_volume!r}"
            )

        try:
            prime_divisor_volumes = canonical_prime_divisor_volumes(
                glsm, divisor_volumes, prime_labels
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        reference = {
            "h11": h11,
            "effective_cone": effective_cone,
            "kappa": kappa,
            "tip": tip,
            "basis_matrix": basis_matrix,
            "glsm": glsm,
            "prime_toric_divisors": prime_labels,
        }

        visible_sector_path = "cytools/geometric/visible_sector"
        has_visible_sector = visible_sector_path in file
        if has_visible_sector:
            # ``reconstruct_potential_from_reference`` appends the QED prime
            # divisor's own charge as an extra potential column
            # ("appended_prime_divisor_e3") when it is not already one of the
            # direct effective-cone charges -- exactly the same construction
            # the EFT-row reconstruction path uses for the same purpose.
            # Route through it (rather than the geometry-only
            # ``_geometry_potential_terms`` directly) so that edge case is
            # handled the established way instead of leaving a column index
            # that could run past the end of a geometry-only Q/L.
            visible = file[visible_sector_path]
            qed_divisor_index = int(visible["qed_divisor_index"][()])
            reconstructed_potential = mg.reconstruct_potential_from_reference(
                reference, {"qed_divisor_index": qed_divisor_index}
            )
            q = np.asarray(reconstructed_potential["Q"], dtype=np.int64)
            l = np.asarray(reconstructed_potential["L"], dtype=np.float64)
            direct_count = int(reconstructed_potential["direct_count"])
            qed_source_index = int(reconstructed_potential["source_index"])
        else:
            terms = mg._geometry_potential_terms(reference)
            q = np.asarray(terms["q"], dtype=np.int64)
            l = np.asarray(terms["l"], dtype=np.float64)

        if q.shape[0] != h11:
            raise RuntimeError(f"reconstructed Q has shape {q.shape}, expected first axis {h11}")
        if l.shape[0] != 2 or l.shape[1] != q.shape[1]:
            raise RuntimeError(f"reconstructed L has shape {l.shape}, expected (2, {q.shape[1]})")

        geometric.create_dataset(
            "divisor_volumes", data=divisor_volumes, compression="gzip", compression_opts=9
        )
        geometric.create_dataset("Kinv", data=kinv, compression="gzip", compression_opts=9)
        geometric.create_dataset(
            "prime_divisor_volumes", data=prime_divisor_volumes,
            compression="gzip", compression_opts=9,
        )
        potential = file["cytools/potential"]
        # Store Q and L transposed on disk. In memory q is (h11, N) and l is
        # (2, N), the canonical Julia orientation. HDF5.jl reads datasets with
        # reversed axes relative to h5py (column-major vs row-major), so a
        # dataset written here as (h11, N) is read in Julia as (N, h11). The
        # raw potential path (`read.potential`/`potential_factored`, used by the
        # spectrum and vacua engines) does not re-orient, so persist (N, h11)
        # and (N, 2) here to make the Julia-side raw read yield (h11, N)/(2, N),
        # matching the reference generator's on-disk layout. `record_potential_match`
        # below keeps the original in-memory (h11, N)/(2, N) arrays.
        potential.create_dataset(
            "Q", data=np.ascontiguousarray(q.T), compression="gzip", compression_opts=9
        )
        potential.create_dataset(
            "L", data=np.ascontiguousarray(l.T), compression="gzip", compression_opts=9
        )

        visible_sector_patch = None
        if has_visible_sector:
            qed_charge = np.asarray(visible["qed_charge"][()], dtype=np.int64)
            match = record_potential_match(q, l, qed_charge, direct_count, qed_source_index)
            for name in (
                "qed_unsorted_potential_index",
                "qed_post_sort_source_position",
            ):
                if name in visible:
                    del visible[name]
                visible.create_dataset(name, data=int(match[name]))
            for name in ("qed_potential_scale", "qed_log10_lambda4"):
                if name in visible:
                    del visible[name]
                visible.create_dataset(name, data=float(match["qed_potential_scale"]))
            visible_sector_patch = match
        file.flush()
    return {
        "divisor_volumes": divisor_volumes,
        "Kinv": kinv,
        "cy_volume": cy_volume,
        "Q_shape": q.shape,
        "L_shape": l.shape,
        "visible_sector_patch": visible_sector_patch,
    }


def write_orientifold_provenance_group(
    h5_path,
    *,
    witness_record,
    ledger_entry,
    h11_plus_from_diagnostic,
    h11_minus_from_diagnostic,
    h21_plus_from_special_shift_diagnostic,
    exact_action_h21_evidence,
    ledger_zst_path,
    ledger_sha256,
    source_commit,
    cytools_version_string,
    run_manifest,
    action_digest,
    certification_info=None,
):
    """Append complete class/action provenance to the staged geometry file."""
    require_exact_action_h21_plus_validation()
    expected_action_digest = orientifold_action_digest(witness_record)
    if action_digest != expected_action_digest:
        raise RuntimeError("supplied action digest does not match the exact witness")
    expected_witness_digest = action_witness_digest(witness_record)
    if witness_record.get("action_witness_digest") != expected_witness_digest:
        raise RuntimeError("supplied action-witness digest does not match the witness")
    ledger_witness = ledger_entry.get("accepted_witness") or {}
    ledger_action_digest = orientifold_action_digest(ledger_witness)
    with h5py.File(h5_path, "r+") as file:
        if "orientifold" in file:
            raise FileExistsError(
                f"{h5_path} already carries an 'orientifold' provenance group"
            )
        group = file.create_group("orientifold")
        group.create_dataset(
            "h2_involution_matrix",
            data=np.asarray(witness_record["h2_involution_matrix"], dtype=np.int64),
            compression="gzip", compression_opts=9,
        )
        group.create_dataset(
            "lattice_matrix",
            data=np.asarray(witness_record["lattice_matrix"], dtype=np.int64),
            compression="gzip", compression_opts=9,
        )
        group.attrs["h2_action_method"] = "exact_full_glsm_quotient_relation"
        group.attrs["h2_action_proof_json"] = json.dumps(
            _jsonable(witness_record.get("h2_action_proof", {})),
            sort_keys=True,
        )
        torus_shift = witness_record["torus_shift"]
        group.create_dataset(
            "torus_shift_numerator",
            data=np.asarray(torus_shift["numerator"], dtype=np.int64),
        )
        group.create_dataset(
            "torus_shift_denominator", data=int(torus_shift["denominator"])
        )
        group.create_dataset("lambda_f", data=int(witness_record["lambda_f"]))
        group.attrs["canonical_action_digest"] = action_digest
        group.attrs["canonical_matrix_digest"] = witness_record["matrix_digest"]
        group.attrs["canonical_action_witness_digest"] = expected_witness_digest
        group.attrs["ledger_summary_action_digest"] = ledger_action_digest
        group.attrs["selected_action_digest_matches_ledger_summary"] = (
            action_digest == ledger_action_digest
        )
        group.attrs["canonical_action_source"] = (
            "live_accepted_action_exact_evaluation;ledger_summary_action_is_cross_reference_only"
        )
        group.create_dataset("h11_minus", data=int(witness_record["h11_minus"]))
        group.create_dataset("h11_plus", data=int(witness_record["h11_plus"]))
        group.create_dataset(
            "h11_minus_diagnostic", data=int(h11_minus_from_diagnostic)
        )
        group.create_dataset(
            "h11_plus_diagnostic", data=int(h11_plus_from_diagnostic)
        )
        group.create_dataset(
            "h21_plus_special_shift_diagnostic",
            data=float(h21_plus_from_special_shift_diagnostic),
        )
        if exact_action_h21_evidence.get("status") != "validated":
            raise RuntimeError("refusing HDF5 output without validated exact-action evidence")
        if exact_action_h21_evidence.get("action_digest") != action_digest:
            raise RuntimeError("exact-action evidence digest does not match canonical action")
        for name in (
            "h11_plus", "h11_minus", "h21_plus", "h21_minus",
            "chi_fixed_locus", "chi_x",
        ):
            group.create_dataset(name + "_exact_action", data=int(exact_action_h21_evidence[name]))
        group.attrs["exact_action_h21_plus_status"] = EXACT_ACTION_H21_PLUS_STATUS
        group.attrs["exact_action_h21_plus_reason"] = EXACT_ACTION_H21_PLUS_REASON
        group.attrs["exact_action_evidence_schema_version"] = exact_action_h21_evidence[
            "schema_version"
        ]
        group.attrs["exact_action_evidence_json"] = json.dumps(
            _jsonable(exact_action_h21_evidence), sort_keys=True
        )
        group.attrs["polytope_id"] = witness_record["polytope_id"]
        group.attrs["frst_class_index"] = int(witness_record["frst_class_index"])
        group.attrs["polytope_normal_form_id"] = ledger_entry.get(
            "polytope_normal_form_id", ""
        )
        group.attrs["polytope_normal_form_id_source"] = (
            "copied_from_immutable_schema_2.5_ledger_class_membership;"
            "live_action_evidence_uses_candidate_schema_3.0"
        )
        group.attrs["frst_hash"] = witness_record["frst_hash"]
        group.attrs["frst_hash_verified_against_ledger"] = bool(
            ledger_entry.get("frst_hash") == witness_record["frst_hash"]
        )
        group.attrs["ledger_class_key"] = (
            f"{witness_record['polytope_id']}::{witness_record['frst_hash']}::"
            f"{int(witness_record['frst_class_index'])}"
        )
        group.attrs["class_identity_join_key"] = (
            "polytope_id::frst_hash::frst_class_index"
        )
        group.attrs["action_identity_join_key"] = (
            "polytope_id::frst_hash::frst_class_index::matrix_id::"
            "matrix_digest::candidate_id::action_digest::torus_shift::lambda_f"
        )
        group.attrs["ledger_class_membership_verified"] = bool(
            ledger_entry.get("accepted_for_table_1", False)
            and ledger_entry.get("frst_hash") == witness_record["frst_hash"]
        )
        group.attrs["class_certification_evidence_json"] = json.dumps(
            {
                "polytope_id": witness_record["polytope_id"],
                "frst_hash": witness_record["frst_hash"],
                "frst_class_index": int(witness_record["frst_class_index"]),
                "record_role": "class_level_membership",
                "ledger_accepted_for_table_1": bool(
                    ledger_entry.get("accepted_for_table_1", False)
                ),
                "frst_hash_verified_against_ledger": bool(
                    ledger_entry.get("frst_hash") == witness_record["frst_hash"]
                ),
                "witness_terminal_status": "accepted_verified_orientifold",
            },
            sort_keys=True,
        )
        group.attrs["witness_source"] = (
            "canonical_L_t_lambda_f_from_live_exact_source_trilayer_reconstruction; "
            "ledger_summary_action_is_cross_reference_only; live_candidate_"
            "rederived_and_verified_in_this_worktree; linkage is the "
            "(polytope_id, frst_class_index) match with exact frst_hash; "
            "ledger action digest is retained for comparison only"
        )
        group.attrs["involution_type"] = witness_record["involution_type"]
        group.attrs["candidate_id"] = witness_record["candidate_id"]
        group.attrs["matrix_id"] = witness_record.get(
            "matrix_id", witness_record.get("matrix_candidate_id", "")
        )
        group.attrs["ledger_accepted_witness_candidate_id"] = (
            ledger_entry.get("accepted_witness", {}).get("candidate_id", "")
        )
        ledger_torus_shift = ledger_entry.get("accepted_witness", {}).get("torus_shift", {})
        group.attrs["ledger_torus_shift_numerator_json"] = json.dumps(
            _jsonable(ledger_torus_shift.get("numerator"))
        )
        group.attrs["ledger_torus_shift_denominator"] = int(
            ledger_torus_shift.get("denominator", 0)
        )
        group.attrs["ledger_lambda_f"] = int(
            ledger_entry.get("accepted_witness", {}).get("lambda_f", -1)
        )
        group.attrs["witness_torus_shift_matches_ledger"] = bool(
            ledger_torus_shift.get("numerator") == torus_shift["numerator"]
            and int(ledger_torus_shift.get("denominator", -1)) == int(torus_shift["denominator"])
        )
        group.attrs["full_action_witness_identity_json"] = json.dumps(
            {
                "record_role": "full_action_witness",
                "polytope_id": witness_record["polytope_id"],
                "frst_hash": witness_record["frst_hash"],
                "frst_class_index": int(witness_record["frst_class_index"]),
                "matrix_id": witness_record.get(
                    "matrix_id", witness_record.get("matrix_candidate_id")
                ),
                "matrix_digest": witness_record["matrix_digest"],
                "candidate_id": witness_record["candidate_id"],
                "action_digest": expected_action_digest,
                "action_witness_digest": expected_witness_digest,
                "torus_shift": _jsonable(witness_record["torus_shift"]),
                "lambda_f": int(witness_record["lambda_f"]),
            },
            sort_keys=True,
        )
        group.attrs["source_ledger_path"] = ledger_zst_path
        group.attrs["source_ledger_sha256"] = ledger_sha256
        group.attrs["source_commit"] = source_commit
        group.attrs["source_tree_digest"] = run_manifest["source_tree_digest"]
        group.attrs["run_manifest_file_sha256"] = run_manifest["manifest_file_sha256"]
        group.attrs["run_manifest_payload_sha256"] = run_manifest["manifest_payload_sha256"]
        group.attrs["run_manifest_path"] = run_manifest["manifest_path"]
        group.attrs["cytools_version"] = cytools_version_string
        group.attrs["normalization_map_version"] = NORMALIZATION_MAP_VERSION
        group.attrs["orientifold_provenance_schema_version"] = (
            ORIENTIFOLD_PROVENANCE_SCHEMA_VERSION
        )
        group.attrs["bridge_schema_version"] = BRIDGE_SCHEMA_VERSION
        group.attrs["trilayer_gate"] = (
            "h11_minus=0; exact_action_h21_plus_status="
            + EXACT_ACTION_H21_PLUS_STATUS
        )
        if certification_info is not None:
            group.attrs["certified_trilayer_count"] = int(
                certification_info["certified_trilayer_count"]
            )
            ceiling = certification_info.get("conditional_ceiling")
            group.attrs["conditional_ceiling"] = -1 if ceiling is None else int(ceiling)
            group.attrs["pending_certification_json"] = json.dumps(
                _jsonable(certification_info.get("pending_certification", [])),
                sort_keys=True,
            )
            group.attrs["certification_status"] = (
                "certified_accepted_verified_orientifold_lambda_f_1"
            )
        bounded_certificate = witness_record.get("mpcp_certificate")
        if isinstance(bounded_certificate, dict):
            # Keep the complete bounded certificate, including fallback
            # reasons and component/H2 evidence, beside the exact action.  A
            # digest alone is not sufficient for replay or audit.
            verification = witness_record.get("mpcp_certificate_verification") or {}
            group.attrs["mpcp_certificate_status"] = verification.get(
                "status", "valid"
            )
            group.attrs["mpcp_certificate_schema_version"] = bounded_certificate.get(
                "certificate_schema_version", ""
            )
            group.attrs["mpcp_formula_schema_version"] = bounded_certificate.get(
                "formula_schema_version", ""
            )
            group.attrs["mpcp_certificate_digest"] = bounded_certificate.get(
                "certificate_digest", ""
            )
            group.attrs["mpcp_certificate_key_digest"] = bounded_certificate.get(
                "certificate_key_digest", ""
            )
            group.attrs["mpcp_certificate_json"] = json.dumps(
                _jsonable(bounded_certificate), sort_keys=True
            )
        file.attrs["orientifold_provenance_complete"] = True
        file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11", type=int, default=2, help="Physical h11 (Phase 1: 2 only).")
    parser.add_argument(
        "--parquet-dir", required=True,
        help="KS Parquet mirror directory (polytopes-4d-*-vertices.parquet).",
    )
    parser.add_argument(
        "--ledger-population-dir", required=True,
        help="Directory holding the preserved *.terminal-ledger.jsonl.summary.json.zst "
        "and SHA256SUMS.txt.",
    )
    parser.add_argument(
        "--ledger-name", required=True,
        help="Basename of the ledger summary *.zst file to use as the source of "
        "accepted h11_minus=0 classes, e.g. h11-2-cartier-nf.terminal-ledger.jsonl.summary.json.zst",
    )
    parser.add_argument(
        "--matrix-catalog",
        type=Path,
        help="zstd accepted-witness matrix catalog built from the immutable terminal ledger",
    )
    parser.add_argument(
        "--matrix-terminal-ledger",
        type=Path,
        help="immutable terminal JSONL used only to verify the matrix catalog source SHA",
    )
    parser.add_argument(
        "--mpcp-certificates",
        type=Path,
        default=None,
        help="JSON or JSON.zst bounded MPCP certificate input keyed by (polytope_id, frst_hash)",
    )
    parser.add_argument("--db-root", required=True, help="Destination database root directory.")
    parser.add_argument(
        "--stage", choices=("select", "full", "certify-pending"), default="full",
        help="'select' stops after Step A population selection/verification and "
        "prints the hard-gate evidence without writing any HDF5 files. "
        "'certify-pending' attempts to certify --pending-classes via a fresh "
        "orientifold-witness re-derivation and builds only the ones that succeed "
        "(see --pending-classes / --np-index-start).",
    )
    parser.add_argument(
        "--pending-classes", default=None,
        help="Comma-separated poly_index:class_index pairs to attempt certification "
        "on, for --stage certify-pending (e.g. '282:0,282:1,640:1').",
    )
    parser.add_argument(
        "--np-index-start", type=int, default=None,
        help="First np_index to assign to a newly-certified class, for --stage "
        "certify-pending. Must not collide with any np_index already used by this "
        "h11's prior build (e.g. one past the certified count from the main build).",
    )
    parser.add_argument(
        "--certified-trilayer-count", type=int, default=None,
        help="For --stage certify-pending: the certified count from the prior main "
        "build, recorded in provenance and used as the base for the running total.",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Optional compressed report output path; must end in .json.zst.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Optional zstd-compressed run-manifest path; defaults under --db-root.",
    )
    parser.add_argument(
        "--np-start", type=int, default=None,
        help="1-based np_index (assigned by sorted poly_index order over the full "
        "verified population) to start Step B/C from, inclusive. Lets one h11's "
        "Step B/C run be split across several bounded foreground invocations "
        "without any resume/overwrite ambiguity: np_index numbering is fixed by "
        "the full verified population regardless of this filter, so chunked runs "
        "never collide or renumber a class already written by an earlier chunk.",
    )
    parser.add_argument(
        "--np-end", type=int, default=None,
        help="1-based np_index to end Step B/C at, inclusive.",
    )
    parser.add_argument(
        "--expected-mismatch-classes", default=None,
        help="Comma-separated poly_index:class_index pairs (e.g. '282:0,282:1,640:1') "
        "naming a maintainer-reviewed, documented, EXACT set of classes that satisfy "
        "the paper's own h21_plus_zero trilayer diagnostic but are absent from the "
        "ledger's accepted_for_table_1 set (a certification gap, not a reproduction "
        "error). When given, Step A proceeds with the remaining verified classes as a "
        "'certified' subset ONLY if the observed mismatches are exactly this set "
        "(same count, same identities) -- any other or additional mismatch still "
        "hard-stops. Every written cyax.h5 records the certified count, the paper's "
        "target as a conditional ceiling, and this pending-certification list in its "
        "orientifold/ provenance group.",
    )
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parent.parent
    # Refuse to create even the manifest until the source tree is a known,
    # clean commit.  This prevents artifacts from being attributed to a
    # partially edited worktree.
    require_clean_git_source(str(package_root))
    population_preflight = run_population_preflight(package_root, args.h11)
    certificate_input = None
    if args.h11 in (4, 5) or args.mpcp_certificates is not None:
        certificate_input = load_mpcp_certificates(args.mpcp_certificates)
    # The identity-action diagnostic is validated, but general-L fixed-locus
    # Euler evidence is not. Stop before a scan, manifest, or HDF5 output.
    require_exact_action_h21_plus_validation()
    ledger_zst_path = os.path.join(args.ledger_population_dir, args.ledger_name)
    sha256sums_path = os.path.join(args.ledger_population_dir, "SHA256SUMS.txt")
    # Create one immutable manifest before any geometry artifact is finalized.
    # HDF5 provenance links to this digest so a file can be traced to the
    # exact source tree, command, environment, and ledger input.
    _ledger_sha_for_manifest = sha256_of_file(ledger_zst_path)
    manifest_path = args.manifest or default_run_manifest_path(args)
    run_manifest = build_run_manifest(
        args=args, package_root=package_root, ledger_path=ledger_zst_path,
        ledger_sha256=_ledger_sha_for_manifest, manifest_path=manifest_path,
    )
    manifest_reference = write_run_manifest(manifest_path, run_manifest)
    run_manifest["manifest_file_sha256"] = manifest_reference["manifest_file_sha256"]
    run_manifest["manifest_path"] = str(Path(manifest_path).resolve())

    if args.stage == "certify-pending":
        if not args.pending_classes or args.np_index_start is None:
            raise SystemExit(
                "--stage certify-pending requires --pending-classes and --np-index-start"
            )
        pending_keys = set()
        for token in args.pending_classes.split(","):
            token = token.strip()
            if not token:
                continue
            poly_text, class_text = token.split(":")
            pending_keys.add((int(poly_text), int(class_text)))
        _ledger_accepted, _ledger, ledger_sha256 = load_ledger_accepted_classes(
            ledger_zst_path, sha256sums_path
        )
        if args.matrix_catalog is None or args.matrix_terminal_ledger is None:
            raise RuntimeError("matrix catalog provenance arguments are required")
        _ledger_accepted, _matrix_provenance = enrich_accepted_witness_matrices(
            _ledger_accepted, args.matrix_catalog, args.matrix_terminal_ledger
        )
        certification_info = {
            "certified_trilayer_count": args.certified_trilayer_count,
            "conditional_ceiling": PAPER_TRILAYER_TARGETS.get(args.h11, {}).get(
                "h11_minus_zero_h21_plus_zero_orientifold_cys"
            ),
            "pending_certification": [
                {"poly_index": p, "class_index": c} for p, c in sorted(pending_keys)
            ],
        }
        certified_results, not_certified = certify_and_build_pending(
            args, pending_keys, ledger_zst_path, ledger_sha256, str(package_root),
            args.np_index_start, run_manifest, certification_info,
            mpcp_certificates=(
                None if certificate_input is None else certificate_input["certificates"]
            ),
        )
        report = {
            "h11": args.h11,
            "stage": "certify-pending",
            "run_manifest_file_sha256": run_manifest["manifest_file_sha256"],
            "run_manifest_payload_sha256": run_manifest["manifest_payload_sha256"],
            "run_manifest_path": run_manifest["manifest_path"],
            "pending_keys": sorted(list(pending_keys)),
            "certified_count": len(certified_results),
            "certified_results": _jsonable(certified_results),
            "not_certified": _jsonable(not_certified),
        }
        if population_preflight is not None:
            report["population_preflight"] = population_preflight
        if certificate_input is not None:
            report["bounded_mpcp_certificates"] = {
                key: value for key, value in certificate_input.items()
                if key != "certificates"
            }
        if args.report is not None:
            write_json_report(args.report, report)
            print(f"\nwrote {args.report}")
        return

    ledger_accepted, ledger, ledger_sha256 = load_ledger_accepted_classes(
        ledger_zst_path, sha256sums_path
    )
    if args.matrix_catalog is None or args.matrix_terminal_ledger is None:
        raise RuntimeError("matrix catalog provenance arguments are required")
    ledger_accepted, matrix_provenance = enrich_accepted_witness_matrices(
        ledger_accepted, args.matrix_catalog, args.matrix_terminal_ledger
    )
    print(
        f"ledger {args.ledger_name}: {len(ledger_accepted)} accepted_for_table_1 "
        f"(h11_minus=0) classes; sha256={ledger_sha256}",
        flush=True,
    )

    selected, mismatches, total_h21_plus_zero, live_population = select_and_verify_trilayer_population(
        args.h11, args.parquet_dir, ledger_accepted,
        mpcp_certificates=(
            None if certificate_input is None else certificate_input["certificates"]
        ),
    )
    population_audit = population_set_audit(live_population, ledger_accepted)

    target = PAPER_TRILAYER_TARGETS.get(args.h11, {}).get(
        "h11_minus_zero_h21_plus_zero_orientifold_cys"
    )
    report = {
        "h11": args.h11,
        "accepted_matrix_provenance": matrix_provenance,
        "run_manifest_file_sha256": run_manifest["manifest_file_sha256"],
        "run_manifest_payload_sha256": run_manifest["manifest_payload_sha256"],
        "run_manifest_path": run_manifest["manifest_path"],
        "ledger_name": args.ledger_name,
        "ledger_sha256": ledger_sha256,
        "ledger_accepted_h11_minus_zero_count": len(ledger_accepted),
        "total_h21_plus_zero_trilayer_classes_found": total_h21_plus_zero,
        "verified_selected_count": len(selected),
        "paper_table1_target": target,
        "mismatches": _jsonable(
            [
                {
                    "poly_index": m["poly_index"],
                    "class_index": m["class_index"],
                    "polytope_id": m["polytope_id"],
                    "frst_hash": m["frst_hash"],
                    "ledger_entry_present": m["ledger_entry"] is not None,
                    "ledger_frst_hash": (m["ledger_entry"] or {}).get("frst_hash"),
                    "exact_action_h21_evidence": m.get("exact_action_h21_evidence"),
                }
                for m in mismatches
            ]
        ),
        "population_set_audit": population_audit,
        "selected_classes": _jsonable(
            [
                {
                    "poly_index": r["poly_index"],
                    "class_index": r["class_index"],
                    "polytope_id": r["polytope_id"],
                    "frst_hash": r["frst_hash"],
                    "h21_plus_special_shift_diagnostic": r[
                        "h21_plus_special_shift_diagnostic"
                    ],
                    "matrix_id": r["witnesses"][0].get(
                        "matrix_id", r["witnesses"][0].get("matrix_candidate_id")
                    ),
                    "matrix_digest": r["witnesses"][0]["matrix_digest"],
                    "torus_shift": r["witnesses"][0]["torus_shift"],
                    "lambda_f": int(r["witnesses"][0]["lambda_f"]),
                    "candidate_id": r["witnesses"][0]["candidate_id"],
                    "action_digest": r.get("orientifold_action_digest"),
                    "action_witness_digest": r.get("action_witness_digest"),
                    "ledger_polytope_index": r["ledger_entry"]["polytope_index"],
                }
                for r in selected
            ]
        ),
    }
    if population_preflight is not None:
        report["population_preflight"] = population_preflight
    if certificate_input is not None:
        report["bounded_mpcp_certificates"] = {
            key: value for key, value in certificate_input.items()
            if key != "certificates"
        }

    print(f"\n=== Step A hard-gate evidence (h11={args.h11}) ===")
    print(f"paper Table 1 target trilayer class count: {target}")
    print(f"re-derived h21_plus_zero trilayer class count (full population scan): {total_h21_plus_zero}")
    print(f"verified against ledger (frst_hash match): {len(selected)}")
    print(f"mismatches (accepted here, absent/inconsistent in ledger): {len(mismatches)}")
    print(
        "bidirectional population audit: "
        f"live-minus-ledger={len(population_audit['live_minus_ledger'])}, "
        f"ledger-minus-live={len(population_audit['ledger_minus_live'])}, "
        f"equal={population_audit['equal']}"
    )
    for r in report["selected_classes"]:
        print(f"  poly_index={r['poly_index']:3d} class_index={r['class_index']} "
              f"polytope_id={r['polytope_id'][:40]}... frst_hash={r['frst_hash'][:16]}... "
              f"special_shift_h21_plus={r['h21_plus_special_shift_diagnostic']:.6f} "
              f"ledger_polytope_index={r['ledger_polytope_index']}")

    observed_mismatch_keys = {(m["poly_index"], m["class_index"]) for m in mismatches}
    expected_mismatch_keys = None
    if args.expected_mismatch_classes is not None:
        expected_mismatch_keys = set()
        for token in args.expected_mismatch_classes.split(","):
            token = token.strip()
            if not token:
                continue
            poly_text, class_text = token.split(":")
            expected_mismatch_keys.add((int(poly_text), int(class_text)))

    certification_gap_matches_exactly = (
        expected_mismatch_keys is not None
        and observed_mismatch_keys == expected_mismatch_keys
        and not population_audit["ledger_minus_live"]
        and not population_audit["live_duplicate_keys"]
        and not population_audit["ledger_duplicate_keys"]
        and target is not None
        and len(selected) + len(mismatches) == target
    )

    if mismatches:
        label = (
            "documented certification gap (maintainer-reviewed, --expected-mismatch-classes "
            "matches exactly)" if certification_gap_matches_exactly
            else "HARD GATE FAILURE: mismatches between re-derived and ledger population"
        )
        print(f"\n{label}.")
        for m in report["mismatches"]:
            print(f"  {m}")
    if target is not None and len(selected) != target and not certification_gap_matches_exactly:
        print(
            f"\nHARD GATE FAILURE: verified count {len(selected)} != paper target {target}."
        )

    report["certified_trilayer_count"] = len(selected)
    report["conditional_ceiling"] = target
    report["pending_certification"] = report["mismatches"] if certification_gap_matches_exactly else []

    if not certification_gap_matches_exactly and (
        mismatches
        or population_audit["ledger_minus_live"]
        or population_audit["live_duplicate_keys"]
        or population_audit["ledger_duplicate_keys"]
        or (target is not None and len(selected) != target)
    ):
        print("\nSTOPPING per Phase 1 hard-gate policy; no HDF5 files written.")
        if args.report is not None:
            write_json_report(args.report, report)
            print(f"\nwrote {args.report}")
        sys.exit(1)

    if certification_gap_matches_exactly:
        print(
            f"\nProceeding with {len(selected)} certified classes; "
            f"{len(mismatches)} pending-certification classes excluded from this "
            f"build and recorded in provenance (conditional ceiling {target})."
        )

    if args.stage == "select":
        if args.report is not None:
            write_json_report(args.report, report)
            print(f"\nwrote {args.report}")
        print("\n--stage=select: stopping after population selection/verification.")
        return

    certification_info = {
        "certified_trilayer_count": len(selected),
        "conditional_ceiling": target,
        "pending_certification": report["pending_certification"],
    }

    print("\n=== Step B/C: witness reconstruction + QCD-vol-40 evaluation + HDF5 write ===")
    build_results, classes_with_no_viable_qcd_divisor = build_database(
        args, selected, ledger_zst_path, ledger_sha256, str(package_root),
        run_manifest, certification_info
    )
    report["build_results"] = _jsonable(build_results)
    if args.report is not None:
        write_json_report(args.report, report)
        print(f"\nupdated {args.report}")

    if classes_with_no_viable_qcd_divisor:
        print(
            f"\nSTOPPING: {len(classes_with_no_viable_qcd_divisor)}/{len(selected)} "
            "ledger-verified classes reached zero viable QCD divisor assignments; "
            "see the report JSON for full per-class rejection reasons before proceeding."
        )
        sys.exit(1)


def _build_one_class(args, np_index, record, ledger_zst_path, ledger_sha256,
        source_commit, cytools_version_string, workdir, run_manifest,
        certification_info=None,
        witnesses=None):
    """Build every viable QCD-divisor cyax.h5 for one verified/certified class.

    Shared by `build_database` (the bulk 260/267-style loop) and
    `certify_and_build_pending` (the one-off per-class certification path):
    identical witness-reconstruction, orientifold_config, canonical_qcd
    dilation, and provenance writing either way. `witnesses` may be passed in
    already computed (certify_and_build_pending needs the result regardless
    of outcome, to report a concrete evidence-gap reason); when omitted it is
    computed here.
    """
    require_exact_action_h21_plus_validation()
    poly = record["poly"]
    triangulation = record["triangulation"]
    polytope_id = record["polytope_id"]
    frst_hash = record["frst_hash"]
    ledger_entry = record["ledger_entry"]

    print(
        f"\n[np_{np_index:07d}] poly_index={record['poly_index']} "
        f"class_index={record['class_index']} polytope_id={polytope_id[:40]}...",
        flush=True,
    )
    if witnesses is None:
        witnesses = record.get("witnesses") or find_exact_trilayer_witnesses(
            poly, triangulation, ledger_entry,
            frst_class_index=int(record["class_index"])
        )
    if not witnesses:
        raise RuntimeError(
            f"no accepted O3/O7 (lambda_f=1, h11_minus=0) witness re-derived for "
            f"poly_index={record['poly_index']} class_index={record['class_index']}; "
            "cannot build an orientifold_config for this ledger-accepted class"
        )
    witness = dict(witnesses[0])
    ledger_witness = ledger_entry.get("accepted_witness") or {}
    # Each accepted live action is its own canonical witness. The preserved
    # summary action is an arbitrary accepted representative and is retained
    # only as a non-gating cross-reference.
    witness["polytope_id"] = polytope_id
    witness["frst_hash"] = frst_hash
    witness["frst_class_index"] = int(record["class_index"])
    witness["matrix_digest"] = lattice_matrix_digest(witness["lattice_matrix"])
    witness["action_witness_digest"] = action_witness_digest(witness)
    exact_action_evidence = exact_action_h21_diagnostic(poly, triangulation, witness)
    if exact_action_evidence.get("status") != "validated":
        raise RuntimeError(
            "canonical witness exact-action Hodge evidence is unavailable: "
            + str(exact_action_evidence.get("reason"))
        )
    if exact_action_evidence["h21_plus"] != 0:
        raise RuntimeError(
            "canonical witness fails h21_plus=0: "
            f"h21_plus={exact_action_evidence['h21_plus']}"
        )
    print(
        f"  re-derived {len(witnesses)} accepted O3/O7 witness(es); using "
        f"candidate_id={witness['candidate_id'][:16]}... "
        f"(ledger accepted_witness candidate_id="
        f"{ledger_entry.get('accepted_witness', {}).get('candidate_id', '')[:16]}...)",
        flush=True,
    )

    orientifold_config_path = workdir / f"orientifold_np{np_index:07d}.json"
    orientifold_config_path.write_text(
        json.dumps(
            {
                "lattice_matrix": witness["lattice_matrix"],
                "involution_type": "O3/O7",
                "label": f"phase1-h11-{args.h11}-np{np_index:07d}-trilayer-witness",
                "torus_shift": witness["torus_shift"],
                "lambda_f": int(witness["lambda_f"]),
                "matrix_id": witness.get("matrix_candidate_id"),
                "matrix_digest": witness["matrix_digest"],
                "polytope_id": witness["polytope_id"],
                "frst_hash": witness["frst_hash"],
                "frst_class_index": int(witness["frst_class_index"]),
                "candidate_id": witness["candidate_id"],
                "action_witness_digest": witness["action_witness_digest"],
                "action_digest": orientifold_action_digest(witness),
                "canonical_action_required": True,
                "ledger_summary_action_cross_reference": witness.get(
                    "ledger_summary_action_cross_reference", {}
                ),
            }
        )
    )

    h11 = args.h11
    n_prime_divisors = h11 + 4
    written = []
    rejections = []
    for qcd_divisor_index in range(n_prime_divisors):
        cy_index = len(written) + 1
        target_path = mg.output_path(args.db_root, h11, np_index, cy_index)
        if os.path.exists(target_path):
            raise FileExistsError(
                f"refusing to overwrite an existing geometry artifact: {target_path}"
            )
        # Keep the final path absent until geometry construction, dense
        # materialization, and provenance validation all succeed.  An
        # interruption therefore leaves only a disposable sibling artifact.
        staging_path = f"{target_path}.building-{os.getpid()}-{time.time_ns()}"
        sampling_metadata = {
            "scheme": "phase1_trilayer_ledger_replay",
            "seed": 0,
            "proposal_seed": None,
            "source_ledger": args.ledger_name,
            "source_ledger_sha256": ledger_sha256,
        }
        seed = int.from_bytes(
            hashlib.sha256(
                f"{polytope_id}:{qcd_divisor_index}".encode("utf-8")
            ).digest()[:4],
            "big",
        )
        try:
            mg.generate_and_save_geometry(
                h11,
                triangulation.get_cy(),
                np.asarray(poly.points(), dtype=int),
                np.asarray(triangulation.simplices(), dtype=int),
                staging_path,
                1_000_000.0,
                100,
                1.0,
                1.0,
                25.0,
                40.0,
                "canonical_qcd",
                QCD_VOLUME_TARGET,
                qcd_divisor_index,
                "intersecting_d7",
                None,
                np.random.default_rng(seed),
                lambda message: None,
                poly=poly,
                triangulation=triangulation,
                polytope_id=polytope_id,
                sampling_metadata=sampling_metadata,
                ks_database_version=f"KS Parquet mirror: {args.parquet_dir}",
                orientifold_config=mg.load_orientifold(str(orientifold_config_path)),
                qed_selection_policy="uniform_eligible",
                qed_selection_seed=seed,
            )
            materialized = materialize_potential_and_kinetic_data(staging_path)
            expected_action_digest = orientifold_action_digest(witness)
            with h5py.File(staging_path, "r") as generated_file:
                generated_orientifold = generated_file["cytools/geometric/orientifold"]
                if generated_orientifold.attrs.get("action_digest") != expected_action_digest:
                    raise RuntimeError(
                        "generated orientifold action digest does not match the "
                        "ledger-selected witness"
                    )
            write_orientifold_provenance_group(
                staging_path,
                witness_record=witness,
                ledger_entry=ledger_entry,
                h11_plus_from_diagnostic=h11 - witness["h11_minus"],
                h11_minus_from_diagnostic=witness["h11_minus"],
                h21_plus_from_special_shift_diagnostic=record[
                    "h21_plus_special_shift_diagnostic"
                ],
                exact_action_h21_evidence=exact_action_evidence,
                ledger_zst_path=ledger_zst_path,
                ledger_sha256=ledger_sha256,
                source_commit=source_commit,
                cytools_version_string=cytools_version_string,
                run_manifest=run_manifest,
                action_digest=expected_action_digest,
                certification_info=certification_info,
            )
            with h5py.File(staging_path, "r") as completed_file:
                if not bool(completed_file.attrs.get("orientifold_provenance_complete", False)):
                    raise RuntimeError("staged geometry is missing complete provenance")
            with open(staging_path, "rb") as staged_stream:
                os.fsync(staged_stream.fileno())
            os.replace(staging_path, target_path)
            try:
                directory_fd = os.open(os.path.dirname(target_path), os.O_RDONLY)
            except OSError:  # pragma: no cover - platform-specific directory fsync
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except EXPECTED_REJECTIONS as exc:
            if os.path.exists(staging_path):
                os.unlink(staging_path)
            print(
                f"  qcd_divisor_index={qcd_divisor_index}: rejected "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            rejections.append(
                {"qcd_divisor_index": qcd_divisor_index,
                 "exception_type": type(exc).__name__, "reason": str(exc)}
            )
            continue
        except Exception:
            if os.path.exists(staging_path):
                os.unlink(staging_path)
            raise
        print(f"  qcd_divisor_index={qcd_divisor_index}: accepted -> {target_path}", flush=True)
        written.append(
            {
                "cy_index": cy_index,
                "qcd_divisor_index": qcd_divisor_index,
                "path": target_path,
                "size_bytes": os.path.getsize(target_path),
                "Q_shape": materialized["Q_shape"],
                "L_shape": materialized["L_shape"],
            }
        )
    if not written:
        print(
            f"  NO VIABLE QCD DIVISOR ASSIGNMENT for poly_index={record['poly_index']} "
            f"class_index={record['class_index']} (tried all {n_prime_divisors} prime "
            "toric divisor indices); recording rejections and continuing to the next "
            "class rather than aborting the whole run.",
            flush=True,
        )
    return {
        "np_index": np_index,
        "poly_index": record["poly_index"],
        "class_index": record["class_index"],
        "polytope_id": polytope_id,
        "rejections": rejections,
        "written": written,
    }


def build_database(args, selected, ledger_zst_path, ledger_sha256, package_root,
        run_manifest, certification_info=None):
    require_exact_action_h21_plus_validation()
    source_commit = git_commit(package_root)
    cytools_version_string = cytools_version()
    results = []
    with tempfile.TemporaryDirectory(prefix="orientifold_config_") as workdir_name:
        workdir = Path(workdir_name)
        for np_index, record in enumerate(
            sorted(selected, key=lambda r: r["poly_index"]), start=1
        ):
            if args.np_start is not None and np_index < args.np_start:
                continue
            if args.np_end is not None and np_index > args.np_end:
                continue
            results.append(_build_one_class(
                args, np_index, record, ledger_zst_path, ledger_sha256,
                source_commit, cytools_version_string, workdir, run_manifest,
                certification_info,
            ))

    classes_with_no_viable_qcd_divisor = [r for r in results if not r["written"]]
    print(
        f"\n=== Step B/C summary: {len(results) - len(classes_with_no_viable_qcd_divisor)}/"
        f"{len(results)} classes wrote at least one cyax.h5; "
        f"{sum(len(r['written']) for r in results)} cyax.h5 total ===",
        flush=True,
    )
    if classes_with_no_viable_qcd_divisor:
        print(
            "\nHARD BLOCKER: the following classes reached zero viable QCD divisor "
            "assignments across every prime toric divisor index (their rejection "
            "reasons are recorded in the report JSON):",
            flush=True,
        )
        for r in classes_with_no_viable_qcd_divisor:
            print(
                f"  np_index={r['np_index']} poly_index={r['poly_index']} "
                f"class_index={r['class_index']} polytope_id={r['polytope_id'][:40]}...",
                flush=True,
            )
    return results, classes_with_no_viable_qcd_divisor


def certify_and_build_pending(args, pending_keys, ledger_zst_path, ledger_sha256,
        package_root, np_index_start, run_manifest, certification_info,
        *, mpcp_certificates=None):
    """Attempt to certify specific classes and build the ones that succeed.

    `pending_keys` is a set of (poly_index, class_index) pairs -- normally the
    exact `mismatches` a prior Step A run already identified: classes that
    satisfy the paper's own h21_plus_zero trilayer diagnostic but have no
    entry in the ledger's accepted_for_table_1 set. Certification here means
    exactly what certified the other classes: an `enumerate_orientifold_candidates`
    re-derivation that finds a genuine `accepted_verified_orientifold`
    (`h11_minus=0`, `lambda_f=1`) witness -- via `find_accepted_o3o7_witness`,
    the same function `build_database` already relies on. A class that finds
    no such witness is left out, with the concrete terminal-status evidence
    recorded, not silently dropped or hand-promoted.
    """
    require_exact_action_h21_plus_validation()
    source_commit = git_commit(package_root)
    cytools_version_string = cytools_version()

    print(f"\n=== re-scanning h11={args.h11} population to recover the {len(pending_keys)} "
          "pending classes' live CYTools objects ===", flush=True)
    records = _selected_records_by_key(
        args, pending_keys, mpcp_certificates=mpcp_certificates
    )
    missing_keys = pending_keys - set(records.keys())
    if missing_keys:
        raise RuntimeError(
            f"the following pending classes were not found as h21_plus_zero trilayer "
            f"candidates on this re-scan (population changed?): {sorted(missing_keys)}"
        )

    certified_results = []
    not_certified = []
    with tempfile.TemporaryDirectory(prefix="orientifold_certify_") as workdir_name:
        workdir = Path(workdir_name)
        np_index = np_index_start
        for key in sorted(pending_keys):
            record = records[key]
            poly_index, class_index = key
            print(f"\n--- certifying poly_index={poly_index} class_index={class_index} ---",
                  flush=True)
            bounded_certificate = _lookup_mpcp_certificate(
                mpcp_certificates,
                polytope_id=record["polytope_id"],
                frst_hash=record["frst_hash"],
            )
            if bounded_certificate is None and args.h11 in (4, 5):
                raise RuntimeError(
                    "missing bounded MPCP certificate for pending identity "
                    f"{record['polytope_id']}::{record['frst_hash']}"
                )
            reconstruction_kwargs = {
                "frst_class_index": int(record["class_index"]),
            }
            if bounded_certificate is not None:
                reconstruction_kwargs.update(
                    mpcp_certificate=bounded_certificate,
                    source_record={"source": {
                        "polytope_id": record["polytope_id"],
                        "global_points": np.asarray(
                            record["poly"].points(), dtype=int
                        ).tolist(),
                    }},
                )
            witnesses = find_exact_trilayer_witnesses(
                record["poly"], record["triangulation"], record["ledger_entry"],
                **reconstruction_kwargs,
            )
            if not witnesses:
                status_counts = {}
                topology = _exact_trilayer_topology(
                    record["poly"], record["triangulation"]
                )
                reconstruction = reconstruct_trilayer_actions(
                    record["poly"], record["triangulation"], topology
                )
                for candidate in reconstruction["candidates"]:
                    status = candidate.get("terminal_status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                print(f"  NOT CERTIFIED: no accepted_verified_orientifold (h11_minus=0, "
                    f"lambda_f=1) witness found among {len(reconstruction['candidates'])} candidates; "
                    f"terminal_status counts: {status_counts}", flush=True)
                not_certified.append({
                    "poly_index": poly_index, "class_index": class_index,
                    "polytope_id": record["polytope_id"], "frst_hash": record["frst_hash"],
                    "candidate_terminal_status_counts": status_counts,
                    "candidate_count": len(reconstruction["candidates"]),
                    "reason": "no exact source-reconstructed trilayer action with "
                        "h21_plus=0 among the complete candidate set",
                })
                continue
            print(f"  CERTIFIED: {len(witnesses)} accepted O3/O7 witness(es) found", flush=True)
            result = _build_one_class(
                args, np_index, record, ledger_zst_path, ledger_sha256,
                source_commit, cytools_version_string, workdir, run_manifest,
                certification_info,
                witnesses=witnesses,
            )
            certified_results.append(result)
            np_index += 1

    print(f"\n=== certification summary: {len(certified_results)}/{len(pending_keys)} "
          f"of the pending classes certified ===", flush=True)
    for entry in not_certified:
        print(f"  NOT CERTIFIED poly_index={entry['poly_index']} "
              f"class_index={entry['class_index']}: {entry['reason']} "
              f"({entry['candidate_terminal_status_counts']})", flush=True)
    return certified_results, not_certified


def _selected_records_by_key(args, wanted_keys, *, mpcp_certificates=None):
    """Re-scan the population and return only the requested (poly,class) records.

    Returns a dict keyed by (poly_index, class_index) holding the same record
    shape `select_and_verify_trilayer_population` produces (live `poly`/
    `triangulation` CYTools objects included), regardless of whether that
    class is present in the ledger's accepted_for_table_1 set -- callers here
    already know these specific classes are absent from it and are trying to
    certify them independently.
    """
    require_exact_action_h21_plus_validation()
    ledger_zst_path = os.path.join(args.ledger_population_dir, args.ledger_name)
    sha256sums_path = os.path.join(args.ledger_population_dir, "SHA256SUMS.txt")
    ledger_accepted, _ledger, _sha = load_ledger_accepted_classes(
        ledger_zst_path, sha256sums_path
    )
    ledger_by_key = {
        (entry["polytope_id"], entry["frst_class_index"]): entry for entry in ledger_accepted
    }
    records = mg.load_mirror_polytopes(args.parquet_dir, h11=args.h11, limit=10**9, favorable=True)
    found = {}
    for poly_index, (poly, provenance) in enumerate(records):
        if not any(pi == poly_index for pi, _ci in wanted_keys):
            continue
        raw, classes = repro._frst_classes(poly)
        points = np.asarray(poly.points(), dtype=int)
        polytope_id = compute_polytope_id(points)
        for class_index, triangulation in enumerate(classes):
            if (poly_index, class_index) not in wanted_keys:
                continue
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            frst_hash = compute_triangulation_hash(simplices)
            bounded_certificate = _lookup_mpcp_certificate(
                mpcp_certificates, polytope_id=polytope_id, frst_hash=frst_hash
            )
            if bounded_certificate is None and args.h11 in (4, 5):
                raise RuntimeError(
                    "missing bounded MPCP certificate for pending identity "
                    f"{polytope_id}::{frst_hash}"
                )
            reconstruction_kwargs = {"frst_class_index": class_index}
            if bounded_certificate is not None:
                reconstruction_kwargs.update(
                    mpcp_certificate=bounded_certificate,
                    source_record={"source": {
                        "polytope_id": polytope_id,
                        "global_points": points.tolist(),
                    }},
                )
            witnesses = find_exact_trilayer_witnesses(
                poly, triangulation, ledger_by_key.get((polytope_id, class_index)),
                **reconstruction_kwargs,
            )
            if not witnesses:
                continue
            h21_diag = witnesses[0]["exact_action_h21_evidence"]
            key = (poly_index, class_index)
            ledger_entry = ledger_by_key.get((polytope_id, class_index))
            found[key] = {
                "poly_index": poly_index, "class_index": class_index,
                "polytope_id": polytope_id, "frst_hash": frst_hash,
                "h21_plus_special_shift_diagnostic": h21_diag["h21_plus"],
                "poly": poly,
                "triangulation": triangulation,
                "witnesses": witnesses,
                "ledger_entry": ledger_entry if ledger_entry is not None else {
                    "accepted_witness": {}, "polytope_index": poly_index,
                },
            }
    return found


if __name__ == "__main__":
    main()
