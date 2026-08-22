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
(``_frst_classes``), and for every polytope admitting the source trilayer
involution (``_trilayer_candidate``), evaluates the paper's own
``h^{2,1}_+(X,I)=0`` gate (``_h21_plus_zero_diagnostic``, eq. 4.51) per FRST
class.  A class is accepted into this database only when *both*:

1. the paper's own diagnostic marks it ``h21_plus_zero`` here, in this
   worktree's code, from a fresh CYTools re-instantiation; and
2. the ledger's own ``accepted_for_table_1`` entry for the same
   ``(polytope_id, frst_class_index)`` records the *same*
   ``frst_hash`` (``glimmers_raw_frst.compute_triangulation_hash``, an
   order-independent hash of the FRST's simplices) -- i.e. the two
   independently generated artifacts agree on which triangulation this is.

Any accepted-here class absent from the ledger's accepted set, or present
with a different ``frst_hash``, is a hard-gate failure: the script stops and
reports the discrepancy rather than silently proceeding (see ``--stage``).

Known schema-version note
--------------------------
The ledger under ``data/orientifold_h11_2_3_population_20260820/`` was built
by a worktree at ``inherited_orientifold_candidates`` candidate schema 2.5
(``polytope_normal_form_id`` normal-form geometry keying).  This package
checkout is schema 2.0 and does not carry that commit.  ``compute_polytope_id``
and ``compute_triangulation_hash`` are documented as unchanged across that
gap, so the ``frst_hash`` cross-check above is schema-independent and is the
hard gate this script enforces.  The per-candidate ``matrix_id`` hash the
ledger's ``accepted_witness`` carries is *not* re-derived byte-for-byte
here (enumeration order of ``enumerate_polytope_involutions`` is not
guaranteed stable across that schema gap); instead this script independently
re-derives its own accepted O3/O7 witness (an ``accepted_verified_orientifold``
candidate with ``h11_minus=0`` and ``lambda_f=1``) via
``inherited_orientifold_candidates.enumerate_orientifold_candidates`` run
fresh, in this worktree, against the frst_hash-verified triangulation, and
records both the ledger's and this run's ``torus_shift``/``lambda_f`` for
comparison in the ``orientifold/`` provenance group.

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
import reproduce_fuzzy_axions_h11_4 as repro
from glimmers_raw_frst import compute_polytope_id, compute_triangulation_hash
from qed_divisor_assignment import (
    NORMALIZATION_MAP_VERSION,
    QCD_VOLUME_TARGET,
    record_potential_match,
)

BRIDGE_SCHEMA_VERSION = "cyaxiverse-phase1-orientifold-axiverse-bridge-1.0"
ORIENTIFOLD_PROVENANCE_SCHEMA_VERSION = "cyaxiverse-phase1-orientifold-provenance-1.0"

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


def git_commit(repo_root: str) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - best effort provenance only
        return "unknown"


def cytools_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("cytools")
    except Exception:  # pragma: no cover - best effort provenance only
        return "unknown"


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


def select_and_verify_trilayer_population(h11, parquet_dir, ledger_accepted):
    """Re-derive the h21_plus=0 trilayer population and verify against the ledger.

    Returns ``(selected, mismatches, total_h21_plus_zero_count)`` where
    ``selected`` is a list of dicts (one per verified accepted class) each
    holding the live CYTools ``poly``/``triangulation`` objects plus identity
    and diagnostic evidence, and ``mismatches`` lists any class accepted here
    but absent from, or hash-inconsistent with, the ledger.
    """
    ledger_by_key = {
        (entry["polytope_id"], entry["frst_class_index"]): entry
        for entry in ledger_accepted
    }

    records = mg.load_mirror_polytopes(parquet_dir, h11=h11, limit=10**9, favorable=True)
    print(f"loaded {len(records)} favorable h11={h11} polytopes from the KS mirror", flush=True)

    selected = []
    mismatches = []
    total_h21_plus_zero = 0
    for poly_index, (poly, provenance) in enumerate(records):
        raw, classes = repro._frst_classes(poly)
        trilayer = repro._trilayer_candidate(poly)
        if trilayer is None:
            continue
        points = np.asarray(poly.points(), dtype=int)
        polytope_id = compute_polytope_id(points)
        for class_index, triangulation in enumerate(classes):
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            frst_hash = compute_triangulation_hash(simplices)
            h21_diag = repro._h21_plus_zero_diagnostic(poly, triangulation, trilayer["p0"])
            if h21_diag["status"] != "h21_plus_zero":
                continue
            total_h21_plus_zero += 1
            key = (polytope_id, class_index)
            ledger_entry = ledger_by_key.get(key)
            record = {
                "poly_index": poly_index,
                "class_index": class_index,
                "polytope_id": polytope_id,
                "frst_hash": frst_hash,
                "h21_plus": h21_diag["h21_plus"],
                "provenance": provenance,
            }
            if ledger_entry is None or ledger_entry["frst_hash"] != frst_hash:
                record["ledger_entry"] = ledger_entry
                mismatches.append(record)
                continue
            record["poly"] = poly
            record["triangulation"] = triangulation
            record["p0"] = trilayer["p0"]
            record["ledger_entry"] = ledger_entry
            selected.append(record)
        print(
            f"  polytope {poly_index + 1}/{len(records)}: trilayer, "
            f"{len(classes)} FRST class(es), running h21_plus_zero total="
            f"{total_h21_plus_zero}",
            flush=True,
        )
    return selected, mismatches, total_h21_plus_zero


def find_accepted_o3o7_witness(poly, triangulation):
    """Re-derive an accepted O3/O7 (lambda_f=1, h11_minus=0) witness.

    Returns the sorted list of matching ``enumerate_orientifold_candidates``
    records (sorted by ``candidate_id`` for determinism); the caller uses the
    first as the canonical witness.
    """
    cy = triangulation.get_cy()
    topology = dict(mg.extract_topology(cy, triangulation))
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
    accepted.sort(key=lambda record: record["candidate_id"])
    return accepted


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

        prime_charges = np.asarray(basis_matrix[:, prime_labels].T, dtype=np.int64)
        prime_divisor_volumes = prime_charges @ divisor_volumes

        reference = {
            "h11": h11,
            "effective_cone": effective_cone,
            "kappa": kappa,
            "tip": tip,
            "basis_matrix": basis_matrix,
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
    h21_plus_from_diagnostic,
    ledger_zst_path,
    ledger_sha256,
    source_commit,
    cytools_version_string,
    certification_info=None,
):
    """Append the top-level ``orientifold/`` provenance group the plan requires."""
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
        torus_shift = witness_record["torus_shift"]
        group.create_dataset(
            "torus_shift_numerator",
            data=np.asarray(torus_shift["numerator"], dtype=np.int64),
        )
        group.create_dataset(
            "torus_shift_denominator", data=int(torus_shift["denominator"])
        )
        group.create_dataset("lambda_f", data=int(witness_record["lambda_f"]))
        group.create_dataset("h11_minus", data=int(witness_record["h11_minus"]))
        group.create_dataset("h11_plus", data=int(witness_record["h11_plus"]))
        group.create_dataset(
            "h11_minus_diagnostic", data=int(h11_minus_from_diagnostic)
        )
        group.create_dataset(
            "h11_plus_diagnostic", data=int(h11_plus_from_diagnostic)
        )
        group.create_dataset(
            "h21_plus", data=float(h21_plus_from_diagnostic)
        )
        group.attrs["polytope_id"] = witness_record["polytope_id"]
        group.attrs["polytope_normal_form_id"] = ledger_entry.get(
            "polytope_normal_form_id", ""
        )
        group.attrs["polytope_normal_form_id_source"] = (
            "copied_from_ledger_not_independently_recomputed_in_this_worktree_"
            "candidate_schema_2.0"
        )
        group.attrs["frst_hash"] = witness_record["frst_hash"]
        group.attrs["frst_hash_verified_against_ledger"] = True
        group.attrs["witness_source"] = (
            "re_derived_fresh_in_this_worktree_candidate_schema_2.0_not_taken_"
            "from_ledger_candidate_schema_2.5; linkage to the preserved ledger "
            "entry is the (polytope_id, frst_class_index) match with an exact "
            "frst_hash agreement (see frst_hash_verified_against_ledger and "
            "ledger_accepted_witness_candidate_id)"
        )
        group.attrs["involution_type"] = witness_record["involution_type"]
        group.attrs["candidate_id"] = witness_record["candidate_id"]
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
        group.attrs["source_ledger_path"] = ledger_zst_path
        group.attrs["source_ledger_sha256"] = ledger_sha256
        group.attrs["source_commit"] = source_commit
        group.attrs["cytools_version"] = cytools_version_string
        group.attrs["normalization_map_version"] = NORMALIZATION_MAP_VERSION
        group.attrs["orientifold_provenance_schema_version"] = (
            ORIENTIFOLD_PROVENANCE_SCHEMA_VERSION
        )
        group.attrs["bridge_schema_version"] = BRIDGE_SCHEMA_VERSION
        group.attrs["trilayer_gate"] = "h11_minus=0 and h21_plus=0 (arXiv:2412.12012 tab:ScanData)"
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
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report output path.")
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
    ledger_zst_path = os.path.join(args.ledger_population_dir, args.ledger_name)
    sha256sums_path = os.path.join(args.ledger_population_dir, "SHA256SUMS.txt")

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
            args.np_index_start, certification_info,
        )
        report = {
            "h11": args.h11,
            "stage": "certify-pending",
            "pending_keys": sorted(list(pending_keys)),
            "certified_count": len(certified_results),
            "certified_results": _jsonable(certified_results),
            "not_certified": _jsonable(not_certified),
        }
        if args.report is not None:
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"\nwrote {args.report}")
        return

    ledger_accepted, ledger, ledger_sha256 = load_ledger_accepted_classes(
        ledger_zst_path, sha256sums_path
    )
    print(
        f"ledger {args.ledger_name}: {len(ledger_accepted)} accepted_for_table_1 "
        f"(h11_minus=0) classes; sha256={ledger_sha256}",
        flush=True,
    )

    selected, mismatches, total_h21_plus_zero = select_and_verify_trilayer_population(
        args.h11, args.parquet_dir, ledger_accepted
    )

    target = PAPER_TRILAYER_TARGETS.get(args.h11, {}).get(
        "h11_minus_zero_h21_plus_zero_orientifold_cys"
    )
    report = {
        "h11": args.h11,
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
                }
                for m in mismatches
            ]
        ),
        "selected_classes": _jsonable(
            [
                {
                    "poly_index": r["poly_index"],
                    "class_index": r["class_index"],
                    "polytope_id": r["polytope_id"],
                    "frst_hash": r["frst_hash"],
                    "h21_plus": r["h21_plus"],
                    "ledger_polytope_index": r["ledger_entry"]["polytope_index"],
                }
                for r in selected
            ]
        ),
    }

    print(f"\n=== Step A hard-gate evidence (h11={args.h11}) ===")
    print(f"paper Table 1 target trilayer class count: {target}")
    print(f"re-derived h21_plus_zero trilayer class count (full population scan): {total_h21_plus_zero}")
    print(f"verified against ledger (frst_hash match): {len(selected)}")
    print(f"mismatches (accepted here, absent/inconsistent in ledger): {len(mismatches)}")
    for r in report["selected_classes"]:
        print(f"  poly_index={r['poly_index']:3d} class_index={r['class_index']} "
              f"polytope_id={r['polytope_id'][:40]}... frst_hash={r['frst_hash'][:16]}... "
              f"h21_plus={r['h21_plus']:.6f} ledger_polytope_index={r['ledger_polytope_index']}")

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

    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.report}")

    if not certification_gap_matches_exactly and (
        mismatches or (target is not None and len(selected) != target)
    ):
        print("\nSTOPPING per Phase 1 hard-gate policy; no HDF5 files written.")
        sys.exit(1)

    if certification_gap_matches_exactly:
        print(
            f"\nProceeding with {len(selected)} certified classes; "
            f"{len(mismatches)} pending-certification classes excluded from this "
            f"build and recorded in provenance (conditional ceiling {target})."
        )

    if args.stage == "select":
        print("\n--stage=select: stopping after population selection/verification.")
        return

    certification_info = {
        "certified_trilayer_count": len(selected),
        "conditional_ceiling": target,
        "pending_certification": report["pending_certification"],
    }

    print("\n=== Step B/C: witness reconstruction + QCD-vol-40 evaluation + HDF5 write ===")
    build_results, classes_with_no_viable_qcd_divisor = build_database(
        args, selected, ledger_zst_path, ledger_sha256, str(package_root), certification_info
    )
    report["build_results"] = _jsonable(build_results)
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nupdated {args.report}")

    if classes_with_no_viable_qcd_divisor:
        print(
            f"\nSTOPPING: {len(classes_with_no_viable_qcd_divisor)}/{len(selected)} "
            "ledger-verified classes reached zero viable QCD divisor assignments; "
            "see the report JSON for full per-class rejection reasons before proceeding."
        )
        sys.exit(1)


def _build_one_class(args, np_index, record, ledger_zst_path, ledger_sha256,
        source_commit, cytools_version_string, workdir, certification_info=None,
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
        witnesses = find_accepted_o3o7_witness(poly, triangulation)
    if not witnesses:
        raise RuntimeError(
            f"no accepted O3/O7 (lambda_f=1, h11_minus=0) witness re-derived for "
            f"poly_index={record['poly_index']} class_index={record['class_index']}; "
            "cannot build an orientifold_config for this ledger-accepted class"
        )
    witness = dict(witnesses[0])
    witness["polytope_id"] = polytope_id
    witness["frst_hash"] = frst_hash
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
                target_path,
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
        except EXPECTED_REJECTIONS as exc:
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
        print(f"  qcd_divisor_index={qcd_divisor_index}: accepted -> {target_path}", flush=True)
        materialized = materialize_potential_and_kinetic_data(target_path)
        write_orientifold_provenance_group(
            target_path,
            witness_record=witness,
            ledger_entry=ledger_entry,
            h11_plus_from_diagnostic=h11 - witness["h11_minus"],
            h11_minus_from_diagnostic=witness["h11_minus"],
            h21_plus_from_diagnostic=record["h21_plus"],
            ledger_zst_path=ledger_zst_path,
            ledger_sha256=ledger_sha256,
            source_commit=source_commit,
            cytools_version_string=cytools_version_string,
            certification_info=certification_info,
        )
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
        certification_info=None):
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
                source_commit, cytools_version_string, workdir, certification_info,
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
        package_root, np_index_start, certification_info):
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
    source_commit = git_commit(package_root)
    cytools_version_string = cytools_version()

    print(f"\n=== re-scanning h11={args.h11} population to recover the {len(pending_keys)} "
          "pending classes' live CYTools objects ===", flush=True)
    records = _selected_records_by_key(args, pending_keys)
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
            witnesses = find_accepted_o3o7_witness(record["poly"], record["triangulation"])
            if not witnesses:
                candidates = ioc.enumerate_orientifold_candidates(
                    record["poly"], record["triangulation"],
                    dict(mg.extract_topology(record["triangulation"].get_cy(),
                        record["triangulation"])),
                )
                status_counts = {}
                for candidate in candidates:
                    status = candidate.get("terminal_status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                print(f"  NOT CERTIFIED: no accepted_verified_orientifold (h11_minus=0, "
                      f"lambda_f=1) witness found among {len(candidates)} candidates; "
                      f"terminal_status counts: {status_counts}", flush=True)
                not_certified.append({
                    "poly_index": poly_index, "class_index": class_index,
                    "polytope_id": record["polytope_id"], "frst_hash": record["frst_hash"],
                    "candidate_terminal_status_counts": status_counts,
                    "candidate_count": len(candidates),
                    "reason": "no accepted_verified_orientifold witness with h11_minus=0 "
                        "and lambda_f=1 among the full re-derived candidate set",
                })
                continue
            print(f"  CERTIFIED: {len(witnesses)} accepted O3/O7 witness(es) found", flush=True)
            result = _build_one_class(
                args, np_index, record, ledger_zst_path, ledger_sha256,
                source_commit, cytools_version_string, workdir, certification_info,
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


def _selected_records_by_key(args, wanted_keys):
    """Re-scan the population and return only the requested (poly,class) records.

    Returns a dict keyed by (poly_index, class_index) holding the same record
    shape `select_and_verify_trilayer_population` produces (live `poly`/
    `triangulation` CYTools objects included), regardless of whether that
    class is present in the ledger's accepted_for_table_1 set -- callers here
    already know these specific classes are absent from it and are trying to
    certify them independently.
    """
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
        trilayer = repro._trilayer_candidate(poly)
        if trilayer is None:
            continue
        points = np.asarray(poly.points(), dtype=int)
        polytope_id = compute_polytope_id(points)
        for class_index, triangulation in enumerate(classes):
            if (poly_index, class_index) not in wanted_keys:
                continue
            simplices = np.asarray(triangulation.simplices(), dtype=int)
            frst_hash = compute_triangulation_hash(simplices)
            h21_diag = repro._h21_plus_zero_diagnostic(poly, triangulation, trilayer["p0"])
            if h21_diag["status"] != "h21_plus_zero":
                continue
            key = (poly_index, class_index)
            ledger_entry = ledger_by_key.get((polytope_id, class_index))
            found[key] = {
                "poly_index": poly_index, "class_index": class_index,
                "polytope_id": polytope_id, "frst_hash": frst_hash,
                "h21_plus": h21_diag["h21_plus"], "poly": poly,
                "triangulation": triangulation,
                "ledger_entry": ledger_entry if ledger_entry is not None else {
                    "accepted_witness": {}, "polytope_index": poly_index,
                },
            }
    return found


if __name__ == "__main__":
    main()
