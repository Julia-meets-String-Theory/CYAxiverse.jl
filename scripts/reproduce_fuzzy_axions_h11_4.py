#!/usr/bin/env python3
"""Reproduce the Table 1 (tab:ScanData) population benchmarks of
arXiv:2412.12012 for a given ``--h11`` (developed and most thoroughly
validated against h11=4; targets are also recorded for h11=3 and h11=5).

This is a source-matched audit driver, not a replacement for the production
geometry generator.  It reads the same KS Parquet mirror as the generator,
counts FRST classes using the paper's two-face equivalence relation, and
records the special trilayer involution together with two independent
per-FRST-class diagnostics: the frozen-conifold smoothness check (Moritz
eqs. 4.48-4.50, a separate orientifold-background smoothness condition)
and the h^{2,1}_+(X,I)=0 Hodge-number identity (eq. 4.51), the latter
being the actual gate for the paper's h11_minus_zero_h21_plus_zero
population.

The orientifold and model stages are intentionally represented as explicit
diagnostic records.  A count is labelled ``exact`` only when the implementation
has the corresponding source criterion and complete input evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
import time
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError:  # Candidate-only replay does not perform HDF5 model writes.
    h5py = None
import numpy as np

from generate_geometric_data_multitriangulation import (
    DIVISOR_VOLUME_TOLERANCE,
    configure_mosek_license,
    evaluate_kaehler_point,
    extract_topology,
    load_mirror_polytopes,
    sample_stretched_kaehler_points,
)
from geometry_charge_conventions import canonicalize_unique_charge_rows
from inherited_orientifold_candidates import (
    CANDIDATE_SCHEMA_VERSION,
    _ambient_intersection_tensor,
    _triangulation_cones,
    enumerate_orientifold_candidates,
    facets_with_non_smooth_cones,
    identity_fixed_surface_n_s_table,
)
from orientifold_terminal_ledger import TerminalLedgerWriter
from trilayer_involutions import (
    enumerate_source_trilayer_candidates,
    reconstruct_trilayer_actions,
)
from orientifold_population_preflight import run_population_preflight


PAPER_TARGETS_BY_H11 = {
    2: {
        "favorable_polytopes": 36,
        "frst_classes": 36,
        "inherited_orientifold_cys": 32,
        "h11_minus_zero_orientifold_cys": 32,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 11,
        "models": 2,
    },
    3: {
        "favorable_polytopes": 243,
        "frst_classes": 274,
        "inherited_orientifold_cys": 253,
        "h11_minus_zero_orientifold_cys": 253,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 66,
        "models": 263,
    },
    4: {
        "favorable_polytopes": 1185,
        "frst_classes": 1760,
        "inherited_orientifold_cys": 1559,
        "h11_minus_zero_orientifold_cys": 1554,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 267,
        "models": 3348,
    },
    5: {
        "favorable_polytopes": 4897,
        "frst_classes": 11713,
        "inherited_orientifold_cys": 9530,
        "h11_minus_zero_orientifold_cys": 9459,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 1033,
        "models": 29898,
    },
    6: {
        "favorable_polytopes": 16608,
        "frst_classes": 74503,
        "inherited_orientifold_cys": 54274,
        "h11_minus_zero_orientifold_cys": 53810,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 3623,
        "models": 231676,
    },
    7: {
        "favorable_polytopes": 48221,
        "frst_classes": 467283,
        "inherited_orientifold_cys": 292158,
        "h11_minus_zero_orientifold_cys": 289684,
        "h11_minus_zero_h21_plus_zero_orientifold_cys": 12253,
        "models": 1565380,
    },
}

REPRODUCTION_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1"
DEPRECATED_COUNT_ALIASES = {
    "counts.source_vertex_evidence_inherited_orientifold_cys": (
        "counts.source_evidence_inherited_orientifold_cys"
    ),
    "counts.source_vertex_evidence_h11_minus_zero_orientifold_cys": (
        "counts.source_evidence_h11_minus_zero_orientifold_cys"
    ),
}
UNAVAILABLE = "unavailable"


def _population_completion_status(
    h11,
    loaded_favorable_polytope_count,
    *,
    explicit_basis=None,
):
    """Return a conservative completeness status for a loaded population.

    Known Hodge numbers use the favorable-polytope target recorded in Table 1.
    An unknown Hodge number remains incomplete unless the caller supplies an
    explicit basis with ``favorable_polytopes`` and ``label`` entries.
    """
    loaded_count = int(loaded_favorable_polytope_count)
    targets = PAPER_TARGETS_BY_H11.get(int(h11))
    if targets is not None:
        expected_count = int(targets["favorable_polytopes"])
        basis = "table_1_favorable_polytope_target"
        if loaded_count == expected_count:
            reason = (
                f"loaded {loaded_count} favorable polytopes, matching the "
                f"Table 1 target for h11={int(h11)}"
            )
            complete = True
        else:
            reason = (
                f"loaded {loaded_count} favorable polytopes, but the Table 1 "
                f"target for h11={int(h11)} is {expected_count}"
            )
            complete = False
    elif explicit_basis is not None:
        try:
            expected_count = int(explicit_basis["favorable_polytopes"])
            basis_label = str(explicit_basis["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "explicit_basis must provide 'favorable_polytopes' and 'label'"
            ) from exc
        basis = f"explicit:{basis_label}"
        complete = loaded_count == expected_count
        if complete:
            reason = (
                f"loaded {loaded_count} favorable polytopes, matching the "
                f"explicit basis '{basis_label}'"
            )
        else:
            reason = (
                f"loaded {loaded_count} favorable polytopes, but explicit basis "
                f"'{basis_label}' requires {expected_count}"
            )
    else:
        expected_count = None
        basis = "no_explicit_basis"
        complete = False
        reason = (
            f"h11={int(h11)} has no Table 1 favorable-polytope target and no "
            "explicit completion basis"
        )

    return {
        "complete": complete,
        "basis": basis,
        "reason": reason,
        "loaded_favorable_polytopes": loaded_count,
        "expected_favorable_polytopes": expected_count,
    }


def _repository_root():
    return Path(__file__).resolve().parent.parent


def _run_repository_command(root, *command):
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _working_tree_identity(root):
    """Hash tracked diffs and untracked file contents for dirty provenance."""
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if diff.returncode != 0 or untracked.returncode != 0:
        return None

    diff_digest = hashlib.sha256(diff.stdout).hexdigest()
    untracked_digest = hashlib.sha256()
    names = [name for name in untracked.stdout.split(b"\0") if name]
    for name in sorted(names):
        path = root / os.fsdecode(name)
        try:
            contents = path.read_bytes()
        except OSError:
            return None
        untracked_digest.update(name)
        untracked_digest.update(b"\0")
        untracked_digest.update(contents)
    tree = _run_repository_command(root, "git", "rev-parse", "HEAD^{tree}")
    return {
        "tree_sha256": tree,
        "diff_sha256": diff_digest,
        "untracked_sha256": untracked_digest.hexdigest(),
        "untracked_file_count": len(names),
    }


def _package_version(root):
    project_path = root / "Project.toml"
    try:
        project_text = project_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return UNAVAILABLE
    match = re.search(r"^\s*version\s*=\s*[\"']([^\"']+)[\"']\s*$", project_text, re.MULTILINE)
    return match.group(1) if match else UNAVAILABLE


def _distribution_version(names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - broken environment metadata
            return UNAVAILABLE
    return UNAVAILABLE


def _normalize_cli_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_cli_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cli_value(item) for item in value]
    return value


def _normalized_cli_arguments(args):
    return {
        key: _normalize_cli_value(value)
        for key, value in sorted(vars(args).items())
    }


def _parquet_partition_sort_key(path):
    match = re.search(r"polytopes-4d-(\d+)-vertices\.parquet$", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def _matching_parquet_paths(parquet_dir):
    try:
        directory = Path(parquet_dir).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return [], "unavailable", "could not resolve parquet directory"
    if not directory.is_dir():
        return [], "unavailable", "parquet directory does not exist"
    try:
        paths = sorted(
            directory.glob("polytopes-4d-*-vertices.parquet"),
            key=_parquet_partition_sort_key,
        )
    except OSError:
        return [], "unavailable", "could not enumerate parquet partitions"
    if not paths:
        return [], "unavailable", "no matching parquet partitions"
    return paths, "available", None


def _scanned_parquet_paths(parquet_dir, records, limit=None):
    paths, status, reason = _matching_parquet_paths(parquet_dir)
    if status != "available":
        return paths, status, reason

    record_paths = set()
    for record in records:
        try:
            source = record[1]
            record_path = source.get("parquet_file")
            if record_path is not None:
                record_paths.add(Path(record_path).expanduser().resolve())
        except (IndexError, OSError, RuntimeError, TypeError, ValueError):
            continue
    if limit is None or len(records) < int(limit) or not record_paths:
        return paths, "all_matching_partitions", None

    matching_record_indices = [
        index for index, path in enumerate(paths) if path in record_paths
    ]
    if not matching_record_indices:
        return paths, "all_matching_partitions", None
    last_scanned_index = max(matching_record_indices)
    return paths[: last_scanned_index + 1], "loader_record_boundary", None


def _hash_file(path):
    digest = hashlib.sha256()
    try:
        size_bytes = path.stat().st_size
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, ValueError):
        return {
            "path": str(path),
            "size_bytes": None,
            "sha256": None,
            "status": UNAVAILABLE,
        }
    return {
        "path": str(path),
        "size_bytes": int(size_bytes),
        "sha256": digest.hexdigest(),
        "status": "available",
    }


def _parquet_input_manifest(parquet_dir, records, limit=None):
    paths, scan_basis, scan_reason = _scanned_parquet_paths(
        parquet_dir,
        records,
        limit=limit,
    )
    partitions = [_hash_file(path) for path in paths]
    if scan_reason is not None:
        status = UNAVAILABLE
    elif any(partition["status"] == UNAVAILABLE for partition in partitions):
        status = "partial"
    else:
        status = "complete"
    return {
        "status": status,
        "scan_basis": scan_basis,
        "scan_reason": scan_reason,
        "directory": str(Path(parquet_dir).expanduser())
        if parquet_dir is not None
        else UNAVAILABLE,
        "pattern": "polytopes-4d-*-vertices.parquet",
        "partitions": partitions,
    }


def _run_provenance(args, records):
    root = _repository_root()
    commit = _run_repository_command(root, "git", "rev-parse", "HEAD")
    dirty_output = _run_repository_command(
        root,
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if dirty_output is None:
        git_dirty = UNAVAILABLE
    else:
        git_dirty = bool(dirty_output)
    input_manifest = _parquet_input_manifest(
        args.parquet_dir,
        records,
        limit=args.limit,
    )
    return {
        "source_commit": commit or UNAVAILABLE,
        "git_dirty": git_dirty,
        "working_tree_identity": (
            _working_tree_identity(root) if git_dirty is True else None
        ),
        "package_version": _package_version(root),
        "runtime_versions": {
            "python": platform.python_version(),
            "cytools": _distribution_version(("cytools",)),
            "numpy": _distribution_version(("numpy",)),
            "scipy": _distribution_version(("scipy",)),
            "sympy": _distribution_version(("sympy",)),
            "pyarrow": _distribution_version(("pyarrow",)),
        },
        "cli_arguments": _normalized_cli_arguments(args),
        "input_partition_manifest_version": "cyaxiverse-parquet-input-manifest-1",
        "input_partition_manifest": input_manifest,
    }


def _as_int_rows(values: Any) -> np.ndarray:
    rows = np.asarray(values, dtype=int)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"expected an (n, 4) integer array, got {rows.shape}")
    return rows


def _frst_classes(poly):
    """Return one representative per paper FRST class.

    The paper identifies triangulations when their restrictions to all
    two-faces agree up to a polytope automorphism.  CYTools exposes this as
    ``Triangulation.is_equivalent(..., on_faces_dim=2)``.
    """

    raw = poly.all_triangulations(
        only_fine=True,
        only_regular=True,
        only_star=True,
        as_list=True,
    )
    representatives = []
    for triangulation in raw:
        if not any(
            triangulation.is_equivalent(reference, on_faces_dim=2)
            for reference in representatives
        ):
            representatives.append(triangulation)
    return raw, representatives


def _trilayer_candidate(poly):
    """Return the first exact source-authorized trilayer action.

    The population driver retains this compatibility shape, but the source
    reconstruction itself is performed by
    :func:`trilayer_involutions.enumerate_source_trilayer_candidates`, which
    examines every primal vertex and records all structural terminal reasons.
    The separate exact-action path consumes the complete candidate manifest;
    this helper is only the historical single-candidate API.
    """
    records = enumerate_source_trilayer_candidates(poly)
    for record in records:
        if record.get("terminal_status") != "structurally_reconstructed":
            continue
        action = record["action"]
        shift = action["torus_shift"]
        return {
            "p0": record["p0"],
            "q0": record["q0"],
            "lattice_matrix": action["lattice_matrix"],
            "torus_shift_numerator": shift["numerator"],
            "torus_shift_denominator": shift["denominator"],
            "lambda_f": int(action["lambda_f"]),
            "criterion": "Moritz eqs. (4.64)-(4.66), exact source-gauge reconstruction",
            "reconstruction_schema_version": record["schema_version"],
            "reconstruction_rule_version": record["reconstruction_rule_version"],
            "action_digest": record["action_digest"],
        }
    return None


def _identity_torus_actions(poly):
    """Return the source identity-linear-action torus representatives.

    CYTools returns integer numerators q for representatives t=q/2 modulo
    lattice automorphisms.  For an O3/O7 hypersurface (lambda_f=1), the
    vertex part of source eq. (4.45) requires q.pairing+1 to be even for all
    dual vertices.  This is only an action-level diagnostic: fixed-locus
    smoothness and the paper's frozen-conifold cut are recorded separately.
    """

    actions = np.asarray(poly.inequivalent_Z2_actions(), dtype=int)
    dual_vertices = _as_int_rows(poly.dual().vertices())
    valid = [
        numerator
        for numerator in actions
        if np.all((dual_vertices @ numerator + 1) % 2 == 0)
    ]
    return actions, np.asarray(valid, dtype=int)


def _frozen_conifold_diagnostic(triangulation, p0):
    """Evaluate the source fixed-surface conifold diagnostic for L=I.

    For L=I, fixed components are indexed by fan cones satisfying
    ``t + 1/2 sum(sigma(1)) in N``.  With t=p0/2 this is the parity condition
    p0+sum(sigma(1)) in 2N.  A two-dimensional fixed component is contained in
    the hypersurface for lambda_f=1.  For a smooth cone generated by p and q,
    eq. (4.49) reduces to

        int_F (K_V^{-1}-D_p)(K_V^{-1}-D_q).

    A positive value is a frozen-conifold obstruction for the paper's model
    population.  Non-smooth cones are reported as unavailable evidence rather
    than silently classified.
    """

    fan = triangulation.fan()
    vectors = _as_int_rows(fan.vectors())
    tensor = _ambient_intersection_tensor(triangulation)
    if tensor.shape[0] != vectors.shape[0] + 1:
        return {
            "status": "unavailable",
            "reason": "ambient intersection tensor does not match fan rays",
            "surface_records": [],
        }

    surface_records = []
    unavailable = []
    for cone in fan.cones(dim=2, formal=True):
        ray_points = _as_int_rows(cone.rays())
        if ray_points.shape[0] != 2:
            continue
        if not cone.is_smooth():
            unavailable.append(ray_points.tolist())
            continue
        if not np.all((np.asarray(p0, dtype=int) + ray_points.sum(axis=0)) % 2 == 0):
            continue
        ray_indices = []
        for point in ray_points:
            matches = np.flatnonzero(np.all(vectors == point, axis=1))
            if matches.size != 1:
                unavailable.append(ray_points.tolist())
                ray_indices = []
                break
            ray_indices.append(int(matches[0]) + 1)
        if not ray_indices:
            continue
        p_index, q_index = ray_indices
        # Vectorised over the fan rays (indices 1..n_rays; index 0 is the
        # canonical divisor, excluded as in the original double loop). Exact
        # integer-valued tensor entries make the numpy sums identical to the
        # sequential Python sums before rounding (finding F4).
        block = tensor[p_index, q_index]
        l2 = block[1:, 1:].sum()
        l_dp = block[1:, p_index].sum()
        l_dq = block[1:, q_index].sum()
        dpdq = block[p_index, q_index]
        n_s = int(round(l2 - l_dp - l_dq + dpdq))
        surface_records.append(
            {
                "rays": ray_points.tolist(),
                "frozen_conifold_count": n_s,
                "formula": "int_F (K_V^-1-D_p)(K_V^-1-D_q)",
            }
        )

    frozen = [record for record in surface_records if record["frozen_conifold_count"] != 0]
    if unavailable:
        status = "unavailable"
    elif frozen:
        status = "frozen_conifold_obstruction"
    else:
        status = "no_frozen_conifold_obstruction"
    return {
        "status": status,
        "surface_records": surface_records,
        "unavailable_cones": unavailable,
        "frozen_surface_count": len(frozen),
    }


def _fixed_locus_components(triangulation, p0):
    """Enumerate the irreducible fixed-locus components F_I(sigma), L=I.

    Per Moritz eqs. (4.33)-(4.35), with L=I the nu label is trivially 0 and
    every cone is pointwise L-invariant, so the smooth-cone components of
    the fixed point set are labelled by cones sigma (of any dimension 0-4)
    satisfying the parity condition t + (1/2) sum(sigma(1)) in N, i.e. with
    t=p0/2, p0 + sum(rays in sigma(1)) in 2N.  Per the reduction described
    just before eq. (4.35) ("one then removes the F~(sigma,nu) that are
    already contained in a higher dimensional component"), a candidate
    cone is discarded whenever a proper face of it also satisfies the
    parity condition, since V(sigma) is then already contained in that
    smaller face's own (larger) stratum.
    """

    fan = triangulation.fan()
    vectors = _as_int_rows(fan.vectors())
    p0 = np.asarray(p0, dtype=int)

    def parity_ok(idxs):
        total = p0 + (vectors[list(idxs)].sum(axis=0) if idxs else 0)
        return bool(np.all(total % 2 == 0))

    def indices_of(cone):
        ray_points = _as_int_rows(cone.rays())
        idxs = []
        for point in ray_points:
            matches = np.flatnonzero(np.all(vectors == point, axis=1))
            if matches.size != 1:
                return None
            idxs.append(int(matches[0]))
        return tuple(sorted(idxs))

    candidates = []
    unavailable = False
    reasons = []
    if parity_ok(()):
        candidates.append((0, ()))
    for k in (1, 2, 3, 4):
        for cone in fan.cones(dim=k, formal=True):
            idxs = indices_of(cone)
            if idxs is None:
                unavailable = True
                reasons.append(f"ray_lookup_failed dim={k}")
                continue
            if not parity_ok(idxs):
                continue
            if k % 2 == 0 and not cone.is_smooth():
                # Eq. (4.46): f vanishes identically on F~(sigma) for even
                # dim(sigma), so all of F~(sigma) becomes part of F_I; the
                # smoothness discussion of Sec. 4.6 requires this toric
                # stratum itself to be smooth for its contribution to
                # chi(F_I) to be well defined.
                unavailable = True
                reasons.append(f"non_smooth_even_dim_component dim={k} rays={idxs}")
                continue
            # For odd dim(sigma), f is a generic section (eq. 4.46) and the
            # paper's own requirement is weaker than cone smoothness ("no
            # orbifold singularities of F~(sigma,nu) intersect the
            # hypersurface", Sec. 4.6) -- independently verified not to
            # change the population count (see the fixed-locus validation
            # note), so a non-smooth odd-dimension cone is kept, not
            # excluded.
            candidates.append((k, idxs))

    sets = [set(idxs) for _, idxs in candidates]
    minimal = [
        (k, idxs)
        for i, (k, idxs) in enumerate(candidates)
        if not any(j != i and sets[j] < sets[i] for j in range(len(candidates)))
    ]
    return minimal, unavailable, reasons


def _fixed_locus_euler_characteristic(poly, triangulation, p0):
    """Compute chi(F_I) for the L=I trilayer fixed locus.

    Dispatches each irreducible component F_I(sigma) from
    ``_fixed_locus_components`` by dim(sigma) parity (Moritz eq. 4.46):

    - even dim(sigma): F_I(sigma)=F~(sigma) in full, a smooth complete
      toric variety whose Euler characteristic equals its number of
      maximal cones (each toric torus-fixed point contributes 1).
    - dim(sigma)=1 (a ray p, F~(sigma)=D_p): F_I(sigma)=D_p . X, a
      generic-hypersurface-section surface in the Calabi-Yau threefold X.
      By adjunction (K_X=0), chi(D_p.X) = D_p^3 + c2(X).D_p -- verified
      to agree with an independent ambient toric-Chern-class derivation
      to numerical precision on real h11=4 examples, see
      validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md.
    - dim(sigma)=3 (rays p,q,r spanning a curve V(sigma)): F_I(sigma) is
      the point set cut by X on that curve, whose count is the ambient
      quadruple intersection D_p.D_q.D_r.X = sum_s kappa[p,q,r,s].
    """

    components, unavailable, reasons = _fixed_locus_components(triangulation, p0)
    if unavailable:
        return {"chi_F_I": None, "status": "unavailable", "reasons": reasons, "components": []}

    fan = triangulation.fan()
    vectors = _as_int_rows(fan.vectors())
    maximal_cones = [set(cone) for cone in fan.cones(as_inds=True)]
    tensor = _ambient_intersection_tensor(triangulation)
    kappa = tensor[1:, 1:, 1:, 1:]

    cy = triangulation.get_cy()
    prime_divs = set(cy.prime_toric_divisors())
    inter_dense = None
    c2_vec = None

    chi_total = 0.0
    detail = []
    for k, idxs in components:
        if k % 2 == 0:
            chi = float(sum(1 for cone in maximal_cones if set(idxs) <= cone))
            detail.append({"dim": k, "rays": idxs, "chi": chi, "kind": "toric_stratum"})
        elif k == 1:
            ray_index = idxs[0]
            point_index = int(poly.points_to_indices(vectors[ray_index].reshape(1, -1))[0])
            if point_index not in prime_divs:
                return {
                    "chi_F_I": None,
                    "status": "unavailable",
                    "reasons": reasons + [f"ray {ray_index} is not a prime toric divisor of X"],
                    "components": detail,
                }
            if inter_dense is None:
                inter_dense = cy.intersection_numbers(in_basis=False, format="dense")
                c2_vec = cy.second_chern_class(in_basis=False, include_origin=True)
            chi = float(
                inter_dense[point_index, point_index, point_index] + c2_vec[point_index]
            )
            detail.append({"dim": k, "rays": idxs, "chi": chi, "kind": "divisor_surface"})
        else:
            p_index, q_index, r_index = idxs
            chi = float(kappa[p_index, q_index, r_index, :].sum())
            detail.append({"dim": k, "rays": idxs, "chi": chi, "kind": "curve_points"})
        chi_total += chi

    return {"chi_F_I": chi_total, "status": "computed", "reasons": reasons, "components": detail}


def _h21_plus_zero_diagnostic(poly, triangulation, p0):
    """Test whether h^{2,1}_+(X,I)=0 exactly, via Moritz eq. (4.51).

    For L=I, h^{1,1}_-(X,I)=0 identically (the identity lattice action
    fixes every toric divisor class), so eq. (4.51) reduces to
    h^{2,1}_-(X,I) = (chi(F_I) - chi(X))/4 - 1, and h^{2,1}_+ = h^{2,1}(X)
    - h^{2,1}_-(X,I).  Independently validated against the paper's own
    h11=2 worked example (Sec. 4.2.1, eq. 4.2): reproduces the stated
    (h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-) = (2,0,0,132) exactly. The
    exact action, fan, GLSM, parity, fixed-component, and smoothness evidence
    is reconstructed by ``trilayer_involutions``; population counts remain a
    separate bounded task.
    """

    # Reconstruct the source trilayer action from the polytope itself and run
    # the exact fan/GLSM/parity/fixed-locus contract. The historical
    # Float64-only fixed-locus helper remains available as a non-gating
    # diagnostic, but is no longer the population acceptance computation.
    topology = dict(extract_topology(triangulation.get_cy(), triangulation))
    triangulation_cones = _triangulation_cones(poly, triangulation)
    topology["fixed_surface_n_s"] = identity_fixed_surface_n_s_table(
        triangulation_cones, triangulation
    )
    expected_p0 = tuple(int(value) for value in p0)
    reconstructed = reconstruct_trilayer_actions(poly, triangulation, topology)
    matching = [
        record
        for record in reconstructed["candidates"]
        if tuple(record.get("p0", ())) == expected_p0
    ]
    if not matching:
        return {
            "status": "unavailable",
            "reasons": ["requested p0 is not a source-reconstructed trilayer action"],
            "chi_F_I": None,
            "h21_minus": None,
            "h21_plus": None,
            "components": [],
        }
    exact = matching[0]
    fixed_euler = exact.get("fixed_locus_euler", {})
    hodge = exact.get("hodge_split", {})
    if exact.get("terminal_status") != "accepted_exact_trilayer_action":
        return {
            "status": "unavailable",
            "reasons": [exact.get("reason") or exact.get("terminal_status")],
            "chi_F_I": fixed_euler.get("chi_F_I"),
            "h21_minus": hodge.get("h21_minus"),
            "h21_plus": hodge.get("h21_plus"),
            "components": fixed_euler.get("components", []),
            "exact_action": exact,
        }
    return {
        "status": "h21_plus_zero" if hodge.get("h21_plus") == 0 else "h21_plus_nonzero",
        "reasons": [],
        "chi_F_I": fixed_euler.get("chi_F_I"),
        "h21_minus": hodge.get("h21_minus"),
        "h21_plus": hodge.get("h21_plus"),
        "components": fixed_euler.get("components", []),
        "exact_action": exact,
    }


def _orientifold_action_audit(
    poly,
    classes,
    *,
    collect_reason_diagnostics=False,
    polytope_index=None,
    terminal_ledger=None,
):
    """Count class-level inherited O3/O7 orientifold candidates.

    Per FRST class, runs the general ``(L, t, lambda_f)`` triple search
    (``enumerate_orientifold_candidates``, Moritz arXiv:2305.06363 eqs.
    4.3-4.51) and asks: does this CY admit at least one
    ``accepted_verified_orientifold`` candidate at all (``inherited``), and
    does it admit at least one with ``h11_minus==0`` (``h11_minus_zero``)?

    For ``L=identity`` candidates, ``n^S_{df=0}`` evidence (source eq.
    around line 649-654: isolated nodal points on a 2-dimensional fixed
    surface) is supplied via ``identity_fixed_surface_n_s_table`` so
    ``classify_smoothness``'s eq. (4.50) check can actually resolve, rather
    than reporting ``smoothness_verification_unavailable`` for every
    candidate with a 2-dimensional fixed component (which is what happened
    unconditionally before this evidence existed anywhere in this
    codebase). This replaces an earlier, ad hoc version of this function
    (source-vertex-parity-only, no fixed-locus/smoothness evidence at all)
    that gave 16 accepted CYs for the real h11=4 population against a
    Table-1 target of 1,559 -- see
    ``validation/fuzzy_axions_2412_12012_h11_3_h11_5_table1_verification_20260818.md``
    Sec. 6 for the investigation that found and fixed this, including a
    real polarity bug in ``classify_smoothness`` found at the same time
    (confirmed against the primary source, not just inferred).
    ``L != identity`` candidates receive general fixed-surface ``n_S``
    evidence from the toric Euler-sequence/adjunction calculation in
    ``_general_fixed_surface_n_s_table``. The helper returns evidence only
    when the local quotient fan and ambient Cartier data certify a smooth
    toric surface; all other components remain conservatively unresolved.

    Also supplies ``non_smooth_facet_dual_vertices`` evidence (source line
    ~629: eq. (4.45)'s dual-vertex parity condition applies not only to
    ``L``-fixed dual vertices but to every dual vertex whose facet of
    ``Delta_circ`` meets a non-simplicial and/or non-smooth cone of the
    triangulation's own fan -- independent of ``L``, computed once per
    triangulation via ``facets_with_non_smooth_cones``).
    """

    if not classes:
        result = {"inherited": 0, "h11_minus_zero": 0, "h11_minus_zero_classes": []}
        if collect_reason_diagnostics:
            result["reason_diagnostics"] = {
                "polytope_index": polytope_index,
                "surface_attempts": [],
                "unresolved_components": [],
                "certified_surfaces": [],
            }
        return result
    inherited_classes = set()
    h11_zero_classes = set()
    surface_attempts = []
    unresolved_components = []
    certified_surfaces = []
    for class_index, triangulation in enumerate(classes):
        cy = triangulation.get_cy()
        topology = dict(extract_topology(cy, triangulation))
        triangulation_cones = _triangulation_cones(poly, triangulation)
        topology["fixed_surface_n_s"] = identity_fixed_surface_n_s_table(
            triangulation_cones, triangulation
        )
        topology["compute_general_fixed_surface_n_s"] = True
        topology["non_smooth_facet_dual_vertices"] = facets_with_non_smooth_cones(
            poly, triangulation
        )
        matrix_diagnostics = {} if collect_reason_diagnostics else None

        def record_terminal(record):
            if terminal_ledger is None:
                return
            enriched = dict(record)
            enriched["polytope_index"] = polytope_index
            enriched["frst_class_index"] = class_index
            terminal_ledger.write(enriched)

        records = enumerate_orientifold_candidates(
            poly,
            triangulation,
            topology,
            general_fixed_surface_diagnostics=matrix_diagnostics,
            record_sink=record_terminal,
        )
        # Table 1's population (main.tex:1272, item 3 of the ensemble
        # definition) is explicitly "an ... orientifold involution of O3/O7
        # type" -- lambda_f=1 only. An accepted lambda_f=0 (O5/O9) candidate
        # is a real, structurally different orientifold and must not count
        # as evidence for this row.
        accepted = [
            record
            for record in records
            if record.get("terminal_status") == "accepted_verified_orientifold"
            and int(record.get("lambda_f", -1)) == 1
        ]
        if accepted:
            inherited_classes.add(class_index)
        if any(record.get("h11_minus") == 0 for record in accepted):
            h11_zero_classes.add(class_index)
        if collect_reason_diagnostics:
            for matrix_id, matrix_data in matrix_diagnostics.items():
                candidate_context_by_component = {}
                for record in records:
                    if (
                        record.get("matrix_candidate_id") != matrix_id
                        or record.get("lambda_f") != 1
                    ):
                        continue
                    context = {
                        "candidate_id": record.get("candidate_id"),
                        "torus_shift": record.get("torus_shift"),
                        "lambda_f": record.get("lambda_f"),
                        "h11_plus": record.get("h11_plus"),
                        "h11_minus": record.get("h11_minus"),
                        "candidate_terminal_status": record.get("terminal_status"),
                    }
                    for fixed_component in record.get("fixed_point_components", []):
                        component_key = json.dumps(
                            {
                                "sigma_rays": fixed_component.get("sigma_rays"),
                                "nu": fixed_component.get("nu"),
                            },
                            sort_keys=True,
                        )
                        candidate_context_by_component.setdefault(
                            component_key, []
                        ).append(context)
                for surface in matrix_data["surface_diagnostics"]:
                    for fixed_component in surface["fixed_components"]:
                        attempt = {
                            "polytope_index": polytope_index,
                            "frst_class_index": class_index,
                            "matrix_id": matrix_id,
                            "lattice_matrix": matrix_data["lattice_matrix"],
                            "fixed_component": fixed_component,
                            "status": surface["status"],
                            "reason_code": surface["reason_code"],
                            "reason": surface["reason"],
                            "surface_data": {
                                key: value
                                for key, value in surface.items()
                                if key not in {"fixed_components"}
                            },
                        }
                        if surface["status"] == "certified":
                            component_key = json.dumps(
                                {
                                    "sigma_rays": fixed_component.get("sigma_rays"),
                                    "nu": fixed_component.get("nu"),
                                },
                                sort_keys=True,
                            )
                            matches = candidate_context_by_component.get(
                                component_key, []
                            )
                            if matches:
                                attempt["candidate_context"] = matches[0]
                        surface_attempts.append(attempt)
                        if surface["status"] == "certified":
                            certified_surfaces.append(attempt)
                by_sigma = {
                    json.dumps(surface["sigma_rays"], sort_keys=True): surface
                    for surface in matrix_data["surface_diagnostics"]
                }
                for record in records:
                    if record.get("matrix_candidate_id") != matrix_id:
                        continue
                    if record.get("lambda_f") != 1:
                        continue
                    for fixed_component in record.get("fixed_point_components", []):
                        if (
                            int(fixed_component.get("fixed_toric_dimension", -1)) != 2
                            or not fixed_component.get("f_vanishes_identically")
                        ):
                            continue
                        surface = by_sigma.get(
                            json.dumps(fixed_component["sigma_rays"], sort_keys=True)
                        )
                        if surface is None or surface["status"] == "certified":
                            continue
                        unresolved_components.append(
                            {
                                "polytope_index": polytope_index,
                                "frst_class_index": class_index,
                                "matrix_id": matrix_id,
                                "candidate_id": record["candidate_id"],
                                "lattice_matrix": record["lattice_matrix"],
                                "torus_shift": record["torus_shift"],
                                "lambda_f": record["lambda_f"],
                                "h11_plus": record.get("h11_plus"),
                                "h11_minus": record.get("h11_minus"),
                                "candidate_terminal_status": record.get("terminal_status"),
                                "fixed_component": fixed_component,
                                "reason_code": surface["reason_code"],
                                "reason": surface["reason"],
                            }
                        )
    result = {
        "inherited": len(inherited_classes),
        "h11_minus_zero": len(h11_zero_classes),
        "h11_minus_zero_classes": sorted(h11_zero_classes),
    }
    if collect_reason_diagnostics:
        result["reason_diagnostics"] = {
            "polytope_index": polytope_index,
            "surface_attempts": surface_attempts,
            "unresolved_components": unresolved_components,
            "certified_surfaces": certified_surfaces,
        }
    return result


def _export_kaehler_point(triangulation):
    """Export the Algorithm-1 reference point t0 for one accepted FRST.

    ``t0`` is the tip of the stretched Kähler cone (arXiv:2412.12012 Sec.
    4.1, Algorithm 1), reusing ``sample_stretched_kaehler_points``'s
    ``attempt_index=1``/``canonical_tip`` convention (attempts=1, so only the
    tip itself is yielded; no qpsolvers/MOSEK angular sampling runs).
    ``kahler_cone.hyperplanes()`` is the *Mori*-cone dual (
    ``toric_kahler_cone() = toric_mori_cone(in_basis=True).dual()``), so
    ``tip_of_stretched_cone(1.0)`` bounds curve volumes, not divisor volumes:
    the paper's criterion 1 (tau_alpha >= 1 for all h11+4 prime toric
    divisors) is genuinely *not* guaranteed to hold at t0 itself -- confirmed
    empirically (divisor volumes ~0.5 at t0 on a live h11=4 sample, while the
    Kahler-cone slack sits at exactly 1.0 as expected). This matches the
    paper's own Algorithm 1, which starts at t0 and only requires criterion 1
    to hold after a homogeneous dilatation lambda*t0 (Sec. 3.3, quoted in the
    scope note). ``evaluate_kaehler_point`` is therefore called with
    ``enforce_divisor_volume_lower_bounds=False`` -- the same deferral the
    existing ``canonical_qcd`` moduli policy uses for its own post-dilation
    check (see generate_geometric_data_multitriangulation.py around line
    2207) -- so this export records t0's raw geometry data for the Phase 3
    lambda solver to evaluate criterion 1 against, rather than silently
    discarding every candidate whose *undilated* tip fails a check the paper
    never requires there.  Only the plain-data outputs needed by the Phase
    2/3 model-stage math are returned -- no live CYTools object is retained,
    so this record is safe to serialize and consume from Julia.

    ``glsm_charge_matrix`` is Q in the ``potential_matrix_convention``
    documented around generate_geometric_data_multitriangulation.py line
    ~3082 (h11 x N, instanton charges are columns); N here indexes all h11+4
    prime toric divisors, matching eq. 3.9's Q^i_alpha.
    """

    cy = triangulation.get_cy()
    topology = extract_topology(cy, triangulation)
    kahler_cone = cy.toric_kahler_cone()
    mosek_license = configure_mosek_license()
    tip_solver = "cytools-default"
    if mosek_license["activated"]:
        try:
            reference_tip = np.asarray(
                kahler_cone.tip_of_stretched_cone(1.0, backend="mosek"), dtype=float
            )
            tip_solver = "mosek"
        except Exception:
            reference_tip = np.asarray(
                kahler_cone.tip_of_stretched_cone(1.0), dtype=float
            )
            tip_solver = "cytools-default-after-mosek-failure"
    else:
        reference_tip = np.asarray(
            kahler_cone.tip_of_stretched_cone(1.0), dtype=float
        )

    qprime_raw = np.asarray(cy.toric_effective_cone().rays(), dtype=float)
    qprime, _ = canonicalize_unique_charge_rows(qprime_raw)
    qprime = np.asarray(qprime, dtype=np.int64)

    glsm = np.asarray(cy.glsm_charge_matrix(include_origin=False), dtype=int)
    volume_context = {
        "volume_backend": "fan",
        "kappa": topology["kappa"],
        "glsm_charge_matrix": glsm,
        "mori_cone": topology["mori_cone"],
    }

    tip_proposal = next(
        sample_stretched_kaehler_points(
            kahler_cone,
            reference_tip,
            np.random.default_rng(),
            1,
            lambda message: None,
            point_seed=None,
            diagnostics=None,
            include_metadata=True,
        )
    )
    point = np.asarray(tip_proposal["point"], dtype=float)

    diagnostic, values = evaluate_kaehler_point(
        cy,
        kahler_cone,
        qprime,
        point,
        attempt_index=1,
        point_kind="canonical_tip",
        solver=tip_solver,
        min_prime_divisor_volume=1.0,
        min_divisor_volume=1.0,
        volume_tolerance=DIVISOR_VOLUME_TOLERANCE,
        enforce_divisor_volume_lower_bounds=False,
        **volume_context,
    )
    if values is None:
        return {
            "status": "rejected",
            "reason": diagnostic.get(
                "failure_reason", "canonical tip failed the Kähler-point domain checks"
            ),
            "diagnostic": diagnostic,
        }
    return {
        "status": "accepted",
        "h11": int(topology["h11"]),
        "point": values["point"].tolist(),
        "cy_volume": float(values["cy_volume"]),
        "prime_divisor_volumes": values["prime_divisor_volumes"].tolist(),
        "inverse_metric": values["inverse_metric"].tolist(),
        "glsm_charge_matrix": glsm.tolist(),
        "potential_matrix_convention": {
            "Q": "h11 x N; instanton charges are columns; N indexes all h11+4 prime toric divisors",
        },
        "diagnostic": diagnostic,
    }


MODEL_STAGE_DRIVER = Path(__file__).resolve().parent / "fuzzy_axion_model_stage_driver.jl"


def _write_model_stage_input(records: list[dict], path: Path) -> None:
    """Write flat (X, FRST) Kähler-point export records to HDF5 for the Julia driver.

    HDF5 is used rather than JSON because it is already a hard dependency on
    both sides of this package (``h5py`` here, ``HDF5.jl`` in
    ``Project.toml``); ``CYAxiverse`` has no JSON-parsing package, and this
    keeps priority 4 from having to add one. ``Q``/``inverse_metric`` are
    written in their Python-documented shapes (h11 x N, h11 x h11); the Julia
    driver is responsible for undoing HDF5.jl's dimension-order reversal on
    read (see its module docstring -- confirmed empirically, not assumed,
    before writing that script).
    """

    with h5py.File(path, "w") as file:
        file.create_dataset("record_count", data=len(records))
        for index, record in enumerate(records):
            group = file.create_group(f"records/{index}")
            group.create_dataset("Q", data=np.array(record["glsm_charge_matrix"], dtype=np.int64))
            group.create_dataset("tau", data=np.array(record["prime_divisor_volumes"], dtype=np.float64))
            group.create_dataset("cy_volume", data=float(record["cy_volume"]))
            group.create_dataset(
                "inverse_metric", data=np.array(record["inverse_metric"], dtype=np.float64)
            )


def _run_model_stage(records: list[dict], args) -> dict[str, Any]:
    """Run priority 3's Julia model-stage evaluator over the exported Kähler points.

    ``gs``/``W0`` are a single documented convention applied uniformly to
    every record -- never tuned per record or towards the 3,348 target (see
    validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md
    Sec. 6, "Acceptance tests for the 3,348 comparison").
    """

    convention = {
        "qcd_divisor_domain": args.qcd_divisor_domain,
        "qcd_divisor_domain_justification": (
            "all_prime is Algorithm 1's literal 'for D in the h11+4 prime toric divisors'; "
            "leading_nonself is the opt-in candidate restriction to the h11 leading-instanton "
            "divisors minus the fuzzy axion's own (see enumerate_fuzzy_axion_models's docstring "
            "and validation/fuzzy_axions_2412_12012_sampler_reverse_engineering_20260818.md). "
            "Counts under leading_nonself are diagnostic only and undershoot Table 1 at h11>=3 "
            "by construction, because the sampler's second half -- an enrichment supplying more "
            "than one Kahler point per surviving (CY, axion, D) combination -- is deliberately "
            "not implemented"
        ),
        "gs": args.gs,
        "gs_justification": (
            "paper's stated main-analysis reference value, eq. 3.28-3.29 (gs=0.5 -> P~5e-4)"
        ),
        "w0_real": args.w0_real,
        "w0_imag": args.w0_imag,
        "w0_justification": (
            "no ensemble-specific W0 value is stated in the source (Sec. 3.4 gives only "
            "the reheating-example illustrative value W0=1e-5, and argues -- eq. 3.26-3.27 "
            "-- that the fuzzy axion's own decay constant is near-insensitive to W0 once its "
            "mass is pinned by construction, though this does not bound the model *count*, "
            "see the scope doc Sec. 3 point 2); defaults to W0=1, matching the paper's own "
            "Sec. 4.2.1 hand-worked-example convention, and is not tuned to approach the "
            "3,348 target"
        ),
    }

    if not records:
        return {
            "input_record_count": 0,
            "total_model_count": None,
            "model_count_per_record": [],
            "models": [] if args.keep_details else None,
            "convention": convention,
            "diagnostic_reason": (
                "no accepted h21_plus_zero-class Kahler-point export records in this run"
            ),
        }

    with tempfile.TemporaryDirectory(prefix="fuzzy_axion_model_stage_") as workdir_name:
        workdir = Path(workdir_name)
        input_path = workdir / "model_stage_input.h5"
        output_path = workdir / "model_stage_output.h5"
        _write_model_stage_input(records, input_path)

        julia_project = args.julia_project or Path(__file__).resolve().parent.parent
        subprocess.run(
            [
                args.julia_binary,
                f"--project={julia_project}",
                str(MODEL_STAGE_DRIVER),
                str(input_path),
                str(output_path),
                str(args.gs),
                str(args.w0_real),
                str(args.w0_imag),
                str(args.qcd_divisor_domain),
            ],
            check=True,
        )

        with h5py.File(output_path, "r") as file:
            total_model_count = int(file["total_model_count"][()])
            model_count_per_record = np.asarray(file["model_count_per_record"]).tolist()
            prefactor_P = float(file["prefactor_P"][()])
            models = None
            if args.keep_details:
                record_index = np.asarray(file["model_record_index"])
                axion_index = np.asarray(file["model_axion_index"])
                qcd_divisor_index = np.asarray(file["model_qcd_divisor_index"])
                lam = np.asarray(file["model_lambda"])
                mass_reference_log10_ev = np.asarray(file["model_mass_reference_log10_ev"])
                tau_reference = np.asarray(file["model_tau_reference"])
                models = [
                    {
                        "record_index": int(record_index[i]),
                        "axion_index": int(axion_index[i]),
                        "qcd_divisor_index": int(qcd_divisor_index[i]),
                        "lambda": float(lam[i]),
                        "mass_reference_log10_ev": float(mass_reference_log10_ev[i]),
                        "tau_reference": float(tau_reference[i]),
                    }
                    for i in range(len(record_index))
                ]

    convention["prefactor_P"] = prefactor_P
    return {
        "input_record_count": len(records),
        "total_model_count": total_model_count,
        "model_count_per_record": model_count_per_record,
        "models": models,
        "convention": convention,
        "diagnostic_reason": None,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _count_reason_rows(rows, key):
    """Count reason codes by one diagnostic scope key."""

    counts = Counter()
    for row in rows:
        counts[str(key(row))] += 1
    return dict(sorted(counts.items()))


def _orientifold_reason_diagnostics_summary(
    h11,
    surface_attempts,
    unresolved_components,
    certified_surfaces,
):
    """Aggregate opt-in general-``L`` fixed-surface evidence by audit scope."""

    skipped = [row for row in surface_attempts if row["status"] == "unavailable"]

    def scope_key(row):
        return f"{row['polytope_index']}:{row['frst_class_index']}:{row['matrix_id']}"

    def component_key(row):
        return json.dumps(
            {
                "polytope_index": row["polytope_index"],
                "frst_class_index": row["frst_class_index"],
                "matrix_id": row["matrix_id"],
                "fixed_component": row["fixed_component"],
            },
            sort_keys=True,
        )

    reason_counts = dict(sorted(Counter(row["reason_code"] for row in skipped).items()))
    return {
        "schema_version": "cyaxiverse-general-L-fixed-surface-diagnostics-1.1",
        "h11": int(h11),
        "surface_attempt_count": len(surface_attempts),
        "skipped_surface_count": len(skipped),
        "certified_surface_count": len(certified_surfaces),
        "reason_counts": reason_counts,
        "reason_counts_by_h11": {str(int(h11)): reason_counts},
        "reason_counts_by_polytope": _count_reason_rows(
            skipped, lambda row: row["polytope_index"]
        ),
        "reason_counts_by_frst_class": _count_reason_rows(
            skipped, lambda row: f"{row['polytope_index']}:{row['frst_class_index']}"
        ),
        "reason_counts_by_matrix": _count_reason_rows(skipped, scope_key),
        "reason_counts_by_fixed_component": _count_reason_rows(skipped, component_key),
        "unresolved_candidate_component_count": len(unresolved_components),
        "unresolved_candidate_reason_counts": dict(
            sorted(Counter(row["reason_code"] for row in unresolved_components).items())
        ),
        "surface_attempts": surface_attempts,
        "unresolved_components": unresolved_components,
        "certified_surfaces": certified_surfaces,
    }


def _legacy_reproduce(args):
    # Higher-h11 runs are replay runs.  Read the prior population handoffs and
    # verify their durable compressed artifacts before importing any parquet
    # rows or constructing a CYTools geometry.  The preflight is deliberately
    # mandatory for both bounded and full h11=4/5 invocations.
    population_preflight = run_population_preflight(
        Path(__file__).resolve().parents[1], args.h11
    )
    targets = PAPER_TARGETS_BY_H11.get(args.h11)
    all_records = load_mirror_polytopes(
        args.parquet_dir,
        h11=args.h11,
        limit=args.limit,
        favorable=True,
    )
    total_favorable = len(all_records)
    shard_count = int(getattr(args, "shard_count", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    is_sharded = shard_count > 1
    # Deterministic strided partition over the global favorable-polytope index.
    # Striding (not a contiguous block) averages the varying per-polytope cost
    # across shards. shard_count==1 selects every record, so the default,
    # unsharded run is byte-for-byte unchanged.
    shard_items = list(enumerate(all_records))[shard_index::shard_count]
    records = [payload for _, payload in shard_items]
    # Provenance describes the full input dataset and is identical across shards.
    run_provenance = _run_provenance(args, all_records)
    terminal_ledger = None
    terminal_ledger_summary = None
    if args.orientifold_audit:
        ledger_path = getattr(args, "terminal_ledger", None)
        if ledger_path is None:
            output_path = getattr(args, "output", None)
            if output_path is None:
                raise ValueError(
                    "--orientifold-audit requires --terminal-ledger when --output is absent"
                )
            output_path = Path(output_path)
            ledger_path = output_path.with_name(
                f"{output_path.stem}.terminal-ledger.jsonl"
            )
        terminal_ledger = TerminalLedgerWriter(
            ledger_path,
            provenance=run_provenance,
            metadata={
                "program": "fuzzy-axion inherited-orientifold Table 1 reproduction",
                "requested_h11": int(args.h11),
                "candidate_schema": CANDIDATE_SCHEMA_VERSION,
                "acceptance_scope": "lambda_f=1 and accepted_verified_orientifold only",
                "search_scope": "supplied_frst_only; lattice involutions; projected torus shifts; both lambda_f values",
            },
        )
    polytopes = []
    total_raw = 0
    total_classes = 0
    trilayer_polytope_count = 0
    trilayer_class_count = 0
    trilayer_nonfrozen_class_count = 0
    trilayer_h21_plus_zero_class_count = 0
    identity_action_cy_count = 0
    identity_valid_action_cy_count = 0
    identity_action_count = 0
    orientifold_inherited_count = 0
    orientifold_h11_zero_count = 0
    kaehler_export_accepted_count = 0
    kaehler_export_rejected_count = 0
    export_kaehler_points = args.export_kaehler_points or args.model_stage
    model_stage_records = [] if args.model_stage else None
    details = []
    reason_surface_attempts = []
    reason_unresolved_components = []
    reason_certified_surfaces = []
    for poly_index, (poly, provenance) in shard_items:
        raw, classes = _frst_classes(poly)
        total_raw += len(raw)
        total_classes += len(classes)
        trilayer = _trilayer_candidate(poly)
        actions, valid_actions = _identity_torus_actions(poly)
        identity_action_count += len(actions)
        if len(actions):
            identity_action_cy_count += len(classes)
        if len(valid_actions):
            identity_valid_action_cy_count += len(classes)
        frozen = None
        frozen_per_class = None
        nonfrozen_class_count_this_polytope = 0
        h21_per_class = None
        h21_plus_zero_class_count_this_polytope = 0
        orientifold = None
        if trilayer is not None:
            trilayer_polytope_count += 1
            trilayer_class_count += len(classes)
            # The special trilayer involution (L=I) is compatible with every
            # FRST -- but both per-class diagnostics below are evaluated on a
            # specific simplicial fan, and different FRST classes of the same
            # polytope can subdivide the fixed toric divisors differently.
            # Measured directly: the frozen-conifold diagnostic's status
            # varies across FRST classes of the same polytope in ~22% of a
            # sampled subset (see
            # validation/fuzzy_axions_2412_12012_frst_dependent_frozen_conifold_20260817.md).
            # Both must therefore be evaluated per FRST class, not propagated
            # from a single representative.
            frozen_per_class = [
                _frozen_conifold_diagnostic(triangulation, trilayer["p0"])
                for triangulation in classes
            ]
            nonfrozen_class_count_this_polytope = sum(
                1
                for result in frozen_per_class
                if result["status"] == "no_frozen_conifold_obstruction"
            )
            trilayer_nonfrozen_class_count += nonfrozen_class_count_this_polytope
            frozen = frozen_per_class[0] if frozen_per_class else None
            # The actual h11_minus_zero_h21_plus_zero population gate (eq.
            # 4.51 applied directly): independently validated to reproduce
            # the paper's own h11=2 worked example's Hodge splitting exactly
            # and the h11=4 population's 267-class target exactly -- unlike
            # `frozen_per_class` above, which is a separate smoothness
            # diagnostic for the orientifold background, not part of this
            # population's definition. See
            # validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md.
            h21_per_class = [
                _h21_plus_zero_diagnostic(poly, triangulation, trilayer["p0"])
                for triangulation in classes
            ]
            h21_plus_zero_class_count_this_polytope = sum(
                1 for result in h21_per_class if result["status"] == "h21_plus_zero"
            )
            trilayer_h21_plus_zero_class_count += h21_plus_zero_class_count_this_polytope
        if args.orientifold_audit:
            orientifold = _orientifold_action_audit(
                poly,
                classes,
                collect_reason_diagnostics=args.orientifold_reason_diagnostics,
                polytope_index=poly_index,
                terminal_ledger=terminal_ledger,
            )
            orientifold_inherited_count += orientifold["inherited"]
            orientifold_h11_zero_count += orientifold["h11_minus_zero"]
            if args.orientifold_reason_diagnostics:
                diagnostics = orientifold["reason_diagnostics"]
                reason_surface_attempts.extend(diagnostics["surface_attempts"])
                reason_unresolved_components.extend(diagnostics["unresolved_components"])
                reason_certified_surfaces.extend(diagnostics["certified_surfaces"])
        kaehler_export_per_class = None
        if export_kaehler_points and h21_per_class is not None:
            # Only the classes actually accepted by the h21_plus_zero
            # population gate (Algorithm 1's model-stage input population,
            # see validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md
            # section 6) get a live-CYTools export; the rest stay `None` so
            # index alignment with `h21_per_class` is preserved.
            kaehler_export_per_class = []
            for class_index, triangulation in enumerate(classes):
                if h21_per_class[class_index]["status"] != "h21_plus_zero":
                    kaehler_export_per_class.append(None)
                    continue
                record = _export_kaehler_point(triangulation)
                kaehler_export_per_class.append(record)
                if record["status"] == "accepted":
                    kaehler_export_accepted_count += 1
                    if args.model_stage:
                        model_stage_records.append(
                            {
                                "polytope_index": poly_index,
                                "frst_class_index": class_index,
                                "glsm_charge_matrix": record["glsm_charge_matrix"],
                                "prime_divisor_volumes": record["prime_divisor_volumes"],
                                "cy_volume": record["cy_volume"],
                                "inverse_metric": record["inverse_metric"],
                            }
                        )
                else:
                    kaehler_export_rejected_count += 1
        details.append(
            {
                "polytope_index": poly_index,
                "provenance": provenance,
                "raw_frst_count": len(raw),
                "frst_class_count": len(classes),
                "trilayer": trilayer,
                "frozen_conifold": frozen,
                "frozen_conifold_per_class": frozen_per_class,
                "nonfrozen_class_count": nonfrozen_class_count_this_polytope,
                "h21_plus_zero_per_class": h21_per_class,
                "h21_plus_zero_class_count": h21_plus_zero_class_count_this_polytope,
                "identity_torus_action_numerators": actions.tolist(),
                "identity_valid_o3o7_action_numerators": valid_actions.tolist(),
                "orientifold_action_audit": orientifold,
                "kaehler_point_export_per_class": kaehler_export_per_class,
            }
        )
        if args.progress and (poly_index + 1) % args.progress == 0:
            print(
                f"processed {poly_index + 1}/{len(records)} polytopes; "
                f"raw={total_raw}, classes={total_classes}, "
                f"trilayer_classes={trilayer_class_count}, "
                f"nonfrozen_trilayer_classes={trilayer_nonfrozen_class_count}, "
                f"h21_plus_zero_trilayer_classes={trilayer_h21_plus_zero_class_count}",
                flush=True,
            )

    if is_sharded:
        # A single shard can never be population-complete on its own. Report it
        # as partial and defer the completeness verdict to the merge step; the
        # expected target is still surfaced for reference.
        population_completion = {
            "complete": False,
            "basis": "sharded_partial_run",
            "reason": (
                f"shard {shard_index + 1}/{shard_count} covers {len(records)} of "
                f"{total_favorable} loaded favorable polytopes; combine shards with "
                "scripts/merge_orientifold_shards.py to determine completeness"
            ),
            "loaded_favorable_polytopes": len(records),
            "expected_favorable_polytopes": _population_completion_status(
                args.h11, total_favorable
            )["expected_favorable_polytopes"],
        }
    else:
        population_completion = _population_completion_status(
            args.h11,
            len(records),
        )
    population_complete = population_completion["complete"]
    population_exact_target = (
        targets is not None
        and trilayer_h21_plus_zero_class_count == targets["h11_minus_zero_h21_plus_zero_orientifold_cys"]
        and population_complete
    )
    model_stage = _run_model_stage(model_stage_records, args) if args.model_stage else None
    if terminal_ledger is not None:
        terminal_ledger_summary = terminal_ledger.close()
    # A non-default QCD-divisor domain is a candidate reading of Algorithm 1,
    # not a correction to it, so its counts can never be a benchmark-match
    # claim -- not even where they coincide with Table 1 (they do at h11=2).
    model_stage_is_literal_algorithm = args.qcd_divisor_domain == "all_prime"
    if model_stage is not None and model_stage["diagnostic_reason"] is None:
        reasons = []
        if not model_stage_is_literal_algorithm:
            reasons.append(
                f"--qcd-divisor-domain={args.qcd_divisor_domain} is an opt-in candidate reading "
                "of Algorithm 1's 'for D' loop, not the paper's literal text; counts under it "
                "are diagnostic only regardless of how they compare to Table 1"
            )
        if not population_complete:
            reasons.append(f"run was limited via --limit; population is not the full h11={args.h11} set")
        if targets is None:
            reasons.append(f"no Table 1 target is recorded for h11={args.h11}; counts are diagnostic only")
        else:
            if trilayer_h21_plus_zero_class_count != targets["h11_minus_zero_h21_plus_zero_orientifold_cys"]:
                reasons.append(
                    f"input population is {trilayer_h21_plus_zero_class_count} h21_plus_zero-accepted "
                    f"FRST classes, not the exact {targets['h11_minus_zero_h21_plus_zero_orientifold_cys']}"
                    "-target population (see "
                    "fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md)"
                )
            if model_stage["total_model_count"] != targets["models"]:
                reasons.append(
                    f"total model count {model_stage['total_model_count']} != paper target "
                    f"{targets['models']}"
                )
        if reasons:
            model_stage["diagnostic_reason"] = "; ".join(reasons)

    summary = {
        "schema_version": REPRODUCTION_SCHEMA_VERSION,
        "schema_metadata": {
            "deprecated_aliases": DEPRECATED_COUNT_ALIASES,
        },
        "paper": "arXiv:2412.12012",
        "orientifold_source": "arXiv:2305.06363",
        "run_provenance": run_provenance,
        "input": {
            "source": "generator.load_mirror_polytopes",
            "parquet_dir": str(Path(args.parquet_dir).resolve()),
            "requested_h11": args.h11,
            "favorable_lattice": "N",
            "record_count": len(records),
            "shard": {
                "index": shard_index,
                "count": shard_count,
                "is_sharded": is_sharded,
                "shard_favorable_polytopes": len(records),
                "total_favorable_polytopes": total_favorable,
            },
            "population_complete": population_complete,
            "population_completion_basis": population_completion["basis"],
            "population_completion_reason": population_completion["reason"],
            "population_completion_loaded_favorable_polytopes": population_completion[
                "loaded_favorable_polytopes"
            ],
            "population_completion_expected_favorable_polytopes": population_completion[
                "expected_favorable_polytopes"
            ],
        },
        "population_preflight": population_preflight,
        "counts": {
            "favorable_polytopes": len(records),
            "raw_frsts": total_raw,
            "frst_classes": total_classes,
            "raw_trilayer_polytopes": trilayer_polytope_count,
            "raw_trilayer_frst_classes": trilayer_class_count,
            # A separate orientifold-background smoothness diagnostic, not
            # this population's own gate -- see h21_plus_zero_trilayer_frst_classes.
            "nonfrozen_trilayer_frst_classes": trilayer_nonfrozen_class_count,
            "h21_plus_zero_trilayer_frst_classes": trilayer_h21_plus_zero_class_count,
            "identity_torus_action_count": identity_action_count,
            "identity_torus_action_cy_count": identity_action_cy_count,
            "identity_valid_o3o7_action_cy_count": identity_valid_action_cy_count,
            "source_evidence_inherited_orientifold_cys": orientifold_inherited_count
            if args.orientifold_audit
            else None,
            "source_evidence_h11_minus_zero_orientifold_cys": orientifold_h11_zero_count
            if args.orientifold_audit
            else None,
            # Deprecated aliases retained for consumers of schema 1.0.
            "source_vertex_evidence_inherited_orientifold_cys": orientifold_inherited_count
            if args.orientifold_audit
            else None,
            "source_vertex_evidence_h11_minus_zero_orientifold_cys": orientifold_h11_zero_count
            if args.orientifold_audit
            else None,
            "kaehler_point_export_accepted_count": kaehler_export_accepted_count
            if export_kaehler_points
            else None,
            "kaehler_point_export_rejected_count": kaehler_export_rejected_count
            if export_kaehler_points
            else None,
        },
        "paper_targets": targets,
        "claim_status": (
            {
                "favorable_polytopes": None,
                "frst_classes": None,
                "h21_plus_zero": None,
                "model_count": None,
            }
            if targets is None
            else {
                "favorable_polytopes": "exact" if len(records) == targets["favorable_polytopes"] else "mismatch",
                "frst_classes": "exact" if total_classes == targets["frst_classes"] else "mismatch",
                "h21_plus_zero": (
                    "benchmark_match_candidate"
                    if trilayer_h21_plus_zero_class_count == targets["h11_minus_zero_h21_plus_zero_orientifold_cys"]
                    else "diagnostic_only"
                ),
                "model_count": (
                    None
                    if model_stage is None
                    else (
                        "benchmark_match_candidate"
                        if population_exact_target
                        and model_stage_is_literal_algorithm
                        and model_stage["total_model_count"] == targets["models"]
                        else "diagnostic_only"
                    )
                ),
            }
        ),
        "model_stage": model_stage,
        "orientifold_reason_diagnostics": (
            _orientifold_reason_diagnostics_summary(
                args.h11,
                reason_surface_attempts,
                reason_unresolved_components,
                reason_certified_surfaces,
            )
            if args.orientifold_reason_diagnostics
            else None
        ),
        "terminal_ledger": terminal_ledger_summary,
        "details": details if args.keep_details else None,
    }
    return _jsonable(summary)


def _shard_suffix_path(path, shard_index, shard_count):
    """Insert a deterministic ``.shardNNN-of-MMM`` tag before the file suffix."""
    path = Path(path)
    tag = f"shard{int(shard_index):03d}-of-{int(shard_count):03d}"
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")
REPLAY_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-exact-replay-3.0"
REPLAY_CHECKPOINT_SCHEMA_VERSION = "cyaxiverse-fuzzy-axions-exact-replay-checkpoint-1.0"
MAX_REPLAY_WORKERS = 4


class ReplayConfigurationError(ValueError):
    """Raise when a replay request is unsafe or scientifically ambiguous."""


class ReplayResumeError(RuntimeError):
    """Raise when a checkpoint cannot be resumed under its frozen contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_input_hash(parquet_dir: Path) -> str:
    """Hash the selected source directory without loading any geometry."""
    entries = []
    for path in sorted(parquet_dir.glob("polytopes-4d-*-vertices.parquet")):
        entries.append({"path": str(path.resolve()), "sha256": _sha256_file(path)})
    if not entries:
        entries = [{"path": str(parquet_dir.resolve()), "exists": parquet_dir.is_dir()}]
    return _sha256_bytes(_canonical_json(entries).encode("utf-8"))


def _code_hash() -> str:
    """Hash the replay driver and exact candidate modules used by it."""
    names = (
        "reproduce_fuzzy_axions_h11_4.py",
        "inherited_orientifold_candidates.py",
        "trilayer_involutions.py",
        "geometry_charge_conventions.py",
        "orientifold_general_l_geometry.py",
        "toric_fixed_component_euler.py",
        "orientifold_population_preflight.py",
        "generate_geometric_data_multitriangulation.py",
        "mpcp_bounded_analysis.py",
        "mpcp_immutable_source.py",
    )
    payload = []
    root = Path(__file__).resolve().parent
    for name in names:
        path = root / name
        payload.append({"name": name, "sha256": _sha256_file(path)})
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _handoff_hash(preflight: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(preflight.get("handoffs", [])).encode("utf-8"))


def _runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cytools": "unavailable",
        "zstd": "unavailable",
    }
    try:
        import cytools

        versions["cytools"] = str(getattr(cytools, "__version__", "unknown"))
    except ImportError:
        pass
    zstd = shutil.which("zstd")
    if zstd:
        try:
            completed = subprocess.run(
                [zstd, "--version"], check=False, capture_output=True, text=True
            )
            versions["zstd"] = (completed.stdout or completed.stderr).strip()
        except OSError:
            pass
    return versions


def _freeze_replay_config(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    h11 = getattr(args, "h11", None)
    if h11 not in (4, 5):
        raise ReplayConfigurationError("exact replay requires explicit --h11 4 or --h11 5")
    workers = int(getattr(args, "workers", 1))
    if not 1 <= workers <= MAX_REPLAY_WORKERS:
        raise ReplayConfigurationError(
            f"--workers must be between 1 and {MAX_REPLAY_WORKERS}"
        )
    shard_count = int(getattr(args, "shard_count", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    if not 1 <= shard_count <= MAX_REPLAY_WORKERS:
        raise ReplayConfigurationError(
            f"--shard-count must be between 1 and {MAX_REPLAY_WORKERS}"
        )
    if not 0 <= shard_index < shard_count:
        raise ReplayConfigurationError("--shard-index must be in [0, --shard-count)")
    max_rows = int(getattr(args, "max_rows", 0))
    if max_rows < 0:
        raise ReplayConfigurationError("--max-rows must be non-negative")
    checkpoint_interval = int(getattr(args, "checkpoint_interval", 32))
    if checkpoint_interval < 1:
        raise ReplayConfigurationError("--checkpoint-interval must be positive")
    parquet_dir = Path(args.parquet_dir).expanduser().resolve()
    source_contract = preflight.get("source_contract")
    if isinstance(source_contract, dict):
        declared_path = Path(str(source_contract.get("source_path", ""))).resolve()
        if parquet_dir != declared_path:
            raise ReplayConfigurationError(
                "source path does not match the implementation-handoff declaration"
            )
        observed_partitions = sorted(
            path.name for path in parquet_dir.glob("polytopes-4d-*-vertices.parquet")
        )
        expected_partitions = sorted(source_contract.get("required_partitions", ()))
        if observed_partitions != expected_partitions:
            raise ReplayConfigurationError(
                f"source partitions must be exactly 05..10; observed {observed_partitions}"
            )
    source_input_sha256 = _source_input_hash(parquet_dir)
    enrichment_arg = getattr(args, "enrichment", None)
    enrichment_contract = None
    if enrichment_arg is not None:
        enrichment_path = Path(enrichment_arg).expanduser().resolve()
        if not enrichment_path.is_file():
            raise ReplayConfigurationError(f"enrichment artifact is missing: {enrichment_path}")
        enrichment_contract = {
            "path": str(enrichment_path),
            "sha256": _sha256_file(enrichment_path),
        }
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "requested_h11": int(h11),
        "selection_query": {
            "mirror_h12_equals": int(h11),
            "favorable": True,
            "physical_h11_verification": "CYTools Polytope.h11 equals requested_h11",
            "target_selected_acceptance": False,
        },
        "source_path": str(parquet_dir),
        "source_input_sha256": source_input_sha256,
        "enrichment": enrichment_contract,
        "source_contract": source_contract,
        "source_partition_hashes": (
            {
                row["partition"]: {
                    "expected": row.get("expected_sha256"),
                    "observed": row.get("observed_sha256"),
                }
                for row in source_contract.get("partitions", [])
            }
            if isinstance(source_contract, dict)
            else {}
        ),
        "source_code_sha256": _code_hash(),
        "population_handoff_sha256": _handoff_hash(preflight),
        "max_rows": max_rows,
        "checkpoint_interval": checkpoint_interval,
        "workers": workers,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "orientifold_reason_diagnostics": bool(
            getattr(args, "orientifold_reason_diagnostics", False)
        ),
        "labels": {
            "validity": "not_validated",
            "selection": "candidate-only",
            "representativeness": "nonrepresentative",
            "execution_mode": "infrastructure_smoke_only",
            "scientific_result": "no_scientific_result",
        },
        "resource_settings": {
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
            "worker_policy": "sequential unless explicitly requested; hard maximum four",
        },
        "runtime_versions": _runtime_versions(),
        "artifact_contract": _artifact_contract(preflight),
        "preflight": preflight,
    }


def _config_digest(config: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(config).encode("utf-8"))


def _artifact_contract(preflight: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return the explicitly manifest-bound merged and gap artifact digests."""

    artifacts = preflight.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReplayConfigurationError("preflight has no artifact digest contract")
    declared = artifacts.get("artifact_digests")
    if not isinstance(declared, dict):
        raise ReplayConfigurationError(
            "preflight must explicitly hash merged and gap artifacts"
        )
    contract: dict[str, dict[str, str]] = {}
    for name in ("merged_artifact", "gap_analysis_artifact"):
        record = declared.get(name)
        if not isinstance(record, dict):
            raise ReplayConfigurationError(f"preflight has no explicit hash for {name}")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ReplayConfigurationError(f"preflight hash entry for {name} is invalid")
        expected_path = artifacts.get(name)
        if not isinstance(expected_path, str) or Path(path).resolve() != Path(expected_path).resolve():
            raise ReplayConfigurationError(f"preflight hash path does not bind {name}")
        contract[name] = {"path": str(Path(path).resolve()), "sha256": digest}
    return contract


def _select_ledger_candidates(
    merged: dict[str, Any],
    *,
    requested_h11: int,
    declared_source_digest: str | None,
    max_rows: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Build a deterministic candidate list from the merged terminal ledger."""

    if merged.get("requested_h11") != int(requested_h11):
        raise ReplayConfigurationError("merged candidate ledger physical h11 does not match request")
    funnel = merged.get("terminal_ledger", {}).get("class_funnel")
    if not isinstance(funnel, list):
        raise ReplayConfigurationError("merged candidate ledger has no terminal class funnel")
    selected = []
    identities: set[str] = set()
    for entry in funnel:
        if not isinstance(entry, dict):
            continue
        # Keep the complete funnel in the merged source artifact for class
        # accounting, but only the source writer's authoritative acceptance
        # predicate enters the replay candidate identity set.
        if entry.get("accepted_for_table_1") is not True:
            continue
        witness = entry.get("accepted_witness")
        if not isinstance(witness, dict):
            continue
        source = {
            "declared_source_digest": declared_source_digest,
            "source_digest": declared_source_digest,
            "canonical_polytope_id": entry.get("polytope_id"),
            "polytope_id": entry.get("polytope_id"),
            "global_coordinates": entry.get("global_points") or witness.get("global_points"),
            "global_points": entry.get("global_points") or witness.get("global_points"),
            # The merged ledger's polytope_index is an upstream scan index,
            # not the row number in the durable 05--10 Parquet partitions.
            # The immutable input certificate supplies the latter after the
            # exact source join.
            "source_row": None,
            "population_polytope_index": entry.get("polytope_index"),
            "frst_hash": entry.get("frst_hash"),
            "mirror_h12": int(requested_h11),
        }
        witness_digest = _sha256_bytes(_canonical_json(witness).encode("utf-8"))
        action_digest = entry.get("action_digest") or witness.get("action_digest") or witness_digest
        candidate = dict(entry)
        candidate["action_digest"] = action_digest
        candidate["witness_digest"] = witness_digest
        identity = _row_identity(
            source,
            int(entry.get("frst_class_index", 0)),
            {
                "frst_hash": entry.get("frst_hash"),
                "action_digest": action_digest,
                "witness_digest": witness_digest,
                "accepted_witness": witness,
            },
        )
        if identity in identities:
            raise ReplayConfigurationError(
                f"duplicate authoritative candidate identity in merged ledger: {identity}"
            )
        identities.add(identity)
        selected.append((identity, source, candidate))
    selected.sort(key=lambda row: row[0])
    return selected[:max_rows]


def _read_merged_ledger(
    path: Path, *, expected_sha256: str, requested_h11: int
) -> dict[str, Any]:
    """Rehash the frozen merged artifact immediately before decoding it."""

    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ReplayConfigurationError(
            f"merged artifact changed after preflight: expected {expected_sha256}, got {observed_sha256}"
        )
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ReplayConfigurationError("zstd is required for merged candidate ledger replay")
    completed = subprocess.run([zstd, "-dcq", str(path)], check=False, capture_output=True)
    if completed.returncode != 0:
        raise ReplayConfigurationError(f"cannot read merged candidate ledger: {path}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReplayConfigurationError(f"merged candidate ledger is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("requested_h11") != int(requested_h11):
        raise ReplayConfigurationError("merged candidate ledger physical h11 does not match request")
    funnel = value.get("terminal_ledger", {}).get("class_funnel")
    if not isinstance(funnel, list):
        raise ReplayConfigurationError("merged candidate ledger has no terminal class funnel")
    return value


def _read_enrichment_rows(path: Path, expected_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    """Read immutable source/input certificates keyed by replay row identity."""
    if not path.is_file():
        raise ReplayConfigurationError(f"declared enrichment artifact is missing: {path}")
    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise ReplayConfigurationError(f"enrichment artifact changed after preflight: {path}")
    rows = _zstd_jsonl_read(path)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type") != "row":
            continue
        identity = row.get("row_identity")
        if not isinstance(identity, str) or not identity:
            raise ReplayConfigurationError("enrichment row has no canonical replay row_identity")
        if identity in indexed:
            raise ReplayConfigurationError(f"duplicate enrichment row identity: {identity}")
        indexed[identity] = row
    return indexed


def _attach_enrichment(
    candidate_rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
    enrichment: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Join source/input certificates without changing the candidate set."""
    attached = []
    for identity, source, candidate in candidate_rows:
        evidence = enrichment.get(identity)
        if evidence is not None:
            candidate = dict(candidate)
            candidate["source_record"] = evidence.get("source_record")
            candidate["mpcp_certificate"] = evidence.get("mpcp_certificate")
        attached.append((identity, source, candidate))
    return attached


def _zstd_jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ReplayConfigurationError("zstd is required for replay checkpoints")
    completed = subprocess.run(
        [zstd, "-dcq", str(path)], check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise ReplayResumeError(
            f"cannot read checkpoint {path}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    rows = []
    for line_number, line in enumerate(completed.stdout.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayResumeError(f"invalid checkpoint JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ReplayResumeError(f"checkpoint line {line_number} is not an object")
        rows.append(value)
    return rows


def _zstd_jsonl_write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a zstd level-19 JSONL checkpoint with atomic replacement."""
    zstd = shutil.which("zstd")
    if zstd is None:
        raise ReplayConfigurationError("zstd is required for replay checkpoints")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_name(f".{path.name}.raw-{os.getpid()}-{time.time_ns()}")
    compressed_path = path.with_name(f".{path.name}.zst-{os.getpid()}-{time.time_ns()}")
    try:
        with raw_path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(_canonical_json(row) + "\n")
        completed = subprocess.run(
            [zstd, "-19", "-q", "-f", "-o", str(compressed_path), str(raw_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ReplayConfigurationError(
                f"zstd checkpoint compression failed: {completed.stderr.strip()}"
            )
        os.replace(compressed_path, path)
    finally:
        raw_path.unlink(missing_ok=True)
        compressed_path.unlink(missing_ok=True)


def _checkpoint_state(path: Path, config: dict[str, Any], *, resume: bool) -> tuple[list[dict[str, Any]], set[str]]:
    rows = _zstd_jsonl_read(path)
    if not rows:
        if resume:
            raise ReplayResumeError(f"cannot resume missing or empty checkpoint: {path}")
        header = {
            "record_type": "header",
            "checkpoint_schema_version": REPLAY_CHECKPOINT_SCHEMA_VERSION,
            "config": config,
            "config_sha256": _config_digest(config),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return [header], set()
    header = rows[0]
    if header.get("record_type") != "header" or header.get("checkpoint_schema_version") != REPLAY_CHECKPOINT_SCHEMA_VERSION:
        raise ReplayResumeError("checkpoint header schema is not supported")
    expected = _config_digest(config)
    if header.get("config_sha256") != expected or header.get("config") != config:
        raise ReplayResumeError("checkpoint frozen config/input/code/handoff hashes do not match")
    identities: set[str] = set()
    for row in rows[1:]:
        if row.get("record_type") != "row" or row.get("row_identity") is None:
            continue
        identity = str(row["row_identity"])
        if identity in identities:
            raise ReplayResumeError(f"checkpoint contains duplicate row identity: {identity}")
        identities.add(identity)
    return rows, identities


def _row_identity(source: dict[str, Any], class_index: int, action: dict[str, Any]) -> str:
    """Return an identity independent of enumeration order or class index."""

    witness = action.get("accepted_witness") or action.get("witness") or action.get("action")
    witness_digest = action.get("witness_digest")
    if witness_digest is None and isinstance(witness, dict):
        witness_digest = _sha256_bytes(_canonical_json(witness).encode("utf-8"))
    action_digest = action.get("action_digest") or action.get("digest") or witness_digest
    if not isinstance(action_digest, str) or not action_digest:
        raise ReplayConfigurationError(
            "candidate action identity requires an action digest or complete accepted_witness"
        )
    frst_hash = action.get("frst_hash") or source.get("frst_hash")
    payload = {
        "declared_source_digest": source.get(
            "declared_source_digest", source.get("source_digest")
        ),
        "canonical_polytope_id": source.get(
            "canonical_polytope_id", source.get("polytope_id")
        ),
        "global_coordinates": source.get(
            "global_coordinates", source.get("global_points")
        ),
        "frst_hash": frst_hash,
        "action_digest": action_digest,
        "witness_digest": witness_digest,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _live_replay_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    """Retain complete live exact-kernel evidence separate from input identity."""
    names = (
        "fixed_components",
        "smoothness",
        "fixed_locus_euler",
        "h2_action",
        "eq_4_45_parity",
        "hodge_split",
        "action_involution",
        "fan_preservation",
    )
    return {name: candidate.get(name) for name in names if name in candidate}


def _select_certificate_action(
    candidates: Any, certificate: dict[str, Any]
) -> dict[str, Any]:
    """Select exactly one live action matching the immutable input witness."""
    witness = certificate.get("action", {}).get("witness")
    if not isinstance(witness, dict):
        raise ReplayConfigurationError("input certificate action witness is missing")
    if not isinstance(candidates, (list, tuple)):
        raise ReplayConfigurationError("exact replay returned no candidate list")

    def matches(candidate: Any) -> bool:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("action"), dict):
            return False
        action = candidate["action"]
        for field in ("lattice_matrix", "torus_shift", "lambda_f"):
            if action.get(field) != witness.get(field):
                return False
        for field in ("matrix_id", "candidate_id"):
            observed = action.get(field)
            if observed is not None and observed != witness.get(field):
                return False
        return True

    matching = [candidate for candidate in candidates if matches(candidate)]
    if len(matching) != 1:
        raise ReplayConfigurationError(
            "input certificate action has "
            f"{len(matching)} exact live matches; expected one"
        )
    selected = dict(matching[0])
    selected_action = dict(selected["action"])
    if not selected.get("action_digest"):
        selected["action_digest"] = certificate["action"].get("digest")
    selected["action"] = selected_action
    return selected


def _verify_replay_row(poly: Any, source: dict[str, Any], requested_h11: int) -> None:
    if int(source.get("mirror_h12", -1)) != int(requested_h11):
        raise ReplayConfigurationError("source mirror_h12 does not equal requested physical h11")
    poly_h11 = getattr(poly, "h11", None)
    if not callable(poly_h11) or int(poly_h11()) != int(requested_h11):
        raise ReplayConfigurationError("CYTools Polytope.h11 does not equal requested physical h11")


def _validate_enriched_evidence(
    source: dict[str, Any],
    candidate: dict[str, Any],
    source_record: Any,
    mpcp_certificate: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Validate optional immutable source and action-keyed MPCP evidence."""

    if not isinstance(source_record, dict) or not isinstance(mpcp_certificate, dict):
        return None, None, "schema-valid source_record and mpcp_certificate are both required"
    record_source = source_record.get("source")
    if not isinstance(record_source, dict):
        return None, None, "source_record.source is not a schema-valid mapping"
    required_source = ("source_row", "polytope_id", "global_points")
    if any(record_source.get(field) in (None, "") for field in required_source):
        return None, None, "immutable source_record is missing required identity fields"
    source_digest = record_source.get("source_sha256", record_source.get("parquet_sha256"))
    if not isinstance(source_digest, str) or not source_digest:
        return None, None, "immutable source_record has no source SHA-256"
    if record_source.get("polytope_id") != source.get("canonical_polytope_id"):
        return None, None, "immutable source_record polytope_id does not match candidate"
    if source.get("source_row") is not None and record_source.get("source_row") != source.get("source_row"):
        return None, None, "immutable source_record source_row does not match candidate"
    for field in ("source_directory_sha256", "source_input_sha256"):
        observed = source_record.get(field, record_source.get(field))
        if observed not in (None, source.get("declared_source_digest")):
            return None, None, f"immutable source_record {field} does not match source contract"
    certificate_source = mpcp_certificate.get("source")
    certificate_frst = mpcp_certificate.get("frst")
    certificate_action = mpcp_certificate.get("action")
    if not all(isinstance(value, dict) for value in (certificate_source, certificate_frst, certificate_action)):
        return None, None, "mpcp_certificate identity sections are incomplete"
    if certificate_source.get("source_sha256") != source_digest:
        return None, None, "mpcp_certificate source digest does not match source_record"
    if certificate_source.get("polytope_id") != record_source.get("polytope_id"):
        return None, None, "mpcp_certificate polytope_id does not match source_record"
    if certificate_source.get("global_points") != record_source.get("global_points"):
        return None, None, "mpcp_certificate coordinates do not match source_record"
    if certificate_frst.get("frst_hash") != source.get("frst_hash"):
        return None, None, "mpcp_certificate FRST hash does not match candidate"
    selected_frst = source_record.get("selected_frst")
    if mpcp_certificate.get("certificate_schema_version") == "cyaxiverse-population-mpcp-certificate-1.0":
        if not isinstance(selected_frst, dict):
            return None, None, "population certificate requires immutable selected_frst evidence"
        for field in ("points", "simplices", "simplices_index_space"):
            if certificate_frst.get(field) != selected_frst.get(field):
                return None, None, f"population certificate FRST {field} does not match source_record"
    witness = candidate.get("accepted_witness")
    witness_digest = _sha256_bytes(_canonical_json(witness).encode("utf-8")) if isinstance(witness, dict) else None
    input_certificate = mpcp_certificate.get("certificate_schema_version") == "cyaxiverse-population-input-certificate-1.0"
    if input_certificate:
        action_witness = certificate_action.get("witness")
        if not isinstance(action_witness, dict):
            return None, None, "population input certificate action witness is missing"
        for field in ("matrix_id", "candidate_id", "torus_shift", "lambda_f"):
            if field in witness and action_witness.get(field) != witness.get(field):
                return None, None, f"population input action {field} does not match ledger witness"
    else:
        expected_action_digest = candidate.get("action_digest") or witness_digest
        if certificate_action.get("digest") != expected_action_digest:
            return None, None, "mpcp_certificate action key does not match candidate"
    try:
        from mpcp_bounded_analysis import validate_replay_certificate
        validator_source = {
            **record_source,
            "source_sha256": source_digest,
            "parquet_sha256": source_digest,
        }
        if mpcp_certificate.get("certificate_schema_version") == "cyaxiverse-population-input-certificate-1.0":
            from mpcp_bounded_analysis import validate_population_input_certificate

            validation = validate_population_input_certificate(
                mpcp_certificate,
                source=validator_source,
                frst_hash=source.get("frst_hash"),
                action=certificate_action.get("witness"),
                requested_h11=int(source.get("mirror_h12")),
            )
        elif mpcp_certificate.get("certificate_schema_version") == "cyaxiverse-population-mpcp-certificate-1.0":
            from mpcp_bounded_analysis import validate_population_replay_certificate

            validation = validate_population_replay_certificate(
                mpcp_certificate,
                source=validator_source,
                frst_hash=source.get("frst_hash"),
                action=certificate_action.get("witness"),
                requested_h11=int(source.get("mirror_h12")),
            )
        else:
            from mpcp_bounded_analysis import validate_replay_certificate

            validation = validate_replay_certificate(
                mpcp_certificate,
                source=validator_source,
                frst_hash=source.get("frst_hash"),
                action=certificate_action.get("witness"),
            )
    except (ImportError, TypeError, ValueError) as exc:
        return None, None, f"mpcp_certificate validator unavailable: {exc}"
    if not isinstance(validation, dict) or validation.get("status") != "valid":
        reasons = validation.get("reasons", []) if isinstance(validation, dict) else []
        detail = "; ".join(str(reason) for reason in reasons) or "certificate validation failed"
        return None, None, detail
    return source_record, mpcp_certificate, None


def _load_replay_geometry(
    source: dict[str, Any],
    candidate: dict[str, Any],
    source_record: dict[str, Any],
    mpcp_certificate: dict[str, Any],
    requested_h11: int,
) -> tuple[Any | None, Any | None, str | None]:
    """Construct and verify the immutable Polytope and selected FRST.

    A certificate is a witness, not a geometry serialization.  The source
    record must therefore carry the exact global lattice coordinates and the
    selected FRST's local coordinates and simplices.  Reconstructing the
    FRST from an identity, a class index, or a certificate summary would
    silently choose a different triangulation and is rejected.
    """

    try:
        from mpcp_bounded_analysis import (
            _construct_polytope,
            _construct_selected_triangulation,
            point_identity,
            triangulation_identity,
        )
    except (ImportError, AttributeError) as exc:
        return None, None, f"CYTools replay loader is unavailable: {type(exc).__name__}: {exc}"

    record_source = source_record.get("source")
    selected_frst = source_record.get("selected_frst")
    if not isinstance(record_source, dict):
        return None, None, "immutable source_record.source is required for geometry reconstruction"
    if not isinstance(selected_frst, dict):
        return None, None, "immutable source_record.selected_frst is required for geometry reconstruction"
    if selected_frst.get("simplices") is None or selected_frst.get("points") is None:
        return None, None, "selected_frst must include exact local points and simplices"
    if selected_frst.get("simplices_index_space") != "triangulation_local":
        return None, None, "selected_frst must declare triangulation_local simplices"

    global_points = record_source.get("global_points")
    polytope_id = record_source.get("polytope_id")
    source_row = record_source.get("source_row")
    source_sha256 = record_source.get("source_sha256", record_source.get("parquet_sha256"))
    if global_points is None or polytope_id in (None, "") or source_row in (None, ""):
        return None, None, "immutable source_record is missing global coordinates or source row identity"
    if not isinstance(source_sha256, str) or not source_sha256:
        return None, None, "immutable source_record is missing the source Parquet SHA-256"
    if polytope_id != source.get("canonical_polytope_id"):
        return None, None, "source_record polytope_id does not match the candidate ledger"
    certificate_source = mpcp_certificate.get("source", {})
    if certificate_source.get("source_sha256") != source_sha256:
        return None, None, "certificate source SHA-256 does not match source_record"
    if certificate_source.get("polytope_id") != polytope_id:
        return None, None, "certificate polytope_id does not match source_record"
    if certificate_source.get("global_points") != global_points:
        return None, None, "certificate global coordinates do not match source_record"
    if source.get("source_row") is not None and record_source.get("source_row") != source.get("source_row"):
        return None, None, "source_record source row does not match the candidate ledger"
    for field in ("source_directory_sha256", "source_input_sha256"):
        observed = source_record.get(field, record_source.get(field))
        if observed not in (None, source.get("declared_source_digest")):
            return None, None, f"source_record {field} does not match the frozen source contract"

    try:
        expected_polytope_id = f"lattice-points-sha256:{point_identity(global_points)}"
    except (TypeError, ValueError) as exc:
        return None, None, f"source global coordinates are invalid: {exc}"
    if expected_polytope_id != polytope_id:
        return None, None, "source polytope_id does not match canonical global coordinates"

    source_evidence = {
        "status": "source_identity_ready",
        "terminal": False,
        "polytope_id": polytope_id,
        "source_row": source_row,
        "source_sha256": source_sha256,
        "global_points": global_points,
        "global_point_count": len(global_points),
    }
    record = {"source": record_source, "selected_frst": selected_frst}
    try:
        poly, poly_status = _construct_polytope(record, source_evidence)
    except (TypeError, ValueError, RuntimeError) as exc:
        return None, None, f"CYTools Polytope construction failed: {type(exc).__name__}: {exc}"
    if poly is None:
        return None, None, str(poly_status.get("reason", "CYTools Polytope construction unavailable"))
    try:
        _verify_replay_row(poly, {"mirror_h12": requested_h11}, requested_h11)
    except (TypeError, ValueError, ReplayConfigurationError) as exc:
        return None, None, f"physical h11 verification failed: {exc}"

    try:
        triangulation, tri_status = _construct_selected_triangulation(poly, record)
    except (TypeError, ValueError, RuntimeError) as exc:
        return None, None, f"selected FRST reconstruction failed: {type(exc).__name__}: {exc}"
    if triangulation is None:
        return None, None, str(tri_status.get("reason", "selected FRST reconstruction unavailable"))
    try:
        actual_frst_hash = triangulation_identity(triangulation)
    except (TypeError, ValueError, AttributeError) as exc:
        return None, None, f"selected FRST identity is unavailable: {exc}"
    expected_frst_hash = candidate.get("frst_hash")
    certificate_frst_hash = mpcp_certificate.get("frst", {}).get("frst_hash")
    if expected_frst_hash != actual_frst_hash:
        return None, None, "reconstructed selected FRST hash does not match the candidate ledger"
    if certificate_frst_hash != actual_frst_hash:
        return None, None, "reconstructed selected FRST hash does not match the MPCP certificate"
    return poly, triangulation, None


def _replay_candidate_rows(
    poly: Any,
    source: dict[str, Any],
    class_index: int,
    triangulation: Any,
    *,
    source_record: dict[str, Any] | None = None,
    mpcp_certificate: dict[str, Any] | None = None,
    ledger_candidate: dict[str, Any] | None = None,
    geometry_error: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one ledger candidate with explicit immutable evidence.

    A missing source record or action-keyed bounded certificate is a terminal
    unsupported result.  It never enters the exact kernels and cannot be
    promoted by a summary or population count.
    """
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    candidate = ledger_candidate or {}
    supplied_source_record = source_record
    supplied_certificate = mpcp_certificate
    source_record = source_record if source_record is not None else source
    common = {
        "source": source,
        "source_record": source_record,
        "frst_class_index": int(class_index),
        "frst_hash": candidate.get("frst_hash") or source.get("frst_hash"),
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate": candidate,
        "formula_gate": {
            "status": "not_reached",
            "eq_4_46": "requires explicit Eq.4.35 certificate",
            "eq_4_50": "requires exact containment",
        },
        "resource_limit_reason": None,
        "attempt": 1,
        "started_utc": started_utc,
    }
    common["replay_output_evidence"] = {}
    common["replay_output_evidence_digest"] = _sha256_bytes(b"{}")
    action = candidate.get("accepted_witness") or candidate.get("action") or {}
    action_digest = candidate.get("action_digest") or action.get("action_digest")
    witness_digest = candidate.get("witness_digest")
    if witness_digest is None and isinstance(action, dict):
        witness_digest = _sha256_bytes(_canonical_json(action).encode("utf-8"))
    identity_action = {
        "frst_hash": common["frst_hash"],
        "action_digest": action_digest or witness_digest,
        "witness_digest": witness_digest,
        "action": action,
    }
    common["row_identity"] = _row_identity(source, class_index, identity_action)
    if geometry_error is not None:
        return [{
            **common,
            "terminal_status": "exact_geometry_unavailable",
            "reason": geometry_error,
            "mpcp_certificate_status": "valid",
            "elapsed_seconds": time.perf_counter() - started,
        }]
    validated_source, validated_certificate, evidence_error = _validate_enriched_evidence(
        source, candidate, supplied_source_record, supplied_certificate
    )
    if evidence_error is not None:
        row = {
            **common,
            "terminal_status": "exact_certificate_unavailable",
            "reason": evidence_error,
            "mpcp_certificate_status": "invalid" if supplied_certificate is not None else "unavailable",
            "elapsed_seconds": time.perf_counter() - started,
        }
        return [row]
    try:
        topology = dict(extract_topology(triangulation.get_cy(), triangulation))
        cones = _triangulation_cones(poly, triangulation)
        topology["fixed_surface_n_s"] = identity_fixed_surface_n_s_table(cones, triangulation)
        topology["compute_general_fixed_surface_n_s"] = True
        topology["non_smooth_facet_dual_vertices"] = facets_with_non_smooth_cones(poly, triangulation)
        if validated_certificate is not None:
            # The input certificate names the authoritative inherited action.
            # Reconstructing the generic trilayer identity-action population
            # here would silently replace it with a different witness.
            from trilayer_involutions import evaluate_exact_trilayer_action

            certificate_action = validated_certificate["action"]["witness"]
            structural = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "terminal_status": "structurally_reconstructed",
                "polytope_id": source.get("canonical_polytope_id"),
                "frst_hash": common["frst_hash"],
                "action": dict(certificate_action),
            }
            reconstruction = {
                "candidates": [
                    evaluate_exact_trilayer_action(
                        poly,
                        triangulation,
                        topology,
                        structural,
                        source_record=validated_source,
                        mpcp_certificate=validated_certificate,
                    )
                ]
            }
        else:
            reconstruction = reconstruct_trilayer_actions(
                poly,
                triangulation,
                topology,
                source_record=validated_source,
                mpcp_certificate=validated_certificate,
            )
    except BaseException as exc:
        return [{
            **common,
            "terminal_status": "candidate_evaluation_failed",
            "failure_category": "exact_reconstruction_failure",
            "exception": f"{type(exc).__name__}: {exc}",
            "resource_limit_reason": "downstream exact kernel failure; fail closed",
            "elapsed_seconds": time.perf_counter() - started,
        }]
    candidates = reconstruction.get("candidates", [])
    if validated_certificate is not None:
        try:
            selected = _select_certificate_action(candidates, validated_certificate)
        except ReplayConfigurationError as exc:
            return [{
                **common,
                "terminal_status": "exact_certificate_unavailable",
                "reason": str(exc),
                "mpcp_certificate_status": "valid",
                "elapsed_seconds": time.perf_counter() - started,
            }]
        candidates = [selected]
    rows = []
    for candidate in candidates:
        action = candidate.get("action", {}) if isinstance(candidate, dict) else {}
        live_evidence = _live_replay_evidence(candidate)
        rows.append(
            {
                "source": source,
                "source_record": source_record,
                "frst_class_index": int(class_index),
                "frst_hash": candidate.get("frst_hash"),
                "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                "candidate": candidate,
                "replay_output_evidence": live_evidence,
                "replay_output_evidence_digest": _sha256_bytes(_canonical_json(live_evidence).encode("utf-8")),
                "terminal_status": candidate.get("terminal_status", "candidate_evaluation_failed"),
                # The input certificate row is the immutable replay identity;
                # downstream kernel metadata must not create a second row on
                # resume merely because its action digest has extra fields.
                "row_identity": common["row_identity"] if validated_certificate is not None else _row_identity(source, class_index, candidate),
                "formula_gate": {
                    "eq_4_46": "conditional on Eq.4.35 certificate; otherwise actual invariant/Newton support",
                    "eq_4_50": "only after exact containment",
                    "status": "kernel_enforced",
                },
                "resource_limit_reason": None,
                "attempt": 1,
                "started_utc": started_utc,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    if not rows:
        rows.append(
            {
                "source": source,
                "frst_class_index": int(class_index),
                "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
                "terminal_status": "no_source_candidate",
                "row_identity": _row_identity(source, class_index, {
                    "frst_hash": common["frst_hash"], "action_digest": action_digest or "unavailable",
                }),
                "candidate": None,
                "formula_gate": {"status": "not_reached"},
                "resource_limit_reason": None,
                "attempt": 1,
                "started_utc": started_utc,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return rows


def exact_replay(args: argparse.Namespace) -> dict[str, Any]:
    """Replay only deterministic candidates declared by the merged ledger."""
    workers = int(getattr(args, "workers", 1))
    if workers > MAX_REPLAY_WORKERS:
        raise ReplayConfigurationError(f"hard worker maximum is {MAX_REPLAY_WORKERS}")
    parquet_dir = Path(args.parquet_dir).expanduser().resolve()
    preflight = run_population_preflight(
        Path(__file__).resolve().parents[1], args.h11, parquet_dir
    )
    if not isinstance(preflight, dict) or preflight.get("status") != "passed":
        raise ReplayConfigurationError("successful five-handoff population preflight is required")
    config = _freeze_replay_config(args, preflight)
    max_rows = int(config["max_rows"])
    checkpoint_arg = getattr(args, "checkpoint", None) or getattr(args, "terminal_ledger", None)
    output_arg = getattr(args, "output", None)
    checkpoint = Path(checkpoint_arg or (str(output_arg) + ".checkpoint.jsonl.zst" if output_arg else "exact-replay.checkpoint.jsonl.zst"))
    resume = bool(getattr(args, "resume", False))
    if checkpoint.exists() and not resume:
        raise ReplayConfigurationError(f"refusing implicit checkpoint overwrite: {checkpoint}")
    if max_rows == 0 or bool(getattr(args, "dry_run", False)):
        rows, identities = _checkpoint_state(checkpoint, config, resume=resume)
        summary = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "status": "dry_run",
            "config": config,
            "checkpoint": str(checkpoint.resolve()),
            "rows_evaluated": 0,
            "terminal_status_counts": {},
            "duplicate_count": 0,
            "database_writes": 0,
            "scientific_labels": config["labels"],
        }
        if not checkpoint.exists():
            _zstd_jsonl_write_atomic(checkpoint, rows)
        return summary

    merged_contract = config["artifact_contract"]["merged_artifact"]
    merged_path = Path(merged_contract["path"])
    if not merged_path.is_file():
        raise ReplayConfigurationError(f"declared merged candidate ledger is missing: {merged_path}")
    merged = _read_merged_ledger(
        merged_path,
        expected_sha256=merged_contract["sha256"],
        requested_h11=int(args.h11),
    )
    source_contract = preflight.get("source_contract", {})
    declared_digest = source_contract.get("expected_directory_digest")
    # The cap is deliberately applied after canonical sorting of candidate
    # identities. It never limits source polytopes or invokes FRST discovery.
    candidate_rows = _select_ledger_candidates(
        merged,
        requested_h11=int(args.h11),
        declared_source_digest=declared_digest,
        max_rows=max_rows,
    )
    enrichment_contract = config.get("enrichment")
    if isinstance(enrichment_contract, dict):
        enrichment_rows = _read_enrichment_rows(
            Path(enrichment_contract["path"]), enrichment_contract["sha256"]
        )
        candidate_rows = _attach_enrichment(candidate_rows, enrichment_rows)

    valid_evidence = 0
    for _, source, candidate in candidate_rows:
        _, _, evidence_error = _validate_enriched_evidence(
            source,
            candidate,
            candidate.get("source_record"),
            candidate.get("mpcp_certificate"),
        )
        if evidence_error is None:
            valid_evidence += 1
    all_unavailable = valid_evidence == 0
    if all_unavailable:
        allow_smoke = bool(getattr(args, "allow_terminal_only_smoke", False))
        if not allow_smoke or max_rows > 2:
            raise ReplayConfigurationError(
                "no schema-valid certificates are available; full execution is blocked "
                "(pass --allow-terminal-only-smoke with --max-rows <= 2 for plumbing smoke)"
            )
    else:
        config["labels"] = {
            **config["labels"],
            "execution_mode": "schema_validated_replay",
            "scientific_result": "not_established",
        }
    rows, identities = _checkpoint_state(checkpoint, config, resume=resume)

    status_counts: dict[str, int] = {}
    for existing in rows[1:]:
        if existing.get("record_type") == "row":
            status = str(existing.get("terminal_status"))
            status_counts[status] = status_counts.get(status, 0) + 1
    source_ledger = merged.get("terminal_ledger", {})
    funnel = source_ledger.get("class_funnel", [])
    source_accounting = {
        "class_funnel_count": len(funnel) if isinstance(funnel, list) else 0,
        "authoritative_candidate_count": sum(
            isinstance(entry, dict)
            and entry.get("accepted_for_table_1") is True
            and isinstance(entry.get("accepted_witness"), dict)
            for entry in funnel
        ) if isinstance(funnel, list) else 0,
        "rejected_or_noncandidate_count": sum(
            not (
                isinstance(entry, dict)
                and entry.get("accepted_for_table_1") is True
                and isinstance(entry.get("accepted_witness"), dict)
            )
            for entry in funnel
        ) if isinstance(funnel, list) else 0,
    }
    duplicate_count = int(
        source_ledger.get("duplicate_count", source_ledger.get("duplicates", 0)) or 0
    )
    evaluated = 0
    since_checkpoint = 0
    cursor = 0
    for identity, source, candidate in candidate_rows:
        if cursor % int(config["shard_count"]) != int(config["shard_index"]):
            cursor += 1
            continue
        cursor += 1
        if identity in identities:
            # Resume skips are not source-ledger duplicates and are never
            # appended as duplicate rows.
            continue
        source_record = candidate.get("source_record")
        mpcp_certificate = candidate.get("mpcp_certificate")
        poly = triangulation = None
        geometry_error = None
        if source_record is not None and mpcp_certificate is not None:
            poly, triangulation, geometry_error = _load_replay_geometry(
                source,
                candidate,
                source_record,
                mpcp_certificate,
                int(args.h11),
            )
        candidate_row = _replay_candidate_rows(
            poly,
            source,
            int(candidate.get("frst_class_index", 0)),
            triangulation,
            source_record=source_record,
            mpcp_certificate=mpcp_certificate,
            ledger_candidate=candidate,
            geometry_error=geometry_error,
        )[0]
        candidate_row["record_type"] = "row"
        candidate_row["cursor"] = cursor
        rows.append(candidate_row)
        identities.add(identity)
        evaluated += 1
        since_checkpoint += 1
        status = str(candidate_row.get("terminal_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if since_checkpoint >= int(config["checkpoint_interval"]):
            _zstd_jsonl_write_atomic(checkpoint, rows)
            since_checkpoint = 0
    if since_checkpoint or not checkpoint.exists():
        _zstd_jsonl_write_atomic(checkpoint, rows)
    summary = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "completed",
        "config": config,
        "checkpoint": str(checkpoint.resolve()),
        "rows_evaluated": evaluated,
        "terminal_status_counts": status_counts,
        "duplicate_count": duplicate_count,
        "database_writes": 0,
        "scientific_labels": config["labels"],
        "execution_mode": "infrastructure_smoke_only" if all_unavailable else "schema_validated_replay",
        "scientific_result": "no_scientific_result" if all_unavailable else "not_established",
        "source_ledger_accounting": source_accounting,
    }
    if output_arg:
        output = Path(output_arg).expanduser().resolve()
        if output.exists():
            raise ReplayConfigurationError(f"refusing to overwrite replay output: {output}")
        _zstd_jsonl_write_atomic(output, rows + [{"record_type": "summary", **summary}])
    return summary


def reproduce(args):
    """Run the bounded replay or the legacy population audit.

    The two entry points use distinct argument namespaces.  This preserves
    the established audit API while making the schema-3.0 replay available as
    the default for callers created by ``build_argument_parser``.
    """
    if any(
        hasattr(args, name)
        for name in ("dry_run", "max_rows", "workers", "checkpoint", "enrichment")
    ):
        return exact_replay(args)
    return _legacy_reproduce(args)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the bounded schema-3.0 replay CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument(
        "--h11",
        type=int,
        required=True,
        choices=(4, 5),
        help="Explicit physical h11 selected by mirror_h12 and checked with CYTools.",
    )
    parser.add_argument(
        "--max-rows", "--limit", dest="max_rows", type=int, default=0,
        help="Maximum favorable source rows to evaluate; 0 performs a bounded dry run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run preflight and freeze provenance without loading geometry rows.")
    parser.add_argument("--workers", type=int, default=1, help="Sequential worker setting (hard maximum 4).")
    parser.add_argument("--shard-count", type=int, default=1, help="Deterministic source-row shard count (maximum 4).")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based deterministic shard index.")
    parser.add_argument("--checkpoint", type=Path, help="Append-only zstd level-19 JSONL checkpoint.")
    parser.add_argument("--terminal-ledger", type=Path, help="Compatibility alias for the append-only terminal checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume only when frozen config and hashes match.")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=32,
        help="Atomically rewrite the zstd checkpoint after this many new rows.",
    )
    parser.add_argument(
        "--allow-terminal-only-smoke",
        action="store_true",
        help="Allow at most two terminal-only rows for infrastructure plumbing validation.",
    )
    parser.add_argument("--orientifold-reason-diagnostics", action="store_true", help="Retain exact-kernel reason diagnostics (schema-3.0 compatibility flag).")
    parser.add_argument("--orientifold-audit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enrichment",
        type=Path,
        help="zstd JSONL source/input-certificate artifact produced by the bounded enrichment runner.",
    )
    return parser


def _parse_args(argv=None):
    """Parse the established source-matched audit-driver arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--h11", type=int, default=4)
    parser.add_argument("--limit", type=int, default=10**9)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--keep-details", action="store_true")
    parser.add_argument("--orientifold-audit", action="store_true")
    parser.add_argument("--terminal-ledger", type=Path)
    parser.add_argument("--orientifold-reason-diagnostics", action="store_true")
    parser.add_argument("--export-kaehler-points", action="store_true")
    parser.add_argument("--model-stage", action="store_true")
    parser.add_argument("--gs", type=float, default=0.5)
    parser.add_argument("--w0-real", type=float, default=1.0)
    parser.add_argument("--w0-imag", type=float, default=0.0)
    parser.add_argument(
        "--qcd-divisor-domain",
        choices=("all_prime", "leading_nonself"),
        default="all_prime",
    )
    parser.add_argument("--julia-binary", default="julia")
    parser.add_argument("--julia-project", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.shard_count > 1:
        if args.output is not None:
            args.output = _shard_suffix_path(args.output, args.shard_index, args.shard_count)
        if args.terminal_ledger is not None:
            args.terminal_ledger = _shard_suffix_path(
                args.terminal_ledger, args.shard_index, args.shard_count
            )
    if args.orientifold_reason_diagnostics and not args.orientifold_audit:
        parser.error("--orientifold-reason-diagnostics requires --orientifold-audit")
    if args.orientifold_audit and args.output is None and args.terminal_ledger is None:
        parser.error("--orientifold-audit requires --terminal-ledger when --output is absent")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.terminal_ledger is not None and (
        args.terminal_ledger.exists()
        or Path(f"{args.terminal_ledger}.summary.json").exists()
    ):
        parser.error(f"refusing to overwrite existing terminal ledger: {args.terminal_ledger}")
    return args


def main(argv=None):
    """Run either the bounded replay CLI or the established audit CLI."""
    values = list(sys.argv[1:] if argv is None else argv)
    replay_flags = {
        "--dry-run",
        "--workers",
        "--checkpoint",
        "--enrichment",
        "--max-rows",
        "--allow-terminal-only-smoke",
    }
    if any(flag in values for flag in replay_flags):
        parser = build_argument_parser()
        args = parser.parse_args(values)
        try:
            result = reproduce(args)
        except (ReplayConfigurationError, ReplayResumeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    args = _parse_args(values)
    result = reproduce(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
