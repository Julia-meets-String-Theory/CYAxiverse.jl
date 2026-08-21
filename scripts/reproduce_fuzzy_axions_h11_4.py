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
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
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
    """Return the source [21] trilayer candidate, if one exists.

    For a primal vertex p0, the dual facet is q.p0=-1.  The special source
    construction requires the dual polytope to be the convex hull of that
    facet and one vertex q0 outside it, with q0.p0=+1.  The associated data are
    L=I, t=p0/2, and lambda_f=1.
    """

    primal_vertices = _as_int_rows(poly.vertices())
    dual = poly.dual()
    dual_vertices = _as_int_rows(dual.vertices())
    for p0 in primal_vertices:
        heights = dual_vertices @ p0
        outside = np.flatnonzero(heights > -1)
        if outside.size != 1:
            continue
        q0_index = int(outside[0])
        if int(heights[q0_index]) != 1:
            continue
        if np.any(heights < -1):
            continue
        return {
            "p0": p0.tolist(),
            "q0": dual_vertices[q0_index].tolist(),
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift_numerator": p0.tolist(),
            "torus_shift_denominator": 2,
            "lambda_f": 1,
            "criterion": "Moritz eqs. (4.64)-(4.66), trilayer sufficient condition",
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
        n_rays = vectors.shape[0]
        l2 = 0.0
        l_dp = 0.0
        l_dq = 0.0
        for r in range(1, n_rays + 1):
            for s in range(1, n_rays + 1):
                l2 += tensor[p_index, q_index, r, s]
            l_dp += tensor[p_index, q_index, r, p_index]
            l_dq += tensor[p_index, q_index, r, q_index]
        dpdq = tensor[p_index, q_index, p_index, q_index]
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
    (h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-) = (2,0,0,132) exactly, and
    reproduces the paper's 267-class h11=4 population target exactly when
    applied across all trilayer FRST classes.  See
    validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md.
    """

    result = _fixed_locus_euler_characteristic(poly, triangulation, p0)
    if result["status"] != "computed":
        return {
            "status": "unavailable",
            "reasons": result["reasons"],
            "chi_F_I": None,
            "h21_minus": None,
            "h21_plus": None,
            "components": result["components"],
        }
    cy = triangulation.get_cy()
    chi_X = cy.chi()
    h21_X = cy.h21()
    chi_FI = result["chi_F_I"]
    h21_minus = (chi_FI - chi_X) / 4.0 - 1.0
    h21_plus = h21_X - h21_minus
    is_zero = abs(h21_plus) < 1e-6
    return {
        "status": "h21_plus_zero" if is_zero else "h21_plus_nonzero",
        "reasons": result["reasons"],
        "chi_F_I": chi_FI,
        "h21_minus": h21_minus,
        "h21_plus": h21_plus,
        "components": result["components"],
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


def reproduce(args):
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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument(
        "--h11",
        type=int,
        default=4,
        help="Physical h11 to reproduce from arXiv:2412.12012 Table 1 (tab:ScanData); "
        "3, 4, and 5 have recorded paper targets, others run as diagnostic-only.",
    )
    parser.add_argument("--limit", type=int, default=10**9)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "Total number of parallel shards. The favorable-polytope population "
            "is partitioned deterministically (strided by global index) across "
            "shards so each shard processes a disjoint subset. A sharded run is "
            "always a partial run; combine the per-shard outputs with "
            "scripts/merge_orientifold_shards.py to recover population totals and "
            "completeness. Default 1 (no sharding, behaviour unchanged)."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based index of this shard in [0, --shard-count).",
    )
    parser.add_argument("--progress", type=int, default=50)
    parser.add_argument("--keep-details", action="store_true")
    parser.add_argument("--orientifold-audit", action="store_true")
    parser.add_argument(
        "--terminal-ledger",
        type=Path,
        help=(
            "Write the lossless orientifold terminal ledger to this JSONL path. "
            "When omitted with --output, use <output-stem>.terminal-ledger.jsonl."
        ),
    )
    parser.add_argument(
        "--orientifold-reason-diagnostics",
        action="store_true",
        help=(
            "Record machine-readable general-L fixed-surface reason rows and "
            "certified Chern-class terms. Requires --orientifold-audit."
        ),
    )
    parser.add_argument(
        "--export-kaehler-points",
        action="store_true",
        help=(
            "Export the Algorithm-1 canonical-tip Kahler point (cy_volume, "
            "prime_divisor_volumes, inverse_metric, GLSM charge matrix Q) for "
            "every h21_plus_zero-accepted FRST class. Requires --keep-details "
            "to appear in the output."
        ),
    )
    parser.add_argument(
        "--model-stage",
        action="store_true",
        help=(
            "Run priority 4: per h21_plus_zero-accepted FRST class, export the "
            "Algorithm-1 canonical-tip Kahler point (implies --export-kaehler-points) "
            "and enumerate (QCD divisor, fuzzy axion) models via "
            "CYAxiverse.paper_benchmarks.enumerate_fuzzy_axion_models (bridged through "
            "Julia via HDF5), comparing the total model count against the paper's "
            "target of 3,348 under the acceptance-test discipline in "
            "validation/fuzzy_axions_2412_12012_kaehler_qcd_model_count_scope_20260818.md "
            "Sec. 6."
        ),
    )
    parser.add_argument(
        "--gs",
        type=float,
        default=0.5,
        help="String coupling used for the prefactor P = gs^4/128 (eq. 3.28-3.29); "
        "the paper's stated main-analysis value is 0.5.",
    )
    parser.add_argument(
        "--w0-real",
        type=float,
        default=1.0,
        help="Real part of the flux superpotential W0 (eq. 3.12); no ensemble-specific "
        "value is given in the source, so this defaults to the paper's own Sec. 4.2.1 "
        "hand-worked-example convention W0=1.",
    )
    parser.add_argument("--w0-imag", type=float, default=0.0)
    parser.add_argument(
        "--qcd-divisor-domain",
        choices=("all_prime", "leading_nonself"),
        default="all_prime",
        help=(
            "Which prime toric divisors Algorithm 1's 'for D' loop ranges over. "
            "all_prime (default) is the paper's literal text: all h11+4 of them, "
            "self-pairing included. leading_nonself opts into the candidate "
            "restriction to the h11 leading-instanton divisors minus the fuzzy "
            "axion's own -- the only one of 48 screened restrictions that survives "
            "Table 1's h11=2 row and predicts h11=4 and 5 to within 10%% "
            "(validation/fuzzy_axions_2412_12012_sampler_reverse_engineering_20260818.md). "
            "It forces model_count claim_status to diagnostic_only, and it undershoots "
            "Table 1 at h11>=3 by construction; do not tune towards Table 1 to close that."
        ),
    )
    parser.add_argument(
        "--julia-binary",
        default="julia",
        help="Julia executable used to run scripts/fuzzy_axion_model_stage_driver.jl.",
    )
    parser.add_argument(
        "--julia-project",
        type=Path,
        default=None,
        help="Julia --project path for the model-stage driver; defaults to this "
        "repository's root (the parent of scripts/).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.shard_count > 1:
        # Give each shard distinct output/ledger paths so the immutability
        # guards below apply per shard and shards never collide on disk.
        if args.output is not None:
            args.output = _shard_suffix_path(args.output, args.shard_index, args.shard_count)
        if args.terminal_ledger is not None:
            args.terminal_ledger = _shard_suffix_path(
                args.terminal_ledger, args.shard_index, args.shard_count
            )
    if args.orientifold_reason_diagnostics and not args.orientifold_audit:
        parser.error("--orientifold-reason-diagnostics requires --orientifold-audit")
    if args.orientifold_audit and args.output is None and args.terminal_ledger is None:
        parser.error(
            "--orientifold-audit requires --terminal-ledger when --output is absent"
        )
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.terminal_ledger is not None and (
        args.terminal_ledger.exists()
        or Path(f"{args.terminal_ledger}.summary.json").exists()
    ):
        parser.error(f"refusing to overwrite existing terminal ledger: {args.terminal_ledger}")
    return args


def main(argv=None):
    args = _parse_args(argv)
    result = reproduce(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
