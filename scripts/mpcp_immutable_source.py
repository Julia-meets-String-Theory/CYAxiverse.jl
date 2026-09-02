"""Immutable bounded MPCP replay rows for h11=2 classes 26, 31, and 33.

The rows are copied from the local KS mirror Parquet source.  The parent
polytope has eight lattice points; CYTools' boundary-only FRST has seven
local points because point 7 is interior to a facet.  Keep the global points
and local simplex index space separate in every replay record.
"""

from __future__ import annotations

from copy import deepcopy


SOURCE_DATASET = "calabi-yau-data/polytopes-4d"
SOURCE_PARQUET = "/private/tmp/cyax-ks-mirror-h11-2-3/polytopes-4d-06-vertices.parquet"
SOURCE_PARQUET_SHA256 = "d9708d4067a7ae145697e170bee67e46e9b482c525e8885efe039e417e70ab9c"


_FRST_A = [
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 6],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 4, 6],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 6],
    [0, 2, 3, 4, 5],
    [0, 2, 3, 4, 6],
]
_FRST_B = [
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 6],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 4, 6],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 6],
    [0, 2, 3, 5, 6],
    [0, 2, 4, 5, 6],
    [0, 3, 4, 5, 6],
]

_FRST_IDS = {
    "a": "a62a2ea1c9e512c1e06fd8e65132fed907fca2e10afcc926ca0b6ee7208587f2",
    "b": "57e8dcae74298839b9a208e9411125bed1c73459f8ce4ae14b23152ac6f7ebb0",
}

_ACTION = {
    "lattice_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "torus_shift": {"numerator": [1, 0, 0, 0], "denominator": 2},
    "lambda_f": 1,
}


_ROWS = {
    26: {
        "source_row": 21,
        "polytope_id": "lattice-points-sha256:6e846f704fa1736c6e7abe36a323a97371ba895314b68e3e13576e2cca40051e",
        "global_points": [
            [0, 0, 0, 0], [1, 0, 0, 0], [-4, -1, -1, -1], [0, 0, 0, 1],
            [0, 0, 1, 0], [-1, -1, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0],
        ],
        "expected_hodge": {"h11": 2, "h21": 120, "chi": -236},
    },
    31: {
        "source_row": 27,
        "polytope_id": "lattice-points-sha256:6c754109e315b80c5361bcab91d34140564d0acf7a4143b4c652b633a07f0b78",
        "global_points": [
            [0, 0, 0, 0], [1, 0, 0, 0], [-2, -1, 1, -1], [0, 0, 0, 1],
            [0, 1, 0, 0], [-3, -1, -1, 0], [0, 0, 1, 0], [-1, 0, 0, 0],
        ],
        "expected_hodge": {"h11": 2, "h21": 128, "chi": -252},
    },
    33: {
        "source_row": 29,
        "polytope_id": "lattice-points-sha256:e777ace9bcfd967af24cfa8f56c098e455c0e665a9dc786d05280a6cb5ea12cf",
        "global_points": [
            [0, 0, 0, 0], [1, 0, 0, 0], [-2, 1, -1, -1], [0, 0, 0, 1],
            [0, 0, 1, 0], [-2, -1, 0, 0], [0, 1, 0, 0], [-1, 0, 0, 0],
        ],
        "expected_hodge": {"h11": 2, "h21": 132, "chi": -260},
    },
}


def source_records() -> dict[int, dict]:
    """Return fresh source-keyed replay records for exactly three classes."""

    records = {}
    for index, row in _ROWS.items():
        local_points = row["global_points"][:7]
        selected_frsts = []
        for label, simplices in (("a", _FRST_A), ("b", _FRST_B)):
            selected_frsts.append({
                "identity": _FRST_IDS[label],
                "simplices": deepcopy(simplices),
                "points": deepcopy(local_points),
                "simplices_index_space": "triangulation_local",
            })
        records[index] = {
            "index": index,
            "source": {
                "dataset": SOURCE_DATASET,
                "parquet_file": SOURCE_PARQUET,
                "parquet_sha256": SOURCE_PARQUET_SHA256,
                "source_row": row["source_row"],
                "polytope_id": row["polytope_id"],
                "global_points": deepcopy(row["global_points"]),
                "expected_point_count": 8,
                "expected_boundary_point_count": 7,
                "expected_hodge": deepcopy(row["expected_hodge"]),
            },
            "selected_frst": deepcopy(selected_frsts[0]),
            "selected_frsts": selected_frsts,
            "actions": [deepcopy(_ACTION)],
        }
    return records
