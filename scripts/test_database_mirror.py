#!/usr/bin/env python3
"""Focused test for the local Hugging Face KS Parquet mirror adapter."""

from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from generate_geometric_data_multitriangulation import load_mirror_polytopes


def main():
    vertices = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, -1, -1, -1],
    ]
    with tempfile.TemporaryDirectory() as directory:
        parquet_path = Path(directory) / "polytopes-4d-05-vertices.parquet"
        pq.write_table(
            pa.table(
                {
                    "vertices": [vertices],
                    "vertex_count": [5],
                    # The P^4 row is h11=101, h12=1 in the mirror orientation.
                    "h11": [101],
                    "h12": [1],
                }
            ),
            parquet_path,
        )
        records = load_mirror_polytopes(directory, h11=1, limit=1, favorable=True)
        assert len(records) == 1
        poly, source = records[0]
        assert int(poly.h11()) == 1
        assert int(poly.h21()) == 101
        assert source["mirror_h11"] == 101
        assert source["mirror_h12"] == 1
        assert source["row_index"] == 0
        assert Path(source["parquet_file"]).resolve() == parquet_path.resolve()
    print("KS Parquet mirror adapter test passed")


if __name__ == "__main__":
    main()
