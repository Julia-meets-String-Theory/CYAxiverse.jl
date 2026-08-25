"""Focused tests for the raw-FRST topology cache codec."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

import generate_stage2_eft_reference as stage2_entrypoint
import glimmers_raw_frst as raw_frst


def sample_topology():
    return {
        "h11": 2,
        "h21": 3,
        "basis": np.asarray([10, 11], dtype=np.int64),
        "basis_matrix": np.asarray(
            [[1, 0, 2, 0, 0], [0, -3, 0, 0, 0]], dtype=np.int64
        ),
        "glsm": np.asarray([[1, 0, 2], [0, -3, 0]], dtype=np.int64),
        "prime_toric_divisors": np.asarray([0, 2, 4], dtype=np.int64),
        "kappa": np.asarray(
            [[0, 0, 0, 1.5], [0, 1, 2, -2.0], [1, 1, 1, 4.25]], dtype=np.float64
        ),
        "c2": np.asarray([24.0, 12.0]),
        "mori_cone": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "kahler_cone_hyperplanes": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "face_restriction_dim2": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "kahler_cone_rays": None,
    }


def cache_metadata():
    return {
        "schema_version": raw_frst.TOPOLOGY_CACHE_SCHEMA_VERSION,
        "h11": 2,
        "h21": 3,
        "geometry_id": "raw-frst:test",
        "raw_geometry_id": "raw-frst:test",
        "polytope_id": "polytope:test",
        "full_triangulation_hash": "triangulation:test",
        "cytools_version": "test-cytools",
        "backend": "qhull",
        "conventions": raw_frst.TOPOLOGY_CACHE_CONVENTIONS,
        "kahler_rays_exported": False,
    }


class TopologyCacheTests(unittest.TestCase):
    def raw_arrays(self):
        return dict(
            h11=2,
            polytope_vertices=np.zeros((5, 4), dtype=np.int64),
            polytope_points=np.arange(24, dtype=np.int64).reshape(6, 4),
            triangulation_labels=np.arange(5, dtype=np.int64),
            triangulation_points=np.arange(20, dtype=np.int64).reshape(5, 4),
            simplices=np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64),
            simplex_indices=np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64),
            metadata={
                "geometry_id": "raw-frst:test",
                "polytope_id": "polytope:test",
                "full_triangulation_hash": "triangulation:test",
            },
        )

    def write_artifact(self, directory, **kwargs):
        path = Path(directory) / "frst.h5"
        arguments = self.raw_arrays()
        arguments.update(kwargs)
        raw_frst.write_raw_frst_artifact(
            path,
            **arguments,
            topology_cache=sample_topology(),
            topology_cache_metadata=cache_metadata(),
        )
        return path

    def test_round_trip_and_sparse_dense_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_artifact(directory)
            persisted = raw_frst.read_raw_frst_artifact(path)
            expected = cache_metadata()
            topology, reason = raw_frst.validate_topology_cache(
                persisted["topology_cache"], expected
            )
            self.assertEqual(reason, "cache identity validated")
            np.testing.assert_array_equal(
                topology["basis_matrix"], sample_topology()["basis_matrix"]
            )
            np.testing.assert_array_equal(topology["basis"], sample_topology()["basis"])
            np.testing.assert_array_equal(topology["kappa"], sample_topology()["kappa"])
            metadata_only = raw_frst.read_raw_frst_artifact(
                path, include_topology_cache=False
            )
            self.assertIsNone(metadata_only["topology_cache"])

    def test_cache_datasets_are_losslessly_compressed_and_shuffled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_artifact(directory)
            with h5py.File(path, "r") as handle:
                cache = handle[raw_frst.TOPOLOGY_CACHE_GROUP]
                self.assertEqual(cache.attrs["compression"], "gzip")
                self.assertEqual(cache.attrs["compression_opts"], 9)
                self.assertTrue(bool(cache.attrs["shuffle"]))

                def assert_properties(group):
                    for item in group.values():
                        if isinstance(item, h5py.Group):
                            assert_properties(item)
                        else:
                            self.assertEqual(item.compression, "gzip")
                            self.assertEqual(item.compression_opts, 9)
                            self.assertTrue(item.shuffle)

                assert_properties(cache)

    def test_identity_mismatch_falls_back_with_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_artifact(directory)
            persisted = raw_frst.read_raw_frst_artifact(path)
            expected = cache_metadata()
            expected["backend"] = "cgal"
            topology, reason = raw_frst.validate_topology_cache(
                persisted["topology_cache"], expected
            )
            self.assertIsNone(topology)
            self.assertIn("identity mismatch", reason)

    def test_cache_write_failure_keeps_raw_frst_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frst.h5"
            with mock.patch.object(
                raw_frst, "_write_topology_cache", side_effect=OSError("simulated cache failure")
            ):
                record = raw_frst.write_raw_frst_artifact(
                    path,
                    **self.raw_arrays(),
                    topology_cache=sample_topology(),
                    topology_cache_metadata=cache_metadata(),
                )
            self.assertEqual(record["stage1_status"], raw_frst.RAW_FRST_ARTIFACT_STATUS)
            self.assertEqual(record["topology_cache_status"], "write_failed")
            persisted = raw_frst.read_raw_frst_artifact(path)
            self.assertEqual(persisted["stage1_status"], raw_frst.RAW_FRST_ARTIFACT_STATUS)
            self.assertIsNone(persisted["topology_cache"])

    def test_stage2_reconstruction_uses_validated_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frst.h5"
            arguments = self.raw_arrays()
            points = np.asarray(arguments["polytope_points"], dtype=int)
            simplices = np.asarray(arguments["simplices"], dtype=int)
            polytope_id = raw_frst.compute_polytope_id(points)
            full_hash = raw_frst.compute_triangulation_hash(simplices)
            geometry_id = raw_frst.build_raw_frst_geometry_id(
                arguments["h11"], polytope_id, full_hash
            )
            metadata = cache_metadata()
            metadata.update(
                {
                    "geometry_id": geometry_id,
                    "raw_geometry_id": geometry_id,
                    "polytope_id": polytope_id,
                    "full_triangulation_hash": full_hash,
                    "cytools_version": getattr(
                        stage2_entrypoint.generator.cytools, "version", None
                    ),
                    "backend": "qhull",
                }
            )
            raw_frst.write_raw_frst_artifact(
                path,
                **arguments,
                topology_cache=sample_topology(),
                topology_cache_metadata=metadata,
            )
            raw_record = {
                key: value
                for key, value in raw_frst.read_raw_frst_artifact(path).items()
                if key in {
                    "h11",
                    "polytope_id",
                    "full_triangulation_hash",
                    "geometry_id",
                    "raw_frst_path",
                }
            }
            fake_polytope = mock.MagicMock()
            fake_polytope.points.return_value = points
            fake_polytope.dim.return_value = 4
            fake_polytope.ambient_dim.return_value = 4
            fake_polytope.is_reflexive.return_value = True
            fake_triangulation = mock.MagicMock()
            fake_triangulation.simplices.return_value = simplices
            fake_polytope.triangulate.return_value = fake_triangulation
            topology_audit = stage2_entrypoint.build_topology_audit_record(
                raw_record, "qhull"
            )
            with mock.patch.object(
                stage2_entrypoint.generator, "Polytope", return_value=fake_polytope
            ), mock.patch.object(
                stage2_entrypoint.generator,
                "validate_frst",
                return_value={"valid": True},
            ):
                persisted, _, _ = stage2_entrypoint.reconstruct_raw_frst(
                    raw_record, "qhull", topology_audit
                )
            self.assertEqual(topology_audit["topology_cache_status"], "hit")
            self.assertEqual(
                topology_audit["topology_cache_reason"], "cache identity validated"
            )
            np.testing.assert_array_equal(
                persisted["topology_override"]["basis_matrix"],
                sample_topology()["basis_matrix"],
            )


if __name__ == "__main__":
    unittest.main()
