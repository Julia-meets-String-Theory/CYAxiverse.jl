"""Regression tests for CYTools sampler adapter argument contracts."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="cytools-test-cache-"))

import generate_geometric_data_multitriangulation as generator
import probe_h11_491_sampler as probe
from generate_geometric_data_multitriangulation import triangulation_candidates


H491_MANIFEST = os.path.join(
    os.path.dirname(__file__), "manifests", "h11_491_11_ks.json"
)


class RecordingPolytope:
    """Record sampler calls without constructing a real CYTools polytope."""

    labels_not_facet = (0, 1, 2)

    def __init__(self):
        self.calls = []

    def triangulate(self, **kwargs):
        self.calls.append(("triangulate", kwargs))
        return "deterministic"

    def random_triangulations_fast(self, **kwargs):
        self.calls.append(("fast", kwargs))
        return iter(("fast-1", "fast-2")[: kwargs["N"]])

    def random_triangulations_fair(self, **kwargs):
        self.calls.append(("fair", kwargs))
        return iter(("fair-1",))

    def ntfe_frts(self, **kwargs):
        self.calls.append(("ntfe", kwargs))
        return iter(("ntfe-1", "ntfe-2"))

    def random_triangulations_gnn(self, **kwargs):
        self.calls.append(("gnn", kwargs))
        return iter(("gnn-1",))


class RecordingTriangulation:
    """Provide stable simplex data for probe hash regression tests."""

    def __init__(self, faces):
        self.faces = faces

    def simplices(self, **kwargs):
        if kwargs.get("on_faces_dim") == 2:
            return self.faces
        return np.asarray([[0, 1, 2, 3]], dtype=np.int32)


def sampler_candidates(poly, scheme):
    return list(
        triangulation_candidates(
            poly,
            scheme,
            2,
            17,
            "cgal",
            123,
            7,
            11,
            13,
            8,
            0.01,
            25,
            0.2,
            "fast",
            0,
            5,
        )
    )


class TriangulationCandidateTests(unittest.TestCase):
    def test_two_face_hash_ignores_simplex_order_within_each_face(self):
        first = RecordingTriangulation(
            (
                np.asarray([[2, 0, 1], [1, 2, 3]], dtype=np.int32),
                np.asarray([[4, 3, 2]], dtype=np.int32),
            )
        )
        second = RecordingTriangulation(
            (
                np.asarray([[3, 1, 2], [1, 0, 2]], dtype=np.int32),
                np.asarray([[2, 4, 3]], dtype=np.int32),
            )
        )
        self.assertEqual(
            probe.canonical_two_face_hash(first), probe.canonical_two_face_hash(second)
        )

    def test_ntfe_uses_direct_extension_with_bounded_face_pool(self):
        poly = RecordingPolytope()
        self.assertEqual(sampler_candidates(poly, "ntfe_fast"), ["ntfe-1", "ntfe-2"])
        self.assertEqual(poly.calls[0][0], "ntfe")
        self.assertEqual(
            poly.calls[0][1],
            {
                "N": 2,
                "make_star": True,
                "seed": 123,
                "max_npts": 0,
                "N_face_triangs": 5,
                "triang_method": "fast",
                "as_generator": True,
                "backend": "cgal",
                "verbosity": 0,
            },
        )

    def test_gnn_ntfe_passes_pool_and_seed_to_cytools(self):
        poly = RecordingPolytope()
        self.assertEqual(sampler_candidates(poly, "gnn_ntfe"), ["gnn-1"])
        self.assertEqual(poly.calls[0][0], "gnn")
        self.assertEqual(
            poly.calls[0][1],
            {
                "N": 2,
                "make_star": True,
                "max_npts": 0,
                "N_face_triangs": 5,
                "as_generator": True,
                "seed": 123,
                "verbosity": 0,
            },
        )

    def test_fast_retains_its_explicit_biased_deterministic_candidate(self):
        poly = RecordingPolytope()
        self.assertEqual(sampler_candidates(poly, "fast"), ["deterministic", "fast-1"])
        self.assertEqual(poly.calls[0][0], "triangulate")
        self.assertEqual(poly.calls[1][0], "fast")

    def test_h491_manifest_plans_without_calling_the_ks_endpoint(self):
        manifest = generator.load_polytope_manifest(H491_MANIFEST)
        with mock.patch.object(
            generator,
            "fetch_polytopes",
            side_effect=AssertionError("local manifest must bypass the KS endpoint"),
        ):
            tasks = generator.plan_tasks(
                491,
                1,
                tempfile.mkdtemp(prefix="cyax-h491-plan-"),
                17,
                1,
                1,
                False,
                1_000_000.0,
                1,
                1.0,
                1.0,
                25.0,
                40.0,
                "adaptive",
                40.0,
                None,
                "none",
                None,
                False,
                "ntfe_fast",
                "cgal",
                None,
                None,
                None,
                8,
                1e-2,
                25,
                0.2,
                "fast",
                0,
                5,
                "local-test-manifest",
                True,
                {"requested": False, "status": "not_requested"},
                False,
                polytope_manifest=manifest,
            )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0][2:4], (491, 1))


if __name__ == "__main__":
    unittest.main()
