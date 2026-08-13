"""Regression tests for CYTools sampler adapter argument contracts."""

import os
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="cytools-test-cache-"))

import generate_geometric_data_multitriangulation as generator
import generate_h11_491_frsts as frst_generator
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

    def face_triangs(self, **kwargs):
        self.calls.append(("face_triangs", kwargs))
        return [["face-0", "face-1", "face-2"], ["face-3", "face-4", "face-5"]]

    def triangface_ineqs(self, **kwargs):
        self.calls.append(("triangface_ineqs", kwargs))
        return [["ineq-0", "ineq-1", "ineq-2"], ["ineq-3", "ineq-4", "ineq-5"]]

    labels = (0, 1, 2, 3)

    def triangulate(self, **kwargs):
        self.calls.append(("extension" if "heights" in kwargs else "triangulate", kwargs))
        return True if "heights" in kwargs else "deterministic"


class RecordingTriangulation:
    """Provide stable simplex data for probe hash regression tests."""

    def __init__(self, faces):
        self.faces = faces

    def simplices(self, **kwargs):
        if kwargs.get("on_faces_dim") == 2:
            return self.faces
        return np.asarray([[0, 1, 2, 3]], dtype=np.int32)


class RecordingFaceTriangulation:
    """Provide stable simplex data for explicit face-combination tests."""

    def __init__(self, label):
        self.label = label

    def simplices(self, **kwargs):
        return np.asarray([[self.label, 0, 1]], dtype=np.int32)


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
    def test_h491_generator_cli_keeps_sampler_and_geometry_defaults(self):
        args = frst_generator.build_parser().parse_args([])
        self.assertEqual(args.h11, 491)
        self.assertEqual(args.sampling_scheme, "ntfe_fast")
        self.assertTrue(args.exact_proposals)
        self.assertEqual(args.proposal_budget, 300)
        self.assertEqual(args.cores, 1)
        self.assertEqual(args.moduli_policy, "adaptive")
        self.assertEqual(args.min_prime_divisor_volume, 1.0)
        self.assertEqual(args.qcd_volume_min, 25.0)
        self.assertEqual(args.qcd_volume_max, 40.0)
        frst_generator.validate_args(args)

    def test_h491_generator_uses_package_hdf5_layout(self):
        output_directory = tempfile.mkdtemp(prefix="cyax-h491-frst-artifact-")
        self.assertEqual(
            frst_generator.package_generator.output_path(
                output_directory, 491, 1, 1
            ),
            os.path.join(
                output_directory,
                "h11_491",
                "np_0000001",
                "cy_0000001",
                "cyax.h5",
            ),
        )

    def test_h491_generator_routes_accepted_frst_to_package_hdf5_writer(self):
        output_directory = tempfile.mkdtemp(prefix="cyax-h491-hdf5-writer-")
        args = frst_generator.build_parser().parse_args(
            ["--outdir", output_directory, "--moduli-policy", "canonical_qcd"]
        )
        triangulation = RecordingTriangulation(
            (np.asarray([[2, 0, 1]], dtype=np.int32),)
        )
        triangulation.get_cy = mock.Mock(return_value=object())
        report = {
            "counts": {
                "valid_frsts": 0,
                "accepted_geometries": 0,
                "rejected_frsts": 0,
                "candidate_errors": 0,
                "duplicate_full_triangulations": 0,
                "duplicate_two_face_classes": 0,
                "written_hdf5": 0,
            }
        }

        def fake_writer(*writer_args, **writer_kwargs):
            del writer_kwargs
            os.makedirs(os.path.dirname(writer_args[4]), exist_ok=True)
            with open(writer_args[4], "wb") as stream:
                stream.write(b"mock hdf5")

        with mock.patch.object(
            frst_generator.package_generator,
            "generate_and_save_geometry",
            side_effect=fake_writer,
        ) as writer:
            candidate = frst_generator.generate_candidate(
                args,
                object(),
                np.zeros((3, 4), dtype=int),
                "polytope-id",
                {"scheme": "canonical_qcd"},
                triangulation,
                1,
                1,
                output_directory,
                {"requested": False, "status": "not_requested"},
                set(),
                set(),
                report,
            )

        self.assertEqual(candidate["terminal_status"], "accepted_geometry")
        self.assertEqual(report["counts"]["written_hdf5"], 1)
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    output_directory,
                    "h11_491",
                    "np_0000001",
                    "cy_0000001",
                    "cyax.h5",
                )
            )
        )
        writer.assert_called_once()

    def test_h491_generator_dry_run_uses_local_manifest(self):
        output_directory = tempfile.mkdtemp(prefix="cyax-h491-frst-dry-run-")
        args = frst_generator.build_parser().parse_args(
            ["--dry-run", "--outdir", output_directory]
        )
        report = frst_generator.run_generation(args)
        self.assertEqual(report["terminal_status"], "dry_run")
        self.assertEqual(report["polytope"]["h11"], 491)
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "report.json")))
        self.assertEqual(
            report["outputs"]["geometry_root"], "h11_491/np_0000001"
        )

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

    def test_exact_harness_stops_at_explicit_budget(self):
        poly = RecordingPolytope()
        poly.face_triangs = mock.Mock(
            return_value=[
                [RecordingFaceTriangulation(0), RecordingFaceTriangulation(1), RecordingFaceTriangulation(2)],
                [RecordingFaceTriangulation(3), RecordingFaceTriangulation(4), RecordingFaceTriangulation(5)],
            ]
        )
        poly.triangface_ineqs = mock.Mock(
            return_value=[[[1], [2], [3]], [[4], [5], [6]]]
        )
        state = {
            "attempted_proposals": 0,
            "yielded_triangulations": 0,
            "non_solid_attempts": 0,
            "solid_attempts": 0,
            "invalid_extensions": 0,
            "duplicate_two_face_classes": 0,
            "seen_two_face_hashes": set(),
            "two_face_combination_hashes": [],
            "extension_errors": [],
            "pool_decomposition": None,
            "terminal_status": None,
        }
        with mock.patch.object(
            probe, "_find_interior_point_highs", return_value=np.ones(4)
        ):
            candidates = list(
                probe.exact_ntfe_candidates(
                    poly,
                    "fast",
                    max_draws=4,
                    seed=123,
                    max_face_points=0,
                    face_pool_size=3,
                    state=state,
                )
            )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(state["attempted_proposals"], 4)
        self.assertEqual(state["solid_attempts"], 4)
        self.assertEqual(state["terminal_status"], "completed")
        self.assertEqual(len(set(state["two_face_combination_hashes"])), 4)

    def test_atomic_report_leaves_only_the_final_json(self):
        report_directory = tempfile.mkdtemp(prefix="cyax-atomic-report-")
        report_path = os.path.join(report_directory, "report.json")
        probe.atomic_json_dump(report_path, {"status": "complete", "attempts": 4})
        with open(report_path, encoding="utf-8") as stream:
            self.assertEqual(json.load(stream), {"attempts": 4, "status": "complete"})
        self.assertEqual(
            [name for name in os.listdir(report_directory) if name != "report.json"], []
        )


if __name__ == "__main__":
    unittest.main()
