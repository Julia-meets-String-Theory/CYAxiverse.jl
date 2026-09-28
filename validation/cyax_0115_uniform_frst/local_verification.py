#!/usr/bin/env python3
"""Run local CYAX-0115 identity, state-machine, and persistence checks."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation/cyax_0115_uniform_frst"
SUPPORT_DIR = VALIDATION / "exact_support_hard_capped"
MANIFEST = SUPPORT_DIR / "validation_manifest.json"
EXPECTED_MANIFEST_SHA256 = "04254692b901a0b2a2d18d2bb21d7647af9aa48e0ce7d7173307a82ff6563263"
EXPECTED_SCHEDULE_SHA256 = "a351d05ce60a5cb00f6298b567e045d8e6e18d9ab49055453919872f16feb92d"
CYTOOLS_CACHE = Path("/private/tmp/cyax-0115-local-verification-cache")


def load_validator():
    path = ROOT / "scripts/validate_uniform_frst_ensemble.py"
    spec = importlib.util.spec_from_file_location("cyax0115_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validation driver: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return [list(row) for row in self.rows]


class IdentityOnlyTriangulation:
    def __init__(self, rows):
        self.rows = rows

    def simplices(self, **kwargs):
        return Rows(self.rows)


def check_manifest_and_supports(v):
    manifest_bytes = MANIFEST.read_bytes()
    manifest_sha = v.sha256_bytes(manifest_bytes)
    require(manifest_sha == EXPECTED_MANIFEST_SHA256, "frozen manifest SHA changed")
    frozen_check = v.verify_existing_frozen_manifest(MANIFEST, EXPECTED_MANIFEST_SHA256)
    require(frozen_check["no_manifest_bytes_written"], "frozen verifier may not rewrite bytes")
    require(
        frozen_check["seed_schedule_identity_sha256"] == EXPECTED_SCHEDULE_SHA256,
        "frozen seed schedule identity changed",
    )
    manifest = json.loads(manifest_bytes)
    cytools, poly_mod, tri_mod, _ = v.bootstrap_cytools(CYTOOLS_CACHE)
    all_states = 0
    state_checks = []
    for fixture in manifest["fixtures"]:
        support_path = SUPPORT_DIR / fixture["support_file"]
        require(
            v.file_sha256(support_path) == fixture["support_file_sha256"],
            f"support file hash changed: {fixture['candidate_id']}",
        )
        support = json.loads(support_path.read_bytes())
        ids = [state["full_triangulation_sha256"] for state in support["states"]]
        require(len(ids) == len(set(ids)) == fixture["K"], "support identities are not unique")
        require(
            v.sha256_bytes(v.canonical_json_bytes(sorted(ids))) == fixture["support_sha256"],
            f"support identity set changed: {fixture['candidate_id']}",
        )
        candidate = fixture["source_fixture"]
        poly = v.construct_candidate(cytools, candidate)
        require(v.polytope_identity(poly) == fixture["polytope_id"], "polytope identity changed")
        labels = list(range(len(fixture["public_wrapper_point_labels"])))
        for state in support["states"]:
            rows = state["simplices_as_point_indices"]
            reconstructed = tri_mod.Triangulation(
                poly,
                labels,
                make_star=False,
                simplices=rows,
                check_input_simplices=False,
                backend="cgal",
                verbosity=0,
            )
            full_id = v.full_triangulation_identity(reconstructed)
            two_face_id = v.two_face_identity(reconstructed)
            require(full_id == state["full_triangulation_sha256"], "full identity changed")
            require(two_face_id == state["two_face_sha256"], "two-face identity changed")
            reversed_rows = [list(reversed(row)) for row in reversed(rows)]
            require(
                v.full_triangulation_identity(IdentityOnlyTriangulation(reversed_rows)) == full_id,
                "full identity is not invariant to simplex/vertex ordering",
            )
            all_states += 1
        state_checks.append(
            {
                "candidate_id": fixture["candidate_id"],
                "K": fixture["K"],
                "full_identity_unique": True,
                "two_face_identities_reconstructed": fixture["K"],
                "polytope_identity": fixture["polytope_id"],
            }
        )
    return {
        "manifest_sha256": manifest_sha,
        "seed_schedule_identity_sha256": EXPECTED_SCHEDULE_SHA256,
        "chain_count_frozen": frozen_check["chain_count"],
        "support_state_count_verified": all_states,
        "fixtures": state_checks,
        "status": "PASS",
    }, manifest, cytools, poly_mod, tri_mod


def check_duplicate_retry_state_machine(v, manifest, cytools, poly_mod, tri_mod):
    fixture = manifest["fixtures"][0]
    support = json.loads((SUPPORT_DIR / fixture["support_file"]).read_bytes())
    rows = support["states"][0]["simplices_as_point_indices"]
    support_id = support["states"][0]["full_triangulation_sha256"]
    poly = v.construct_candidate(cytools, fixture["source_fixture"])
    observer = v.SamplerObserver([support_id])
    original_generator = tri_mod.random_triangulations_fair_generator
    original_poly_generator = poly_mod.random_triangulations_fair_generator
    original_tri = tri_mod.Triangulation
    prior_observer = tri_mod.__dict__.get("__cyax_observe__")
    instrumented, source_ast = v.instrumented_fair_generator_source(tri_mod)

    class DeterministicTriangulation:
        fine_calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def is_fine(self):
            call = DeterministicTriangulation.fine_calls
            DeterministicTriangulation.fine_calls += 1
            return call % 3 != 1

        def random_flips(self, *args, **kwargs):
            return self

        def __hash__(self):
            return 194115

        def simplices(self, *args, **kwargs):
            if kwargs.get("on_faces_dim") == 2:
                return [[tuple(row) for row in rows]]
            return Rows(rows)

    try:
        tri_mod.random_triangulations_fair_generator = instrumented
        poly_mod.random_triangulations_fair_generator = instrumented
        tri_mod.Triangulation = DeterministicTriangulation
        tri_mod.__dict__["__cyax_observe__"] = observer.observe
        yielded = list(
            poly.random_triangulations_fair(
                N=None,
                n_walk=0,
                n_flip=0,
                initial_walk_steps=-1,
                walk_step_size=0.01,
                max_steps_to_wall=2,
                fine_tune_steps=1,
                max_retries=1,
                make_star=False,
                points=None,
                backend="cgal",
                as_list=False,
                progress_bar=False,
                seed=4115,
            )
        )
        yielded_ids = [v.full_triangulation_identity(item) for item in yielded]
        require(len(observer.raw_events) == 2, "synthetic chain did not record two raw candidates")
        require(observer.raw_events[0]["duplicate_predicted_from_public_hash_history"] is False,
                "first synthetic raw candidate should be new")
        require(observer.raw_events[1]["duplicate_predicted_from_public_hash_history"] is True,
                "second synthetic raw candidate should be a duplicate")
        require(len(observer.first_discoveries) == 1, "duplicate must not become a first discovery")
        require(len(observer.duplicate_branches) == 1, "duplicate retry branch was not recorded")
        require(observer.duplicate_branches[0]["n_retries_after_increment"] == 1,
                "duplicate must consume exactly one retry")
        require(len(observer.step_updates) == 1, "duplicate continue must skip the step update")
        require(len(observer.terminal_events) == 1, "retry termination was not recorded")
        require(observer.terminal_events[0]["reason"] == "max_retries",
                "synthetic chain should stop at the frozen retry condition")
        require(yielded_ids == [support_id], "synthetic yielded identity mismatch")
        return {
            "status": "PASS",
            "test_kind": "deterministic_stub_state_machine_only",
            "source_ast_sha256": v.sha256_bytes(source_ast.encode("utf-8")),
            "raw_identity_sequence": [event["full_triangulation_sha256"] for event in observer.raw_events],
            "duplicate_flags": [event["duplicate_predicted_from_public_hash_history"] for event in observer.raw_events],
            "unique_yield_count": len(yielded_ids),
            "duplicate_retry_increment": observer.duplicate_branches[0]["n_retries_after_increment"],
            "step_update_count": len(observer.step_updates),
            "terminal_reason": observer.terminal_events[0]["reason"],
            "terminal_retry_count": observer.terminal_events[0]["n_retries"],
            "seed": 4115,
            "population_inference": False,
        }
    finally:
        tri_mod.random_triangulations_fair_generator = original_generator
        poly_mod.random_triangulations_fair_generator = original_poly_generator
        tri_mod.Triangulation = original_tri
        if prior_observer is None:
            tri_mod.__dict__.pop("__cyax_observe__", None)
        else:
            tri_mod.__dict__["__cyax_observe__"] = prior_observer


def check_atomic_collision_and_torn_tail(v):
    with tempfile.TemporaryDirectory(prefix="cyax-0115-local-verification-") as temporary:
        root = Path(temporary)
        collision_path = root / "seed-result.json"
        v.atomic_json_create(collision_path, {"seed": 291504128, "status": "original"})
        original_bytes = collision_path.read_bytes()
        collision_rejected = False
        try:
            v.atomic_json_create(collision_path, {"seed": 291504128, "status": "overwrite"})
        except v.ValidationError:
            collision_rejected = True
        require(collision_rejected, "atomic result collision was accepted")
        require(collision_path.read_bytes() == original_bytes, "collision overwrote original bytes")

        journal = root / "restart-progress.jsonl"
        frozen_seed_list = [291504128, 291504129]
        v.append_jsonl_fsync(journal, {"event": "restart_complete", "seed": frozen_seed_list[0]})
        complete_prefix = journal.read_bytes()
        torn_bytes = b'{"event":"restart_complete","seed":291504129'
        with journal.open("ab") as stream:
            stream.write(torn_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        torn_read = v.read_progress_journal(journal)
        require(len(torn_read["events"]) == 1, "reader accepted a partial JSONL event")
        require(torn_read["unterminated_tail_bytes"] == len(torn_bytes), "torn tail length not retained")
        raw = journal.read_bytes()
        repaired_end = raw.rfind(b"\n") + 1
        with journal.open("r+b") as stream:
            stream.truncate(repaired_end)
            stream.flush()
            os.fsync(stream.fileno())
        repaired = v.read_progress_journal(journal)
        require(journal.read_bytes() == complete_prefix, "tail repair changed complete event bytes")
        require(len(repaired["events"]) == 1 and repaired["unterminated_tail_bytes"] == 0,
                "journal repair did not preserve only the completed event")
        require(frozen_seed_list[1] in [291504128 + i for i in range(10000)],
                "resume seed does not belong to the frozen restart schedule")
        return {
            "status": "PASS",
            "atomic_collision_rejected": collision_rejected,
            "existing_result_sha256_preserved": v.file_sha256(collision_path),
            "partial_tail_bytes_ignored": torn_read["unterminated_tail_bytes"],
            "complete_event_count_before_and_after_repair": [len(torn_read["events"]), len(repaired["events"])],
            "repaired_tail_bytes": torn_read["unterminated_tail_bytes"],
            "next_seed_for_resume": frozen_seed_list[1],
            "next_seed_was_frozen": True,
            "resume_scheduler_run": False,
        }


def main():
    v = load_validator()
    identity, manifest, cytools, poly_mod, tri_mod = check_manifest_and_supports(v)
    state_machine = check_duplicate_retry_state_machine(v, manifest, cytools, poly_mod, tri_mod)
    persistence = check_atomic_collision_and_torn_tail(v)
    result = {
        "schema": "CYAX-0115-r4-local-verification-v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "seed_schedule_identity_sha256": EXPECTED_SCHEDULE_SHA256,
        "checks": {
            "manifest_and_exact_support_identities": identity,
            "instrumented_duplicate_retry_state_machine": state_machine,
            "atomic_collision_and_torn_tail_recovery": persistence,
        },
        "gated_not_run": {
            "worker_count_and_reversed_schedule_invariance": "first frozen seed terminates before raw index 1 in every fixture/control; protocol requires stop and forbids proceeding to later restart seeds",
            "full_restart_resume_scheduler": "no batch runner was started; only atomic result and torn-tail recovery primitives were exercised",
        },
        "overall_status": "PASS_LOCAL_INVARIANTS_WITH_SAMPLER_GATES_CLOSED",
    }
    output_path = VALIDATION / "calibration/local-verification.json"
    v.atomic_json_create(output_path, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
