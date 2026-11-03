"""CI cache lifecycle (CYAX-0223): workflow wiring and cache-maintenance selection."""

from __future__ import annotations

import re
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "CI.yml"
FAST_CACHE_NAME = "julia-cache;workflow=CI;job=fast"
GC_STEP = (
    "      - name: Collect unused depot entries before the cache is saved\n"
    "        # julia-actions/cache runs Pkg.gc() only in the post step that\n"
    "        # delete-old-caches: 'false' turns off (CYAX-0223).\n"
    "        run: julia --startup-file=no -e 'using Pkg; Pkg.gc()'\n"
)
BEGIN = "# BEGIN cache-maintenance"
END = "# END cache-maintenance"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job_names(workflow: str) -> list[str]:
    jobs = workflow[workflow.index("\njobs:\n"):]
    return re.findall(r"^  ([A-Za-z0-9_-]+):\n", jobs, flags=re.M)


def _job_block(workflow: str, job: str) -> str:
    """Return the text of one job: its key line and the more-indented lines after it."""
    lines = workflow.splitlines(keepends=True)
    start = lines.index(f"  {job}:\n")
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith("   "):
            break
        block.append(line)
    return "".join(block)


def _section(block: str, key: str) -> str:
    """Return the lines of a job-level key (e.g. 'steps', 'env') and its body."""
    lines = block.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line == f"    {key}:\n")
    out = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith("     "):
            break
        out.append(line)
    return "".join(out)


def _maintenance_namespace() -> dict:
    """Execute the inline maintenance script from CI.yml without running main()."""
    workflow = _workflow()
    # From the start of the BEGIN line, so that every line keeps the block's indent.
    start = workflow.rindex("\n", 0, workflow.index(BEGIN)) + 1
    body = workflow[start:workflow.index(END)]
    namespace: dict = {"__name__": "cache_maintenance_under_test"}
    exec(compile(textwrap.dedent(body), "CI.yml:cache-maintenance", "exec"), namespace)
    return namespace


class WorkflowWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = _workflow()

    def test_push_runs_on_vmm_main_and_tags(self) -> None:
        trigger = self.workflow[self.workflow.index("  push:\n"):self.workflow.index("  pull_request:\n")]
        self.assertIn("      - main\n", trigger)
        self.assertIn("      - vmm\n", trigger)
        self.assertIn("    tags: '*'\n", trigger)

    def test_pull_request_fast_job_restores_only(self) -> None:
        fast = _job_block(self.workflow, "fast")
        self.assertIn("    if: github.event_name == 'pull_request'\n", fast)
        self.assertIn("    cache-mode: read\n", fast)
        self.assertIn(f"          cache-name: {FAST_CACHE_NAME}\n", fast)

    def test_push_fast_cache_job_produces_the_fast_cache(self) -> None:
        producer = _job_block(self.workflow, "fast-cache")
        # vmm only: a tag ref's cache is usable only by runs on that tag (owner decision).
        self.assertIn("    if: github.event_name == 'push' && github.ref == 'refs/heads/vmm'\n",
                      producer)
        self.assertIn("    cache-mode: write\n", producer)
        self.assertIn(f"          cache-name: {FAST_CACHE_NAME}\n", producer)

    def test_producer_runs_the_same_environment_and_steps_as_fast(self) -> None:
        fast = _job_block(self.workflow, "fast")
        producer = _job_block(self.workflow, "fast-cache")
        self.assertEqual(_section(fast, "env"), _section(producer, "env"))
        # The producer adds only the depot garbage collection before its save.
        self.assertEqual(_section(fast, "steps").rstrip("\n") + "\n" + GC_STEP,
                         _section(producer, "steps").rstrip("\n") + "\n")

    def test_every_job_that_saves_a_julia_cache_collects_garbage_last(self) -> None:
        saving = []
        for job in _job_names(self.workflow):
            block = _job_block(self.workflow, job)
            if "uses: julia-actions/cache@" not in block:
                continue
            steps = _section(block, "steps").rstrip("\n") + "\n"
            with self.subTest(job=job):
                if "    cache-mode: read\n" in block:
                    self.assertNotIn("Pkg.gc()", block)
                else:
                    saving.append(job)
                    self.assertTrue(steps.endswith(GC_STEP), steps[-300:])
        self.assertEqual(sorted(saving), ["fast-cache", "full", "lifecycle", "qualification"])

    def test_only_fast_and_its_producer_share_a_cache_name(self) -> None:
        named = {job: re.findall(r"cache-name: (.+)\n", _job_block(self.workflow, job))
                 for job in _job_names(self.workflow)}
        sharing = sorted(job for job, names in named.items() if FAST_CACHE_NAME in names)
        self.assertEqual(sharing, ["fast", "fast-cache"])
        others = [name for job, names in named.items() if job not in sharing for name in names]
        self.assertEqual(others, [])

    def test_every_julia_cache_step_keeps_old_caches(self) -> None:
        for job in _job_names(self.workflow):
            block = _job_block(self.workflow, job)
            for step in block.split("      - ")[1:]:
                if step.startswith("uses: julia-actions/cache@"):
                    with self.subTest(job=job):
                        self.assertIn("delete-old-caches: 'false'", step)

    def test_only_the_maintenance_job_has_write_permission(self) -> None:
        # The workflow-level block is exactly contents: read, nothing more.
        top = re.search(r"^permissions:\n((?:  .*\n)*)", self.workflow, flags=re.M)
        self.assertIsNotNone(top)
        self.assertEqual(top.group(1), "  contents: read\n")
        self.assertEqual(_section(_job_block(self.workflow, "cache-maintenance"), "permissions"),
                         "    permissions:\n      actions: write\n      pull-requests: read\n")
        for job in _job_names(self.workflow):
            block = _job_block(self.workflow, job)
            with self.subTest(job=job):
                if job == "cache-maintenance":
                    self.assertIn(
                        "    permissions:\n      actions: write\n      pull-requests: read\n", block)
                else:
                    self.assertNotIn("    permissions:", block)
                    self.assertNotIn(": write", block.replace("cache-mode: write", ""))

    def test_maintenance_job_is_push_only_serialised_and_runs_no_repository_code(self) -> None:
        job = _job_block(self.workflow, "cache-maintenance")
        self.assertIn("    if: github.event_name == 'push'\n", job)
        self.assertIn("    concurrency:\n      group: ci-cache-maintenance\n"
                      "      cancel-in-progress: false\n", job)
        self.assertNotIn("actions/checkout", job)
        self.assertNotIn("needs:", job)
        self.assertNotIn("scripts/", job)
        self.assertIn(BEGIN, job)
        self.assertIn(END, job)

    def test_full_suite_does_not_run_on_push_to_vmm(self) -> None:
        full = _job_block(self.workflow, "full")
        self.assertIn(
            "    if: (github.event_name != 'pull_request' && "
            "!(github.event_name == 'push' && github.ref == 'refs/heads/vmm')) || "
            "github.base_ref == 'main' || "
            "contains(github.event.pull_request.labels.*.name, 'full-tests')\n", full)


def _cache(cache_id: int, ref: str, job: str, created: str, run: int | None = None,
           version: str = "b3708dc8") -> dict:
    run = cache_id if run is None else run
    key = f"julia-cache;workflow=CI;job={job};os=Linux;run_id={run};run_attempt=1"
    return {"id": cache_id, "ref": ref, "key": key, "created_at": created, "version": version}


class FakeApi:
    def __init__(self, caches, states=None, per_page=2, fail_list=False,
                 fail_state=False, fail_delete=(), gone=(), move_to_front_after_page=None):
        self.caches = list(caches)
        self.move_to_front_after_page = move_to_front_after_page
        self.states = states or {}
        self.per_page = per_page
        self.fail_list = fail_list
        self.fail_state = fail_state
        self.fail_delete = set(fail_delete)
        self.gone = set(gone)
        self.pages = []
        self.deleted = []
        self.calls = []

    def list_caches_page(self, page, per_page):
        self.calls.append(("list", page))
        if self.fail_list:
            raise RuntimeError("list failed")
        size = self.per_page
        self.pages.append(page)
        served = self.caches[(page - 1) * size:page * size]
        if self.move_to_front_after_page == page:
            # A cache used between page reads moves position, as under the
            # API's default last-accessed ordering.
            self.caches.insert(0, self.caches.pop())
        return served, len(self.caches)

    def pr_state(self, number):
        self.calls.append(("state", number))
        if self.fail_state:
            raise RuntimeError("state failed")
        return self.states[number]

    def delete_cache(self, cache_id):
        self.calls.append(("delete", cache_id))
        if cache_id in self.fail_delete:
            raise RuntimeError(f"delete {cache_id} failed")
        if cache_id in self.gone:
            raise self.not_found
        self.deleted.append(cache_id)


class MaintenanceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = _maintenance_namespace()

    def run_main(self, api):
        api.not_found = self.ns["NotFound"]()
        lines = []
        code = self.ns["main"](api, out=lines.append)
        return code, lines

    def test_keeps_newest_entry_per_ref_and_key_prefix(self) -> None:
        caches = [
            _cache(1, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z"),
            _cache(2, "refs/heads/vmm", "fast", "2026-11-02T00:00:00Z"),
            _cache(3, "refs/heads/vmm", "lifecycle", "2026-11-01T00:00:00Z"),
            _cache(4, "refs/pull/7/merge", "full", "2026-11-01T00:00:00Z"),
            _cache(5, "refs/pull/7/merge", "full", "2026-11-03T00:00:00Z"),
        ]
        api = FakeApi(caches, states={7: "open"})
        code, _ = self.run_main(api)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(api.deleted), [1, 4])

    def test_newest_by_creation_not_by_listing_order_or_run_id(self) -> None:
        caches = [
            _cache(9, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z", run=900),
            _cache(8, "refs/heads/vmm", "fast", "2026-11-05T00:00:00Z", run=100),
        ]
        api = FakeApi(caches)
        self.run_main(api)
        self.assertEqual(api.deleted, [9])

    def test_creation_times_compare_as_times_not_strings(self) -> None:
        # As strings "…:30Z" sorts after "…:30.5Z", although it is half a second older.
        for older, newer in (("2026-11-03T15:31:30Z", "2026-11-03T15:31:30.5Z"),
                             ("2026-11-03T15:31:30.119977Z", "2026-11-03T15:31:30.9Z"),
                             ("2026-11-03T15:31:29.999999Z", "2026-11-03T15:31:30Z")):
            with self.subTest(older=older, newer=newer):
                caches = [_cache(1, "refs/heads/vmm", "fast", older),
                          _cache(2, "refs/heads/vmm", "fast", newer)]
                api = FakeApi(caches)
                self.run_main(api)
                self.assertEqual(api.deleted, [1])

    def test_entries_with_different_cache_versions_are_kept_apart(self) -> None:
        caches = [
            _cache(1, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z", version="aaaa"),
            _cache(2, "refs/heads/vmm", "fast", "2026-11-02T00:00:00Z", version="bbbb"),
        ]
        api = FakeApi(caches)
        self.run_main(api)
        self.assertEqual(api.deleted, [])

    def test_deletes_every_entry_of_a_closed_pull_request(self) -> None:
        caches = [
            _cache(1, "refs/pull/215/merge", "fast", "2026-11-03T00:00:00Z"),
            _cache(2, "refs/pull/215/merge", "full", "2026-11-03T00:00:00Z"),
            _cache(3, "refs/pull/190/merge", "lifecycle", "2026-11-03T00:00:00Z"),
        ]
        api = FakeApi(caches, states={215: "closed", 190: "open"})
        self.run_main(api)
        self.assertEqual(sorted(api.deleted), [1, 2])

    def test_reopened_pull_request_keeps_its_newest_entries(self) -> None:
        caches = [_cache(1, "refs/pull/12/merge", "lifecycle", "2026-11-03T00:00:00Z")]
        api = FakeApi(caches, states={12: "open"})
        self.run_main(api)
        self.assertEqual(api.deleted, [])

    def test_fast_tests_entries_on_open_pull_requests_are_deleted(self) -> None:
        # Pull requests no longer save the Fast tests cache, so any such entry is
        # stale and would be restored instead of the vmm base.
        caches = [
            _cache(1, "refs/pull/190/merge", "fast", "2026-11-03T00:00:00Z"),
            _cache(2, "refs/pull/190/merge", "lifecycle", "2026-11-03T00:00:00Z"),
            _cache(3, "refs/pull/190/merge", "fastx", "2026-11-03T00:00:00Z"),
            _cache(4, "refs/heads/vmm", "fast", "2026-11-03T00:00:00Z"),
        ]
        api = FakeApi(caches, states={190: "open"})
        self.run_main(api)
        self.assertEqual(api.deleted, [1])

    def test_only_strict_pull_merge_refs_are_treated_as_pull_requests(self) -> None:
        caches = [
            _cache(1, "refs/pull/5/head", "fast", "2026-11-03T00:00:00Z"),
            _cache(2, "refs/pull/5/merge/x", "fast", "2026-11-03T00:00:00Z"),
            _cache(3, "refs/pull/x5/merge", "fast", "2026-11-03T00:00:00Z"),
        ]
        api = FakeApi(caches, states={5: "closed"})
        self.run_main(api)
        self.assertEqual(api.deleted, [])
        self.assertNotIn(("state", 5), api.calls)

    def test_never_deletes_the_newest_vmm_or_main_entry(self) -> None:
        caches = [
            _cache(1, "refs/heads/vmm", "fast", "2026-11-03T00:00:00Z"),
            _cache(2, "refs/heads/main", "full", "2026-11-03T00:00:00Z"),
        ]
        api = FakeApi(caches)
        self.run_main(api)
        self.assertEqual(api.deleted, [])

    def test_reads_every_page_before_any_deletion(self) -> None:
        caches = [_cache(i, "refs/heads/vmm", "fast", f"2026-11-0{i}T00:00:00Z")
                  for i in range(1, 6)]
        api = FakeApi(caches, per_page=2)
        self.run_main(api)
        self.assertEqual(api.pages, [1, 2, 3])
        first_delete = next(i for i, call in enumerate(api.calls) if call[0] == "delete")
        last_list = max(i for i, call in enumerate(api.calls) if call[0] == "list")
        self.assertLess(last_list, first_delete)
        self.assertEqual(sorted(api.deleted), [1, 2, 3, 4])

    def test_an_entry_seen_twice_across_pages_is_not_deleted(self) -> None:
        caches = [
            _cache(2, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z"),
            _cache(3, "refs/heads/vmm", "lifecycle", "2026-11-01T00:00:00Z"),
            _cache(4, "refs/pull/9/merge", "full", "2026-11-01T00:00:00Z"),
        ]
        api = FakeApi(caches, states={9: "open"}, per_page=2, move_to_front_after_page=1)
        code, _ = self.run_main(api)
        self.assertEqual(code, 0)
        self.assertEqual(api.deleted, [])

    def test_lists_caches_oldest_first_by_creation(self) -> None:
        paths = []

        class Recording(self.ns["GitHubApi"]):
            def _call(self, method, path):
                paths.append((method, path))
                return {"actions_caches": [], "total_count": 0}

        Recording("owner/repo", "token").list_caches_page(1, 100)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0][0], "GET")
        self.assertIn("sort=created_at", paths[0][1])
        self.assertIn("direction=asc", paths[0][1])

    def test_selection_failure_deletes_nothing_and_fails(self) -> None:
        for kwargs in ({"fail_list": True}, {"fail_state": True}):
            with self.subTest(**kwargs):
                caches = [
                    _cache(1, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z"),
                    _cache(2, "refs/heads/vmm", "fast", "2026-11-02T00:00:00Z"),
                    _cache(3, "refs/pull/4/merge", "fast", "2026-11-02T00:00:00Z"),
                ]
                api = FakeApi(caches, states={4: "closed"}, **kwargs)
                code, lines = self.run_main(api)
                self.assertNotEqual(code, 0)
                self.assertEqual(api.deleted, [])
                self.assertTrue(any(line.startswith("::error::") for line in lines))

    def test_delete_failure_stops_reports_and_logs_what_was_deleted(self) -> None:
        caches = [_cache(i, "refs/heads/vmm", "fast", f"2026-11-0{i}T00:00:00Z")
                  for i in range(1, 5)]
        api = FakeApi(caches, per_page=10, fail_delete={2})
        code, lines = self.run_main(api)
        self.assertNotEqual(code, 0)
        self.assertEqual(api.deleted, [1])
        self.assertNotIn(("delete", 3), api.calls)
        self.assertTrue(any("deleted 1" in line for line in lines))
        self.assertTrue(any(line.startswith("::error::") and "2" in line for line in lines))

    def test_an_already_removed_cache_is_logged_not_fatal(self) -> None:
        caches = [_cache(i, "refs/heads/vmm", "fast", f"2026-11-0{i}T00:00:00Z")
                  for i in range(1, 4)]
        api = FakeApi(caches, per_page=10, gone={1})
        code, lines = self.run_main(api)
        self.assertEqual(code, 0)
        self.assertEqual(api.deleted, [2])
        self.assertTrue(any("1" in line and "already" in line for line in lines))

    def test_logged_keys_and_refs_cannot_inject_workflow_commands(self) -> None:
        hostile = _cache(1, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z")
        hostile["key"] = hostile["key"] + "\n::error::injected"
        hostile["ref"] = "refs/heads/vmm"
        caches = [hostile, _cache(2, "refs/heads/vmm", "fast", "2026-11-02T00:00:00Z")]
        # The suffix follows run_id=, so entry 1 shares entry 2's group and is selected.
        api = FakeApi(caches)
        _, lines = self.run_main(api)
        self.assertTrue(any("injected" in line for line in lines))
        for line in lines:
            with self.subTest(line=line):
                self.assertNotIn("\n", line)
                self.assertNotIn("\r", line)

    def test_every_selected_id_is_logged_before_deleting(self) -> None:
        caches = [
            _cache(1, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z"),
            _cache(2, "refs/heads/vmm", "fast", "2026-11-02T00:00:00Z"),
        ]
        api = FakeApi(caches)
        _, lines = self.run_main(api)
        selected = [i for i, line in enumerate(lines) if line.startswith("selected 1 ")]
        deleted = [i for i, line in enumerate(lines) if line == "deleted 1"]
        self.assertTrue(selected and deleted and selected[0] < deleted[0])


if __name__ == "__main__":
    unittest.main()
