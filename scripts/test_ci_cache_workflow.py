"""CI cache lifecycle (CYAX-0223): workflow wiring and cache-maintenance selection.

Also the CPU and package-image report after every Julia cache step (CYAX-0230 R-003)
and the one CPU target for package images (CYAX-0230 R-005, R-008).
"""

from __future__ import annotations

import re
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "CI.yml"
FAST_CACHE_NAME = "julia-cache;workflow=CI;job=fast"
GC_STEP = (
    "      - name: Collect unused depot entries before the tests\n"
    "        # julia-actions/cache runs Pkg.gc() only in the post step that\n"
    "        # delete-old-caches: 'false' turns off (CYAX-0223). Run before the\n"
    "        # tests: packages only the tests use look unused to Pkg.gc(), so any\n"
    "        # that it deletes are reinstalled by the tests and saved. Housekeeping\n"
    "        # only, as upstream: a failure must not fail the tests.\n"
    "        continue-on-error: true\n"
    "        run: julia --startup-file=no -e 'using Pkg; Pkg.gc()'\n"
)
# The step that each cache-saving job runs right after Pkg.gc(): its tests.
GC_FOLLOWED_BY = {
    "lifecycle": "      - uses: actions/setup-python@v5\n",
    "fast-cache": "      - uses: julia-actions/julia-runtest@v1\n",
    "qualification": "      - name: Watchdog self-test, then fast and public qualification",
    "full": "      - name: Run package tests\n",
}
# CYAX-0230 R-001: the step directly after every julia-actions/cache step.
REPORT_NAME = "Report CPU and restored package images"
REPORT_SCRIPT = "scripts/ci_report_package_images.jl"
WORKFLOWS = {
    "CI": WORKFLOW,
    "Documentation": ROOT / ".github" / "workflows" / "Documentation.yml",
}
# The six jobs that restore a Julia depot cache (CYAX-0230 baseline).
CACHED_JOBS = [
    ("CI", "lifecycle"), ("CI", "fast"), ("CI", "fast-cache"), ("CI", "qualification"),
    ("CI", "full"), ("Documentation", "build"),
]
# CYAX-0230 R-005: the only place either workflow names JULIA_CPU_TARGET.
CPU_TARGET_LINE = "  JULIA_CPU_TARGET: 'haswell'\n"
# The forms of a julia-actions/cache step the R-003 check must find (R-008).
CACHE_STEP_FORMS = {
    "plain": "      - uses: julia-actions/cache@v2\n",
    "single-quoted": "      - uses: 'julia-actions/cache@v2'\n",
    "double-quoted": '      - uses: "julia-actions/cache@v2"\n',
    "trailing comment": "      - uses: julia-actions/cache@v2  # depot cache\n",
    "quoted with comment": "      - uses: 'julia-actions/cache@v2' # depot cache\n",
    "after a name": "      - name: Cache the depot\n        uses: julia-actions/cache@v2\n",
}
# A step line that uses julia-actions/cache: as "- uses:" or as a "uses:" key
# after "- name:", the value plain, single- or double-quoted, with or without a
# trailing comment.
CACHE_USES = re.compile(
    r"""^      (?:- |  )uses: *(?P<q>['"]?)julia-actions/cache@[^'"\s#]+(?P=q) *(?:#.*)?$""",
    flags=re.M)
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


def _gc_calls(text: str) -> list[str]:
    """Return every occurrence of the word gc in the non-comment lines of text."""
    return [match
            for line in text.splitlines() if not line.lstrip().startswith("#")
            for match in re.findall(r"\bgc\b", line)]


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


def _steps(block: str) -> list[str]:
    """Return the steps of a job, each as its text without trailing blank lines."""
    steps: list[str] = []
    for line in _section(block, "steps").splitlines(keepends=True)[1:]:
        if line.startswith("      - "):
            steps.append(line)
        elif steps:
            steps[-1] += line
    return [step.rstrip() + "\n" for step in steps]


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
        # The producer adds only the depot garbage collection, before its tests.
        runtest = "      - uses: julia-actions/julia-runtest@v1\n"
        self.assertEqual(_section(fast, "steps").replace(runtest, GC_STEP + runtest, 1),
                         _section(producer, "steps"))

    def _assert_no_gc_calls(self, block: str, job: str = "") -> None:
        """A job that only restores a cache must not collect garbage (CYAX-0222 R-006)."""
        self.assertEqual(_gc_calls(block), [], f"job {job}: restore-only" if job else None)

    def test_restore_only_check_sees_every_spelling_but_not_comments(self) -> None:
        # Hand-written blocks; the check must raise for a call and stay quiet otherwise.
        for call in ("Pkg.gc(collect_delay=Dates.Day(0))", "import Pkg as P; P.gc()",
                     "Pkg.gc()", "GC.gc()"):
            with self.subTest(call=call):
                with self.assertRaises(AssertionError):
                    self._assert_no_gc_calls(f"    steps:\n      - run: julia -e '{call}'\n")
        self._assert_no_gc_calls("    steps:\n      # Pkg.gc() runs only in the saving jobs\n"
                                 "      - run: gcc --version\n")

    def _check_gc_placement(self, workflow: str) -> None:
        """Raise AssertionError unless restore-only jobs never collect garbage and the
        jobs that save a Julia cache do it once, right before their tests.

        No subTest here: subTest records a failure instead of raising it, so a caller
        that expects the failure (assertRaises) would not see it. The first failure
        raises and names its job.
        """
        saving = []
        for job in _job_names(workflow):
            block = _job_block(workflow, job)
            if "uses: julia-actions/cache@" not in block:
                continue
            steps = _section(block, "steps")
            if "    cache-mode: read\n" in block:
                self._assert_no_gc_calls(block, job)
            else:
                saving.append(job)
                # Exactly one call. The count is of the word gc in non-comment
                # lines, so aliased, keyword and repeated calls count, and so does
                # GC.gc(); comment lines may mention it. A text search cannot see
                # a name built at run time.
                calls = _gc_calls(steps)
                self.assertEqual(len(calls), 1, f"job {job}: {calls}")
                self.assertIn(GC_STEP + GC_FOLLOWED_BY[job], steps, f"job {job}")
        self.assertEqual(sorted(saving), sorted(GC_FOLLOWED_BY))

    def test_every_job_that_saves_a_julia_cache_collects_garbage_before_its_tests(self) -> None:
        self._check_gc_placement(self.workflow)

    def test_placement_check_fails_for_an_aliased_call_in_the_restore_only_job(self) -> None:
        # The check must see a call that the old assertNotIn("Pkg.gc()") missed.
        runtest = "      - uses: julia-actions/julia-runtest@v1\n"
        aliased = ("      - name: Collect garbage under an alias\n"
                   "        run: julia --startup-file=no -e 'import Pkg as P; P.gc()'\n")
        fast = _job_block(self.workflow, "fast")
        self.assertEqual(fast.count(runtest), 1)
        broken = self.workflow.replace(fast, fast.replace(runtest, aliased + runtest, 1), 1)
        self.assertNotEqual(broken, self.workflow)
        with self.assertRaises(AssertionError):
            self._check_gc_placement(broken)

    def test_placement_check_fails_for_a_second_call_in_a_saving_job(self) -> None:
        # R-005: a saving job collects exactly once. A second step directly after the
        # first still has the tests right behind a gc step, so only the count can
        # catch it.
        producer = _job_block(self.workflow, "fast-cache")
        self.assertEqual(producer.count(GC_STEP), 1)
        broken = self.workflow.replace(producer, producer.replace(GC_STEP, GC_STEP + GC_STEP, 1), 1)
        self.assertNotEqual(broken, self.workflow)
        with self.assertRaises(AssertionError):
            self._check_gc_placement(broken)

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

    def test_maintenance_job_has_a_time_limit(self) -> None:
        job = _job_block(self.workflow, "cache-maintenance")
        # Job level (four spaces), not on a step.
        self.assertRegex(job, r"(?m)^    timeout-minutes: 10$")
        self.assertNotRegex(job, r"(?m)^ {5,}timeout-minutes:")

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


class PackageImageReport(unittest.TestCase):
    """CYAX-0230 R-003: the R-001 report step in every job that restores a Julia cache."""

    def setUp(self) -> None:
        self.workflows = {name: path.read_text(encoding="utf-8")
                          for name, path in WORKFLOWS.items()}

    def _check_report_steps(self, workflows: dict[str, str]) -> None:
        """Raise AssertionError unless every job with a julia-actions/cache step has
        the report step directly after it, non-fatal, with one text in all jobs.

        No subTest, so that a caller expecting the failure sees it (see
        _check_gc_placement).
        """
        reports = {}
        for name, workflow in workflows.items():
            for job in _job_names(workflow):
                steps = _steps(_job_block(workflow, job))
                caches = [i for i, step in enumerate(steps) if CACHE_USES.search(step)]
                if not caches:
                    continue
                self.assertEqual(len(caches), 1, f"{name} job {job}")
                after = caches[0] + 1
                self.assertLess(after, len(steps), f"{name} job {job}: no step after the cache")
                report = steps[after]
                self.assertTrue(report.startswith(f"      - name: {REPORT_NAME}\n"),
                                f"{name} job {job}: step after the cache is {report.splitlines()[0]!r}")
                # A step key (eight spaces), not text inside the run block.
                self.assertRegex(report, r"(?m)^        continue-on-error: true$",
                                 f"{name} job {job}")
                # R-001: no word gc anywhere in the step, comments included (the
                # CYAX-0223 placement test counts it in non-comment lines).
                self.assertNotRegex(report, r"\bgc\b", f"{name} job {job}")
                reports[(name, job)] = report
        self.assertEqual(sorted(reports), sorted(CACHED_JOBS))
        texts = set(reports.values())
        self.assertEqual(len(texts), 1, f"report step texts differ: {sorted(texts)}")

    def test_every_cached_job_reports_cpu_and_package_images_after_the_cache(self) -> None:
        self._check_report_steps(self.workflows)

    def test_the_report_runs_a_script_that_exists(self) -> None:
        fast = _job_block(self.workflows["CI"], "fast")
        report = next(step for step in _steps(fast) if REPORT_NAME in step)
        self.assertIn(f"julia --startup-file=no {REPORT_SCRIPT}", report)
        self.assertTrue((ROOT / REPORT_SCRIPT).is_file())

    def test_report_check_fails_when_a_job_lacks_the_step(self) -> None:
        build = _steps(_job_block(self.workflows["Documentation"], "build"))
        report = next((step for step in build if REPORT_NAME in step), None)
        self.assertIsNotNone(report)
        broken = dict(self.workflows)
        broken["Documentation"] = self.workflows["Documentation"].replace(report, "", 1)
        self.assertNotEqual(broken["Documentation"], self.workflows["Documentation"])
        with self.assertRaises(AssertionError):
            self._check_report_steps(broken)

    def test_report_check_fails_when_one_job_has_a_different_text(self) -> None:
        full = _job_block(self.workflows["CI"], "full")
        report = next((step for step in _steps(full) if REPORT_NAME in step), None)
        self.assertIsNotNone(report)
        changed = report.replace("continue-on-error: true\n",
                                 "continue-on-error: true\n        timeout-minutes: 5\n", 1)
        self.assertNotEqual(changed, report)
        broken = dict(self.workflows)
        broken["CI"] = self.workflows["CI"].replace(full, full.replace(report, changed, 1), 1)
        self.assertNotEqual(broken["CI"], self.workflows["CI"])
        with self.assertRaises(AssertionError):
            self._check_report_steps(broken)

    def test_report_check_fails_when_continue_on_error_is_only_in_the_script(self) -> None:
        # The key must be a step key, not text inside the run block.
        key = "        continue-on-error: true\n        run: |\n"
        moved = "        run: |\n          continue-on-error: true\n"
        broken = {name: text.replace(key, moved) for name, text in self.workflows.items()}
        self.assertNotEqual(broken["CI"], self.workflows["CI"])
        self.assertNotEqual(broken["Documentation"], self.workflows["Documentation"])
        with self.assertRaises(AssertionError):
            self._check_report_steps(broken)

    def test_report_check_fails_when_the_step_mentions_gc_in_a_comment(self) -> None:
        key = "        continue-on-error: true\n        run: |\n"
        commented = "        # Pkg.gc() is not run here.\n" + key
        broken = {name: text.replace(key, commented) for name, text in self.workflows.items()}
        self.assertNotEqual(broken["CI"], self.workflows["CI"])
        with self.assertRaises(AssertionError):
            self._check_report_steps(broken)

    def test_report_check_fails_when_the_step_is_fatal(self) -> None:
        lifecycle = _job_block(self.workflows["CI"], "lifecycle")
        report = next((step for step in _steps(lifecycle) if REPORT_NAME in step), None)
        self.assertIsNotNone(report)
        fatal = report.replace("        continue-on-error: true\n", "", 1)
        self.assertNotEqual(fatal, report)
        broken = {name: text.replace(report, fatal) for name, text in self.workflows.items()}
        with self.assertRaises(AssertionError):
            self._check_report_steps(broken)

    def _with_fast_cache_step(self, form: str) -> dict[str, str]:
        fast = _job_block(self.workflows["CI"], "fast")
        changed = fast.replace(CACHE_STEP_FORMS["plain"], form, 1)
        workflows = dict(self.workflows)
        workflows["CI"] = self.workflows["CI"].replace(fast, changed, 1)
        return workflows

    def test_report_check_finds_the_cache_step_in_every_uses_form(self) -> None:
        # R-008: the review of ba2364b found that the check saw only the plain
        # "- uses:" form, so a quoted or commented cache step had no report check.
        for label, form in CACHE_STEP_FORMS.items():
            with self.subTest(form=label):
                workflows = self._with_fast_cache_step(form)
                if label != "plain":
                    self.assertNotEqual(workflows["CI"], self.workflows["CI"])
                self._check_report_steps(workflows)

    def test_report_check_fails_for_a_cache_step_without_a_report_in_every_uses_form(self) -> None:
        # The changes job restores no cache; give it one with no report step after it.
        changes = _job_block(self.workflows["CI"], "changes")
        for label, form in CACHE_STEP_FORMS.items():
            with self.subTest(form=label):
                changed = changes.replace("    steps:\n", "    steps:\n" + form, 1)
                self.assertNotEqual(changed, changes)
                broken = dict(self.workflows)
                broken["CI"] = self.workflows["CI"].replace(changes, changed, 1)
                with self.assertRaises(AssertionError):
                    self._check_report_steps(broken)


def _top_level_section(workflow: str, key: str) -> str:
    """Return a top-level key line (e.g. 'env:') and the indented lines after it."""
    lines = workflow.splitlines(keepends=True)
    start = lines.index(f"{key}:\n")
    out = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and not line.startswith(" "):
            break
        out.append(line)
    return "".join(out)


class CpuTarget(unittest.TestCase):
    """CYAX-0230 R-005 and R-008: one CPU target for package images, set once."""

    def setUp(self) -> None:
        self.workflows = {name: path.read_text(encoding="utf-8")
                          for name, path in WORKFLOWS.items()}

    def _check_cpu_target(self, workflows: dict[str, str]) -> None:
        """Raise AssertionError unless each workflow sets JULIA_CPU_TARGET to haswell
        in its workflow-level env, with a comment that cites CYAX-0230, and names
        JULIA_CPU_TARGET nowhere else (a job or step env:, a run: export or
        $GITHUB_ENV write, a with: input, or that comment). No subTest, so that a
        caller expecting the failure sees it (see _check_gc_placement).
        """
        for name, workflow in workflows.items():
            self.assertIn("\nenv:\n", workflow, f"{name}: no workflow-level env")
            env = _top_level_section(workflow, "env")
            self.assertIn(CPU_TARGET_LINE, env, name)
            self.assertRegex(env, r"(?m)^  #.*CYAX-0230", f"{name}: comment citing the spec")
            self.assertEqual(workflow.count("JULIA_CPU_TARGET"), 1,
                             f"{name}: JULIA_CPU_TARGET outside the workflow-level env line")

    def _assert_check_fails(self, name: str, old: str, new: str) -> None:
        self.assertEqual(self.workflows[name].count(old) >= 1, True, f"{name}: {old!r}")
        broken = dict(self.workflows)
        broken[name] = self.workflows[name].replace(old, new, 1)
        self.assertNotEqual(broken[name], self.workflows[name])
        with self.assertRaises(AssertionError):
            self._check_cpu_target(broken)

    def test_both_workflows_set_haswell_once_at_workflow_level(self) -> None:
        self._check_cpu_target(self.workflows)

    def test_wiki_refresh_does_not_set_a_cpu_target(self) -> None:
        # R-005: WikiRefresh.yml keeps no cache and is not changed.
        wiki = (ROOT / ".github" / "workflows" / "WikiRefresh.yml").read_text(encoding="utf-8")
        self.assertNotIn("JULIA_CPU_TARGET", wiki)

    def test_check_fails_when_the_variable_is_missing(self) -> None:
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                self._assert_check_fails(name, CPU_TARGET_LINE, "")

    def test_check_fails_for_another_value(self) -> None:
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                self._assert_check_fails(name, CPU_TARGET_LINE,
                                         "  JULIA_CPU_TARGET: 'native'\n")

    def test_check_fails_when_the_comment_names_the_variable(self) -> None:
        self._assert_check_fails("CI", CPU_TARGET_LINE,
                                 "  # Sets JULIA_CPU_TARGET.\n" + CPU_TARGET_LINE)

    def test_check_fails_when_the_comment_does_not_cite_the_spec(self) -> None:
        env = _top_level_section(self.workflows["CI"], "env")
        uncited = env.replace("CYAX-0230", "this change")
        self.assertNotEqual(uncited, env)
        self._assert_check_fails("CI", env, uncited)

    def test_check_fails_when_the_setting_moves_to_a_job_env(self) -> None:
        # R-005 places it at workflow level: one occurrence elsewhere is not enough.
        moved = dict(self.workflows)
        moved["CI"] = (self.workflows["CI"]
                       .replace(CPU_TARGET_LINE, "", 1)
                       .replace("    env:\n      CYAXIVERSE_TEST_FULL: '0'\n",
                                "    env:\n      JULIA_CPU_TARGET: 'haswell'\n"
                                "      CYAXIVERSE_TEST_FULL: '0'\n", 1))
        self.assertEqual(moved["CI"].count("JULIA_CPU_TARGET"), 1)
        self.assertNotIn(CPU_TARGET_LINE, _top_level_section(moved["CI"], "env"))
        with self.assertRaises(AssertionError):
            self._check_cpu_target(moved)

    def test_check_fails_for_a_job_env(self) -> None:
        self._assert_check_fails("CI", "    env:\n      CYAXIVERSE_TEST_FULL: '0'\n",
                                 "    env:\n      JULIA_CPU_TARGET: native\n"
                                 "      CYAXIVERSE_TEST_FULL: '0'\n")

    def test_check_fails_for_a_step_env(self) -> None:
        self._assert_check_fails("CI", "        env:\n          JULIA_PKG_PRECOMPILE_AUTO: '0'\n",
                                 "        env:\n          JULIA_CPU_TARGET: native\n"
                                 "          JULIA_PKG_PRECOMPILE_AUTO: '0'\n")

    def test_check_fails_for_a_run_export(self) -> None:
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                self._assert_check_fails(name, "        run: |\n",
                                         "        run: |\n"
                                         "          export JULIA_CPU_TARGET=native\n")

    def test_check_fails_for_a_github_env_write(self) -> None:
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                self._assert_check_fails(name, "        run: |\n",
                                         "        run: |\n"
                                         "          echo 'JULIA_CPU_TARGET=native' >> \"$GITHUB_ENV\"\n")

    def test_check_fails_for_a_with_input(self) -> None:
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                self._assert_check_fails(name, "        with:\n          version: '1.12'\n",
                                         "        with:\n          version: '1.12'\n"
                                         "          JULIA_CPU_TARGET: native\n")


class GarbageCollectionCount(unittest.TestCase):
    """The placement test counts the word gc in non-comment lines (CYAX-0222 R-005)."""

    def test_counts_each_spelling_it_can_see_and_ignores_comments(self) -> None:
        # Counts derived by hand from each text, not from the helper.
        for text, expected in (
            ("import Pkg as P; P.gc()", 1),
            ("Pkg.gc(collect_delay=Dates.Day(0))", 1),
            ("Pkg.gc(); Pkg.gc()", 2),
            ("GC.gc()", 1),
            ("# Pkg.gc()", 0),
            ("        # Pkg.gc()", 0),
            ("gcc --version", 0),
            ("Collect unused depot entries", 0),
        ):
            with self.subTest(text=text):
                self.assertEqual(len(_gc_calls(text)), expected)


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

    def test_listing_stops_at_a_page_limit_and_deletes_nothing(self) -> None:
        # An API that keeps returning the same non-empty page must not loop forever.
        class Endless(FakeApi):
            def list_caches_page(self, page, per_page):
                self.calls.append(("list", page))
                if len(self.calls) > 500:
                    raise AssertionError("listing did not stop")
                return [_cache(1, "refs/heads/vmm", "fast", "2026-11-01T00:00:00Z"),
                        _cache(2, "refs/heads/vmm", "fast", "2026-11-02T00:00:00Z")], 10**6

        api = Endless([])
        code, lines = self.run_main(api)
        self.assertNotEqual(code, 0)
        self.assertEqual(api.deleted, [])
        self.assertEqual(self.ns["MAX_PAGES"], 100)
        self.assertEqual(api.calls, [("list", page) for page in range(1, 101)])
        self.assertTrue(any(line.startswith("::error::") and "page" in line for line in lines))

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
