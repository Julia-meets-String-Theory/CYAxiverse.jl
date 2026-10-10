#!/usr/bin/env python3
"""Read-only mechanical checks for the installed CYAxiverse task candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
HASH = re.compile(r"^[0-9a-f]{64}$")
CASE_IDS = [f"SK-{i:02d}" for i in range(1, 13)]
METHODS = {
    "diagnosing-bugs", "tdd", "code-review", "codebase-design",
    "improve-codebase-architecture", "writing-for-agents",
    "domain-modeling", "retro",
}
CASE_KIND = {
    "SK-01": "executable_red_green",
    "SK-09": "payload_manifest",
    "SK-11": "identity_control_and_assessment",
}
PROC = [
    ".agents/skills/cyaxiverse-julia-quality/references/diagnostic-test-loop.md",
    ".agents/skills/cyaxiverse-julia-quality/references/implementation-self-review.md",
    ".agents/skills/cyaxiverse-sdd/references/architecture-assessment.md",
    ".agents/skills/cyaxiverse-sdd/references/agent-writing-domain-modeling.md",
    ".agents/skills/cyaxiverse-sdd/references/retrospective.md",
]
INSTRUCTION_PATHS = [
    ".agents/skills/cyaxiverse-julia-quality/SKILL.md",
    ".agents/skills/cyaxiverse-sdd/SKILL.md",
    ".agents/skills/cyaxiverse-agent-orchestration/SKILL.md",
    *PROC,
    "specs/cyax-skills-2026-10-09/spec.md",
    "specs/cyax-skills-2026-10-09/plan.md",
    "specs/cyax-skills-2026-10-09/tasks.md",
    "specs/cyax-skills-2026-10-09/sources.md",
    "specs/cyax-skills-2026-10-09/cases.json",
    "specs/cyax-skills-2026-10-09/verify.py",
]
SCOPE = set(INSTRUCTION_PATHS) | {
    "specs/cyax-scientific-certification/spec.md",
    "specs/cyax-scientific-certification/plan.md",
    "specs/cyax-scientific-certification/tasks.md",
    "src/CYAxiverse.jl", "src/scientific_certification.jl",
    "src/scientific_certification/intervals.jl",
    "src/scientific_certification/stationary.jl",
    "src/scientific_certification/spectra.jl",
    "src/scientific_certification/certificates.jl",
    "test/scientific_certification.jl",
    "test/fixtures/scientific_certification/public.toml",
    "test/fixtures/scientific_certification/oracles.jl",
    "test/fixtures/scientific_certification/backend_probe.jl",
    "test/fixtures/scientific_certification/run_fixture.jl",
    "scripts/verify_scientific_certificate.jl",
    "docs/src/scientific_certification.md", "docs/make.jl",
    "Project.toml", "test/runtests.jl",
}
SKILL_SPEC_SHA = "2e9853ad4de7b79b470dd5dccc997d8cce7c3318dbf38efb21e6b0c3311f462a"
SCIENCE_SPEC_SHA = "12ab0de319ebbfef895cfaecef6a19850c2d488778394e723dc1583e77800b8b"
FROZEN_STATUS = (b"> **Status:** Frozen detached pre-edit review candidate. This document is not yet installed in the target repository, independently approved, implementation-ready, merged, or a standalone dispatch grant. The governing task has one exact authorised S2 scope; private control-plane authority and actor configuration are separately bound.")
INSTALLED_STATUS = (b"> **Status:** Installed public task documentation. This document is not a readiness, acceptance, or integration record. Current acceptance state is tracked through the governing task evidence.")
EXAMPLE_MARKER = b"\n## Historical serialized examples (development-only)"


class CheckFailure(Exception):
    def __init__(self, code: str, detail: str, path: str | None = None):
        super().__init__(detail)
        self.code, self.detail, self.path = code, detail, path


def git(args: list[str], *, cwd: Path = ROOT, check: bool = True) -> bytes:
    result = subprocess.run(["git", *args], cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if check and result.returncode:
        raise CheckFailure("GIT_ERROR", result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_spec_bytes(path: str, content: bytes) -> bytes:
    if content.count(INSTALLED_STATUS) != 1:
        raise CheckFailure("SPEC_STATUS_INVALID", "candidate document must identify installed public task documentation", path)
    content = content.replace(INSTALLED_STATUS, FROZEN_STATUS, 1)
    if path == "specs/cyax-scientific-certification/spec.md":
        if content.count(EXAMPLE_MARKER) != 1:
            raise CheckFailure("HISTORICAL_EXAMPLES_MISSING", "scientific spec must contain one historical examples appendix", path)
        content = content[:content.index(EXAMPLE_MARKER)]
    return content


def _example_block(markdown: bytes, heading: bytes, path: str) -> bytes:
    pattern = re.escape(heading) + rb"\n\n```toml\n(.*?)```"
    match = re.search(pattern, markdown, re.S)
    if match is None:
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", f"missing TOML example block {heading.decode()}", path)
    return match.group(1)


def _parse_example(markdown: bytes, heading: bytes, path: str):
    content = _example_block(markdown, heading, path)
    try:
        record = tomllib.loads(content.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", f"invalid {heading.decode()} TOML: {exc}", path) from exc
    return content, record


def _check_historical_examples(science: bytes):
    path = "specs/cyax-scientific-certification/spec.md"
    if science.count(EXAMPLE_MARKER) != 1:
        raise CheckFailure("HISTORICAL_EXAMPLES_MISSING", "scientific spec must contain one historical examples appendix", path)
    inputs, input_record = _parse_example(science, b"### Shared `inputs.toml`", path)
    root_bytes, root = _parse_example(science, b"### Root-only `certificate.toml`", path)
    combined_bytes, combined = _parse_example(science, b"### Root-and-spectrum `certificate.toml`", path)
    env_bytes, env = _parse_example(science, b"### `environment.toml`", path)
    if input_record.get("schema") != "cyax-scientific-input-v1":
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "input sample has the wrong schema", path)
    if (root.get("kind"), combined.get("kind")) != ("root", "root_and_spectrum"):
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "certificate samples must show both exact kinds", path)
    input_sha = sha(inputs)
    if root.get("input_sha256") != input_sha or combined.get("input_sha256") != input_sha:
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "certificate samples do not bind the shared input bytes", path)
    source = root.get("source_ref")
    if not isinstance(source, dict) or source != combined.get("source_ref"):
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "certificate samples must share their actual producer identity", path)
    if source.get("commit") != "4ecdb53d0f224f879646279bfa2f9128d7b14dfa" or source.get("tree") != "9608f60a0d4f770d8296f96d1ecf4eb78dbf5d0f":
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "producer commit/tree differs from the recorded development run", path)
    environment = root.get("environment_ref")
    if not isinstance(environment, dict) or environment != combined.get("environment_ref"):
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "certificate samples must share one environment receipt", path)
    if environment.get("artifact_name") != "environment.toml" or environment.get("receipt_sha256") != sha(env_bytes):
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "sample environment receipt hash does not match its exact bytes", path)
    if env.get("interval_arithmetic_version") != "0.22.36" or env.get("julia_version") != "1.12.6":
        raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", "sample environment does not identify the qualified development profile", path)
    metadata = {
        "shared_input": input_sha,
        "root_certificate": sha(root_bytes),
        "root_and_spectrum_certificate": sha(combined_bytes),
        "environment": sha(env_bytes),
    }
    expected_lines = {
        "- Shared input SHA-256": input_sha,
        "- Root-only certificate SHA-256": metadata["root_certificate"],
        "- Root-and-spectrum certificate SHA-256": metadata["root_and_spectrum_certificate"],
        "- Environment receipt SHA-256": metadata["environment"],
    }
    appendix = science[science.index(EXAMPLE_MARKER):]
    for label, digest in expected_lines.items():
        if f"{label}: `{digest}`".encode() not in appendix:
            raise CheckFailure("HISTORICAL_EXAMPLE_INVALID", f"appendix digest does not match its {label} bytes", path)
    return {
        "shared_input": {"bytes": len(inputs), "sha256": metadata["shared_input"]},
        "root_certificate": {"bytes": len(root_bytes), "sha256": metadata["root_certificate"]},
        "root_and_spectrum_certificate": {"bytes": len(combined_bytes), "sha256": metadata["root_and_spectrum_certificate"]},
        "environment": {"bytes": len(env_bytes), "sha256": metadata["environment"]},
    }


def commit_tree(commit: str) -> str:
    return git(["rev-parse", f"{commit}^{{tree}}"]).decode().strip()


def show(commit: str, path: str) -> bytes:
    return git(["show", f"{commit}:{path}"])


def list_tree(commit: str) -> dict[str, tuple[str, str]]:
    raw = git(["ls-tree", "-r", "-z", "--full-tree", commit])
    result: dict[str, tuple[str, str]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        mode, _kind, oid = meta.split(b" ", 2)
        result[name.decode("utf-8", "surrogateescape")] = (
            mode.decode("ascii"), oid.decode("ascii"))
    return result


def changed_paths(base: str, candidate: str) -> set[str]:
    raw = git(["diff", "--name-status", "--no-renames", "-z", base, candidate])
    fields = raw.split(b"\0")
    found: set[str] = set()
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i].decode("ascii")
        i += 1
        if status.startswith("R") or status.startswith("C"):
            raise CheckFailure("RENAME_FORBIDDEN", "rename/copy detected")
        if i >= len(fields):
            break
        path = fields[i].decode("utf-8", "surrogateescape")
        found.add(path)
        i += 1
    return found


def _objects(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise CheckFailure("JSON_DUPLICATE_KEY", f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _constant(value):
    raise CheckFailure("JSON_NONFINITE", f"nonstandard JSON numeric value {value}")


def _walk_limits(value, depth=0):
    if depth > 12:
        raise CheckFailure("JSON_RESOURCE_LIMIT", "JSON nesting depth exceeds 12")
    if isinstance(value, str) and len(value.encode("utf-8")) > 4096:
        raise CheckFailure("JSON_RESOURCE_LIMIT", "JSON string exceeds 4096 UTF-8 bytes")
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_limits(key, depth + 1)
            _walk_limits(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _walk_limits(child, depth + 1)


def parse_json(path: Path):
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024:
        raise CheckFailure("JSON_RESOURCE_LIMIT", "JSON file exceeds 4 MiB", str(path))
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_objects,
                           parse_constant=_constant)
    except CheckFailure:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckFailure("JSON_INVALID", str(exc), str(path)) from exc
    _walk_limits(value)
    return data, value


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise CheckFailure("CASE_FIELD_INVALID", f"{name} must be a nonempty string")


def validate_case_pack(path: Path):
    data, cases = parse_json(path)
    if not isinstance(cases, list):
        raise CheckFailure("CASE_PACK_INVALID", "case pack must be a JSON array")
    seen_ids = set()
    for item in cases:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            cid = item["id"]
            if cid in seen_ids:
                raise CheckFailure("CASE_ID_DUPLICATE", f"duplicate case id {cid}")
            seen_ids.add(cid)
    ids = []
    expected = {"id", "methods", "task_input", "supplied_fixture",
                "required_observations", "forbidden_observations", "evidence_kind"}
    allowed_kind = {"executable_red_green", "assessed_reasoning", "payload_manifest",
                    "identity_control_and_assessment"}
    for item in cases:
        if not isinstance(item, dict) or set(item) != expected:
            raise CheckFailure("CASE_PACK_INVALID", "case object has missing or unknown keys")
        cid = item["id"]
        if cid in ids:
            raise CheckFailure("CASE_ID_DUPLICATE", f"duplicate case id {cid}")
        ids.append(cid)
        if cid not in CASE_IDS:
            raise CheckFailure("CASE_PACK_INVALID", f"unknown case id {cid!r}")
        _string(item["task_input"], f"{cid}.task_input")
        _string(item["supplied_fixture"], f"{cid}.supplied_fixture")
        if not isinstance(item["methods"], list) or not item["methods"] or any(
            not isinstance(m, str) or m not in METHODS for m in item["methods"]
        ):
            raise CheckFailure("CASE_PACK_INVALID", f"{cid}.methods is invalid")
        if len(item["methods"]) != len(set(item["methods"])):
            raise CheckFailure("CASE_PACK_INVALID", f"{cid}.methods contains duplicates")
        for key in ("required_observations", "forbidden_observations"):
            if not isinstance(item[key], list) or not item[key] or any(
                not isinstance(x, str) or not x.strip() for x in item[key]
            ):
                raise CheckFailure("CASE_PACK_INVALID", f"{cid}.{key} is invalid")
        if not isinstance(item["evidence_kind"], str) or item["evidence_kind"] not in allowed_kind:
            raise CheckFailure("CASE_PACK_INVALID", f"{cid}.evidence_kind is invalid")
        expected_kind = CASE_KIND.get(cid, "assessed_reasoning")
        if item["evidence_kind"] != expected_kind:
            raise CheckFailure("CASE_PACK_INVALID", f"{cid}.evidence_kind must be {expected_kind}")
    if ids != CASE_IDS:
        raise CheckFailure("CASE_PACK_INVALID", "case IDs must be SK-01 through SK-12 in order")
    covered = {m for item in cases for m in item["methods"]}
    if covered != METHODS:
        raise CheckFailure("CASE_METHOD_COVERAGE", "case pack does not cover all eight methods")
    return data, cases


def _expected_blocks(spec: bytes):
    text = spec.decode("utf-8")
    pattern = re.compile(r"### (5\.[1-8]) [^\n]+\n\n```markdown\n(.*?)\n```", re.S)
    found = {key: body.encode("utf-8") for key, body in pattern.findall(text)}
    if set(found) != {f"5.{i}" for i in range(1, 9)}:
        raise CheckFailure("SPEC_BLOCKS_INVALID", "could not extract all eight frozen blocks")
    return found


def _check_appends(base, candidate, spec):
    blocks = _expected_blocks(spec)
    pairs = [
        (".agents/skills/cyaxiverse-julia-quality/SKILL.md", "5.1"),
        (".agents/skills/cyaxiverse-sdd/SKILL.md", "5.2"),
        (".agents/skills/cyaxiverse-agent-orchestration/SKILL.md", "5.3"),
    ]
    for path, block in pairs:
        before, after = show(base, path), show(candidate, path)
        separator = b"" if before.endswith(b"\n") else b"\n"
        if after != before + separator + blocks[block] + b"\n":
            raise CheckFailure("PARENT_MODIFIED", "skill is not baseline bytes plus the exact frozen suffix", path)
    for path, block in zip(PROC, ("5.4", "5.5", "5.6", "5.7", "5.8")):
        if show(candidate, path) != blocks[block]:
            raise CheckFailure("PROCEDURE_MISMATCH", "procedure bytes differ from frozen block", path)


def _check_instruction_budgets(base, candidate, spec):
    blocks = _expected_blocks(spec)
    entries = [
        (".agents/skills/cyaxiverse-julia-quality/SKILL.md", "5.1"),
        (".agents/skills/cyaxiverse-sdd/SKILL.md", "5.2"),
        (".agents/skills/cyaxiverse-agent-orchestration/SKILL.md", "5.3"),
    ]
    entry_bytes = []
    for path, key in entries:
        before, after = show(base, path), show(candidate, path)
        suffix = after[len(before) + (0 if before.endswith(b"\n") else 1):]
        if suffix != blocks[key] + b"\n":
            raise CheckFailure("PARENT_MODIFIED", "installed entry suffix differs from its frozen block", path)
        entry_bytes.append(suffix)

    entry_words = sum(len(value.decode("utf-8").split()) for value in entry_bytes)
    entry_size = sum(map(len, entry_bytes))
    if entry_words > 260:
        raise CheckFailure("WORD_BUDGET_EXCEEDED", f"appended entries contain {entry_words} words; limit is 260")

    procedure_sizes = {}
    procedure_words = 0
    for path in PROC:
        content = show(candidate, path)
        words = len(content.decode("utf-8").split())
        if words > 600:
            raise CheckFailure("PROCEDURE_WORD_LIMIT", f"procedure contains {words} words; limit is 600", path)
        procedure_sizes[path] = {"bytes": len(content), "words": words}
        procedure_words += words
    if procedure_words > 2000:
        raise CheckFailure("WORD_BUDGET_EXCEEDED", f"procedures contain {procedure_words} words; limit is 2000")

    parent_links = {
        PROC[0]: (entries[0][0], "references/diagnostic-test-loop.md", "5.1"),
        PROC[1]: (entries[0][0], "references/implementation-self-review.md", "5.1"),
        PROC[2]: (entries[1][0], "references/architecture-assessment.md", "5.2"),
        PROC[3]: (entries[1][0], "references/agent-writing-domain-modeling.md", "5.2"),
        PROC[4]: (entries[1][0], "references/retrospective.md", "5.2"),
    }
    for procedure, (parent, target, block_key) in parent_links.items():
        links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", blocks[block_key].decode("utf-8"))
        resolved = os.path.normpath(str(PurePosixPath(parent).parent.joinpath(target)))
        if target not in links or resolved != procedure:
            raise CheckFailure("PARENT_LINK_MISSING", "procedure has no explicit link from its owning skill", procedure)

    return {
        "appended_entries": {"bytes": entry_size, "words": entry_words},
        "procedures": {"bytes": sum(x["bytes"] for x in procedure_sizes.values()),
                       "words": procedure_words, "files": procedure_sizes},
    }


def _check_links(base: str, candidate: str, spec: bytes):
    sources = {}
    for path in INSTRUCTION_PATHS[:3]:
        before, after = show(base, path), show(candidate, path)
        separator = 0 if before.endswith(b"\n") else 1
        suffix = after[len(before) + separator:]
        sources[path] = suffix.decode("utf-8")
    for path in PROC:
        sources[path] = show(candidate, path).decode("utf-8")
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for source, body in sources.items():
        for target in link_re.findall(body):
            if "://" in target or target.startswith("mailto:") or target.startswith("#"):
                continue
            rel = target.split("#", 1)[0]
            resolved = PurePosixPath(source).parent.joinpath(rel)
            norm = os.path.normpath(str(resolved))
            if norm == ".." or norm.startswith("../"):
                raise CheckFailure("BROKEN_LINK", f"link escapes repository: {target}", source)
            if norm not in list_tree(candidate):
                raise CheckFailure("BROKEN_LINK", f"link target is missing: {target}", source)


def _check_modes_and_mirrors(base, candidate):
    old, new = list_tree(base), list_tree(candidate)
    for path in SCOPE:
        if path not in new:
            raise CheckFailure("SCOPED_PATH_MISSING", "required path is absent", path)
        if path in old and new[path][0] != old[path][0]:
            raise CheckFailure("MODE_CHANGED", "existing file mode changed", path)
        if path not in old and new[path][0] != "100644":
            raise CheckFailure("MODE_INVALID", "new file must have mode 100644", path)
    mirrors = sorted(p for p in old if p.startswith((".codex/skills/", ".claude/skills/")))
    if len(mirrors) != 14:
        raise CheckFailure("MIRROR_BASE_INVALID", f"expected 14 mirror symlinks, found {len(mirrors)}")
    for path in mirrors:
        if old[path][0] != "120000" or new.get(path) != old[path]:
            raise CheckFailure("MIRROR_CHANGED", "mirror mode/link bytes differ from base", path)
        target = show(candidate, path).decode("utf-8")
        resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
        mirror_root = ".codex/skills/" if path.startswith(".codex/") else ".claude/skills/"
        canonical = ".agents/skills/" + path[len(mirror_root):]
        if resolved != canonical:
            raise CheckFailure("MIRROR_TARGET_INVALID", f"mirror resolves to {resolved}", path)


def _check_narrow_edits(base, candidate):
    replacements = {
        "src/CYAxiverse.jl": [(b'include("axion_photon.jl")\n',
                                b'include("axion_photon.jl")\ninclude("scientific_certification.jl")\n')],
        "test/runtests.jl": [(b'include(joinpath(@__DIR__, "optional_plotting.jl"))\n',
                              b'include(joinpath(@__DIR__, "optional_plotting.jl"))\ninclude(joinpath(@__DIR__, "scientific_certification.jl"))\n')],
        "docs/make.jl": [(b'        "Pipelines" => "pipelines.md",\n',
                           b'        "Pipelines" => "pipelines.md",\n        "Scientific certification" => "scientific_certification.md",\n')],
        "Project.toml": [(b'StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"\n',
                           b'StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"\nTOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"\n'),
                          (b'Test = "1"\n', b'Test = "1"\nTOML = "1"\n')],
    }
    for path, edits in replacements.items():
        expected = show(base, path)
        for old, new in edits:
            if expected.count(old) != 1:
                raise CheckFailure("BASELINE_EXPECTATION_INVALID", "narrow edit anchor not unique", path)
            expected = expected.replace(old, new, 1)
        if show(candidate, path) != expected:
            raise CheckFailure("NARROW_EDIT_MISMATCH", "content exceeds the approved narrow edit", path)


def _check_clean(candidate):
    head = git(["rev-parse", "HEAD"]).decode().strip()
    if head != candidate:
        raise CheckFailure("CANDIDATE_HEAD_MISMATCH", "HEAD differs from requested candidate")
    status = git(["status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        raise CheckFailure("DIRTY_CANDIDATE", "worktree/index contains uncommitted or untracked content")


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckFailure("EVIDENCE_INVALID", f"{name} must be an integer >= {minimum}")


def _file_digest(path: Path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def validate_evidence(path: Path, candidate: str, tree: str, checker_hash: str,
                      case_hash: str, procedure_hashes: dict[str, str]):
    _raw, receipt = parse_json(path)
    required = {"schema", "candidate_commit", "candidate_tree", "checker_sha256",
                "case_pack_sha256", "procedure_sha256", "cases"}
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise CheckFailure("EVIDENCE_INVALID", "receipt has missing/unknown keys or schema")
    if not isinstance(receipt["schema"], str) or receipt["schema"] != "cyax-skills-smoke-1":
        raise CheckFailure("EVIDENCE_INVALID", "receipt schema is invalid")
    for key in ("candidate_commit", "candidate_tree"):
        if not isinstance(receipt[key], str) or not FULL_SHA.fullmatch(receipt[key]):
            raise CheckFailure("EVIDENCE_INVALID", f"{key} must be a full lower-case Git identity")
    if receipt["candidate_commit"] != candidate or receipt["candidate_tree"] != tree:
        raise CheckFailure("EVIDENCE_CANDIDATE_MISMATCH", "receipt candidate identity differs")
    for key in ("checker_sha256", "case_pack_sha256"):
        if not isinstance(receipt[key], str) or not HASH.fullmatch(receipt[key]):
            raise CheckFailure("EVIDENCE_INVALID", f"{key} must be a lower-case SHA-256")
    if receipt["checker_sha256"] != checker_hash or receipt["case_pack_sha256"] != case_hash:
        raise CheckFailure("EVIDENCE_HASH_MISMATCH", "receipt checker/case hash is stale")
    if receipt["procedure_sha256"] != procedure_hashes:
        raise CheckFailure("EVIDENCE_HASH_MISMATCH", "receipt procedure hashes are stale")
    cases = receipt["cases"]
    if not isinstance(cases, list):
        raise CheckFailure("EVIDENCE_INVALID", "receipt cases must be an array")
    case_ids = []
    seen_case_ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise CheckFailure("EVIDENCE_INVALID", "receipt case id must be a string")
        cid = case["id"]
        if cid in seen_case_ids:
            raise CheckFailure("CASE_ID_DUPLICATE", f"duplicate receipt case id {cid}")
        seen_case_ids.add(cid)
        case_ids.append(cid)
    if case_ids != CASE_IDS:
        raise CheckFailure("EVIDENCE_INCOMPLETE", "receipt must contain each case exactly once in order")
    ckeys = {"id", "evidence_status", "criterion_status", "simulated_action_status",
             "actor_ref", "assessment_ref", "evidence", "commands"}
    ekeys = {"artifact_ref", "local_path", "sha256", "bytes"}
    cmdkeys = {"argv", "exit_code", "expected_exit", "sequence"}
    evidence_root = path.resolve().parent
    observed_bytes = {}
    for item in cases:
        cid = item["id"]
        if set(item) != ckeys:
            raise CheckFailure("EVIDENCE_INVALID", f"{cid} case has missing/unknown keys")
        if (not isinstance(item["evidence_status"], str) or item["evidence_status"] not in {"observed", "not_run"}
                or not isinstance(item["criterion_status"], str) or item["criterion_status"] not in {"met", "unmet", "unassessed"}):
            raise CheckFailure("EVIDENCE_INVALID", f"{cid} status field is invalid")
        if item["evidence_status"] != "observed" or item["criterion_status"] != "met":
            raise CheckFailure("EVIDENCE_INCOMPLETE", f"{cid} is not observed and assessed met")
        if (not isinstance(item["simulated_action_status"], str)
                or item["simulated_action_status"] not in {"not_applicable", "BLOCKED"}):
            raise CheckFailure("EVIDENCE_INVALID", f"{cid} simulated_action_status is invalid")
        if cid == "SK-12" and item["simulated_action_status"] != "BLOCKED":
            raise CheckFailure("EVIDENCE_INCOMPLETE", "SK-12 must record its simulated BLOCKED action")
        for key in ("actor_ref", "assessment_ref"):
            _string(item[key], f"{cid}.{key}")
        evs = item["evidence"]
        if not isinstance(evs, list) or not evs:
            raise CheckFailure("EVIDENCE_INCOMPLETE", f"{cid} has no observed evidence")
        for ev in evs:
            if not isinstance(ev, dict) or set(ev) != ekeys:
                raise CheckFailure("EVIDENCE_INVALID", f"{cid} evidence record has wrong keys")
            for key in ("artifact_ref", "local_path"):
                _string(ev[key], f"{cid}.{key}")
            if not isinstance(ev["sha256"], str) or not HASH.fullmatch(ev["sha256"]):
                raise CheckFailure("EVIDENCE_INVALID", f"{cid} evidence SHA-256 is malformed")
            _int(ev["bytes"], f"{cid}.evidence.bytes")
            rel = Path(ev["local_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise CheckFailure("EVIDENCE_PATH_INVALID", "evidence path must stay under its receipt directory")
            target = (path.parent / rel).resolve()
            if evidence_root not in (target, *target.parents) or not target.is_file():
                raise CheckFailure("EVIDENCE_PATH_INVALID", "evidence file is missing or escapes its directory")
            actual_bytes, actual_sha = _file_digest(target)
            if actual_bytes != ev["bytes"] or actual_sha != ev["sha256"]:
                raise CheckFailure("EVIDENCE_HASH_MISMATCH", "evidence bytes/hash do not match")
            observed_bytes.setdefault(cid, []).append(target)
        commands = item["commands"]
        if not isinstance(commands, list):
            raise CheckFailure("EVIDENCE_INVALID", f"{cid}.commands must be an array")
        seqs = []
        for command in commands:
            if not isinstance(command, dict) or set(command) != cmdkeys:
                raise CheckFailure("EVIDENCE_INVALID", f"{cid} command has wrong keys")
            if not isinstance(command["argv"], list) or not command["argv"] or any(not isinstance(x, str) or not x for x in command["argv"]):
                raise CheckFailure("EVIDENCE_INVALID", f"{cid} command argv is invalid")
            _int(command["sequence"], f"{cid}.sequence", 1)
            if isinstance(command["exit_code"], bool) or not isinstance(command["exit_code"], int):
                raise CheckFailure("EVIDENCE_INVALID", f"{cid}.exit_code must be an integer")
            if not isinstance(command["expected_exit"], str) or command["expected_exit"] not in {"zero", "nonzero"}:
                raise CheckFailure("EVIDENCE_INVALID", f"{cid}.expected_exit is invalid")
            if (command["expected_exit"] == "zero") != (command["exit_code"] == 0):
                raise CheckFailure("EVIDENCE_INVALID", f"{cid} expected_exit does not match the observed exit code")
            seqs.append(command["sequence"])
        if len(seqs) != len(set(seqs)):
            raise CheckFailure("EVIDENCE_INVALID", f"{cid} command sequence duplicates")
        if cid == "SK-01":
            ordered = sorted(commands, key=lambda c: c["sequence"])
            if len(ordered) < 2 or ordered[0]["exit_code"] == 0 or ordered[1]["exit_code"] != 0:
                raise CheckFailure("EVIDENCE_INCOMPLETE", "SK-01 requires ordered nonzero-red and zero-green commands")
        if cid == "SK-11" and not any(c["exit_code"] != 0 for c in commands):
            raise CheckFailure("EVIDENCE_INCOMPLETE", "SK-11 requires an observed failing identity control")
        if cid == "SK-11":
            payloads = []
            for artifact in observed_bytes.get(cid, []):
                try:
                    payloads.append(artifact.read_text(encoding="utf-8"))
                except (UnicodeError, OSError):
                    continue
            if not any("EVIDENCE_CANDIDATE_MISMATCH" in body for body in payloads):
                raise CheckFailure("EVIDENCE_INCOMPLETE", "SK-11 evidence must retain the named candidate-mismatch control output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evidence")
    args = parser.parse_args()
    errors = []
    out = {"status": "error", "base_commit": args.base,
           "candidate_commit": args.candidate, "errors": errors}
    try:
        if not FULL_SHA.fullmatch(args.base) or not FULL_SHA.fullmatch(args.candidate):
            raise CheckFailure("COMMIT_ID_INVALID", "base and candidate must be full lower-case Git identities")
        git(["cat-file", "-e", f"{args.base}^{{commit}}"])
        git(["cat-file", "-e", f"{args.candidate}^{{commit}}"])
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", args.base, args.candidate],
                                  cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if ancestor.returncode != 0:
            raise CheckFailure("BASE_NOT_ANCESTOR", "base is not an ancestor of candidate")
        _check_clean(args.candidate)
        btree, ctree = commit_tree(args.base), commit_tree(args.candidate)
        paths = changed_paths(args.base, args.candidate)
        if paths != SCOPE:
            missing, extra = sorted(SCOPE-paths), sorted(paths-SCOPE)
            raise CheckFailure("OUT_OF_SCOPE", f"changed paths mismatch; missing={missing}; extra={extra}")
        modes = _check_modes_and_mirrors(args.base, args.candidate)
        spec = show(args.candidate, "specs/cyax-skills-2026-10-09/spec.md")
        science = show(args.candidate, "specs/cyax-scientific-certification/spec.md")
        skill_core = _canonical_spec_bytes("specs/cyax-skills-2026-10-09/spec.md", spec)
        science_core = _canonical_spec_bytes("specs/cyax-scientific-certification/spec.md", science)
        if sha(skill_core) != SKILL_SPEC_SHA or sha(science_core) != SCIENCE_SPEC_SHA:
            raise CheckFailure("APPROVED_SPEC_MISMATCH", "approved normative spec bytes differ")
        historical_examples = _check_historical_examples(science)
        _check_links(args.base, args.candidate, spec)
        _check_appends(args.base, args.candidate, spec)
        instruction_sizes = _check_instruction_budgets(args.base, args.candidate, spec)
        _check_narrow_edits(args.base, args.candidate)
        case_bytes, cases = validate_case_pack(ROOT/"specs/cyax-skills-2026-10-09/cases.json")
        size_paths = {
            "source_provenance": "specs/cyax-skills-2026-10-09/sources.md",
            "instruction_spec": "specs/cyax-skills-2026-10-09/spec.md",
            "scientific_spec": "specs/cyax-scientific-certification/spec.md",
            "case_pack": "specs/cyax-skills-2026-10-09/cases.json",
        }
        content_sizes = {}
        for label, path in size_paths.items():
            content = show(args.candidate, path)
            content_sizes[label] = {"bytes": len(content), "sha256": sha(content)}
        proc_hashes = {path: sha(show(args.candidate, path)) for path in PROC}
        checker_hash = sha(show(args.candidate, "specs/cyax-skills-2026-10-09/verify.py"))
        if args.evidence:
            validate_evidence(Path(args.evidence).resolve(), args.candidate, ctree,
                              checker_hash, sha(case_bytes), proc_hashes)
        out.update({"status": "ok", "base_tree": btree, "candidate_tree": ctree,
                    "checker_sha256": checker_hash, "case_pack_sha256": sha(case_bytes),
                    "procedure_sha256": proc_hashes, "checked_paths": sorted(paths),
                    "changed_path_count": len(paths), "instruction_path_count": len(INSTRUCTION_PATHS),
                    "case_count": len(cases), "procedure_count": len(PROC),
                    "instruction_sizes": instruction_sizes,
                    "source_provenance_spec_case_sizes": content_sizes,
                    "historical_example_sizes": historical_examples,
                    "evidence_checked": bool(args.evidence)})
    except CheckFailure as exc:
        errors.append({"code": exc.code, **({"path": exc.path} if exc.path else {}),
                       "detail": exc.detail})
    except (OSError, ValueError) as exc:
        errors.append({"code": "INPUT_UNAVAILABLE", "detail": str(exc)})
    print(json.dumps(out, sort_keys=True))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
