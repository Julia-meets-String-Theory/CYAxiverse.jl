# CYAX-0125 Gate A candidate evidence

This record belongs to the premerge Gate A candidate. It does not designate a
historical release, adopt a DEV version, close an iteration, publish a release,
or establish live GitHub protection. The approved specification is
`afc90e63f4703a53a51f3fd296e52edbac325862`; the feature branch starts
from `origin/vmm` at `995163f0058488ea183ac645045ed8b1636bef4a`.

## Canonical storage and synthetic release packet

The future canonical mutable event location is the protected orphan
`release-events` branch, sole file `release-events.jsonl`. Prospective static
metadata is selected from `refs/heads/vmm:iterations.toml`; protected
`iterations/X.Y.Z` anchors carry closure identity. Candidate refs and public
`vX.Y.Z` tags are distinct Git identities. A `released` event precedes GitHub
Release publication. Publication evidence is a separate immutable or content
addressed artifact keyed by event ID and public tag, with its digest and
location recorded after publication. No production event authority or release
artifact is created by this candidate.

The checked-in **synthetic** packet is
[`fixtures/synthetic-release-evidence.json`](fixtures/synthetic-release-evidence.json).
It contains principal and maintenance examples with every R-042 released-event
identity: event ID/type/time, final version and line, anchor ref/SHA/tree,
candidate ref/SHA/tree, final SHA/tree, line-specific `main` identity,
certification binding/subject/policy/harness/environment/evidence references,
closure UTC time, public tag, and evidence references. Separate synthetic
publication artifacts are
[`fixtures/synthetic-publication-principal.json`](fixtures/synthetic-publication-principal.json)
and
[`fixtures/synthetic-publication-maintenance.json`](fixtures/synthetic-publication-maintenance.json).
Their exact raw bytes match the SHA-256 digests in the packet. The fixture test
revalidates both complete tag/event/Release/publication tuples as
`terminal_consistent`. Repeated hexadecimal characters and numeric release
IDs in these files are fixture values only.

## Observed checks

| Check | Observed result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_version_lifecycle_*.py'` | Passed: 101 tests. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts/test_check_version_bump.py scripts/test_documentation_routing.py` | Passed: 11 tests (3 version, 8 documentation). |
| `DOCS_DEPLOY=false CYAX_DOCS_REF=refs/heads/vmm julia --project=docs/ docs/make.jl` | Passed on the regular local host; Documenter rendered all configured pages. |
| Python-unavailable core `using CYAxiverse` | Passed on the regular local host with `PYTHON` and `PYTHONHOME` set to unavailable paths and the optional PyCall extension absent. |
| `python3 scripts/agent_verify.py package` | Failed: existing phase/volume detuning Hessian test at `test/runtests.jl:1267` expects `4π²`; observed `2π²`. |
| `julia --project=. bin/audit.jl` via `agent_verify.py run` | Failed: two existing JET reports for undefined `i` in benchmark modules. |
| `git diff --exit-code origin/vmm -- src test Project.toml bin/audit.jl` | Passed; the failing Julia source, tests, package metadata, and audit script are unchanged by Gate A. |
| `python3 scripts/agent_verify.py snapshot` and `python3 scripts/agent_verify.py diff-check` | Passed; local snapshot captured and no whitespace errors. |
| `git diff --check` and Ruby YAML parse of both changed workflows | Passed. |

The failed broad Julia gates remain failed. The scientific normalization and
benchmark code are outside this approved lifecycle change. Remote CI and live
protection enforcement are not yet observed. An explicit owner merge decision
and a fresh postmerge settings/refs review are still required before Gate A
can be declared complete.
