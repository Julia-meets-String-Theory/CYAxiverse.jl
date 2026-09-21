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

## Independent review history

The first frozen implementation candidate was commit
`638ee4dfcd237d91646d963ac77cb473fcaccef1` (tree
`dbf2dda371b213c317d10606fb38d859658de7a5`). Independent SPEC and
STANDARDS reviews both returned `REQUEST_CHANGES`. The blocking findings
covered different-final reservation consumption; duplicate or late candidate
withdrawal; certification identity binding; default remote event observation,
topology and race checks; and public identity sanitation. The review verdicts
belong only to that exact candidate. A corrected candidate requires fresh
reviews before any premerge readiness claim.

The second frozen candidate was commit
`478cf194bc6c9702598bba28e39792bd2d17006b` (tree
`8c0a304ac4cdaddc2e85a85ec1941fbdee3284c2`). Independent SPEC and
STANDARDS reviews again returned `REQUEST_CHANGES`. Findings covered
pre-entry abort reuse in allocation, canonical public-tag protection proof,
durable certification-transfer evidence, candidate refs, exact schema
versions, public evidence validation, and maintenance base/line binding.
Those verdicts belong only to the second candidate. The corrected successor
requires fresh reviews.

The third frozen candidate was commit
`932b6a94230e9b93d1dabf10d7033d5c50f8c7f3` (tree
`1c7c0d3f3feb9f02f77636d7f60e0c1298470ee6`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered reservation
non-entry proof, intent certification identity, public URL sanitation,
retrospective truth, documentation certification references, release mismatch
classification, event-tree topology and authority callback failure handling.
Those verdicts belong only to the third candidate. The corrected successor
requires fresh reviews.

The fourth frozen candidate was commit
`391420c5b4b97e9fad60628a364730ff9cdfdd43` (tree
`df25e78db667083caf78a0ffea3efe3f1b64eaf1`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. The findings covered duplicate
reservation opening, early remote unsupported-binding handling, incomplete
closure identities, private IP and URL userinfo sanitation, remote event-branch
recreation, static/event exclusion, and the absence of a runnable pinned
checkout certification harness. Those verdicts belong only to the fourth
candidate. A corrected successor requires fresh reviews.

The fifth frozen candidate was commit
`168858862372d44c8c46220ac2921bbd7d0b5c13` (tree
`9f6be909e502d1b75c65932094de41833023266f`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered alternate
private-host spellings, branch-only iteration anchors, nonempty remote
bootstrap, missing bootstrap exclusion, boolean exclusion checks that did not
hold through remote mutation, caller-forged static authority, and unbound
certification test commands and runtime identities. Those verdicts belong
only to the fifth candidate. The corrected successor requires fresh reviews.

The sixth frozen candidate was commit
`ed37febddc4b72147ce1244248ae68023a50fb72` (tree
`4edf48d4030fd63cbaa2dccceaca9ea2cf356a87`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered transaction
leases across complete mutations, the unguarded local event bootstrap, and
private or malformed public repository locators. Those verdicts belong only
to the sixth candidate. The corrected successor requires fresh reviews.

The seventh frozen candidate was commit
`7c04338c44688a897b6b31be1e0d090c4f5260f6` (tree
`4c86f7d11c579762cb61d6c80e553c8aab4c7315`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered reserved
special-use source domains, public IPv6 locator handling, and query/fragment
delimiters. A manager integration probe also found that nested event writes
must reuse a transaction's nonreentrant external lease. Those verdicts belong
only to the seventh candidate. The corrected successor requires fresh reviews.

The eighth frozen candidate was commit
`fb612fbf511d6d905f8fcd3893dfeea4e97a21f1` (tree
`0db30ea0a103093e79ab8c713568939b7d3f02f6`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered recovery from
an existing remote event branch without a local cache, the required
unsupported-certification-binding reason, and bare private IP values in
durable evidence or repository slugs. Those verdicts belong only to the
eighth candidate. The corrected successor requires fresh reviews.

The ninth frozen candidate was commit
`ad98b29a1ef1f7c23a4591681d85ac8caa064a23` (tree
`173ab53556082c48b2dc466223e76fb084fc226b`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered abbreviated
private IPv4 addresses and embedded endpoint forms in durable evidence, plus
IP-like source components hidden behind `.git`, SCP, and URL forms. Those
verdicts belong only to the ninth candidate. The corrected successor requires
fresh reviews.

The tenth frozen candidate was commit
`d34f3caa3d0f6d90e2e6b8ced798673ee77eba95` (tree
`58cab748122edb19c582de900bc9c769bf7db7de`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Findings covered false
rejection of canonical maintenance refs as private addresses, Julia version
component bounds and duplicate version validators, local-cache-only remote
idempotency, default CLI path redaction corrupting version identities, and
remote append without an exact-head compare-and-swap guard. Those verdicts
belong only to the tenth candidate. The corrected successor requires fresh
reviews.

The eleventh frozen candidate was commit
`7335e167661d958b3976f73a06fa9388aaf4aafd` (tree
`ff0bc6f721d16971af053e9de8b9b9ec2ebae924`). Independent SPEC and
STANDARDS reviews returned `REQUEST_CHANGES`. Both found that verified remote
transaction replay from a stale local cache was misclassified as a stale-head
block, including same-ID payload collisions. STANDARDS also found that the
read-only CLI could emit remote transport credentials or local locators in JSON
reports and exception details. Those verdicts belong only to the eleventh
candidate. The corrected successor requires fresh reviews.

The twelfth frozen candidate was commit
`d50b75d9f304f23e45d8f43f70c0950307715db9` (tree
`2dfdc7fd19dd8cee9dbc266de615c26b97e94411`). The independent SPEC
review returned `PASS`; the independent STANDARDS review returned
`REQUEST_CHANGES`. Standards findings covered option-like remote values that
Git could interpret as transport flags and argparse failures that bypassed the
CLI's documented JSON error contract. Those verdicts belong only to the
twelfth candidate. The corrected successor requires fresh reviews.

The thirteenth frozen candidate was commit
`eec4ed0aaf101af4a961f94e4ef8150058abe9da` (tree
`699d938516aa53ce03df742be7f7d3dcfe8d0e93`). A public draft-PR exact-state
review returned `REQUEST_CHANGES`. The blocking findings covered a
caller-forgeable allocation occupancy proof, incomplete verification of the
canonical event branch's Git history, and a lifecycle CI fixture whose
publisher clone did not start from `vmm`. Those findings belong only to the
thirteenth candidate. The corrected successor requires fresh independent SPEC
and STANDARDS reviews.

The fourteenth frozen candidate was commit
`2d506e627b9362db4ebcbba0673f1f3423750ed2` (tree
`53a10d98b4fdc471c6e71c51c2e112957685494e`). The independent STANDARDS
review returned `PASS`; the independent SPEC review returned
`REQUEST_CHANGES`. SPEC found that verified event heads bound commit, bytes and
events but did not bind the canonical `release-events` ref and stream, so a
second valid ledger branch could reach allocation. Those verdicts belong only
to the fourteenth candidate. The corrected successor requires fresh reviews.

The fifteenth frozen candidate was commit
`877dca147599e8e35756f7ded1241c25715019d6` (tree
`e44eda688f55926cd00a41ab5f364e0984ccb6fe`). The public draft-PR review
returned additional blocking findings: static and mutable authority could come
from different repositories; future canonical-tag protection was not global;
forward reconciliation accepted an incomplete intent; maintenance patch
allocation was not bounded to Julia's `UInt32` domain; principal/maintenance
main identities were under-validated; the read-only CLI fetched into the
inspected repository; serialized snapshot mappings were always stale; docs
routing accepted nested tag refs and did not apply the shared component bound;
the certification executable banner and pre-tag certification identities used
weaker public-value checks; and documentation release evidence trusted Git
identities copied from the released event. Those findings belong only to the
fifteenth candidate, which is superseded.

The bounded correction implementation is commit
`d15124d534516d488c7e9ac5b0181b09882eee78` (tree
`123cb304e62250a79859405224a4fed22106a234`). It closes the accepted findings
above with positive and adversarial regressions. It also re-verifies the three
post-`eec4ed0a` blockers: allocation accepts only a writer-verified ledger head;
the complete `release-events` history must reach the empty orphan bootstrap and
bind each event to its actual parent; and the CI publishing fixture clones
`vmm` explicitly. This commit has no fresh exact-candidate SPEC or STANDARDS
verdict. The mechanical evidence/tasks successor does not change lifecycle,
scientific, package, or workflow behavior and must be reviewed together with
this implementation commit as the final PR state.

The sixteenth frozen candidate was commit
`52b2569a57bd07fd3e9742e8fa8269984456f8c3` (tree
`0c27c7622e1cb7689303afbf98b4cab9dbd783f6`). Fresh independent SPEC and
STANDARDS reviews both returned `REQUEST_CHANGES`. STANDARDS found that the
documentation verifier accepted malformed or conflicting duplicate
`ls-remote` advertisements and that the CLI duplicated writer internals. SPEC
found that reconciliation and the pre-tag transaction boundary compared only
a subset of the durable intent's candidate, anchor, release-line and
certification identity. Those verdicts belong only to the sixteenth candidate.

The review-loop correction implementation is commit
`870c36500fb1a2b894060ee68ba20cdca95a29f3` (tree
`3f3bbcd36806f94d0576970c8bcd12dc72f10a4c`). It adds one strict shared remote
advertisement parser, a public verified remote-ledger read path, full canonical
intent validation before tag creation, and exact intent-to-released-event
identity comparison. Adversarial tests cover malformed, duplicate, conflicting,
unrelated and peeled-only advertisements plus candidate, anchor, release-line
and certification mismatches. Fresh exact-successor reviews are required.

The seventeenth frozen candidate was commit
`6d9a4d5e10094105d4e19d437e8c2e87a1bc1fe2` (tree
`78be489009c1bf63253e3c5ce1fbd9ef5e4f0118`). Its fresh independent SPEC
review returned `PASS`. Its fresh independent STANDARDS review returned
`REQUEST_CHANGES`: peeled-tag records were enabled for branch lookups, Unicode
line splitting could hide control-character framing, and the remote-ledger and
documentation readers did not prove that the advertised refs remained stable
through their fetches. The writer's public read path also duplicated its
existing reconciliation primitive. Those verdicts belong only to the
seventeenth candidate.

The second review-loop correction implementation is commit
`0b6e76cf346958018273b9ef2b1c5835e238cffb` (tree
`b60e0fd652affcf613087c21f4bce325f68222f1`). It restricts peeling to tag
refs, requires strict ASCII/LF remote advertisements, rechecks documentation
refs after isolated fetch, and routes the public remote-ledger read through
the shared race-detecting observation primitive. Adversarial regressions cover
branch peeling, control and unterminated records, post-fetch ref revalidation,
and a branch advance between advertisement and fetch. Fresh exact-successor
reviews are required.

## Observed checks

| Check | Observed result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_version_lifecycle_*.py'` | Passed: 241 tests in 66.950 seconds on `0b6e76c`. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts/test_check_version_bump.py scripts/test_documentation_routing.py` | Passed: 13 tests (3 version, 10 documentation) in 10.925 seconds. |
| `python3 scripts/check_version_bump.py --base 995163f0058488ea183ac645045ed8b1636bef4a --head HEAD` | Passed: lifecycle-only changes require no package-version update. |
| `DOCS_DEPLOY=false CYAX_DOCS_REF=refs/heads/vmm julia --compiled-modules=no --project=docs/ docs/make.jl` | Passed on `0b6e76c` with offline installed dependencies and a writable temporary Julia depot; Documenter rendered all configured pages. |
| Python-unavailable core `using CYAxiverse` | Passed on `0b6e76c` with `PYTHON` and `PYTHONHOME` set to unavailable paths and the optional PyCall extension absent. |
| `python3 scripts/agent_verify.py package` | Failed: existing phase/volume detuning Hessian test at `test/runtests.jl:1267` expects `4π²`; observed `2π²`. |
| `julia --project=. bin/audit.jl` via `agent_verify.py run` | Failed: the same two JET reports for undefined `i` in `reduced_models.jl` and `poly102_inflation.jl`; file-monitor exhaustion warnings also appeared after the two reports. |
| `git diff --exit-code 995163f0058488ea183ac645045ed8b1636bef4a -- src test Project.toml bin/audit.jl` and the same comparison against `877dca1` | Passed; the failing Julia source, tests, package metadata, and audit script are unchanged by the full Gate A diff and this correction pass. |
| `python3 scripts/agent_verify.py snapshot` and `python3 scripts/agent_verify.py diff-check` | Passed; local snapshot captured and no whitespace errors. |
| Remote CI on superseded `52b2569` | Lifecycle and documentation passed; fast tests reproduced only the recorded Hessian baseline failure; full suite skipped by workflow. |
| `git diff --check`, Python compilation, and Ruby YAML parse of both workflows | Passed. |

The failed broad Julia gates remain failed at their recorded baseline. The
scientific normalization and benchmark code are outside this approved
lifecycle change, `Project.toml` remains `0.2.0`, and no production release
state changed. Remote CI and fresh exact-final-candidate SPEC/STANDARDS review
for the second review-loop successor are not yet observed. Live protection/event-authority setup, an explicit owner
merge decision, and fresh post-settings review remain later work; Gate B stays
separate.
