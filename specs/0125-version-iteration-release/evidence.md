# CYAX-0125 Gate A candidate evidence

This record belongs to a provisional premerge Gate A worktree candidate. It
does not designate a historical release, adopt a DEV version, close an
iteration, publish a release, or establish live GitHub protection. The exact
normative amendment reviewed for I2 is commit
`2c65bfd5b332f57b9bc2b3e4ab11e9bfdd336aeb` (tree
`7620f1c966cb37c5d2c7ed5c54778758ed2c12eb`). The latest correction
implementation is frozen at commit
`dc8cb513d9a20a9c20ebc7a370477aac6d8a111e` (tree
`6b698382ff44d28a5fe25f6642a0e20462eafb45`). The feature branch
uses target PR head `20b3935ace0e01fcee2808681c56595e3afa7667` as its
execution base.

The manager is operating under the exact handoff packet SHA
`19fa93e2b4b1439b1cc4217cf7ff9e42846980d46df66793156f72789894e40a`, review
result SHA
`a88d7dde2f19b867152ad0bd1b858bf1aaa6008abddae61df1a316fafc3da0bf`, owner
approval receipt SHA
`140d1fe1c7c09db3d40a04c66c1b258f18e8c42025a3453c2472ed682124637c`, and
canonical review-rubric SHA
`418f2d5a276cbdb74b8ad331b532d59219ef33b4e9fabc2b3d4a21c55bc06c72`.

## Canonical storage and synthetic release packet

The reduced candidate supersedes the predecessor's append-only event-ledger
design. Its prospective authority is `refs/heads/vmm:iterations.toml`,
protected `iterations/X.Y.Z` anchors, canonical `vX.Y.Z` tags, and protected
create-once `refs/heads/lifecycle/v1/*` refs. Each lifecycle ref contains one
immutable `manifest.json`; no mutable stream is canonical and Gate A creates no
production lifecycle ref, tag, or release.

The checked-in **synthetic principal-only** packet is
[`fixtures/synthetic-lifecycle-principal.json`](fixtures/synthetic-lifecycle-principal.json).
It contains a complete immutable released/publication manifest pair with a
content-derived manifest identity, deterministic publication identity, and
matching released-manifest digest and owner-authorization bindings. Repeated
hexadecimal identities are fixture values only. Maintenance bootstrap/release
automation is deferred to a later approved S2 gate. Gate A includes only the
validation-only `maintenance-bootstrap-validation-v1` validator and synthetic
positive, non-entry, mismatch, uncertain, and failed-activation cases in
`scripts/test_version_lifecycle_maintenance_validation.py`; it includes no
production maintenance writer. The predecessor event-ledger packet and its
review chronology below remain historical evidence and are not authority for
this reduced candidate.

## Current reduced-candidate checks

The exact reviewed normative authority is commit
`2c65bfd5b332f57b9bc2b3e4ab11e9bfdd336aeb` (tree
`7620f1c966cb37c5d2c7ed5c54778758ed2c12eb`). Independent SPEC and STANDARDS
reviews both returned `PASS` using `gpt-5.6-sol` at `high` reasoning. External
approval record `reviews/n1-approval-v3.json`, SHA-256
`cf0cce050dd39ec508a4528464b889a5a0784807d6ce844f3868a742aee6057c`, binds
that exact five-file revision and both exact review records without changing
the reviewed normative bytes.

These observations apply to the frozen implementation commit above. This
evidence-only successor changes no implementation bytes. The complete final
candidate, including this record, still requires fresh exact-state review.

| Check | Observed result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_version_lifecycle_*.py'` | Passed: 127 tests. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_documentation_routing.py'` | Passed: 10 tests. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_check_version_bump.py'` | Passed: 3 tests. |
| `python3 scripts/check_version_bump.py --base 20b3935ace0e01fcee2808681c56595e3afa7667 --head HEAD` | Passed: no package implementation or `Project.toml` change requires a version bump. |
| `python3 scripts/agent_verify.py snapshot` and `python3 scripts/agent_verify.py diff-check` | Passed: worktree snapshot captured and no whitespace errors. |
| `DOCS_DEPLOY=false CYAX_DOCS_REF=refs/heads/vmm julia --compiled-modules=no --project=docs/ docs/make.jl` | Passed; Documenter rendered all configured pages. |
| Python-unavailable `julia --project=. -e 'using CYAxiverse'` | Passed with `PYTHON` and `PYTHONHOME` set to unavailable paths. |
| `python3 scripts/agent_verify.py package` | Failed at the unchanged `test/runtests.jl:1267` Hessian baseline: expected `4π²`, observed `2π²`. The same failure is already recorded below and `src`, `test`, `Project.toml`, `Manifest.toml`, and `bin/audit.jl` have no candidate delta. |
| `julia --project=. bin/audit.jl` | Failed with the same two unchanged JET undefined-`i` reports in `reduced_models.jl` and `poly102_inflation.jl` recorded below. The audited source and audit script have no candidate delta. |
| Remote CI for the exact implementation candidate | Pending until the candidate is committed and pushed; no result is claimed. |

The superseded correction candidate `aa3795610ba67784fb9c4546be73aa4bbf325db1`
(tree `e1546855d82ac2653ee52878327f481734ab52bf`) received independent SPEC and
STANDARDS `REQUEST_CHANGES` verdicts. Those verdicts remain attached only to
that immutable candidate. Commit `9314c05592e0a0659dc9e405e278f5b33da91a74`
is the bounded implementation correction and has no review verdict. The exact
successor containing the preceding evidence update was commit
`5fc8817651eb834b236624e11cd339363d5036e1` (tree
`c9eff2df02ac54b222cae5ca56900f593ff27b69`). Its independent SPEC and
STANDARDS reviews both returned `REQUEST_CHANGES`; those verdicts remain
attached only to that immutable candidate. Commit
`a88ed41d514b1c6aa1e0bb78ffa6af418854bc81` is the next bounded correction
implementation. Its evidence successor was commit
`28c2476994a18302f8eda05c4326825d807d42c0` (tree
`c4159e3a59b17784a9b88f34d1a28136dd891c9f`); independent SPEC and STANDARDS
reviews both returned `REQUEST_CHANGES`, and those verdicts remain attached
only to that immutable candidate. Commit
`2431504ce1c8885c70737eff9be2ce251b8c486c` is the next bounded correction
implementation. Its evidence successor was commit
`2531135962c5c2e3430a48b45b518f56d5badee5` (tree
`dc46d7cb8690d0f9e3e9972660ba383335eda471`); independent SPEC and STANDARDS
reviews both returned `REQUEST_CHANGES`, and those verdicts remain attached
only to that immutable candidate. Commit
`dd5f9fa36cc22ef3c052f5e31a38214e981c2476` is the next bounded correction
implementation. Its evidence successor was commit
`1d623ca11b96d57abcb1d9c8041a559cdd11c1d9` (tree
`ed8f836f7f2091c2345535135553e463537ba780`); independent SPEC and STANDARDS
reviews both returned `REQUEST_CHANGES`, and those verdicts remain attached
only to that immutable candidate. Commit
`adabe4d23b11ac673582d2a5243ce61e74f26e89` is the next bounded correction
implementation. Its evidence successor was commit
`8d4e14ae103013850a22ce0d0e43f7a50cf05b50` (tree
`e2b60caab5c54bf733ff5834389acf4af997aaec`); independent SPEC and STANDARDS
reviews both returned `REQUEST_CHANGES`, and those verdicts remain attached
only to that immutable candidate. Commit
`f89709ed62b64cd4eb896160479075362254c4b8` is the bounded correction
implementation and has no review verdict. The exact successor containing the
preceding evidence update was commit
`ac22d98f3456ffbadbe5084ef1b089bbc6801244` (tree
`3c94f6e93eacdc02b211e08ddc8397e4d9e0f690`). Its independent SPEC and
STANDARDS reviews both returned `REQUEST_CHANGES`; those verdicts remain
attached only to that immutable candidate. Commit
`bd64963a055e4bdef8faa3ad8e5dc082b99304f2` is the next bounded correction
implementation and has no review verdict. The exact evidence successor was
commit `8ac5e489c2d37ae2e7408f2379058cc00c488a22` (tree
`7c20e154fde59edf0809259c761ddbe0fd86d880`). Its independent SPEC and
STANDARDS reviews using `gpt-6-sol` at `high` both returned
`REQUEST_CHANGES`; those verdicts remain attached only to that exact
candidate. Commit `74b53bbf62be52c649524a8182ca7e82f12b375d` is the next
bounded correction implementation and has no review verdict. Its evidence
successor is commit `9c58514648fe2fef1389a3e8fb3a615540103d23` (tree
`f6e6c2b124c8fb043ddb78ebf1b8a9807474fc46`). Independent SPEC and STANDARDS
reviews using `gpt-6-sol` at `high` both returned `REQUEST_CHANGES`; those
verdicts remain attached only to that exact candidate. Commit
`dc8cb513d9a20a9c20ebc7a370477aac6d8a111e` is the next bounded correction
implementation and has no review verdict. Its evidence successor requires
fresh exact-state review.

## Historical predecessor review history (non-authoritative)

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

The eighteenth frozen candidate was commit
`49147d7d39000a5493a11c7a0bf210ff8fde031c` (tree
`085d31130488fe375bbf8b10ca179a9491525c86`). Fresh independent SPEC and
STANDARDS reviews both returned `REQUEST_CHANGES`. Both found that wildcard
documentation tag enumeration and the static remote namespace reader still
bypassed strict framing; STANDARDS also found the protected-ref reader's
duplicate parser. SPEC additionally found that `candidate_id`, although
required by both event schemas, was omitted from the shared durable-intent
binding. Those verdicts belong only to the eighteenth candidate.

The third review-loop correction implementation is commit
`0fddf3c1fbb13ef32dbc5badc60d224baebfd15e` (tree
`f5530d7698897a9edb036516aa609274932879c7`). It makes one strict
ASCII/LF, duplicate-free parser authoritative for exact refs, wildcard tag
enumeration, the static namespace, protected-ref reads and the documentation
workflow bootstrap check. It also adds `candidate_id` to the shared durable
intent binding and verifies the candidate-open receipt before tag creation.
Adversarial regressions cover malformed, duplicate, control-framed and
unterminated records at each reader plus forged candidate identifiers. Fresh
exact-successor reviews are required.

The nineteenth frozen candidate was commit
`2cb8ad4569acd0f811654398e00ccc8520a1df7b` (tree
`a866c803b55825554a8327d00ab081f88c432020`). Its fresh independent SPEC
review returned `PASS`. Its fresh independent STANDARDS review returned
`REQUEST_CHANGES`: the released-event required fields and public validator
still allowed `candidate_id` to be absent, and the transaction controller did
not independently compare the returned released event with every durable
intent binding field. Those verdicts belong only to the nineteenth candidate.

The fourth review-loop correction implementation is commit
`96f26cda2bb463028964621ecf9ce3864375e8cd` (tree
`0374fc8e77289f5f2246282a7b4a102778b52131`). It requires `candidate_id` in
the canonical released-event schema and public validator, removes the
missing-ID transition compatibility, and independently compares every shared
intent-binding field before publication. Regressions prove that a missing or
forged released candidate ID fails closed. Fresh exact-successor reviews are
required.

The twentieth frozen candidate was commit
`cea78c028e13f4e3f3c6354d3adfd273cc5a6315` (tree
`9d8b40377075f088857b8037976e3c81d020a6a2`). Fresh independent SPEC and
STANDARDS reviews both returned `PASS` with no actionable findings. They
confirmed the complete candidate identity chain, strict shared remote
authority parser, prior authority and exhaustion corrections, unchanged
scientific/package scope, and all 245 lifecycle plus 13 version/documentation
tests. This evidence-only convergence update follows those exact-state
reviews; its successor still requires exact-state confirmation before the
owner merge decision.

## Historical observed checks (non-authoritative)

| Check | Observed result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p 'test_version_lifecycle_*.py'` | Passed: 245 tests in 61.385 seconds on `96f26cd`. |
| `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts/test_check_version_bump.py scripts/test_documentation_routing.py` | Passed: 13 tests (3 version, 10 documentation) in 9.840 seconds. |
| `python3 scripts/check_version_bump.py --base 995163f0058488ea183ac645045ed8b1636bef4a --head HEAD` | Passed: lifecycle-only changes require no package-version update. |
| `DOCS_DEPLOY=false CYAX_DOCS_REF=refs/heads/vmm julia --compiled-modules=no --project=docs/ docs/make.jl` | Passed on `96f26cd` with offline installed dependencies and a writable temporary Julia depot; Documenter rendered all configured pages. |
| Python-unavailable core `using CYAxiverse` | Passed on `0fddf3c` with `PYTHON` and `PYTHONHOME` set to unavailable paths and the optional PyCall extension absent. |
| `python3 scripts/agent_verify.py package` | Failed: existing phase/volume detuning Hessian test at `test/runtests.jl:1267` expects `4π²`; observed `2π²`. |
| `julia --project=. bin/audit.jl` via `agent_verify.py run` | Failed: the same two JET reports for undefined `i` in `reduced_models.jl` and `poly102_inflation.jl`; file-monitor exhaustion warnings also appeared after the two reports. |
| `git diff --exit-code 995163f0058488ea183ac645045ed8b1636bef4a -- src test Project.toml bin/audit.jl` and the same comparison against `877dca1` | Passed; the failing Julia source, tests, package metadata, and audit script are unchanged by the full Gate A diff and this correction pass. |
| `python3 scripts/agent_verify.py snapshot` and `python3 scripts/agent_verify.py diff-check` | Passed; local snapshot captured and no whitespace errors. |
| Remote CI on superseded `52b2569` | Lifecycle and documentation passed; fast tests reproduced only the recorded Hessian baseline failure; full suite skipped by workflow. |
| Remote CI on reviewed `cea78c0` | Version lifecycle and documentation passed. Fast tests ran 10 pass / 1 fail and reproduced only the unchanged Hessian normalization failure at `test/runtests.jl:1267` (`2π²` observed, `4π²` expected). Full suite skipped by workflow. |
| `git diff --check`, Python compilation, and Ruby YAML parse of both workflows | Passed. |

The failed broad Julia gates remain failed at their recorded baseline. The
scientific normalization and benchmark code are outside this approved
lifecycle change, `Project.toml` remains `0.2.0`, and no production release
state changed. Remote CI and fresh exact-candidate SPEC/STANDARDS reviews are
observed on `cea78c0`; the evidence-only convergence successor requires final
exact-state confirmation. Live protection/event-authority setup, an explicit owner
merge decision, and fresh post-settings review remain later work; Gate B stays
separate.
