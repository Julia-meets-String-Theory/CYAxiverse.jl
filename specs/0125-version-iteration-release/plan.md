# CYAX-0125 Gate A prospective-amendment plan

Status: derived from the prospective `spec.md` amendment in this worktree.
Approval is effective only through the external approval record that binds the
exact frozen five-file revision and fresh independent SPEC and STANDARDS PASS
records under rubric SHA-256
`418f2d5a276cbdb74b8ad331b532d59219ef33b4e9fabc2b3d4a21c55bc06c72` before
implementation may claim conformance.

Dispatch context is the exact handoff SHA
`19fa93e2b4b1439b1cc4217cf7ff9e42846980d46df66793156f72789894e40a`, target
PR head `20b3935ace0e01fcee2808681c56595e3afa7667`, and owner approval receipt
SHA `140d1fe1c7c09db3d40a04c66c1b258f18e8c42025a3453c2472ed682124637c`.

The predecessor approved revision is
`cf0f9c39256b7a58e179af92a46fbf1cb651ff76`, with owner provenance in Issue
#125 comments `5746793852` and `5752268640`. Its append-only
`release-events` implementation, review results and evidence remain historical
and are not proof of this reduced candidate. This plan does not back-project
the amendment's requirements onto predecessor work.

## Boundaries and baseline

Keep `Project.toml = 0.2.0`; preserve `main` at its independent released
version, legacy `v-0.1`, Issue #172 history, scientific behavior, public Julia
APIs and persisted scientific schemas. Do not perform Gate B, historical
designation, production DEV adoption, closure, publication, public-tag
creation or `vmm -> main` reconciliation. The target iteration is
`version-lifecycle-retrofit-2026-09`, with package-infrastructure **patch**
impact and no package-version adoption.

The replacement authority is `iterations.toml` plus immutable
`iterations/X.Y.Z` anchors, protected canonical public tags, and protected
create-once `refs/heads/lifecycle/v1/*` refs. Distinct claim, reservation, candidate,
intent, release and publication object types each have their own protected
ref namespace and small immutable canonical evidence manifest; progression is
an immutable predecessor-ref graph. There is no mutable
`release-events` branch or stream. Create-once/CAS, owner authorization,
complete-ref snapshots, and fail-closed recovery preserve permanent no-reuse,
strict canonical versions, exact-tree certification, and principal/maintenance
line semantics.

Gate A requires automation for the first principal lifecycle path. Maintenance
bootstrap/release and rare recovery automation are deferred to a later approved
S2 gate; deferral does not relax any invariant or permit an uncertain mutation.
Python is bounded repository/CI control-plane tooling only and is not required
for `using CYAxiverse`.

## Implementation slices

### A1 — Reconcile governing policy and release-neutral source

Reconcile `AGENTS.md` and this integration-release skill with the reduced
create-once-ref/manifest authority before spec approval. Preserve principal and
maintenance roles, owner authorization, strict version grammar, permanent
no-reuse, exact-tree certification, protected tags, release-neutral docs and
Gate A/Gate B boundaries. Keep Python lifecycle helpers bounded to repository
and CI control-plane work; retain Python-free core import. Any API, scientific,
persisted-schema or version-boundary change stops for a separate decision.

### A2 — Static authority and immutable anchors

Retain the exact `refs/heads/vmm:iterations.toml` selector and immutable
`iterations/X.Y.Z` anchor validation. Build canonical static snapshots with
source/ref/tree, public-tag bindings, and annotated-anchor bindings that include
the direct tag-object SHA, object type, peeled commit/tree and validated closure
UTC payload. Add lightweight-tag and same-target/different-object negatives.
Preserve Gate B retrospective truth and supersession chronology;
do not populate real historical designations.

### A3 — Create-once lifecycle authority

Replace mutable event-ledger design with protected create-once lifecycle refs
and one-manifest commits. Define exact versioned claim/reservation, candidate,
intent, release and publication namespaces; manifest schema,
content-derived IDs with no global sequence allocator, predecessor binding,
exact Git identities and content digests; and a complete
replayable `lifecycle_ref_snapshot`. A publication manifest requires exactly
one released-manifest predecessor, canonical tag commit/tree, GitHub Release
identity and publication-evidence digest; its `pub-` ID and ref are derived
from the released-manifest ID/tag pair. Require
create-if-absent/CAS, no deletion/repointing/force update, global claim
occupation and no reuse. A proven pre-entry reservation abort permanently
consumes the reservation identity but can release a not-yet-claimed final
version; uncertain outcomes remain unavailable. An uncertain remote result
freezes the affected line/version and returns a blocked result; it never
appends or retries a different payload.

Define a canonical immutable owner-authorization record and verifier. Bind its
ID, source reference and digest in each manifest. The verifier resolves the
configured owner authority and compares repository, stable owner account,
validity interval, transaction, action, owner line, final version and exact
target refs under the held exclusion boundary before each mutation. Missing,
changed, stale or cross-operation grants block before writes.

### A4 — First-principal lifecycle automation

Implement and test the first principal path: deterministic principal DEV
reservation, closure/anchor correspondence, candidate durability, exact-tree
certification, main freeze/ancestry disposition, immutable intent, protected
canonical tag boundary, release manifest and exactly one deterministic
publication manifest/ref with GitHub Release identity/evidence. Preserve
owner authorization and forward-only post-tag reconciliation. Maintenance-line
bootstrap/release and rare recovery are contract-only deferred S2 work in this
Gate A candidate; no production maintenance automation is claimed here.
Implement only the validation-only `maintenance-bootstrap-validation-v1`
schema and synthetic positive, non-entry, mismatch and uncertain/failure
fixtures required by R-030/R-031. Do not expose it as a lifecycle manifest
writer or create its reserved production refs.

Documentation deployment must independently resolve and validate the selected
release/publication manifest's tag, commit/tree, package version and digest.

### A5 — Deterministic verification and convergence

Test canonical final/DEV parsing, strict ref/tag grammar, static and lifecycle
snapshots, claim/no-reuse races, create-once idempotence/conflict/uncertainty,
principal closure, certification transfer, tag/manifest/tree consistency,
publication positive binding and duplicate/conflict rejection, and
the fixed expected publication-hash/ref fixture in the specification, and
release-neutral documentation routing. Run focused checks before package,
audit, docs, Python-free import and CI checks. Record failures and unavailable
gates accurately; prior ledger test results remain historical.

### A6 — Fresh review, owner decision and live controls

Obtain fresh independent SPEC and STANDARDS review of this exact candidate and
implementation. Review is not merge authority. After explicit owner merge
authorization, establish and verify actual create-once ref protections,
iteration-anchor/tag protections, freeze controls and principal automation;
record identities, digests and observed enforcement. Obtain fresh post-settings
closure review before claiming Gate A completion. Keep maintenance/recovery S2
work and Gate B separate.

## Contract-to-evidence map

| Requirement | Candidate owner / proof |
| --- | --- |
| R-001 | A1 policy reconciliation; prior Issue decisions retained as historical provenance; fresh amendment review before authority. |
| R-002–R-004 | A1 line roles, target identity and stable-iteration checks. |
| R-005 | A2 canonical SemVer parser and alias/prerelease rejection tests. |
| R-006–R-011 | A3 principal/maintenance allocation rules, global claim refs, terminal consumption and no-reuse tests. |
| R-012, R-015, R-019–R-021 | A2 `iterations.toml`, exact annotated-anchor object/payload bindings and prospective/retrospective truth fixtures. |
| R-013–R-018 | A3 protected create-once refs, immutable manifests, complete lifecycle snapshot and serialized CAS boundary. |
| R-022 | A1/A5 verified-ref/manifest documentation routing with no tracked release edit. |
| R-023–R-024, R-044 | A4 pinned exact-tree certification and explicit tree/commit binding transfer tests. |
| R-025–R-027 | A4 single candidate/intent/release-manifest chain and principal/maintenance line checks; maintenance production path deferred. |
| R-028–R-031 | A4 principal closure/reopen path; R-030/R-031 validation-only maintenance-bootstrap schema/fixtures now, with production automation reserved for later S2. |
| R-032–R-035 | A4 durable candidate refs, withdrawal retention, irreversible protected tags and forward-only reconciliation. |
| R-036–R-039 | A1/A6 grandfathering, package boundary, Gate A/B separation and fresh final-state review. |
| R-040–R-043 | A3/A4 manifest schema/identity, closure UTC binding, explicit publication fields/predecessor/key, release evidence and bidirectional tag/manifest/publication validation with duplicate/conflict negatives. |
| R-045 | A1/A5 bounded Python control-plane checks and Python-free `using CYAxiverse` verification. |
| R-046 | A3/A4 immutable owner-authorization verification with positive and pre-mutation negative tests. |

## Stop and escalation points

Stop on an unresolved static selector, incomplete lifecycle-ref snapshot,
missing create-once/CAS protection, cross-repository authority, conflicting
manifest identity, uncertain remote create, unavailable exact principal
sentinel, unproven owner authorization, changed certified tree, unproven main
freeze, missing protected tag controls, or an operation requiring Gate B or
the deferred maintenance/rare-recovery S2 automation. Preserve the frozen
line/version and return the specified blocked reason. Do not revive the old
mutable ledger as a fallback.

The final pre-merge review and owner merge decision are separate gates. A
review verdict is evidence, not merge authorization, release authority or
Issue closure. No task in the companion checklist may claim completion from
the predecessor implementation's ledger evidence.
