# CYAxiverse.jl Codex Agent Instructions

Before planning or carrying out any work in this repository, read and follow
both of these repository instruction files:

- `.copilot/AGENTS.md`
- `.github/copilot-instructions.md`

Treat those files as normative supplemental instructions for all Codex agents
and delegated subagents working in this repository. In particular, Julia
commands for CYAxiverse.jl package development must run directly in the
regular local host environment, not in a sandbox, container, Docker image, or
other isolated environment. If the execution tool defaults to a sandbox,
request approved local/unsandboxed execution before running Julia.

## Generative AI contribution policy

Every AI coding tool used against this repository — Codex, Claude Code,
GitHub Copilot, or any other agent — and every contributor directing one
must follow `AI_POLICY.md`. It is normative for all agents equally; no
tool is exempt. In particular, label your own commits with a
`Co-Authored-By:` trailer (or your tool's equivalent) rather than
submitting AI-assisted work unlabelled, and do not alter or bypass
existing tests to force a pass.

## Persistent coding workflow

- Apply the Code Foundations workflow from `https://github.com/ryanthedev/code-foundations` to every coding task in this repository.
- For non-trivial work, clarify the requirement, identify the affected scope, write an implementation plan, and map each requirement to verification before editing.
- Keep implementation changes scoped. Preserve unrelated working-tree changes, avoid destructive Git operations, and stop before expanding into an unrequested feature.
- Verify each coherent change with focused checks, then run the applicable package tests and review the final diff for correctness, scope, and whitespace errors.
- Report the exact verification commands and their observed outcomes, including warnings and any checks that could not run.

## Always-on project skill

- At the start of every new Codex run in this project, read and apply `.codex/skills/i-have-adhd/SKILL.md` before responding or acting.
- Keep the `i-have-adhd` output mode active for every response in this project. It may be disabled only when the user says `stop adhd mode` or `normal mode`; a new chat starts with it enabled again.

## Versioning and release policy

- Treat the `version` field in `Project.toml` as the package version source of truth. Use three-part SemVer and release tags of the form `vMAJOR.MINOR.PATCH`; do not create tags such as `v-0.1`.
- Before `1.0.0`, use Julia's package-versioning convention: increment the patch number for compatible bug fixes and the minor number for breaking changes. After `1.0.0`, breaking changes require a major-version increment and new public API requires a minor-version increment.
- Before changing package implementation code, classify the version impact. A major development includes a public API change, a reader/writer or persisted-data contract change, a required Julia or dependency support change, or a scientific behavior change that can alter published results.
- Feature branches and worktrees must state their version impact but should not independently bump `Project.toml`; parallel branches must not all claim the same release version. Apply the actual bump on the reviewed integration or release boundary, normally the `vmm` to `main` release PR. At that boundary, use at least a patch bump for a compatible behavior change and a minor bump for a breaking change while the package is below `1.0.0`.
- Keep scientific artifact and database schema versions separate from the package version. Record those schema versions, the package version, the source commit, Julia version, dependency manifest, and relevant external-tool versions in generated artifacts when possible.
- In the final handoff, state the version impact and whether the bump is included or deferred to the release boundary. The release pull-request version check enforces the final bump; TagBot remains responsible for creating release tags after registration.
