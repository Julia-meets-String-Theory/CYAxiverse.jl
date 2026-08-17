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
