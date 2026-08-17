# Generative AI Policy for CYAxiverse.jl

This policy governs the use of generative AI tools — including but not
limited to GitHub Copilot, Claude Code, Codex/ChatGPT, Gemini, and similar
coding assistants — when contributing to CYAxiverse.jl. It applies equally
to every AI tool and to every contributor who uses one; no tool is
privileged over another. It is adapted from the
[CPython Generative AI policy for contributors](https://devguide.python.org/getting-started/generative-ai/).

## Core responsibility

The person submitting an issue or pull request is responsible for its
content, regardless of whether AI tools were used to help create it.
Generative AI tools can produce output quickly, but discretion, good
judgment, and critical thinking remain the foundation of every good
contribution to this package.

## Considerations for success

- Review AI-generated work in detail before proposing it as a pull request
  or filing it as an issue. Confirm it makes sense for CYAxiverse.jl's
  domain — Calabi-Yau geometry and axion physics — and for the numerical
  and Julia conventions set out in `AGENTS.md`, `.copilot/AGENTS.md`, and
  `.github/copilot-instructions.md`.
- Be able to explain proposed changes in your own words, including the
  physics or numerics behind them.
- Whether or not AI tools were used, every contribution should:
  - Consider whether the change is necessary.
  - Make minimal, focused changes.
  - Follow the existing Julia coding style and package conventions.
  - Include tests that exercise the change, using `Test.jl` in
    `test/runtests.jl`.
  - Preserve backward compatibility and follow the versioning policy in
    `AGENTS.md`.
- Pay close attention to AI-generated test recommendations. Guide the tool
  with CYAxiverse.jl's own testing conventions rather than accepting
  generic ones — type stability, precision handling, and the physical
  invariants in `.copilot/AGENTS.md` need domain-aware tests, not
  boilerplate.
- Always review the full output — including proposed titles and
  descriptions — before opening a pull request or issue.

## Disclosure and labelling

Disclosure of AI tool use is encouraged, though not required, in PR
descriptions and commit messages. State which tool was used and what it
changed.

Any agent authoring a commit on a contributor's behalf must label its own
authorship in that commit, for example with a `Co-Authored-By:` trailer
naming the tool. This applies uniformly regardless of which agent is used
— Claude Code, Copilot, Codex, or otherwise — so keep such trailers intact
rather than stripping them, and add one if the tool you are using
supports it but did not add it automatically.

## Acceptable uses

- Assistance writing comments, docstrings, or issue text, especially for
  non-native English speakers.
- Gaining understanding of existing code, including unfamiliar Julia,
  PyCall/CYTools, or HDF5 data-layout details.
- Supplementing contributor knowledge for code, tests, and documentation.

## Unacceptable uses

- Altering or bypassing existing tests, or removing functionality, in
  order to make a failing test pass. That is not a real fix.
- Repeatedly opening unproductive issues or pull requests. Contributors
  who do so may be blocked, because it is disruptive and disrespectful of
  maintainers' time.

Maintainers may close issues and PRs that are not useful or productive,
without explanation, whether or not AI tools were involved.

## Scope

This policy applies to every AI coding assistant used against this
repository, not to any single vendor. It is referenced from `AGENTS.md`,
`.copilot/AGENTS.md`, `.github/copilot-instructions.md`, and `CLAUDE.md` so
that every agent operating here — and every contributor directing one —
sees it.
