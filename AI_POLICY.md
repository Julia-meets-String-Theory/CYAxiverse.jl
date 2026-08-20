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

## Context-capacity checkpoint and compaction

This procedure is mandatory for every agent and every continuation of an
agent working in this repository. When the active context reaches 85% of its
capacity, stop substantive work and send the following prompt to the
agent's continuation or compaction mechanism:

```markdown
Create a detailed summary machine-readable json doc for continuing this coding session. Place it in ../checkpoints/$DATE using the filename convention $TIME-$AGENT-checkpoint.md.  Include:

1. **Completed work**: What tasks were finished
2. **Current state**: Files modified, their current status
3. **In progress**: What is being worked on now
4. **Next steps**: Clear actions to take
5. **Constraints**: User preferences, project requirements, key decisions made
6. **Critical context**: Any information essential for continuing

Be concise but preserve enough detail that work can continue seamlessly.
```

Execute that prompt immediately. Use `YYYY-MM-DD` for `$DATE`, a
collision-resistant local time for `$TIME`, and the stable agent name for
`$AGENT`. The `.md` file must contain one valid JSON object, not a prose
summary or a fenced code block, with machine-readable keys corresponding to
the six requested sections. Create the parent directory outside this Git
checkout as required by the documentation-placement rules.

After the checkpoint is written, use that JSON file as the authoritative
context for a custom compaction. Start a fresh continuation or session when
the runtime supports it, and seed it with the checkpoint path. The first
action in the new context must be to read and parse that JSON file. Preserve
the system, developer, repository, and active user instructions, but do not
carry the full prior transcript or unverified assumptions into the new task
state.

If the runtime supports native compaction but cannot accept a file as its
seed, invoke native compaction immediately and then read the newest JSON
checkpoint before resuming. If it exposes neither a fresh continuation nor
native compaction, end the current turn and begin the next turn with the
checkpoint path as the first instruction; this is the file-based fallback,
not a claim that the active hidden context was replaced. Do not continue
substantive work between writing the checkpoint and starting the fresh
continuation or compaction. If the runtime does not expose an automatic
context percentage, trigger this procedure at the earliest reliable estimate
of 85% rather than waiting for context exhaustion.
