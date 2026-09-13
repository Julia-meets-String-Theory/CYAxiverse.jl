# Difficulty and leakage check

The #157/#158 ablation had internally aligned current-state evidence and scored
at ceiling in both conditions. This #155/#156 pilot is meaningfully harder:

- a later current GitHub state differs from an earlier explicit keep-open workflow instruction;
- the closure event has an actor and time but no rationale;
- repository implementation completion and Projects-v2 view configuration are distinct;
- a later Project status event does not expose evidence about saved-view configuration;
- governing-Issue, approved-spec, supporting-PR, historical-comment, and current-observation authority must remain distinct; and
- a complete response must decline to fill a real evidence gap while still stating the definite current GitHub state.

The contexts include the source statements and the mechanically bounded absence
record. They do not state whether closure satisfies or supersedes the earlier
condition, do not tell the subject to reopen or retain the Issue, and do not say
that closure proves or disproves view configuration. The subject must synthesize
those conclusions. This check must be confirmed by the independent reviewer
before context freeze.
