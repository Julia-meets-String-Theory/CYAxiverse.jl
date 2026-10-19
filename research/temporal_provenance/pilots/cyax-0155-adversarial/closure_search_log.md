# CYAX-0155 closure and saved-view evidence search

Observation: 2026-09-13T04:35:55Z.

Searched before snapshot freeze:

- Issue #155 body, all comments, current fields, Project item/status, and timeline events through the observation time;
- PR #156 body, comments, reviews, commits, merge fields, and merge commit;
- GitHub repository issue/PR search for `"Research & Chats"` and `"155"`;
- all local Git refs/commit messages for `#155`, `Research & Chats`, closure, completion, and supersession terms; and
- repository files for `Research & Chats`, #155 closure/completion, and saved-view evidence.

`sources/source_search_audit.json` freezes the exact queries, complete-page and
result counts, all ten normalized timeline events, and the matching file set.
`sources/public_ref_inventory.txt` freezes the 66 public remote/tag refs searched
and their object IDs; the normalized inventory SHA-256 is
`103b7931b7abf066d64f55c2693b33f3a67c5851a4b07dc2fc4c6528861a55ab`.

Findings:

- The latest #155 comment is the 2026-09-11T00:01:47Z checkpoint. It says to keep #155 open until the saved view/filter is configured and says then-current tooling could neither configure nor directly verify it.
- The #155 timeline contains a closure event by `vmmhep` at 2026-09-11T01:09:18Z with no associated commit and no GitHub App. It contains no closure rationale.
- The PR #156 body says view creation/configuration is separate from the repository diff and does not claim the Project mutation occurred.
- The merge commit contains no auto-close keyword for #155.
- Search found no later public artifact proving the view was configured and no explicit durable statement superseding the keep-open condition.
- The later PR #164 research README calls the Projects view pending/unverified.
  It does not explain closure or prove configuration and is excluded from subject
  contexts because it is experiment-design meta-commentary.

Mechanically established absence is scoped to this finite search and observation time. It does not prove that the Project view does not exist; it proves only that the frozen captured evidence does not demonstrate it.
