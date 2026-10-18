# CYAX-0170 execution and candidate provenance

Status: provenance correction after the independent review of candidate
`5c90bfe3ce91dcdca442af9df72cf1f32390aea6`.

Scientific source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.

This record distinguishes the Git revision checked out when a measurement ran,
the uncommitted evidence bytes used by that run, and the later immutable Git
candidate that integrated those bytes.  A generated result cannot contain the
hash of the future commit that first adds that result.  Content hashes close
that provenance edge without relabeling a measurement as if it ran later.

## Execution-to-artifact map

| Evidence | Git execution base | Executed harness SHA-256 | Retained result SHA-256 | Candidate relationship |
|---|---|---|---|---|
| Environment and B1 load | `99d702c8b28e674d34e29566f8ecc483d76c7f7f` | `environment_and_load.jl`: `56c9f86f0b19f3db9c1712ec0f65febe0656c64065914439c1f2385549ffcd76` | `environment_and_load.md`: `79565339b17d41ebf59b001c5789a1612999b60045bbe9fe7c453a5029f5bd4a` | Integrated unchanged in `5c90bfe3ce91dcdca442af9df72cf1f32390aea6`; production source equals the scientific source. |
| F1-F13 Julia replay | Working tree based on `8dab6e6185867c62d962b44ca4e664f749df55db` | `fixture_julia_replay.jl`: `fcb17299355ea14542996ee682415f26982d069563bde8b22516219422a4d0d8` | `numerical_semantics_and_fixtures.md`: `e8ed83809a5c52156878d3925725bcfddceb61603694b09461b04e56d63bd326` | Harness and report first integrated in `5c90bfe3ce91dcdca442af9df72cf1f32390aea6`.  The Manager and Independent Numerical Reviewer v2 reran the same harness at exact commit `5c90bfe3ce91dcdca442af9df72cf1f32390aea6`; both runs exited 0 with `source_tree_equal=true` and `package_load=ok`. |
| B2-B7 benchmark | Working tree based on `8dab6e6185867c62d962b44ca4e664f749df55db` | `benchmark_baseline.jl`: `906c471fa79e2633014e3f88c56eb586f1307da6ae812a04d0e2cafed9feee98` | report `cee00e665f1cd9b6eb685b6146c5f7e3875d7db8f7d64eef409bd61e272c9f44`; TSV `274ba44db46e394f53cdf47c07155048c9772891d348ecded7b0ff0ce3d7692a`; metadata `81541ae468698e900f25b422ae527504054b23767fa47cfbdf11a271fe53a22a` | Harness and retained outputs first integrated in `5c90bfe3ce91dcdca442af9df72cf1f32390aea6`; production source equals the scientific source. |

Both normalized lockfiles and every artifact above are retained by repository-
relative path and covered by `SHA256SUMS`.  The benchmark metadata's
`execution_revision` names the execution base, not the later candidate commit.

## Candidate and review identity

- Candidate `5c90bfe3ce91dcdca442af9df72cf1f32390aea6` contains the exact artifact
  bytes named above and no changes to `Project.toml`, `src/`, `test/`, or `ext/`
  relative to the scientific source.
- Independent Numerical Reviewer v2 reviewed that exact candidate and returned
  `REVISE / BLOCKED`.  Its durable record is `independent_review.md`.
- This provenance record and the v2 verdict are later bookkeeping additions.
  Their integration creates a new candidate identity.  A fresh G2 reviewer must
  name and review that exact new commit; the later verdict file is necessarily a
  post-review mechanical record outside the reviewed commit.

No numerical result, fixture, tolerance, source, test, dependency, or schema is
changed by this provenance correction.
