# Engineering-method sources, source identities and complete MIT notice

> **Status:** Frozen detached pre-edit review candidate. This document is not yet installed in the target repository, independently approved, implementation-ready, merged, or a standalone dispatch grant. The governing task has one exact authorised S2 scope; private control-plane authority and actor configuration are separately bound.
>
> **Construction:** Scientific and engineering requirements below are verbatim public-safe extracts of the approved complete R2. The headings and ordering are reorganised into one umbrella feature and a subordinate instruction annex, without changing the scientific methods, options, thresholds, schema meanings, fixtures, required evidence or allowed file set. If any transformation introduces a conflict, the exact governing R2 controls and independent reviewers must block acceptance pending correction.

The recorded upstream method inventory is attribution and verification material only. The pinned public source commits/blobs and unchanged license text below are not installed at runtime. No automatic upstream fetch, second agent service or upstream global setup is authorised. Private owner and ledger references are intentionally excluded.

| Ref | Source and purpose |
|---|---|
| R1 | [CYAxiverse AGENTS.md](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/AGENTS.md), blob `35a9845224f5bdb3f2ee182a16083e940ef11020`: repository-wide code, verification and governance constraints. |
| R2 | [Surface allocation](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/.agents/POLICIES/SURFACE_ALLOCATION.md), blob `d19d236f78c2b5df8f9da5cbbe9e1d6a10bf46d3`: task/surface boundaries. |
| R3 | [SDD](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/.agents/skills/cyaxiverse-sdd/SKILL.md), blob `49269b1a5c2fcefa88661ee631b6f33e620a4ba5`; [Julia quality](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/.agents/skills/cyaxiverse-julia-quality/SKILL.md), `579f6c6f4633410577c9002a11c95852406cffdb`; [orchestration](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/.agents/skills/cyaxiverse-agent-orchestration/SKILL.md), `ea721b7565ef2257d71c2b4b703b1ed332716862`; [scientific reproduction](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/.agents/skills/cyaxiverse-scientific-reproduction/SKILL.md), `3f162a2c1d038f3cb11981b1b1a3b676fae5a44a`. |
| R4 | [Project.toml](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/Project.toml), blob `f4e67d50ffdfcecf2db0c95c925d2f7d0905d02b`: inspected Julia/IntervalArithmetic compatibility, existing dependencies and package version. |
| R5 | [Verification wrapper](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/scripts/agent_verify.py), blob `de470b41b3667f0acf049c638c9f769b1d8d74c4`; [existing audit](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/bin/audit.jl), `cb71703606c67c3b8ce8b60b061a165b5ccd4726`: actual commands, logs and environment behavior. |
| R6 | [Existing tests](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/test/runtests.jl), blob `f2fb9eebfecf86742f521fd4db5255b3e3bf7518`; [numerical candidate path](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/src/inflation_points.jl), `67654606f3f73c6595ee249395b1c8ec2880a345`: named compatibility fixture and composition. |
| R7 | [CI workflow](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/.github/workflows/CI.yml), blob `22093343b449e60c2df4725f52e1ab56977ef5bb`; [docs entry point](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/docs/make.jl), `3138d198e6011fee7f68f3fa12b92413d0b7201c`; [docs environment](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/blob/a5eacd8cd4a5905161cab239a461ba252c64e8e0/docs/Project.toml), `bcbda1749541e7a2ebab39d728ba830ab390efd5`: actual existing integration/preview routes. |
| R8 | [Matt Pocock skills source](https://github.com/mattpocock/skills/tree/49dd158d1076134a641b33efb035946536778336), commit `49dd158d1076134a641b33efb035946536778336`, tree `3ce485319972aae46dec0c2ac34cf4a7f28f7414`: eight selected methods, inspected references and MIT attribution; full source inventory below. |
| R9 | [IntervalArithmetic construction](https://juliaintervals.github.io/IntervalArithmetic.jl/stable/manual/construction/) and [guarantees](https://juliaintervals.github.io/IntervalArithmetic.jl/stable/manual/guarantee/); pinned 0.22.36 source identities below. Stable documentation is explanatory; the exact resolved source/host profile controls certification. |
| R10 | S. M. Rump, “Verification methods: Rigorous results using floating-point arithmetic,” Acta Numerica 19 (2010), [author-hosted paper](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf): validated numerical proof background. The fixed theorem and algorithm implemented here are stated in §6. |
| R11 | [IntervalLinearAlgebra eigenvalue API](https://juliaintervals.github.io/IntervalLinearAlgebra.jl/stable/api/eigenvalues/): documented eigenbox/verify_eigen limitations; no new dependency on this package. |
| R12 | [BootLoops harness](https://bootloops.ai/harness.html): methodological context for independent checks, planted truths, two precisions, interpretation and tool stewardship. The supplied CYAxiverse adaptation determines scope; the live website grants no project authority or automatic expansion. |

The newly inspected IA 0.22.36 registered package tree is `79342df41c3c24664e5bf29395cfdf2f2a599412`, recorded in the General registry's IntervalArithmetic Versions.toml, observed blob `7d5d37675e5c1a7da8640cbe03bd453fa1f7ffbb`. Pinned source reads include:

| IA 0.22.36 path | Observed Git blob |
|---|---|
| [Project.toml](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/v0.22.36/Project.toml) | `eebe3dc752616870973809c87957aee896d78839` |
| [src/IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/v0.22.36/src/IntervalArithmetic.jl) | `51ba4c7e15685aea52eb6e1b46f4bed61d27792a` |
| [src/intervals/rounding.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/v0.22.36/src/intervals/rounding.jl) | `8b576144056d62ca2cb06ef448c74e6a74ae2a34` |
| [src/intervals/intervals.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/v0.22.36/src/intervals/intervals.jl) | `374531089d95596e230f2b60cd2c0a665205f367` |
| [src/intervals/arithmetic/trigonometric.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/v0.22.36/src/intervals/arithmetic/trigonometric.jl) | `19add65e02d394110ab0b08b4460138305536d34` |
| [docs/src/manual/construction.md](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/v0.22.36/docs/src/manual/construction.md) | `dcc13c20dd60659b48d656540289e2108c2cc265` |

The observed v0.22.36 tag resolves to commit `ad85dab23d76c5399369fdd71ee10f2416cbb5ce`, whose tree matches the registered tree `79342df41c3c24664e5bf29395cfdf2f2a599412`. Tag, commit and tree are distinct Git object types; qualify the registered tree and resolved code actually executed, never infer equivalence from a version string alone. Runtime dispatch and all used primitive behavior remain P0/candidate execution obligations.

### 15.2 Public-safe source record and upstream inventory

Materialize §5.9's eight-to-five mapping, omissions, the inventory below and the complete notice in the allowed sources.md. The upstream inventory is preserved from the supplied Skills R2's recorded seventeen-file inspection; this drafting session freshly retrieved the license and checked the selected repository-policy/backend sources. Do not claim that every upstream file was freshly reread in this session. Upstream references are attribution/audit material, outside routine procedure context.

The public umbrella/specification files contain public scientific and engineering requirements and public-safe provenance. Keep private ledger/runtime/capability text, authority locators, actor identities and raw host paths in the existing private handoff/evidence destination. Do not copy this private preparation record wholesale into GitHub. Sanitization preserves every normative public requirement; it never turns unknown authority into an affirmative claim.

| Upstream path | Git blob SHA-1 | UTF-8 bytes | SHA-256 |
|---|---|---:|---|
| [LICENSE](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/LICENSE) | f1dd2c09108dde1a5f56097cee8461b3ea834499 | 1068 | 0e7ac423bf2c6e223b7c5b156f8cf72da49d748e56a1641402c31f22ad07dbb5 |
| [skills/engineering/code-review/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/code-review/SKILL.md) | 373a4f26e6cfa3617778397945bb069d9cb184bc | 6670 | 09eb147f793e4647949edd32dbad278f0554dfb56e0d46b699cd4c2c55bf0357 |
| [skills/engineering/codebase-design/DEEPENING.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/codebase-design/DEEPENING.md) | cd94075cfd754d147555c5d747a16431ed4c7dd8 | 2553 | f3dd099ce99289bd213914d8ee3e2429b78309c3957ca4583f7659551b1d53c1 |
| [skills/engineering/codebase-design/DESIGN-IT-TWICE.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/codebase-design/DESIGN-IT-TWICE.md) | d0990138c35c800bd821289a73115e52680c1620 | 2665 | 52b52e75655a27aea59c6966b410bac82448968ad46ef1ebb9f07c274e3ae010 |
| [skills/engineering/codebase-design/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/codebase-design/SKILL.md) | 3f63c8146dd2604b419c929e9876b90c30d410e9 | 6446 | 2c20617f87ec8af6a434859f381b2f061a69b530444e74eb39e78bb016a6d1e2 |
| [skills/engineering/diagnosing-bugs/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/diagnosing-bugs/SKILL.md) | d039541ddd1c46e6ddbd662fe99d49b686fc5e12 | 8664 | 849198df88cdb65fd0d97236604c1fca7ff2f22d2f705f9724a800d0266b118d |
| [skills/engineering/domain-modeling/ADR-FORMAT.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/domain-modeling/ADR-FORMAT.md) | d7e61f30a9fd8ca70d9ee68f019c2a9ebf2d13f7 | 2733 | 944c92aa790e8fbdc9199640b170979abb8a34ba8d0fe18c2a01a63bce140ca0 |
| [skills/engineering/domain-modeling/GLOSSARY-FORMAT.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/domain-modeling/GLOSSARY-FORMAT.md) | 9c0b4c7a2e8507c23cbb5e0baa66d81699321cc6 | 2300 | 21dcc40cdec8e151a829b98ab1517c1867bc444e3665134a98c6983411996801 |
| [skills/engineering/domain-modeling/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/domain-modeling/SKILL.md) | a4438c6879c517d1a08e2b5fbca049b632688dfe | 3346 | 7b925d7b1e341a2eeae33ad68a8a8c0ab889a38ddd22a7598e4c240e5a7556a3 |
| [skills/engineering/improve-codebase-architecture/HTML-REPORT.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/improve-codebase-architecture/HTML-REPORT.md) | e39e8255b8797cf8257d014fb6658fb533f3a399 | 6641 | 581e8bb5a521e46bbda8ca7e19b15948bed882187108092ebb90c62513b77528 |
| [skills/engineering/improve-codebase-architecture/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/improve-codebase-architecture/SKILL.md) | 44cc7d3136c44534d68ac523fa6cd37e759d2846 | 6000 | 552240a5ab5cec6b67c15dd1ad6e9f6962b1bca6ce870059a8c2713760c20392 |
| [skills/engineering/retro/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/retro/SKILL.md) | 12149acf2dd23b4514dec57b04a6293eaa1352fa | 4369 | 33fbc28568e3146f20fd908b12998bc0ab30aa4f4c87288199a40793efa94696 |
| [skills/engineering/tdd/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/tdd/SKILL.md) | 01eadaa34e7c9a63a67d6dc3cce1cc81b0e49985 | 3629 | 5505d25bbaa8fc14a79e4d20163936958992a2995b7f1dd1d6197edc9ed722e0 |
| [skills/engineering/tdd/mocking.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/tdd/mocking.md) | 71cbfee674d93244ce81d1830b930ca9a69200bd | 1481 | 3ceb807fdf4a47d6a93d4d9a891e5ba6d362a6247bd08adc451feebfc17361ef |
| [skills/engineering/tdd/tests.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/engineering/tdd/tests.md) | 7ab86479f925a1f9e8ba680af33cb3b12e015381 | 2214 | 859f9e592c188fda4fc7277dd180e4ce9c7a2e13f6efe1f6f29eccc9d28c106a |
| [skills/productivity/writing-for-agents/SKILL-MECHANICS.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/productivity/writing-for-agents/SKILL-MECHANICS.md) | 9cdbdb22aadc438755392216e75ad9a14cc9832e | 2629 | c768e6307c7c10728c401c213f2c4ba71c542127eeb7ad2956aabd15a0fa0059 |
| [skills/productivity/writing-for-agents/SKILL.md](https://github.com/mattpocock/skills/blob/49dd158d1076134a641b33efb035946536778336/skills/productivity/writing-for-agents/SKILL.md) | a37608daf6e835e767deecfb498facecaaba82ba | 10886 | 551adca942227b44192edba88acd4e8db911f0121ce58ad16944ccf6a896a74a |

### 15.3 Complete upstream MIT notice

Preserve the following notice verbatim in sources.md along with the pinned source identity. The exact retrieved LICENSE has 1,068 UTF-8 bytes and SHA-256 `0e7ac423bf2c6e223b7c5b156f8cf72da49d748e56a1641402c31f22ad07dbb5`.

```text
MIT License

Copyright (c) 2026 Matt Pocock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Historical provenance notice (post-dispatch clarification, not a change to R2):** The assignment-disposition paragraph immediately below is retained verbatim from the pre-dispatch R2 drafting text. The owner has subsequently manually dispatched R2; that paragraph is historical source provenance only, not an instruction to repeat dispatch or a statement of present reviewer, readiness, implementation, or integration status. Current status and authority come solely from the exact separately governed task records.

**Assignment disposition:** This complete R2 is prepared for the owner's manual Manager bootstrap dispatch with its external exact digest. First persist it and its continuity anchor; second initialise fresh task-specific ChatGPT.com desks and required actors; then establish actual readiness and continue into implementation. Runtime qualification, independent reviews, scientific success and integration remain observable actions, not drafting claims. Work toward completion within 24 hours and continue legitimate contract-aligned recovery beyond that goal when necessary. Raise the owner only under §12.7's last-resort boundary.
