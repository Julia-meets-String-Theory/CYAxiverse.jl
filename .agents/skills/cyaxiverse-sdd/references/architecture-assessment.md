# Bounded architecture assessment

Trigger: A requested design assessment or concrete architectural friction in an authorized task. Inputs: named subsystem/interface, consumers, governing contracts, and relevant source/tests.

Assess only the named boundary and necessary dependencies. Identify what its consumers need to know and which implementation details they currently carry. Cite actual inspected evidence before describing a repository problem.

Compare keeping the existing design with one simpler interface that hides justified complexity. Where another materially different design resolves a real tradeoff, describe it in the same assessment. Do not launch design agents solely to generate alternatives.

Return: current interface and consumers; concrete friction; proposed interface and hidden responsibilities; keep-as-is comparison; consequences for tests, reproducibility, compatibility, and scientific meaning; and a recommendation with remaining uncertainty. A recommendation to keep the design is a valid outcome.

Preserve useful regression, invariant, internal, and interface tests. Replacing or deleting a particular test requires an authorized change with retained coverage demonstrated; interface-level tests do not automatically make existing tests obsolete. Existing domain meanings and dependency rules control terminology and design.

The assessment is read-only unless a separate task expressly authorizes implementation. It grants no refactor, schema, glossary, ADR, Issue, publication, or dispatch authority. Do not generate external HTML reports, fetch CDN scripts, run a whole-repository survey, or invoke another upstream workflow. Stop at the bounded recommendation.