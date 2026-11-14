#!/usr/bin/env bash
# Shoroq reference case: full reproduction of the frozen dataset from the committed polytope manifest (stage 1, the KS
# rescan, is optional: references/shoroq/sample/run_sample.sh). Producers (stages 2-3) then the engine (stages 4-5) with the
# Shoroq run file and expectations. Expensive: about 1.5 h wall with WORKERS = 4 (measured 2026-10-29).
# Comparison with the frozen data: references/shoroq/tests/regression_full.sh.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$D/../.." && pwd)"
export POLYTOPE_MANIFEST="${POLYTOPE_MANIFEST:-$D/data/polytope_manifest_h11_004_050.json}"
export RELIC_RUN="${RELIC_RUN:-$D/configs/run.json}" RELIC_EXPECTATIONS="${RELIC_EXPECTATIONS-$D/configs/expectations.json}"
export EXPECTED_TACHYONIC="${EXPECTED_TACHYONIC-46}"
"$ROOT/producers/run_geometry.sh"
"$ROOT/producers/run_vacuum_repair.sh"
"$ROOT/engine/run_relic.sh"
"$ROOT/engine/run_theta.sh"
