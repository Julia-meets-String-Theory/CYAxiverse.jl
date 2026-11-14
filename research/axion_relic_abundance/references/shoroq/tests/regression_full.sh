#!/usr/bin/env bash
# Full reproduction of stages 2-5 (all 4700 polytopes of the committed manifest) compared with the frozen dataset.
# Expensive: about 2-3 h with WORKERS = 4. Nothing outside WORK is written.
#  1. before: the frozen reference is verified against all its checksum lists; a time marker is written; the state of
#     the CYAxiverse checkout is recorded
#  2. stages 2 and 3 (real drivers); every dataset compared in both directions (compare_geometry_subset.py,
#     compare_vacuum_repair.py, check_full_population.py --full)
#  3. stages 4 and 5 on the regenerated relic input; compared with the frozen relic and theta results
#  4. after: checksums verified again; no file under WORKSPACE_ROOT may be newer than the marker; checkout unchanged;
#     sha256 of every output file under WORK
# A failed stage or comparison stops the remaining stages (marked SKIPPED); step 4 always runs.
# Usage: references/shoroq/tests/regression_full.sh FROZEN_ROOT WORK
#   FROZEN_ROOT    the frozen reference (layout in references/shoroq/tests/verify_frozen_reference.py)
#   WORK           new, empty output directory; summary in WORK/regression_full.json
#   WORKSPACE_ROOT (environment, default FROZEN_ROOT) directory that must stay unmodified
source "$(dirname "$0")/../../../engine/common.sh"
require_producer_toolchain
[[ $# -eq 2 ]] || die "usage: $0 FROZEN_ROOT WORK"
[[ -d "$1" ]] || die "FROZEN_ROOT $1 is not a directory (see references/shoroq/tests/verify_frozen_reference.py for the expected layout)"
FR="$(cd "$1" && pwd)"; WORK="$2"; export WORK
for d in geometry materialized spectra validation vacuum_repair relic_input relic final_analysis; do
  [[ -e "$WORK/$d" ]] && die "$WORK/$d exists: use a new, empty WORK directory"
done
mkdir -p "$WORK/logs" "$WORK/comparison"
WS="$(cd "${WORKSPACE_ROOT:-$FR}" && pwd)"
case "$(cd "$WORK" && pwd)/" in "$WS"/*) die "WORK must not lie inside the protected workspace $WS";; esac
L="$WORK/logs"; C="$WORK/comparison"; RES="$WORK/regression_full.json"
exec 3>&1
NAMES=(); STATUS=(); STOP=0
record() { NAMES+=("$1"); STATUS+=("$2"); echo "[$(date -u +%FT%TZ)] $2  $1" >&3; echo "[$(date -u +%FT%TZ)] $2  $1" >> "$L/regression_full.log"; }
run() { local name="$1"; shift; if "$@"; then record "$name" PASS; else record "$name" FAIL; fi; }
# a stage step: skipped after an earlier failure, and a failure stops the following stage steps
step() { local name="$1"; shift; if (( STOP )); then record "$name" SKIPPED; elif "$@"; then record "$name" PASS; else record "$name" FAIL; STOP=1; fi; }

# 1. before
date -u +%FT%TZ > "$WORK/start_marker"
git -C "$CYAX_REPO" rev-parse HEAD > "$C/cyax_head_before.txt"; git -C "$CYAX_REPO" status --porcelain --ignored > "$C/cyax_status_before.txt"
step "frozen reference intact before the run (all checksum lists)" \
    "$PYTHON" "$REPO/references/shoroq/tests/verify_frozen_reference.py" "$FR" "$C/frozen_reference_before.json" all > "$L/frozen_reference_before.log"

# 2. stages 2-3
FU="$FR/thesis_g2_adaptive/unified_h11_004_050"
export POLYTOPE_MANIFEST="$REPO/references/shoroq/data/polytope_manifest_h11_004_050.json"
step "stage 2 driver (G2 on 4700 polytopes, materialize, audit, Julia read-back, spectra, gates, integrity)" \
    "$REPO/producers/run_geometry.sh" > "$L/stage2_driver.log" 2>&1
step "stage 2 final gate PASS" grep -q '"overall": "PASS"' "$WORK/validation/final_gate.json"
step "stage 2 integrity sample: exact-rational chain and independent spectrum recompute (production criteria)" \
    "$PYTHON" "$REPO/references/shoroq/tests/check_integrity_sample.py" "$WORK/validation/integrity" "$C/integrity_check.json" > "$L/integrity_check.log"
step "stage 2 outputs identical to frozen (raw, materialized, spectra; accepted set)" \
    "$PYTHON" "$REPO/references/shoroq/tests/compare_geometry_subset.py" "$WORK" "$FU" "$C/compare_geometry.json" > "$L/compare_geometry.log"
step "stage 3 driver (inertia scan, minima, validation, corrected spectra, relic input; 46 tachyonic expected)" \
    env EXPECTED_TACHYONIC=46 "$REPO/producers/run_vacuum_repair.sh" > "$L/stage3_driver.log" 2>&1
step "stage 3 records identical to frozen (new -> frozen: evidence, corrected spectra, classification, manifest fields)" \
    "$PYTHON" "$REPO/references/shoroq/tests/compare_vacuum_repair.py" "$WORK" "$FR/spectrum_count_audit/results/accounting_all.jsonl" \
    "$FR/tachyonic_vacua_audit/results" "$FR/tachyonic_vacua_audit/corrected_spectra" \
    "$FR/relic_abundance_input/relic_abundance_input_manifest.json" "$C/compare_vacuum_repair.json" > "$L/compare_vacuum_repair.log"
step "population completeness frozen -> new and population totals (4700/4491/209/46/16250/16313/539)" \
    "$PYTHON" "$REPO/references/shoroq/tests/check_full_population.py" "$WORK" "$FR" "$C/population.json" --full > "$L/population.log"

# 3. stages 4-5 on the regenerated relic input, with the Shoroq run file and expectations
export RELIC_RUN="$REPO/references/shoroq/configs/run.json" RELIC_EXPECTATIONS="$REPO/references/shoroq/configs/expectations.json"
step "stage 4 driver (A1 table, relic abundance, schema validation, independent verification)" \
    "$REPO/engine/run_relic.sh" > "$L/stage4_driver.log" 2>&1
step "stage 5 driver (theta input checks, theta analysis, Python cross-check)" \
    "$REPO/engine/run_theta.sh" > "$L/stage5_driver.log" 2>&1
step "relic and theta results identical to frozen" \
    "$PYTHON" "$REPO/references/shoroq/tests/compare_relic_theta_full.py" "$WORK" "$FR" "$C/compare_relic_theta.json" > "$L/compare_relic_theta.log"

# 4. after (always)
run "frozen reference intact after the run (all checksum lists)" \
    "$PYTHON" "$REPO/references/shoroq/tests/verify_frozen_reference.py" "$FR" "$C/frozen_reference_after.json" all > "$L/frozen_reference_after.log"
find "$WS" -newer "$WORK/start_marker" \( -type f -o -type l -o -type d \) > "$C/workspace_modified_after_start.txt" 2>/dev/null || true
run "no file or directory under the protected workspace modified during the run" test ! -s "$C/workspace_modified_after_start.txt"
git -C "$CYAX_REPO" rev-parse HEAD > "$C/cyax_head_after.txt"; git -C "$CYAX_REPO" status --porcelain --ignored > "$C/cyax_status_after.txt"
run "CYAxiverse checkout unchanged (HEAD and status)" \
    bash -c "cmp -s '$C/cyax_head_before.txt' '$C/cyax_head_after.txt' && cmp -s '$C/cyax_status_before.txt' '$C/cyax_status_after.txt'"
(cd "$WORK" && find geometry materialized spectra validation vacuum_repair relic_input relic final_analysis -type f 2>/dev/null | sort \
    | xargs shasum -a 256) > "$C/SHA256SUMS_outputs.txt" || true

"$PYTHON" - "$RES" "${#NAMES[@]}" "${NAMES[@]}" "${STATUS[@]}" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2]); names, st = sys.argv[3:3 + n], sys.argv[3 + n:3 + 2 * n]
res = {"test": "regression_full", "checks": [{"check": a, "status": b} for a, b in zip(names, st)],
       "n_pass": st.count("PASS"), "n_fail": st.count("FAIL"), "n_skipped": st.count("SKIPPED")}
res["verdict"] = "PASS" if res["n_fail"] == 0 and res["n_skipped"] == 0 else "FAIL"
json.dump(res, open(out, "w"), indent=1)
print(f"REGRESSION {res['verdict']}: {res['n_pass']} passed, {res['n_fail']} failed, {res['n_skipped']} skipped -> {out}")
sys.exit(0 if res["verdict"] == "PASS" else 1)
PY
