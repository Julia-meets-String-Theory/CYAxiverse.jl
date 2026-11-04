#!/usr/bin/env bash
# Stage 4: relic abundance of every mode for the reheating scenarios of the run file (Julia 1.8.4, seconds), followed by
# the output-schema validation and the independent verification (float64 implementation C on every row, 50-digit
# implementation B on a stratified sample).
#   RELIC_RUN           run file (scenarios, approval; configs/run.example.json); required
#   RELIC_EXPECTATIONS  optional expectations of a reference case (references/shoroq/configs/expectations.json)
source "$(dirname "$0")/common.sh"
R="$WORK/relic"; S="$REPO/engine/relic"
[[ -n "${RELIC_RUN:-}" ]] || die "set RELIC_RUN to the run file (see configs/run.example.json)"
require "$RELIC_RUN" "$WORK/relic_input/relic_abundance_input_manifest.json" "$WORK/relic_input/SHA256SUMS.txt" "$WORK/relic_input/spectra" "$REPO/data/standardmodel2018.dat"
[[ -e "$R" ]] && die "$R exists: refusing to overwrite a previous run"
mkdir -p "$R"
# The run file's schema selects the methodology: cyaxiverse_relic_run/1 -> methodology v1 (w_R = 0 fixed), /2 -> v2 (constant w_R per scenario)
RUN_SCHEMA="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("schema", ""))' "$RELIC_RUN")"
case "$RUN_SCHEMA" in
    cyaxiverse_relic_run/1) MV=1 ;;
    cyaxiverse_relic_run/2) MV=2 ;;
    *) die "$RELIC_RUN: schema must be cyaxiverse_relic_run/1 (methodology v1) or cyaxiverse_relic_run/2 (methodology v2), got '$RUN_SCHEMA'" ;;
esac
log "stage 4: run configuration from the v$MV methodology, the run file${RELIC_EXPECTATIONS:+ and the expectations}"
"$PYTHON" "$S/00_compose_run_config.py" relic "$REPO/configs/methodology_v$MV.json" "$RELIC_RUN" "$R/template.json" \
    ${RELIC_EXPECTATIONS:+--expectations "$RELIC_EXPECTATIONS"} --checks "$R/scenario_checks.json" | tee "$WORK/logs/04_compose.log"
log "stage 4: A1 canonical mode table"
"$PYTHON" "$S/01_build_mode_table.py" "$R/template.json" "$WORK/relic_input" "$R/a1" > "$WORK/logs/04_a1.log"
"$PYTHON" "$S/02_write_run_config.py" "$R/template.json" "$WORK/relic_input" "$R/a1" "$R" "$R/run_config.json"
log "stage 4: relic abundance"
jana "$S/compute_relic_abundance.jl" "$R/run_config.json" "$R/a1/canonical_modes.csv" "$R/tables" abundance | tee "$WORK/logs/04_relic.log"
export SHOROQ_RELIC_CONFIG="$R/run_config.json"
EXPECTED_MODES="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"].get("expected_modes", ""))' "$R/run_config.json")"
"$PYTHON" "$S/03_validate_outputs.py" "$R/tables" $EXPECTED_MODES > "$WORK/logs/04_validate.log"
"$PYTHON" "$S/04_verify_relic_outputs.py" "$R" > "$WORK/logs/04_verify.log"
log "stage 4 done"
