# CYAX-0168 G1 Generator-2.3 Failure Evidence v1

Status: **CYAX-0168 G1 NOT SATISFIED — inconclusive/invalid execution**

This is a new, versioned G1 evidence packet. It does not amend the approved
specification, lifecycle state, or historical generator-2.2 evidence.

## Frozen input identity

- Repository: `Julia-meets-String-Theory/CYAxiverse.jl`
- Branch: `research/cyax-0168-materialization-benchmark`
- Authorized start head: `84fb8d786f52653bf62cf7404c68c79ea555ac5c`
- Normative generator-2.3 head: `fe31fed192bfbefa28946736cdb5d2ec134985e0`
- G1 scope: calibration-only C0–C3; G2–G4 not entered
- Historical report and failure artifact were not modified

The start identity was checked with:

```text
$ git rev-parse 84fb8d786f52653bf62cf7404c68c79ea555ac5c
84fb8d786f52653bf62cf7404c68c79ea555ac5c
```

Historical immutability was checked with:

```text
$ sha256sum specs/0168-structured-graph-materialization/g1-execution-report.md research/cyax0168/evidence/generator-independence-failure.json
f8403b8115fe4f8ecbb004103d2cc21da5c8c43ac710b3287d05156de4103452  specs/0168-structured-graph-materialization/g1-execution-report.md
9f7bea4411b9d805e5fbfe621e0b6c349513fb96919e29e927e66dcd40ae972f  research/cyax0168/evidence/generator-independence-failure.json
```

## Generator-2.3 conformance

Primary implementation commits:

- `79c5a969f8233cbbdea8263f917d0a51c0966809` — primary generator-2.3 calibration path
- `ceb63478ab6f2ec093ff474f25d5e36607ef2cbf` — full construction/PRF trace retention
- `bfb8039c7cc49516e0aab1622c51bf022cad9101` — manager-facing comparison harness

Independent implementation commits:

- `4dd2cc840c508a4afa7176a5a0f834b319f228b5` — independent generator-2.3 reproduction
- `8901e01683d2eb3f1c0004ef91fc3bb46004dbf2` — independent calibration-only barrier
- `5f12148379ad3798731e91c4d6a91b1707190551` — separate logical checksum and snapshot ID

The exact comparison was run only for the approved C0 calibration cell:

```text
$ python3 -m research.cyax0168.compare_generator_23 --tier C0 --profile P-medium --seed 168900
{"checks":{"assertion_ids":true,"construction_trace":true,"logical_snapshot_checksum":true,"physical_records":true,"prf_choices":true,"snapshot_id":true,"source_bytes":true},"independent":{"assertion_count":5000,"construction_trace_count":5008,"entity_count":1000,"logical_snapshot_checksum":"e73063a46e1c585bfb15aa0db5e920a0f6b4a37f2b5a44a7a0b372f7aecd3b18","prf_choice_count":3566,"source_revision_count":5100},"passed":true,"primary":{"assertion_count":5000,"construction_trace_count":5008,"entity_count":1000,"logical_snapshot_checksum":"e73063a46e1c585bfb15aa0db5e920a0f6b4a37f2b5a44a7a0b372f7aecd3b18","prf_choice_count":3566,"source_revision_count":5100},"profile_id":"P-medium","seed":168900,"tier":"C0"}
```

Result: **PASS**. Primary and independent records, source bytes, complete
construction trace, PRF choices/retries, assertion IDs, logical checksum, and
snapshot ID are byte-for-byte equal. No decision fixture was generated or
accessed. The only decision-label operation was a guard test for `T0` that
returned `GenerationError` before construction; no decision IDs, bytes,
backend data, queries, or tuning results were produced.

## Ladybug artifact and offline smoke

The approved candidate was installed from the local wheelhouse only:

- Candidate: `ladybug==0.20.4`
- Wheel: `ladybug-0.20.4-cp314-cp314-macosx_15_0_arm64.whl`
- Wheel SHA-256: `7a36d5b051ddc954d7ee5d5fa6165fb49d48a4785a132bafdd311723897fa649`
- Upstream commit: `df58ee387c4e5e9f02bb9d518636b52cd4abe5f7`
- License: MIT
- License file SHA-256: `1c495c9546d0de02e83c9d50d5f7eb21f0085bc8f77a0ee333081a123a9c8d0c`
- Tags: `cp314-cp314-macosx_15_0_arm64`

Wheel identity commands and results:

```text
$ shasum -a 256 <offline-wheelhouse>/ladybug-0.20.4-cp314-cp314-macosx_15_0_arm64.whl
7a36d5b051ddc954d7ee5d5fa6165fb49d48a4785a132bafdd311723897fa649  <offline-wheelhouse>/ladybug-0.20.4-cp314-cp314-macosx_15_0_arm64.whl

$ python3 -m venv <ephemeral-venv> && PIP_NO_INDEX=1 <ephemeral-venv>/bin/python -m pip install --no-index --find-links=<offline-wheelhouse> ladybug==0.20.4
Looking in links: <offline-wheelhouse>
Processing <offline-wheelhouse>/ladybug-0.20.4-cp314-cp314-macosx_15_0_arm64.whl
Installing collected packages: ladybug
Successfully installed ladybug-0.20.4
```

The durable network-blocked smoke harness is
`research/cyax0168/ladybug_g1_offline_smoke.py`. It was run with:

```text
$ PYTHONPATH=. PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 <ephemeral-venv>/bin/python -m research.cyax0168.ladybug_g1_offline_smoke
{"candidate": "ladybug==0.20.4", "clean_graph_creation": true, "complete_logical_export": true, "deterministic_rebuild": true, "graph_bytes": 184320, "logical_export_sha256": "804d9626ecfe42a17436eec6c7ed531355cb9206639de50a60723244a7daca79", "network_blocked": true, "partial_build_rejection": true, "recovery": true, "reopen": true, "required_traversal": true, "smoke_import": true, "tamper_rejection": true, "threads": 1, "traversal_rows": [["impl-1", "Implementation"], ["req-1", "Requirement"]], "version": "0.20.4"}
```

Result: **PASS** for the candidate artifact and the required tiny offline
smoke. The adapter unit test also passed 2/2 in the same isolated venv.

## Campaign-host control gate

Sanitized facts captured on the execution host (no machine name, account,
serial, UUID, or other machine-unique field was recorded):

```text
operating_system=macOS
kernel_release=25.6.0
product_version=26.6.2
build=25G83
cpu_model_class=Apple silicon ARM64
core_topology=logical_cpus=12
physical_ram_bytes=25769803776
filesystem_type=APFS
benchmark_volume_capacity_bytes=494384795648
ordinary_available_volume_capacity_bytes=172847177728
python_version=3.14.6
sqlite_version=3.53.4
power_source=AC Power
energy_mode=Automatic (pmset lowpowermode=0)
descendants=()
Mach diagnostics=unavailable (collector-unavailable)
```

Host manifest capture and validation returned `valid=true` for the structural
manifest, but this does not establish the required live thermal control. The
exact thermal command was:

```text
$ pmset -g therm
Error:Failed to get thermal warning level with error code 0xe00002bc
Error: Failed to get performance warning level with error code 0xe00002bc
Error: No CPU power status with error code 0xe00002bc
```

Related control observations were `memory_pressure`: system-wide memory free
percentage 67%; `pmset -g batt`: AC Power; and `child_pids()`: empty. `diskutil
info /` could not use the DiskManagement framework, and direct `sysctl
hw.memsize` was not permitted; the portable host capture obtained the RAM
value above. These are disclosed limitations, not inferred passes.

Smallest failure: required nominal thermal-state evidence cannot be established
on this run because the approved thermal-control command returned an unavailable
status. This is classified as **Inconclusive / invalid execution — host/control
evidence unavailable**, not as a backend or resource-envelope failure. The
approved host contract was not weakened.

## Verification record

The following checks were executed after the generator repair:

```text
$ python3 -m py_compile research/cyax0168/generator_primary.py research/cyax0168/compare_generator_23.py research/cyax0168/__init__.py research/cyax0168/test_generator_primary.py
PASS

$ python3 -m unittest research.cyax0168.test_generator_primary -q
Ran 7 tests in 11.467s
OK

$ python3 -m unittest discover -s research/cyax0168 -p 'test*.py'
Ran 69 tests in 10.160s
OK (skipped=2)

$ python3 -m unittest discover -s research/cyax0168/tests -p 'test*.py'
Ran 63 tests in 120.464s
OK

$ PYTHONPATH=. <ephemeral-venv>/bin/python -m unittest research.cyax0168.test_ladybug_backend -v
Ran 2 tests in 0.832s
OK
```

The two skipped tests in the system-Python suite are the Ladybug tests; they
require the frozen wheel and passed separately in the isolated wheel venv.
`git diff --check` and the historical SHA-256 checks above passed. Generated
`__pycache__` directories were removed before committing this packet.

## Disposition and scope boundary

- Overall result: **CYAX-0168 G1 NOT SATISFIED**; stop at host/control gate.
- `G` was not implemented or exercised because the handoff permits it only
  after the exact-host gate is established.
- Calibration statistical-precision ratification was not run.
- 48-hour campaign-duration projection was not run.
- No C1–C3 calibration evidence was accepted or frozen.
- No T0–T4 decision fixture was generated, accessed, queried, profiled, or used
  for tuning.
- CYAX-0168 G2, G3, and G4 were not entered.
- FalkorDBLite or any other graph fallback was not substituted.
