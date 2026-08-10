# Archived author/inflation run artifacts — 2026-08-10

The following temporary outputs were moved out of their active locations before
the author-prefactor correction is used for new comparisons:

```text
/private/tmp/cyaxiverse-author-bridge-fixed-smoke.json
/private/tmp/cyaxiverse-author-bridge-h11-010-331/
/private/tmp/cyaxiverse-author-bridge-random20-h11-004-050/
/private/tmp/cyaxiverse-author-fixed-raw/
/private/tmp/inflation-scale-continuation/
/private/tmp/cyaxiverse-inflation-scan-prep.csv
/private/tmp/cyaxiverse-inflation-scan-prep-cap.csv
/private/tmp/cyaxiverse-inflation-scan-prep-discovered.csv
/private/tmp/cyaxiverse-inflation-scan-prep-failure.csv
/private/tmp/cyaxiverse-inflation-scan-prep-regression.csv
```

Archive destination:

```text
/private/tmp/cyaxiverse-archived-runs-20260810/
```

Additionally, the pre-fix author source was preserved at:

```text
/private/tmp/cyaxiverse-archived-runs-20260810/author-source/Camcode_full_2.py.original
```

The archived pre-fix source SHA-256 is
`d5587f82b4283a0a4b6902fbf404b7fece42d857dba1f380eea11ba0bdf4eb01`.
The corrected active source SHA-256 is
`d45bc67235cff9d683ea08d9dc18f77277e882c49828fa5e12c8df6e7972fda5`.

These are recoverable, but must not be used as current comparison evidence:
the author-bridge directories include outputs made before the corrected
cross-term prefactor was propagated. The archived author source copy records
the pre-fix file; the active author source and repository comparison entry
point default to
`CN_Axiverse_code/ks_axiverse_python_collaborator/validation/cyaxiverse_fixed/`,
and new runs should use a fresh output directory.

Unrelated temporary spectrum, vacua, audit, and geometry-generation outputs
were intentionally left in place.
