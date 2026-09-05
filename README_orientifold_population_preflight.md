# Higher-h11 orientifold run preflight

The h11=4 and h11=5 orientifold paths are replay paths. Before geometry
loading, `reproduce_fuzzy_axions_h11_4.py` requires
`orientifold_population_preflight.py` to:

1. read and fingerprint the five relevant population handoffs in
   `../handoffs_checkpoints/`;
2. verify every file listed in the durable h11=4 or h11=5 `SHA256SUMS.txt`;
3. run `zstd -t` on every compressed artifact; and
4. check the stored h11, favorable-polytope, FRST-class, trilayer, and
   `population_complete` metadata.

The preflight fails closed if a handoff, artifact, checksum, compression
check, or metadata check is missing. It does not read a parquet source and it
does not run CYTools.

Run the read-only preflight directly with:

```sh
python scripts/orientifold_population_preflight.py --h11 4
python scripts/orientifold_population_preflight.py --h11 5
```

The resulting `population_preflight` object is included in the reproduction
artifact, so the run records which handoffs and artifacts were acknowledged.
