import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_geometric_data_multitriangulation.py"
DEFAULT_POLYTOPE_MANIFEST = Path(
    "/private/tmp/cyaxiverse-glimmers-local-mirror-manifest.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "/private/tmp/cyaxiverse-glimmers-geometry-dataset-20260814G"
)


ORIENTIFOLD_DATA = {
    "label": "trivial-h11-minus-zero-involution",
    "lattice_matrix": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    "involution_type": "O3/O7",
    "coefficient_constraints": {},
}


def create_orientifold_file(filepath):
    """
    Creates the required O3/O7 explicit orientifold handoff file.
    The visible-sector policy 'intersecting_d7' requires a validated O3/O7 
    orientifold with h11_minus=0.
    """
    filepath = Path(filepath).expanduser().resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        with filepath.open() as stream:
            existing = json.load(stream)
        if existing != ORIENTIFOLD_DATA:
            raise ValueError(f"existing orientifold file has unexpected contents: {filepath}")
        return str(filepath)

    with filepath.open("x") as stream:
        json.dump(ORIENTIFOLD_DATA, stream, indent=2)
        stream.write("\n")
    return str(filepath)


def local_polytope_manifest():
    """Return the mirror-derived manifest used for offline generation."""
    filepath = Path(
        os.environ.get("GLIMMERS_POLYTOPE_MANIFEST", str(DEFAULT_POLYTOPE_MANIFEST))
    ).expanduser().resolve()
    if not filepath.is_file():
        raise FileNotFoundError(
            "Local KS mirror manifest not found: "
            f"{filepath}. Set GLIMMERS_POLYTOPE_MANIFEST to a valid JSON manifest."
        )
    return filepath


def fresh_output_directory():
    """Choose a non-existing output root without overwriting prior attempts."""
    configured = os.environ.get("GLIMMERS_OUTPUT_DIR")
    if configured:
        return str(Path(configured).expanduser().resolve())

    if not DEFAULT_OUTPUT_DIRECTORY.exists():
        return str(DEFAULT_OUTPUT_DIRECTORY)

    for index in range(1, 1000):
        candidate = Path(f"{DEFAULT_OUTPUT_DIRECTORY}-rerun-{index:03d}")
        if not candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "Could not find a fresh output directory after 999 attempts; set "
        "GLIMMERS_OUTPUT_DIR explicitly."
    )


def run_glimmers_generation(outdir, seed, cores=8):
    """
    Executes the repository's CYTools multitriangulation script with the
    approved schema-1.1 EFT geometry plan and the local KS mirror manifest.
    """
    polytope_manifest = local_polytope_manifest()
    outdir = Path(outdir).expanduser().resolve()
    orientifold_path = create_orientifold_file(
        outdir.parent / f"{outdir.name}-o3_o7_involution.json"
    )

    command = [
        sys.executable,
        str(GENERATOR_SCRIPT),
        "--eft",
        "--eft-geometry-plan", "50:500,100:500,200:300,491:100",
        "--eft-minimum-rows", "100000",
        "--eft-maximum-rows", "200000",
        "--eft-output-format", "parquet",
        "--database-source", "manifest",
        "--polytope-manifest", str(polytope_manifest),
        "--ks-database-version", "local current KS Parquet mirror manifest 20260814",
        "--sampling-scheme", "ntfe_fast",
        "--moduli-policy", "canonical_qcd",
        "--qcd-volume-target", "40.0",
        "--visible-sector-policy", "intersecting_d7",
        "--orientifold-file", orientifold_path,
        "--qed-volume-max", "127.5",
        "--backend", "cgal",
        "--outdir", str(outdir),
        "--seed", str(seed),
        "--cores", str(cores),
        "--verbose"
    ]

    print(f"Executing geometry generation:\n{' '.join(command)}")
    environment = os.environ.copy()
    environment.setdefault(
        "XDG_CACHE_HOME",
        str(outdir.parent / f"{outdir.name}-cytools-cache"),
    )
    subprocess.run(command, check=True, cwd=REPOSITORY_ROOT, env=environment)

if __name__ == "__main__":
    output_directory = fresh_output_directory()
    run_glimmers_generation(outdir=output_directory, seed=20260814)
