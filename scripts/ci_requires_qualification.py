#!/usr/bin/env python3
"""Decide whether a pull request must run process-isolated scientific qualification."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable

# Package source and the qualification fixtures themselves: a change to either can
# alter the producer/replay evidence that only the qualification job exercises.
QUALIFICATION_PATH_PREFIXES = ("src/", "test/fixtures/scientific_certification/")

# Single files the qualification job depends on. run_fixture.jl includes
# test/coverage_support.jl and replays through scripts/verify_scientific_certificate.jl;
# the root Project.toml and Manifest.toml select the packages whose versions the
# release pins check; LocalPreferences.toml can change what the job loads, because
# Julia applies its sections for packages the root project lists; the job requires
# a clean source tree, and the root .gitignore hides the root Manifest.toml that CI
# resolves, so a change to it can make the tree dirty (a nested .gitignore cannot
# reach that file); the root JuliaLocalPreferences.toml, when present, is read by
# Julia in place of LocalPreferences.toml (Base.preferences_names, CYAX-0222 R-008);
# the root JuliaProject.toml and JuliaManifest.toml are read in place of Project.toml
# and Manifest.toml when present (Base.project_names, Base.manifest_names in Julia
# 1.12, CYAX-0222 R-009); the versioned root manifest names are matched below;
# the gate script and CI.yml decide whether the job runs.
QUALIFICATION_PATHS = frozenset({
    "Project.toml",
    "Manifest.toml",
    "JuliaProject.toml",
    "JuliaManifest.toml",
    "LocalPreferences.toml",
    "JuliaLocalPreferences.toml",
    ".gitignore",
    "scripts/verify_scientific_certificate.jl",
    "test/coverage_support.jl",
    "scripts/ci_requires_qualification.py",
    ".github/workflows/CI.yml",
})

# Versioned root manifest names: Julia 1.12 lists Manifest-v<major>.<minor>.toml and
# JuliaManifest-v<major>.<minor>.toml in Base.manifest_names (CYAX-0222 R-009). Digits
# only, and fullmatch on a root path (it contains no "/"), so nested files, a
# missing minor version, and suffixes such as ".orig" do not match.
VERSIONED_MANIFEST = re.compile(r"(?:Julia)?Manifest-v\d+\.\d+\.toml")


def requires_qualification(paths: Iterable[str]) -> bool:
    """Return whether any changed repository path needs scientific qualification."""
    return any(path in QUALIFICATION_PATHS
               or path.startswith(QUALIFICATION_PATH_PREFIXES)
               or VERSIONED_MANIFEST.fullmatch(path) is not None
               for path in paths)


def changed_paths(base: str, head: str) -> list[str]:
    """Return the paths changed between the merge base of base and head, and head."""
    # --no-renames: with rename detection Git prints only the new path, so moving an
    # input (such as the root .gitignore or a file under src/) to a path that is not
    # an input would give qualification=false. Without it a rename lists the old path
    # as deleted and the new path as added (CYAX-0222 R-007).
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", f"{base}...{head}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return [line for line in result.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="pull-request base commit")
    parser.add_argument("--head", required=True, help="pull-request head commit")
    arguments = parser.parse_args(argv)

    required = requires_qualification(changed_paths(arguments.base, arguments.head))
    line = f"qualification={'true' if required else 'false'}\n"
    sys.stdout.write(line)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
