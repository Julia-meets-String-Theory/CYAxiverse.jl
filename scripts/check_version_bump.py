#!/usr/bin/env python3
"""Check package-version handling at a release boundary."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path


VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
PACKAGE_PATH_PREFIXES = ("src/", "ext/", "add_functions/")


def git_output(*arguments: str) -> bytes:
    """Return the output of a Git command or stop with its error."""
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        raise SystemExit(result.returncode)
    return result.stdout


def parse_version(value: str, source: str) -> tuple[int, int, int]:
    """Parse a three-part package version."""
    match = VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise SystemExit(f"ERROR: {source} has non-SemVer package version {value!r}")
    return tuple(int(part) for part in match.groups())


def project_version(ref: str) -> tuple[int, int, int]:
    """Read the package version from Project.toml at a Git reference."""
    project = (
        Path("Project.toml").read_bytes()
        if ref in {"WORKTREE", "working-tree"}
        else git_output("show", f"{ref}:Project.toml")
    )
    try:
        value = tomllib.loads(project.decode())["version"]
    except (KeyError, tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise SystemExit(f"ERROR: could not read version from {ref}:Project.toml: {error}")
    if not isinstance(value, str):
        raise SystemExit(f"ERROR: {ref}:Project.toml version is not a string")
    return parse_version(value, f"{ref}:Project.toml")


def changed_files(base: str, head: str) -> list[str]:
    """List paths changed from the merge base to the requested head."""
    if head in {"WORKTREE", "working-tree"}:
        tracked_output = git_output("diff", "--name-only", "-z", base)
        untracked_output = git_output("ls-files", "--others", "--exclude-standard", "-z")
        output = tracked_output + untracked_output
    else:
        output = git_output("diff", "--name-only", "-z", f"{base}...{head}")
    return sorted({path.decode() for path in output.split(b"\0") if path})


def package_implementation_changed(paths: list[str]) -> bool:
    """Return whether a package implementation or metadata file changed."""
    return any(
        path == "Project.toml"
        or path.startswith(PACKAGE_PATH_PREFIXES)
        for path in paths
    )


def main() -> int:
    """Check the package version against the selected Git base."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Git base reference")
    parser.add_argument("--head", default="HEAD", help="Git head reference")
    parser.add_argument(
        "--require-bump",
        action="store_true",
        help="Require an increased version instead of deferring the bump",
    )
    arguments = parser.parse_args()

    paths = changed_files(arguments.base, arguments.head)
    if not package_implementation_changed(paths):
        print("PASS: no package implementation or Project.toml changes require a version bump")
        return 0

    if not arguments.require_bump:
        print("PASS: package implementation changed; version bump deferred to the release boundary")
        return 0

    base_version = project_version(arguments.base)
    head_version = project_version(arguments.head)
    if head_version <= base_version:
        print(
            "FAIL: package implementation changed without increasing the package version "
            f"({'.'.join(map(str, base_version))} -> {'.'.join(map(str, head_version))})",
            file=sys.stderr,
        )
        return 1

    print(
        "PASS: package implementation changed with a version bump "
        f"({'.'.join(map(str, base_version))} -> {'.'.join(map(str, head_version))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
