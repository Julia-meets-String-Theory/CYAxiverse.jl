#!/usr/bin/env python3
"""Decide whether a pull request must run process-isolated scientific qualification."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable

# Package source and the qualification fixtures themselves: a change to either can
# alter the producer/replay evidence that only the qualification job exercises.
QUALIFICATION_PATH_PREFIXES = ("src/", "test/fixtures/scientific_certification/")


def requires_qualification(paths: Iterable[str]) -> bool:
    """Return whether any changed repository path needs scientific qualification."""
    return any(path.startswith(QUALIFICATION_PATH_PREFIXES) for path in paths)


def changed_paths(base: str, head: str) -> list[str]:
    """Return the paths changed between the merge base of base and head, and head."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
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
