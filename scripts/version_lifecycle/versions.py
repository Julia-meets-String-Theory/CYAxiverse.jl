"""Strict package-version and public-tag parsing.

The package contract deliberately accepts the approved ``-DEV`` sentinel and
rejects every other pre-release/build spelling.  Julia's ``VersionNumber``
accepts a wider language, so this module validates the repository grammar
before constructing the small immutable value used by lifecycle code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
import re
from typing import Union


_COMPONENT = r"(?:0|[1-9][0-9]*)"
_PACKAGE_RE = re.compile(
    rf"^(?P<major>{_COMPONENT})\.(?P<minor>{_COMPONENT})\.(?P<patch>{_COMPONENT})(?P<dev>-DEV)?$"
)
_TAG_RE = re.compile(
    rf"^v(?P<major>{_COMPONENT})\.(?P<minor>{_COMPONENT})\.(?P<patch>{_COMPONENT})$"
)


@total_ordering
@dataclass(frozen=True, slots=True)
class Version:
    """A canonical governed package version.

    ``is_dev`` is the only supported prerelease distinction.  The
    :attr:`canonical` property is the exact raw spelling accepted by the
    parser, which makes it safe for equality and digest inputs.
    """

    major: int
    minor: int
    patch: int
    is_dev: bool = False

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.is_dev, bool):
            raise ValueError("is_dev must be bool")

    @property
    def canonical(self) -> str:
        suffix = "-DEV" if self.is_dev else ""
        return f"{self.major}.{self.minor}.{self.patch}{suffix}"

    @property
    def final(self) -> "Version":
        """Return this version with the development suffix removed."""

        return Version(self.major, self.minor, self.patch, False)

    @property
    def is_final(self) -> bool:
        return not self.is_dev

    @property
    def prerelease(self) -> str | None:
        return "DEV" if self.is_dev else None

    @property
    def tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def __str__(self) -> str:
        return self.canonical

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        # A final identity and its DEV identity have the same numeric point;
        # prerelease is ordered before final for deterministic comparisons.
        return (self.tuple, not self.is_dev) < (other.tuple, not other.is_dev)


VersionLike = Union[str, Version]


def _parse(match: re.Match[str], raw: str, *, is_dev: bool) -> Version:
    version = Version(
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        is_dev,
    )
    if version.canonical != raw:
        raise ValueError(f"noncanonical version spelling: {raw!r}")
    return version


def parse_package_version(value: str) -> Version:
    """Parse exactly ``X.Y.Z`` or ``X.Y.Z-DEV``.

    Components use canonical decimal spelling.  Build metadata and all other
    prerelease labels are rejected even where Julia would parse them.
    """

    if not isinstance(value, str):
        raise TypeError("package version must be a string")
    match = _PACKAGE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid governed package version: {value!r}")
    return _parse(match, value, is_dev=match.group("dev") is not None)


def parse_public_tag(value: str) -> Version:
    """Parse exactly a canonical future public tag ``vX.Y.Z``."""

    if not isinstance(value, str):
        raise TypeError("public tag must be a string")
    match = _TAG_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid governed public tag: {value!r}")
    return _parse(match, value[1:], is_dev=False)


parse_version = parse_package_version


def as_version(value: VersionLike) -> Version:
    """Normalize a version value accepted by lifecycle APIs."""

    return value if isinstance(value, Version) else parse_package_version(value)


def final_version(value: VersionLike) -> Version:
    """Normalize a value and require its final identity."""

    version = as_version(value)
    if version.is_dev:
        raise ValueError(f"development version is not a final identity: {version}")
    return version


def public_tag(value: VersionLike) -> str:
    """Return the canonical public tag for a final version."""

    return f"v{final_version(value).canonical}"


def principal_sentinel(closed: VersionLike) -> Version:
    """Return the exact next principal ``X.Y.(Z+1)-DEV`` sentinel."""

    version = final_version(closed)
    return Version(version.major, version.minor, version.patch + 1, True)


def maintenance_line(value: str) -> tuple[int, int]:
    """Parse a maintenance line name such as ``maintenance/0.7``."""

    if not isinstance(value, str):
        raise TypeError("maintenance line must be a string")
    match = re.fullmatch(rf"maintenance/({_COMPONENT})\.({_COMPONENT})", value)
    if match is None or value != f"maintenance/{int(match.group(1))}.{int(match.group(2))}":
        raise ValueError(f"invalid maintenance line: {value!r}")
    return int(match.group(1)), int(match.group(2))


__all__ = [
    "Version",
    "VersionLike",
    "as_version",
    "final_version",
    "maintenance_line",
    "parse_package_version",
    "parse_version",
    "parse_public_tag",
    "principal_sentinel",
    "public_tag",
]
