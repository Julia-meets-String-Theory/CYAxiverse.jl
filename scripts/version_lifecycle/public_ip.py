"""Pure parsing for legacy numeric IPv4 locator spellings.

URL clients can interpret abbreviated, octal, and hexadecimal addresses that
``ipaddress.ip_address`` deliberately rejects.  This parser only classifies
those numeric spellings; it never resolves names or performs network access.
"""

from __future__ import annotations

import ipaddress
import re


def parse_ipv4_compat(value: str) -> ipaddress.IPv4Address | None:
    """Return the IPv4 address denoted by an inet_aton-style numeric value."""

    if not isinstance(value, str):
        return None
    parts = value.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    numbers: list[int] = []
    for part in parts:
        if re.fullmatch(r"0[xX][0-9A-Fa-f]+", part):
            number = int(part[2:], 16)
        elif re.fullmatch(r"0[0-7]+", part):
            number = int(part, 8)
        elif re.fullmatch(r"(?:0|[1-9][0-9]*)", part):
            number = int(part, 10)
        else:
            return None
        numbers.append(number)
    if any(number > 255 for number in numbers[:-1]):
        return None
    remaining_bits = 8 * (5 - len(numbers))
    if numbers[-1] >= 1 << remaining_bits:
        return None
    address = numbers[-1]
    for index, number in enumerate(numbers[:-1]):
        address |= number << (24 - 8 * index)
    return ipaddress.IPv4Address(address)


__all__ = ["parse_ipv4_compat"]
