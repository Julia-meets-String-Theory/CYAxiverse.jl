"""CYAX-0232 R-008 (T071): the A96 reproduction JSON carries its label.

``author_poly102_reference.py`` writes an A96 reproduction (the author's
raw-radian matrix); its top-level ``kinetic_normalization`` key must be
``"a96_reproduction_raw_radian"``. The trajectory integration is replaced by
a stub, so the test needs NumPy and SciPy (for the import) but no long solve.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"author_poly102_reference.py needs NumPy and SciPy: {exc}")

import author_poly102_reference as reference


class AuthorPoly102ReferenceLabelTests(unittest.TestCase):
    def test_json_has_a96_label(self):
        self.assertEqual(reference.KINETIC_NORMALIZATION, "a96_reproduction_raw_radian")

        def stub(delta_k, direction, **_kwargs):
            return {"delta_k": delta_k, "efolds": 0.0, "end_event": "stub",
                    "samples": []}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            with mock.patch.object(reference, "trajectory", stub), \
                    mock.patch("builtins.print"):
                reference.main(output, (1e-7,))
            data = json.loads(output.read_text())
        self.assertEqual(data["kinetic_normalization"], "a96_reproduction_raw_radian")
        self.assertEqual(len(data["results"]), 2)


if __name__ == "__main__":
    unittest.main()
