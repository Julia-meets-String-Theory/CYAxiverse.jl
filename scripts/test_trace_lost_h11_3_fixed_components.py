"""CYAX-0232 T061: the F5 checks in trace_lost_h11_3_fixed_components.py."""

from local_only import require_cytools

require_cytools()

import json
from pathlib import Path
import tempfile
import unittest

from f5_legacy_layout_api import synthesize
import kinetic_normalization_labels as labels
import trace_lost_h11_3_fixed_components as trace


class TracePreSummaryLabelTests(unittest.TestCase):
    def _write(self, directory, summary):
        path = Path(directory) / "pre.json"
        path.write_text(json.dumps(summary))
        return path

    def test_recorded_1_1_layout_is_accepted(self):
        layout = labels.f5_legacy_layouts("1.1")[0]
        summary = synthesize(layout.required)
        summary["schema_version"] = "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1"
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(trace._load_pre_summary(self._write(directory, summary)), summary)

    def test_unrecorded_1_1_file_is_rejected(self):
        summary = {"schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1",
                   "details": []}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "matches no recorded layout"):
                trace._load_pre_summary(self._write(directory, summary))


if __name__ == "__main__":
    unittest.main()
