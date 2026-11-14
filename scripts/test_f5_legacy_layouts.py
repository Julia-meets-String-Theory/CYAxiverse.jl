"""CYAX-0232 T061: the F5 legacy layout records and the A2 inference.

One subtest per recorded layout (Clarification A2): a file holding exactly
the required paths, and one holding every recorded path, are both inferred as
``legacy_c_half``. Files with an unrecorded path, or a legacy version that
carries a label, are rejected. Standard library only.
"""

from __future__ import annotations

import unittest

import kinetic_normalization_labels as labels
from f5_legacy_layout_api import normalize, synthesize

VERSIONS = {
    "1.0": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.0",
    "1.1": "cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1",
}


def _file(paths, version):
    summary = synthesize(paths)
    summary["schema_version"] = VERSIONS[version]
    return summary


class F5LegacyLayoutTests(unittest.TestCase):
    def test_record_counts(self):
        counts = {}
        for version in VERSIONS:
            for layout in labels.f5_legacy_layouts(version):
                counts[layout.group] = counts.get(layout.group, 0) + 1
        self.assertEqual(
            counts,
            {"1.0-G1": 5, "1.0-G2": 8, "1.0-G3": 8, "1.0-G4": 16, "1.0-G5": 16,
             "1.0-G6": 36, "1.1-G1": 26, "1.1-G2": 30, "1.1-G2b": 30,
             "1.1-G3": 30, "1.1-G4": 45},
        )
        ids = [layout.id for v in VERSIONS for layout in labels.f5_legacy_layouts(v)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_recorded_layout_is_inferred(self):
        for version in VERSIONS:
            for layout in labels.f5_legacy_layouts(version):
                self.assertIn("schema_version", layout.required)
                for name, paths in (("required", layout.required),
                                    ("all", layout.required | layout.optional)):
                    with self.subTest(layout=layout.id, file=name):
                        summary = _file(paths, version)
                        self.assertEqual(normalize(summary, layout.map_nodes), paths)
                        self.assertIn(layout.id,
                                      labels.match_f5_legacy_layouts(summary, version))
                        self.assertEqual(
                            labels.f5_reproduction_label(summary, "f.json"),
                            ("legacy_c_half", True),
                        )

    def test_unrecorded_path_is_rejected(self):
        for version in VERSIONS:
            layout = labels.f5_legacy_layouts(version)[0]
            summary = _file(layout.required, version)
            summary["unrecorded_block"] = {"value": 1}
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, r"f\.json.*unrecorded_block"):
                    labels.f5_reproduction_label(summary, "f.json")

    def test_missing_required_path_is_rejected(self):
        summary = _file(labels.f5_legacy_layouts("1.0")[0].required, "1.0")
        del summary["counts"]
        with self.assertRaisesRegex(ValueError, "matches no recorded layout"):
            labels.f5_reproduction_label(summary, "f.json")

    def test_labelled_legacy_version_is_rejected(self):
        for version in VERSIONS:
            summary = _file(labels.f5_legacy_layouts(version)[0].required, version)
            summary["kinetic_normalization"] = "sugra_c1"
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "carries a kinetic_normalization"):
                    labels.f5_reproduction_label(summary, "f.json")

    def test_current_and_1_2_rules(self):
        current = {"schema_version": labels.F5_REPRODUCTION_SCHEMA,
                   "kinetic_normalization": "sugra_c1"}
        self.assertEqual(labels.f5_reproduction_label(current, "f.json"), ("sugra_c1", False))
        legacy = {"schema_version": labels.F5_LEGACY_REPRODUCTION_SCHEMA}
        self.assertEqual(labels.f5_reproduction_label(legacy, "f.json"),
                         ("legacy_c_half", True))
        for bad in (
            {"schema_version": labels.F5_REPRODUCTION_SCHEMA},
            dict(current, kinetic_normalization="legacy_c_half"),
            dict(legacy, kinetic_normalization="sugra_c1"),
            {"schema_version": "cyaxiverse-fuzzy-axions-h11-4-reproduction-9.9"},
        ):
            with self.subTest(case=bad):
                with self.assertRaises(ValueError):
                    labels.f5_reproduction_label(bad, "f.json")

    def test_reproduction_labels_rejected(self):
        # CYAX-0232 R-008 (T071): F5 is a physical family; its reader rejects
        # both reproduction labels, and names the file.
        for label in ("p96_reproduction_c_half", "a96_reproduction_raw_radian"):
            summary = {"schema_version": labels.F5_REPRODUCTION_SCHEMA,
                       "kinetic_normalization": label}
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "f.json"):
                    labels.f5_reproduction_label(summary, "f.json")


if __name__ == "__main__":
    unittest.main()
