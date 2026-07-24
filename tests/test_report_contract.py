import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report import declared_rounds, validate_output_name


class ReportContractTests(unittest.TestCase):
    def test_declared_rounds_includes_every_configured_round(self):
        self.assertEqual(declared_rounds(3, 3), [0, 1, 2, 3])

    def test_declared_rounds_rejects_truncated_report(self):
        with self.assertRaises(ValueError):
            declared_rounds(1, 0)

    def test_output_name_accepts_recovery_directory(self):
        self.assertEqual(validate_output_name("report_m0_m1"), "report_m0_m1")

    def test_output_name_rejects_paths(self):
        for value in ("../report", "report/subdir", "", "."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_output_name(value)


if __name__ == "__main__":
    unittest.main()
