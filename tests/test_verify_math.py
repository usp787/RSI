import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verify_math import (
    extract_final_answer,
    extract_last_boxed,
    normalize_gsm8k,
    verify_completion,
)


class AnswerVerificationTests(unittest.TestCase):
    def test_nested_box(self):
        self.assertEqual(extract_last_boxed(r"Thus \boxed{\frac{1}{2}}."), r"\frac{1}{2}")

    def test_last_box_wins(self):
        self.assertEqual(extract_final_answer(r"First \boxed{2}, finally \boxed{-3}."), "-3")

    def test_gsm8k_numeric_normalization(self):
        self.assertEqual(normalize_gsm8k("1,200.50"), "2401/2")
        self.assertEqual(normalize_gsm8k("-3/6"), "-1/2")
        self.assertEqual(normalize_gsm8k(r"\frac{1}{2}"), "1/2")
        self.assertEqual(normalize_gsm8k("25%"), "percent:25")

    def test_gsm8k_correct_and_incorrect(self):
        correct = verify_completion("gsm8k", r"Work. \boxed{1,200}", "1200")
        wrong = verify_completion("gsm8k", r"Work. \boxed{1199}", "1200")
        self.assertTrue(correct["correct"])
        self.assertFalse(wrong["correct"])

    def test_math_verify_api_and_equivalence(self):
        result = verify_completion("math", r"Therefore \boxed{\frac{1}{2}}.", "0.5")
        self.assertTrue(result["parsed"])
        self.assertTrue(result["correct"])


if __name__ == "__main__":
    unittest.main()
