import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_passk import pass_at_k


class PassAtKTests(unittest.TestCase):
    def test_no_correct_samples(self):
        self.assertEqual(pass_at_k(10, 0, 1), 0.0)
        self.assertEqual(pass_at_k(10, 0, 10), 0.0)

    def test_pass_at_one_is_empirical_accuracy(self):
        self.assertTrue(math.isclose(pass_at_k(10, 3, 1), 0.3))

    def test_full_budget_succeeds_if_any_sample_is_correct(self):
        self.assertEqual(pass_at_k(10, 1, 10), 1.0)

    def test_known_combinatorial_value(self):
        expected = 1.0 - (7 * 6) / (10 * 9)
        self.assertTrue(math.isclose(pass_at_k(10, 3, 2), expected))

    def test_invalid_arguments_fail(self):
        with self.assertRaises(ValueError):
            pass_at_k(10, 11, 1)
        with self.assertRaises(ValueError):
            pass_at_k(10, 1, 0)


if __name__ == "__main__":
    unittest.main()
