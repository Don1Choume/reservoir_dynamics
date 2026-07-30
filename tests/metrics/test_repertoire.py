import math
import unittest

from reservoir_dynamics.metrics.repertoire import effective_repertoire_size


class EffectiveRepertoireSizeTest(unittest.TestCase):
    def test_uniform_probabilities_return_number_of_repertoires(self) -> None:
        self.assertAlmostEqual(effective_repertoire_size((0.25,) * 4), 4.0)

    def test_single_certain_repertoire_returns_one(self) -> None:
        self.assertAlmostEqual(effective_repertoire_size((1.0, 0.0)), 1.0)

    def test_imbalanced_probabilities_reduce_effective_size(self) -> None:
        effective_size = effective_repertoire_size((0.9, 0.1))

        self.assertGreater(effective_size, 1.0)
        self.assertLess(effective_size, 2.0)

    def test_rejects_probabilities_that_do_not_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "合計"):
            effective_repertoire_size((0.2, 0.2))

    def test_rejects_negative_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "非負"):
            effective_repertoire_size((1.1, -0.1))

    def test_rejects_non_finite_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限"):
            effective_repertoire_size((math.nan, 1.0))

    def test_rejects_empty_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "1つ以上"):
            effective_repertoire_size(())


if __name__ == "__main__":
    unittest.main()
