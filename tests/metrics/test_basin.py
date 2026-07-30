import unittest

from reservoir_dynamics.metrics.basin import estimate_basin_stability


class BasinStabilityTest(unittest.TestCase):
    def test_estimates_probabilities_and_wilson_intervals(self) -> None:
        estimates = estimate_basin_stability(("A", "A", "A", "B"))

        self.assertEqual(tuple(estimate.label for estimate in estimates), ("A", "B"))
        self.assertEqual(estimates[0].count, 3)
        self.assertAlmostEqual(estimates[0].probability, 0.75)
        self.assertLessEqual(estimates[0].lower_confidence_bound, 0.75)
        self.assertGreaterEqual(estimates[0].upper_confidence_bound, 0.75)

    def test_preserves_first_observation_order(self) -> None:
        estimates = estimate_basin_stability(("second", "first", "second"))

        self.assertEqual(tuple(estimate.label for estimate in estimates), ("second", "first"))

    def test_probabilities_sum_to_one(self) -> None:
        estimates = estimate_basin_stability((1, 2, 3, 1, 2))

        self.assertAlmostEqual(sum(estimate.probability for estimate in estimates), 1.0)

    def test_rejects_empty_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "1つ以上"):
            estimate_basin_stability(())

    def test_rejects_invalid_confidence_level(self) -> None:
        for invalid_confidence in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(confidence=invalid_confidence):
                with self.assertRaisesRegex(ValueError, "0より大きく1より小さい"):
                    estimate_basin_stability(("A",), confidence=invalid_confidence)


if __name__ == "__main__":
    unittest.main()
