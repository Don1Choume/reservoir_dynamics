import unittest

from reservoir_dynamics.simulation.weight_perturbation import (
    sample_entrywise_bounded_perturbation,
)


class WeightPerturbationTest(unittest.TestCase):
    def test_same_seed_scales_one_fixed_off_diagonal_direction(self) -> None:
        weights = (
            (1.5, 0.2, -0.1),
            (0.1, 1.5, 0.3),
            (-0.2, 0.05, 1.5),
        )

        smaller = sample_entrywise_bounded_perturbation(
            weights,
            maximum_absolute_change=0.1,
            random_seed=1801,
        )
        larger = sample_entrywise_bounded_perturbation(
            weights,
            maximum_absolute_change=0.2,
            random_seed=1801,
        )

        for row in range(3):
            self.assertEqual(smaller[row][row], weights[row][row])
            self.assertEqual(larger[row][row], weights[row][row])
            for column in range(3):
                if row == column:
                    continue
                smaller_change = smaller[row][column] - weights[row][column]
                larger_change = larger[row][column] - weights[row][column]
                self.assertLessEqual(abs(smaller_change), 0.1)
                self.assertLessEqual(abs(larger_change), 0.2)
                self.assertAlmostEqual(larger_change, 2.0 * smaller_change)

    def test_zero_amplitude_returns_equal_immutable_matrix(self) -> None:
        weights = ((1.5, 0.2), (0.1, 1.5))

        perturbed = sample_entrywise_bounded_perturbation(
            weights,
            maximum_absolute_change=0.0,
            random_seed=1802,
        )

        self.assertEqual(perturbed, weights)
        self.assertIsInstance(perturbed, tuple)
        self.assertIsInstance(perturbed[0], tuple)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "正方"):
            sample_entrywise_bounded_perturbation(
                ((1.0, 0.0),),
                maximum_absolute_change=0.1,
                random_seed=1803,
            )
        with self.assertRaisesRegex(ValueError, "非負"):
            sample_entrywise_bounded_perturbation(
                ((1.0, 0.0), (0.0, 1.0)),
                maximum_absolute_change=-0.1,
                random_seed=1803,
            )
        with self.assertRaisesRegex(ValueError, "整数"):
            sample_entrywise_bounded_perturbation(
                ((1.0, 0.0), (0.0, 1.0)),
                maximum_absolute_change=0.1,
                random_seed=True,
            )


if __name__ == "__main__":
    unittest.main()
