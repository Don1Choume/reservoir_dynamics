import math
import unittest

from reservoir_dynamics.theory.contraction import (
    iterated_lipschitz_distance_bound,
)


class IteratedLipschitzDistanceBoundTest(unittest.TestCase):
    def test_returns_bound_from_initial_time_through_requested_steps(self) -> None:
        bounds = iterated_lipschitz_distance_bound(
            initial_distance=4.0,
            lipschitz_constant=0.5,
            steps=3,
        )

        self.assertEqual(bounds, (4.0, 2.0, 1.0, 0.5))

    def test_zero_lipschitz_constant_collapses_after_one_step(self) -> None:
        bounds = iterated_lipschitz_distance_bound(
            initial_distance=3.0,
            lipschitz_constant=0.0,
            steps=2,
        )

        self.assertEqual(bounds, (3.0, 0.0, 0.0))

    def test_rejects_negative_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "初期距離"):
            iterated_lipschitz_distance_bound(
                initial_distance=-1.0,
                lipschitz_constant=0.5,
                steps=1,
            )

    def test_rejects_non_finite_lipschitz_constant(self) -> None:
        with self.assertRaisesRegex(ValueError, "Lipschitz"):
            iterated_lipschitz_distance_bound(
                initial_distance=1.0,
                lipschitz_constant=math.nan,
                steps=1,
            )

    def test_rejects_negative_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps"):
            iterated_lipschitz_distance_bound(
                initial_distance=1.0,
                lipschitz_constant=0.5,
                steps=-1,
            )


if __name__ == "__main__":
    unittest.main()
