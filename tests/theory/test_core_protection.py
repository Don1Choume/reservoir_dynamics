import math
import unittest

from reservoir_dynamics.theory.core_protection import (
    core_deviation_bound_curve,
)


class CoreDeviationBoundCurveTest(unittest.TestCase):
    def test_zero_feedback_preserves_equal_core_states_exactly(self) -> None:
        result = core_deviation_bound_curve(
            core_lipschitz=0.6,
            feedback_lipschitz=0.0,
            reserve_difference_bound=2.0,
            steps=4,
        )

        self.assertEqual(result, (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_accumulates_bounded_feedback_geometrically(self) -> None:
        result = core_deviation_bound_curve(
            core_lipschitz=0.5,
            feedback_lipschitz=0.1,
            reserve_difference_bound=2.0,
            steps=3,
        )

        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.2)
        self.assertAlmostEqual(result[2], 0.3)
        self.assertAlmostEqual(result[3], 0.35)

    def test_includes_initial_core_mismatch(self) -> None:
        result = core_deviation_bound_curve(
            core_lipschitz=0.5,
            feedback_lipschitz=0.0,
            reserve_difference_bound=1.0,
            steps=2,
            initial_core_distance=0.8,
        )

        self.assertEqual(result, (0.8, 0.4, 0.2))

    def test_rejects_invalid_contractivity_and_bounds(self) -> None:
        for invalid_lipschitz in (-0.1, 1.0, math.nan):
            with self.subTest(core_lipschitz=invalid_lipschitz):
                with self.assertRaisesRegex(ValueError, "core_lipschitz"):
                    core_deviation_bound_curve(
                        core_lipschitz=invalid_lipschitz,
                        feedback_lipschitz=0.0,
                        reserve_difference_bound=1.0,
                        steps=2,
                    )

        with self.assertRaisesRegex(ValueError, "feedback_lipschitz"):
            core_deviation_bound_curve(
                core_lipschitz=0.5,
                feedback_lipschitz=-0.1,
                reserve_difference_bound=1.0,
                steps=2,
            )

        with self.assertRaisesRegex(ValueError, "steps"):
            core_deviation_bound_curve(
                core_lipschitz=0.5,
                feedback_lipschitz=0.1,
                reserve_difference_bound=1.0,
                steps=-1,
            )


if __name__ == "__main__":
    unittest.main()
