import math
import unittest

from reservoir_dynamics.metrics.conditional_lyapunov import (
    finite_time_conditional_lyapunov_exponent,
)


class FiniteTimeConditionalLyapunovExponentTest(unittest.TestCase):
    def test_returns_mean_logarithmic_derivative_growth(self) -> None:
        exponent = finite_time_conditional_lyapunov_exponent(
            (0.5, 0.5, 0.5),
        )

        self.assertAlmostEqual(exponent, math.log(0.5))

    def test_applies_washout_before_averaging(self) -> None:
        exponent = finite_time_conditional_lyapunov_exponent(
            (2.0, 0.25, 0.25),
            washout=1,
        )

        self.assertAlmostEqual(exponent, math.log(0.25))

    def test_uses_declared_floor_for_zero_derivative(self) -> None:
        exponent = finite_time_conditional_lyapunov_exponent(
            (0.0,),
            derivative_floor=1e-12,
        )

        self.assertAlmostEqual(exponent, math.log(1e-12))

    def test_rejects_empty_post_washout_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "評価区間"):
            finite_time_conditional_lyapunov_exponent(
                (0.5,),
                washout=1,
            )

    def test_rejects_negative_derivative_magnitude(self) -> None:
        with self.assertRaisesRegex(ValueError, "非負"):
            finite_time_conditional_lyapunov_exponent((-0.5,))

    def test_rejects_non_finite_derivative_magnitude(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限"):
            finite_time_conditional_lyapunov_exponent((math.inf,))

    def test_rejects_invalid_derivative_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "derivative_floor"):
            finite_time_conditional_lyapunov_exponent(
                (0.5,),
                derivative_floor=0.0,
            )

    def test_rejects_invalid_washout(self) -> None:
        with self.assertRaisesRegex(ValueError, "washout"):
            finite_time_conditional_lyapunov_exponent(
                (0.5,),
                washout=-1,
            )


if __name__ == "__main__":
    unittest.main()
