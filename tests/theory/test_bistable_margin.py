import math
import unittest

from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
    positive_bistable_fixed_point,
)


class BistableTanhCertificateTest(unittest.TestCase):
    def test_derives_tangency_and_robust_invariant_boundary(self) -> None:
        certificate = bistable_tanh_certificate(1.5)

        expected_boundary = math.sqrt(1.0 / 3.0)
        expected_forcing = (
            1.5 * expected_boundary - math.atanh(expected_boundary)
        )
        self.assertAlmostEqual(
            certificate.invariant_boundary,
            expected_boundary,
        )
        self.assertAlmostEqual(
            certificate.critical_forcing,
            expected_forcing,
        )
        self.assertAlmostEqual(
            1.5 * (1.0 - certificate.invariant_boundary**2),
            1.0,
        )
        self.assertAlmostEqual(
            math.tanh(
                1.5 * certificate.invariant_boundary
                - certificate.critical_forcing
            ),
            certificate.invariant_boundary,
        )
        self.assertAlmostEqual(
            certificate.certified_uniform_fraction,
            1.0 - expected_boundary,
        )

    def test_positive_fixed_point_is_outside_certified_boundary(self) -> None:
        fixed_point = positive_bistable_fixed_point(1.5)
        certificate = bistable_tanh_certificate(1.5)

        self.assertGreater(
            fixed_point,
            certificate.invariant_boundary,
        )
        self.assertAlmostEqual(
            fixed_point,
            math.tanh(1.5 * fixed_point),
            places=11,
        )

    def test_rejects_non_bistable_gain_and_invalid_solver_settings(self) -> None:
        for invalid_gain in (1.0, 0.5, math.inf, math.nan):
            with self.subTest(recurrent_gain=invalid_gain):
                with self.assertRaisesRegex(ValueError, "recurrent_gain"):
                    bistable_tanh_certificate(invalid_gain)

        with self.assertRaisesRegex(ValueError, "tolerance"):
            positive_bistable_fixed_point(1.5, tolerance=0.0)
        with self.assertRaisesRegex(ValueError, "max_iterations"):
            positive_bistable_fixed_point(1.5, max_iterations=True)


if __name__ == "__main__":
    unittest.main()
