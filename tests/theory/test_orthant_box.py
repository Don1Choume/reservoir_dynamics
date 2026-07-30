import unittest

from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
)
from reservoir_dynamics.theory.orthant_box import (
    robust_orthant_box_certificate,
)


class RobustOrthantBoxCertificateTest(unittest.TestCase):
    def test_reduces_to_scalar_bistable_margin(self) -> None:
        scalar = bistable_tanh_certificate(1.5)
        certificate = robust_orthant_box_certificate(
            recurrent_weights=((1.5,),),
            attractor_signs=(1,),
        )

        self.assertAlmostEqual(
            certificate.invariant_boundary,
            scalar.invariant_boundary,
            places=9,
        )
        self.assertAlmostEqual(
            certificate.maximum_uniform_disturbance,
            scalar.critical_forcing,
            places=9,
        )
        self.assertEqual(certificate.limiting_coordinates, (0,))
        self.assertTrue(certificate.is_certified)

    def test_distinguishes_aligned_and_opposed_coupled_modes(self) -> None:
        recurrent_weights = (
            (1.5, 0.1),
            (0.1, 1.5),
        )
        aligned = robust_orthant_box_certificate(
            recurrent_weights=recurrent_weights,
            attractor_signs=(1, 1),
        )
        opposed = robust_orthant_box_certificate(
            recurrent_weights=recurrent_weights,
            attractor_signs=(1, -1),
        )

        expected_aligned = bistable_tanh_certificate(1.6)
        expected_opposed_margin = (
            bistable_tanh_certificate(1.5).critical_forcing - 0.1
        )
        self.assertAlmostEqual(
            aligned.maximum_uniform_disturbance,
            expected_aligned.critical_forcing,
            places=9,
        )
        self.assertAlmostEqual(
            opposed.maximum_uniform_disturbance,
            expected_opposed_margin,
            places=9,
        )
        self.assertGreater(
            aligned.maximum_uniform_disturbance,
            opposed.maximum_uniform_disturbance,
        )

    def test_reports_zero_when_no_common_inner_box_is_certified(self) -> None:
        certificate = robust_orthant_box_certificate(
            recurrent_weights=(
                (1.5, 0.25),
                (0.25, 1.5),
            ),
            attractor_signs=(1, -1),
        )

        self.assertFalse(certificate.is_certified)
        self.assertEqual(certificate.maximum_uniform_disturbance, 0.0)
        self.assertLess(certificate.raw_uniform_disturbance_margin, 0.0)

    def test_rejects_invalid_matrix_signs_and_search_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "正方"):
            robust_orthant_box_certificate(
                recurrent_weights=((1.5, 0.0),),
                attractor_signs=(1,),
            )
        with self.assertRaisesRegex(ValueError, "attractor_signs"):
            robust_orthant_box_certificate(
                recurrent_weights=((1.5,),),
                attractor_signs=(0,),
            )
        with self.assertRaisesRegex(ValueError, "search_iterations"):
            robust_orthant_box_certificate(
                recurrent_weights=((1.5,),),
                attractor_signs=(1,),
                search_iterations=True,
            )


if __name__ == "__main__":
    unittest.main()
