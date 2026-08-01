import unittest

from reservoir_dynamics.theory.orthant_box import (
    robust_orthant_box_certificate,
)
from reservoir_dynamics.theory.orthant_rectangle import (
    matrix_infinity_norm_difference,
    orthant_rectangle_certificate,
)


class OrthantRectangleCertificateTest(unittest.TestCase):
    def test_block_diagonal_margin_matches_component_minimum(self) -> None:
        first_module = ((1.5, 0.04), (0.04, 1.5))
        second_module = ((1.5, -0.08), (-0.08, 1.5))
        first = robust_orthant_box_certificate(
            recurrent_weights=first_module,
            attractor_signs=(1, 1),
        )
        second = robust_orthant_box_certificate(
            recurrent_weights=second_module,
            attractor_signs=(1, -1),
        )
        full = orthant_rectangle_certificate(
            recurrent_weights=(
                (1.5, 0.04, 0.0, 0.0),
                (0.04, 1.5, 0.0, 0.0),
                (0.0, 0.0, 1.5, -0.08),
                (0.0, 0.0, -0.08, 1.5),
            ),
            attractor_signs=(1, 1, 1, -1),
            lower_boundaries=(
                first.invariant_boundary,
                first.invariant_boundary,
                second.invariant_boundary,
                second.invariant_boundary,
            ),
        )

        self.assertAlmostEqual(
            full.maximum_uniform_disturbance,
            min(
                first.maximum_uniform_disturbance,
                second.maximum_uniform_disturbance,
            ),
        )

    def test_margin_loss_is_bounded_by_matrix_infinity_norm(self) -> None:
        base_weights = (
            (1.5, 0.05, 0.0, 0.0),
            (0.05, 1.5, 0.0, 0.0),
            (0.0, 0.0, 1.5, -0.07),
            (0.0, 0.0, -0.07, 1.5),
        )
        perturbed_weights = (
            (1.5, 0.05, -0.02, 0.0),
            (0.05, 1.5, 0.0, 0.02),
            (-0.02, 0.0, 1.5, -0.07),
            (0.0, 0.02, -0.07, 1.5),
        )
        arguments = {
            "attractor_signs": (1, -1, 1, 1),
            "lower_boundaries": (0.55, 0.55, 0.60, 0.60),
        }

        base = orthant_rectangle_certificate(
            recurrent_weights=base_weights,
            **arguments,
        )
        perturbed = orthant_rectangle_certificate(
            recurrent_weights=perturbed_weights,
            **arguments,
        )
        perturbation_norm = matrix_infinity_norm_difference(
            base_weights,
            perturbed_weights,
        )

        self.assertAlmostEqual(perturbation_norm, 0.02)
        self.assertGreaterEqual(
            perturbed.raw_uniform_disturbance_margin,
            base.raw_uniform_disturbance_margin - perturbation_norm - 1e-12,
        )

    def test_rejects_invalid_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower_boundaries"):
            orthant_rectangle_certificate(
                recurrent_weights=((1.5, 0.0), (0.0, 1.5)),
                attractor_signs=(1, 1),
                lower_boundaries=(0.5, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
