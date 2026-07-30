import unittest

from reservoir_dynamics.metrics.network_diagnostics import (
    local_jacobian_infinity_norm,
    matrix_nonnormality_commutator_norm,
    off_diagonal_infinity_norm,
    signed_minimum_coordinate,
)


class NetworkDiagnosticsTest(unittest.TestCase):
    def test_diagonal_matrix_has_zero_structural_baselines(self) -> None:
        matrix = ((1.5, 0.0), (0.0, 1.5))

        self.assertEqual(off_diagonal_infinity_norm(matrix), 0.0)
        self.assertEqual(
            matrix_nonnormality_commutator_norm(matrix),
            0.0,
        )

    def test_feedforward_matrix_is_nonnormal(self) -> None:
        matrix = ((1.5, 0.2), (0.0, 1.5))

        self.assertAlmostEqual(
            off_diagonal_infinity_norm(matrix),
            0.2,
        )
        self.assertGreater(
            matrix_nonnormality_commutator_norm(matrix),
            0.0,
        )

    def test_local_jacobian_uses_tanh_derivative(self) -> None:
        matrix = ((1.5, 0.2), (0.0, 1.5))

        at_origin = local_jacobian_infinity_norm(
            recurrent_weights=matrix,
            state=(0.0, 0.0),
        )
        at_saturation = local_jacobian_infinity_norm(
            recurrent_weights=matrix,
            state=(0.8, 0.8),
        )

        self.assertAlmostEqual(at_origin, 1.7)
        self.assertAlmostEqual(at_saturation, 1.7 * (1.0 - 0.8**2))

    def test_signed_minimum_coordinate_uses_target_orthant(self) -> None:
        self.assertAlmostEqual(
            signed_minimum_coordinate(
                state=(0.7, -0.4, 0.8),
                attractor_signs=(1, -1, 1),
            ),
            0.4,
        )

        with self.assertRaisesRegex(ValueError, "次元"):
            signed_minimum_coordinate(
                state=(0.7, -0.4),
                attractor_signs=(1,),
            )


if __name__ == "__main__":
    unittest.main()
