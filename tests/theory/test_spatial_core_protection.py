import unittest

from reservoir_dynamics.theory.spatial_core_protection import (
    bistable_coordinate_protection,
    energy_matched_global_weights,
    matrix_frobenius_distance_squared,
    row_gated_matrix,
    time_varying_core_deviation_bound,
)


class SpatialCoreProtectionTest(unittest.TestCase):
    def test_row_gate_and_global_control_have_matched_energy(self) -> None:
        feedback = ((0.2, -0.1), (0.05, 0.15))
        gates = (0.25, 0.75)
        gated_feedback = row_gated_matrix(feedback, gates)
        local_energy = matrix_frobenius_distance_squared(
            feedback,
            gated_feedback,
        )
        full_recurrent = (
            (1.5, 0.0, 0.2, -0.1),
            (0.0, 1.5, 0.05, 0.15),
            (0.1, 0.0, 0.6, 0.2),
            (0.0, 0.1, -0.1, 0.5),
        )

        global_weights = energy_matched_global_weights(
            full_recurrent,
            target_energy=local_energy,
        )

        self.assertAlmostEqual(
            matrix_frobenius_distance_squared(
                full_recurrent,
                global_weights,
            ),
            local_energy,
            places=14,
        )
        expected_feedback = ((0.05, -0.025), (0.0375, 0.1125))
        for observed_row, expected_row in zip(
            gated_feedback,
            expected_feedback,
            strict=True,
        ):
            for observed, expected in zip(
                observed_row,
                expected_row,
                strict=True,
            ):
                self.assertAlmostEqual(observed, expected)

    def test_bistable_certificate_uses_signed_load_and_noise_bound(self) -> None:
        certificate = bistable_coordinate_protection(
            recurrent_gains=(1.5, 1.5),
            feedback_loads=(0.01, -0.02),
            disturbance_bounds=(0.01, 0.01),
        )

        self.assertTrue(certificate.all_certified)
        self.assertEqual(certificate.applied_forcing_bounds, (0.02, 0.03))
        self.assertGreater(certificate.minimum_margin, 0.0)

        failed = bistable_coordinate_protection(
            recurrent_gains=(1.1,),
            feedback_loads=(0.1,),
            disturbance_bounds=(0.0,),
        )
        self.assertFalse(failed.all_certified)

    def test_time_varying_bound_accumulates_realized_loads(self) -> None:
        result = time_varying_core_deviation_bound(
            core_lipschitz=0.5,
            forcing_loads=(0.2, 0.1, 0.0),
            initial_core_distance=0.4,
        )

        self.assertEqual(result, (0.4, 0.4, 0.30000000000000004, 0.15000000000000002))

    def test_rejects_invalid_shapes_and_energy(self) -> None:
        with self.assertRaisesRegex(ValueError, "gates"):
            row_gated_matrix(((1.0, 0.0),), (0.5, 0.5))
        with self.assertRaisesRegex(ValueError, "target_energy"):
            energy_matched_global_weights(((1.0,),), target_energy=-0.1)
        with self.assertRaisesRegex(ValueError, "次元"):
            bistable_coordinate_protection(
                recurrent_gains=(1.5,),
                feedback_loads=(0.0, 0.0),
                disturbance_bounds=(0.0,),
            )


if __name__ == "__main__":
    unittest.main()
