import unittest

from reservoir_dynamics.experiments.aligned_sign_memory import (
    ALIGNED_DIRECTION_CODES,
    evaluate_aligned_sign_memory_network,
)


class AlignedSignMemoryTest(unittest.TestCase):
    def test_diagonal_bistable_network_reports_four_aligned_codes(self) -> None:
        profile = evaluate_aligned_sign_memory_network(
            recurrent_weights=((1.5, 0.0), (0.0, 1.5)),
            disturbance_bounds=(0.08, 0.16),
            task_steps=20,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )

        self.assertEqual(profile.raw_attractor_count, 4)
        self.assertEqual(len(profile.orthants), 4)
        self.assertGreater(profile.mean_uniform_disturbance_margin, 0.0)
        self.assertEqual(
            tuple(
                evaluation.code
                for evaluation in profile.disturbance_evaluations[0]
                .direction_evaluations
            ),
            ALIGNED_DIRECTION_CODES,
        )
        self.assertTrue(
            all(
                evaluation.total_challenges == 4
                for disturbance in profile.disturbance_evaluations
                for evaluation in disturbance.direction_evaluations
            )
        )
        self.assertTrue(
            all(
                disturbance.guarantee_gap >= -1e-12
                for disturbance in profile.disturbance_evaluations
            )
        )

    def test_coordinate_offset_changes_alternating_masks_only(self) -> None:
        first = evaluate_aligned_sign_memory_network(
            recurrent_weights=((1.5,),),
            disturbance_bounds=(0.1,),
            task_steps=5,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
            coordinate_offset=0,
        )
        second = evaluate_aligned_sign_memory_network(
            recurrent_weights=((1.5,),),
            disturbance_bounds=(0.1,),
            task_steps=5,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
            coordinate_offset=1,
        )

        first_by_code = {
            value.code: value.task_retention
            for value in first.disturbance_evaluations[0]
            .direction_evaluations
        }
        second_by_code = {
            value.code: value.task_retention
            for value in second.disturbance_evaluations[0]
            .direction_evaluations
        }
        self.assertEqual(first_by_code["inward"], second_by_code["inward"])
        self.assertEqual(first_by_code["outward"], second_by_code["outward"])
        self.assertEqual(
            first_by_code["alternating"],
            second_by_code["reverse_alternating"],
        )

    def test_explicit_coordinate_indices_align_noncontiguous_component_masks(self) -> None:
        profile = evaluate_aligned_sign_memory_network(
            recurrent_weights=((1.5, 0.0), (0.0, 1.5)),
            disturbance_bounds=(0.1,),
            task_steps=2,
            autonomous_steps=100,
            convergence_tolerance=1e-9,
            coordinate_indices=(1, 4),
        )

        self.assertEqual(profile.coordinate_indices, (1, 4))
        self.assertEqual(profile.coordinate_offset, 0)

    def test_rejects_duplicate_coordinate_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "coordinate_indices"):
            evaluate_aligned_sign_memory_network(
                recurrent_weights=((1.5, 0.0), (0.0, 1.5)),
                disturbance_bounds=(0.1,),
                task_steps=2,
                autonomous_steps=100,
                convergence_tolerance=1e-9,
                coordinate_indices=(2, 2),
            )


if __name__ == "__main__":
    unittest.main()
