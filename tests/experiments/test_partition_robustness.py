import unittest

from reservoir_dynamics.experiments.partition_robustness import (
    PREREGISTERED_DEVELOPMENT_SEEDS,
    PREREGISTERED_RELATIVE_AMPLITUDES,
    run_partition_robustness_confirmation,
    run_partition_robustness_development,
)
from reservoir_dynamics.experiments.partition_robustness_cli import (
    structure_result_payload,
)


class PartitionRobustnessExperimentTest(unittest.TestCase):
    def test_small_task_free_grid_certifies_every_subradius_partition(self) -> None:
        result = run_partition_robustness_development(
            trial_seeds=(2401, 2402),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.02,),
            perturbation_direction_count=2,
            relative_amplitudes=(0.0, 0.9, 1.1),
        )

        self.assertEqual(result.phase, "development")
        self.assertEqual(result.base_network_count, 2)
        self.assertEqual(len(result.points), 12)
        self.assertFalse(result.task_values_generated)
        self.assertTrue(result.decisions.base_recovery)
        self.assertTrue(result.decisions.positive_radius)
        self.assertTrue(result.decisions.affinity_lipschitz)
        self.assertTrue(result.decisions.subradius_exactness)
        self.assertTrue(result.decisions.subradius_pair_distance)
        self.assertTrue(result.decisions.strict_boundary)
        self.assertTrue(result.decisions.task_free)
        self.assertTrue(result.decisions.seed_independence)
        self.assertTrue(result.decisions.all_passed)
        subradius_points = tuple(
            point for point in result.points if point.relative_amplitude < 1.0
        )
        self.assertTrue(subradius_points)
        self.assertTrue(all(point.partition_recovered for point in subradius_points))
        self.assertTrue(
            all(point.pair_disagreement == 0.0 for point in subradius_points)
        )
        self.assertTrue(
            all(
                point.maximum_affinity_change
                <= point.absolute_amplitude + 1e-12
                for point in result.points
            )
        )

    def test_confirmation_rejects_development_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "development"):
            run_partition_robustness_confirmation(
                trial_seeds=(PREREGISTERED_DEVELOPMENT_SEEDS[0], 2501),
                internal_gains=(0.025,),
                maximum_bridge_strengths=(0.02,),
                perturbation_direction_count=1,
                relative_amplitudes=(0.0, 0.9, 1.1),
            )

    def test_registered_amplitudes_exclude_unproved_equality_boundary(self) -> None:
        self.assertNotIn(1.0, PREREGISTERED_RELATIVE_AMPLITUDES)
        self.assertTrue(any(value < 1.0 for value in PREREGISTERED_RELATIVE_AMPLITUDES))
        self.assertTrue(any(value > 1.0 for value in PREREGISTERED_RELATIVE_AMPLITUDES))

    def test_rejects_grid_without_both_sides_of_radius(self) -> None:
        with self.assertRaisesRegex(ValueError, "半径"):
            run_partition_robustness_development(
                trial_seeds=(2401, 2402),
                internal_gains=(0.025,),
                maximum_bridge_strengths=(0.02,),
                perturbation_direction_count=1,
                relative_amplitudes=(0.0, 0.5),
            )

    def test_payload_preserves_decisions_and_amplitude_summaries(self) -> None:
        result = run_partition_robustness_development(
            trial_seeds=(2401, 2402),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.02,),
            perturbation_direction_count=2,
            relative_amplitudes=(0.0, 0.9, 1.1),
        )

        payload = structure_result_payload(result)

        self.assertEqual(payload["experiment_id"], "EXP-2026-018")
        self.assertFalse(payload["task_values_generated"])
        self.assertEqual(payload["base_network_count"], 2)
        self.assertEqual(payload["point_count"], 12)
        self.assertTrue(payload["decisions"]["all_passed"])
        self.assertEqual(len(payload["amplitude_summaries"]), 3)
        subradius = tuple(
            value
            for value in payload["amplitude_summaries"]
            if value["relative_amplitude"] < 1.0
        )
        self.assertTrue(
            all(value["partition_recovery_rate"] == 1.0 for value in subradius)
        )
        self.assertEqual(len(payload["points"]), 12)


if __name__ == "__main__":
    unittest.main()
