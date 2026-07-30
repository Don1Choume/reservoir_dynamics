import unittest

from reservoir_dynamics.experiments.robust_repertoire_task import (
    STUDY_ID,
    run_robust_repertoire_task_diagnostics,
)


class RobustRepertoireTaskDiagnosticsTest(unittest.TestCase):
    def test_certified_fraction_lower_bounds_sign_memory_task(self) -> None:
        result = run_robust_repertoire_task_diagnostics(
            trial_seeds=(701, 703, 709, 719),
            dimension=3,
            coupling_gains=(0.04, 0.07),
            disturbance_bounds=(0.08, 0.16),
            task_steps=30,
            autonomous_steps=300,
            training_seed_count=2,
        )

        self.assertEqual(result.study_id, STUDY_ID)
        self.assertEqual(len(result.points), 16)
        self.assertEqual(len(result.predictive_summaries), 2)
        self.assertTrue(
            all(point.raw_attractor_count == 8 for point in result.points)
        )
        self.assertTrue(
            all(
                point.off_diagonal_infinity_norm > 0.0
                and point.maximum_local_jacobian_infinity_norm > 0.0
                and point.minimum_fixed_point_coordinate > 0.0
                for point in result.points
            )
        )
        self.assertTrue(
            all(
                point.nonnormality_commutator_norm == 0.0
                for point in result.points
            )
        )
        self.assertTrue(
            all(point.guarantee_gap >= -1e-12 for point in result.points)
        )
        self.assertTrue(
            all(
                summary.guarantee_violation_count == 0
                for summary in result.predictive_summaries
            )
        )

    def test_diagnostics_are_deterministic(self) -> None:
        arguments = {
            "trial_seeds": (727, 733, 739, 743),
            "dimension": 3,
            "coupling_gains": (0.04, 0.07),
            "disturbance_bounds": (0.1,),
            "task_steps": 20,
            "autonomous_steps": 300,
            "training_seed_count": 2,
        }

        first = run_robust_repertoire_task_diagnostics(**arguments)
        second = run_robust_repertoire_task_diagnostics(**arguments)

        self.assertEqual(first, second)

    def test_supports_nonnormal_network_family(self) -> None:
        result = run_robust_repertoire_task_diagnostics(
            trial_seeds=(751, 757, 761, 769),
            network_family="feedforward_nonnormal",
            dimension=3,
            coupling_gains=(0.04, 0.07),
            disturbance_bounds=(0.1,),
            task_steps=20,
            autonomous_steps=300,
            training_seed_count=2,
        )

        self.assertEqual(result.network_family, "feedforward_nonnormal")
        self.assertTrue(
            all(
                point.network_family == "feedforward_nonnormal"
                for point in result.points
            )
        )
        self.assertTrue(
            all(point.raw_attractor_count == 8 for point in result.points)
        )
        self.assertTrue(
            all(
                point.nonnormality_commutator_norm > 0.0
                for point in result.points
            )
        )

    def test_rejects_invalid_split_and_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_seed_count"):
            run_robust_repertoire_task_diagnostics(
                trial_seeds=(701, 703, 709, 719),
                training_seed_count=3,
            )

        with self.assertRaisesRegex(ValueError, "coupling_gains"):
            run_robust_repertoire_task_diagnostics(
                coupling_gains=(0.07, 0.04),
            )

        with self.assertRaisesRegex(ValueError, "disturbance_bounds"):
            run_robust_repertoire_task_diagnostics(
                disturbance_bounds=(0.1, 0.1),
            )

        with self.assertRaisesRegex(ValueError, "network_family"):
            run_robust_repertoire_task_diagnostics(
                network_family="unknown",
            )


if __name__ == "__main__":
    unittest.main()
