import unittest

from reservoir_dynamics.experiments.cross_family_robust_task_confirmation import (
    EXPERIMENT_ID,
    FamilyTaskSpecification,
    run_cross_family_robust_task_confirmation,
)


class CrossFamilyRobustTaskConfirmationTest(unittest.TestCase):
    def test_compares_disjoint_seeds_across_network_families(self) -> None:
        result = run_cross_family_robust_task_confirmation(
            discovery_seeds=(1001, 1003, 1007, 1009),
            confirmation_seeds=(1103, 1109, 1117, 1123),
            family_specifications=(
                FamilyTaskSpecification(
                    network_family="dense_symmetric",
                    coupling_gains=(0.04, 0.07),
                    disturbance_bound=0.12,
                ),
                FamilyTaskSpecification(
                    network_family="feedforward_nonnormal",
                    coupling_gains=(0.04, 0.07),
                    disturbance_bound=0.12,
                ),
            ),
            dimension=3,
            task_steps=20,
            autonomous_steps=300,
            bootstrap_resamples=20,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.family_comparisons), 2)
        self.assertEqual(len(result.points), 32)
        self.assertTrue(result.decisions.raw_count_matched)
        self.assertTrue(result.decisions.certificate_lower_bound_valid)
        self.assertTrue(
            all(
                comparison.comparison.guarantee_violation_count == 0
                for comparison in result.family_comparisons
            )
        )
        self.assertTrue(
            all(
                comparison.local_jacobian_test_mae >= 0.0
                and comparison.off_diagonal_norm_test_mae >= 0.0
                and comparison.minimum_coordinate_test_mae >= 0.0
                and comparison.nonnormality_test_mae >= 0.0
                for comparison in result.family_comparisons
            )
        )
        self.assertGreaterEqual(
            result.pooled_local_jacobian_minus_margin_error.resamples,
            20,
        )

    def test_rejects_seed_overlap_and_duplicate_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "重複"):
            run_cross_family_robust_task_confirmation(
                discovery_seeds=(1001, 1003, 1007, 1009),
                confirmation_seeds=(1009, 1103, 1109, 1117),
            )

        duplicate = FamilyTaskSpecification(
            network_family="dense_symmetric",
            coupling_gains=(0.04, 0.07),
            disturbance_bound=0.12,
        )
        with self.assertRaisesRegex(ValueError, "family"):
            run_cross_family_robust_task_confirmation(
                discovery_seeds=(1001, 1003, 1007, 1009),
                confirmation_seeds=(1103, 1109, 1117, 1123),
                family_specifications=(duplicate, duplicate),
            )


if __name__ == "__main__":
    unittest.main()
