import unittest

from reservoir_dynamics.experiments.robust_repertoire_task_confirmation import (
    EXPERIMENT_ID,
    run_robust_repertoire_task_confirmation,
)


class RobustRepertoireTaskConfirmationTest(unittest.TestCase):
    def test_builds_disjoint_train_confirmation_comparison(self) -> None:
        result = run_robust_repertoire_task_confirmation(
            discovery_seeds=(801, 803, 807, 809),
            confirmation_seeds=(901, 907, 911, 919),
            dimension=3,
            coupling_gains=(0.04, 0.07),
            disturbance_bounds=(0.08, 0.16),
            task_steps=20,
            autonomous_steps=300,
            bootstrap_resamples=20,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.comparisons), 2)
        self.assertTrue(result.decisions.raw_count_matched)
        self.assertTrue(result.decisions.certificate_lower_bound_valid)
        self.assertTrue(
            all(
                comparison.guarantee_violation_count == 0
                for comparison in result.comparisons
            )
        )

    def test_rejects_overlapping_discovery_and_confirmation_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "重複"):
            run_robust_repertoire_task_confirmation(
                discovery_seeds=(801, 803, 807, 809),
                confirmation_seeds=(809, 901, 907, 911),
            )

    def test_confirmation_is_deterministic(self) -> None:
        arguments = {
            "discovery_seeds": (821, 823, 827, 829),
            "confirmation_seeds": (921, 929, 937, 941),
            "dimension": 3,
            "coupling_gains": (0.04, 0.07),
            "disturbance_bounds": (0.1,),
            "task_steps": 15,
            "autonomous_steps": 300,
            "bootstrap_resamples": 10,
        }

        first = run_robust_repertoire_task_confirmation(**arguments)
        second = run_robust_repertoire_task_confirmation(**arguments)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
