import unittest

from reservoir_dynamics.experiments.replica_readout_validation import (
    EXPERIMENT_ID,
    FOCUSED_CONDITIONS,
    run_replica_readout_validation,
)


class ReplicaReadoutValidationTest(unittest.TestCase):
    def test_small_study_reports_seed_intervals_and_paired_contrasts(self) -> None:
        result = run_replica_readout_validation(
            condition_pairs=FOCUSED_CONDITIONS,
            trial_seeds=(17, 23),
            state_dimension=3,
            washout=10,
            training_steps=40,
            testing_steps=20,
            max_delay=3,
            tail_window=5,
            bootstrap_resamples=100,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.points), 12)
        self.assertEqual(len(result.condition_summaries), 6)
        self.assertGreaterEqual(len(result.paired_contrasts), 4)
        self.assertTrue(
            all(
                summary.seed_count == 2
                for summary in result.condition_summaries
            )
        )
        self.assertTrue(
            all(
                summary.local_memory_mean.lower
                <= summary.local_memory_mean.estimate
                <= summary.local_memory_mean.upper
                for summary in result.condition_summaries
            )
        )

    def test_small_study_is_deterministic(self) -> None:
        experiment_arguments = {
            "condition_pairs": ((0.6, 0.1),),
            "trial_seeds": (31, 37),
            "state_dimension": 3,
            "washout": 10,
            "training_steps": 30,
            "testing_steps": 20,
            "max_delay": 3,
            "tail_window": 5,
            "bootstrap_resamples": 100,
        }

        first_result = run_replica_readout_validation(
            **experiment_arguments,
        )
        second_result = run_replica_readout_validation(
            **experiment_arguments,
        )

        self.assertEqual(first_result, second_result)

    def test_rejects_insufficient_seed_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial_seeds"):
            run_replica_readout_validation(trial_seeds=(11,))


if __name__ == "__main__":
    unittest.main()
