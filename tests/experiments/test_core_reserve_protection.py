import unittest

from reservoir_dynamics.experiments.core_reserve_protection import (
    EXPERIMENT_ID,
    run_core_reserve_protection_study,
)


class CoreReserveProtectionStudyTest(unittest.TestCase):
    def test_small_study_separates_protected_and_entangled_updates(self) -> None:
        result = run_core_reserve_protection_study(
            trial_seeds=(13, 17),
            core_dimension=2,
            reserve_dimension=2,
            feedback_gains=(0.0, 0.05),
            adaptation_candidates=((0.3, 0.5), (0.6, 0.5)),
            calibration_washout=10,
            calibration_training_steps=40,
            calibration_testing_steps=20,
            evaluation_washout=10,
            evaluation_training_steps=50,
            evaluation_testing_steps=30,
            max_delay=3,
            bootstrap_resamples=100,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.protected_points), 4)
        self.assertEqual(len(result.entangled_points), 2)
        zero_feedback_points = tuple(
            point
            for point in result.protected_points
            if point.feedback_gain == 0.0
        )
        self.assertTrue(
            all(point.max_core_deviation == 0.0 for point in zero_feedback_points)
        )
        self.assertTrue(
            all(
                point.core_retention == 1.0
                for point in zero_feedback_points
            )
        )
        self.assertTrue(
            all(point.bound_satisfied for point in result.protected_points)
        )
        self.assertTrue(
            all(
                point.post_novel_capacity > point.pre_novel_capacity
                for point in result.protected_points
            )
        )

    def test_small_study_is_deterministic(self) -> None:
        arguments = {
            "trial_seeds": (19, 23),
            "core_dimension": 2,
            "reserve_dimension": 2,
            "feedback_gains": (0.0,),
            "adaptation_candidates": ((0.3, 0.5),),
            "calibration_washout": 5,
            "calibration_training_steps": 20,
            "calibration_testing_steps": 10,
            "evaluation_washout": 5,
            "evaluation_training_steps": 25,
            "evaluation_testing_steps": 15,
            "max_delay": 2,
            "bootstrap_resamples": 50,
        }

        first = run_core_reserve_protection_study(**arguments)
        second = run_core_reserve_protection_study(**arguments)

        self.assertEqual(first, second)

    def test_rejects_mismatched_dimensions_and_single_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial_seeds"):
            run_core_reserve_protection_study(trial_seeds=(11,))

        with self.assertRaisesRegex(ValueError, "同じ"):
            run_core_reserve_protection_study(
                core_dimension=2,
                reserve_dimension=3,
            )


if __name__ == "__main__":
    unittest.main()
