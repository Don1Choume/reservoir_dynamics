import unittest

from reservoir_dynamics.experiments.bistable_core_protection import (
    EXPERIMENT_ID,
    run_bistable_core_protection_study,
)


class BistableCoreProtectionStudyTest(unittest.TestCase):
    def test_small_study_separates_safe_and_tipping_regimes(self) -> None:
        result = run_bistable_core_protection_study(
            trial_seeds=(13, 17),
            analytical_recurrent_gains=(1.2, 1.5),
            feedback_ratios=(0.5, 1.5),
            adaptation_candidates=((1.2, 0.75), (1.5, 1.5)),
            calibration_trials=16,
            evaluation_trials=32,
            analytical_steps=1_000,
            evaluation_steps=300,
            bootstrap_resamples=100,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.analytical_points), 4)
        self.assertEqual(len(result.protection_points), 4)
        self.assertTrue(result.decisions.analytic_safe_invariance)
        self.assertTrue(result.decisions.analytic_threshold_tipping)
        self.assertTrue(result.decisions.reserve_attractor_acquisition)
        self.assertTrue(result.decisions.certified_core_preservation)
        self.assertTrue(result.decisions.margin_predicts_failure)

        safe_points = tuple(
            point
            for point in result.protection_points
            if point.feedback_ratio < 1.0
        )
        unsafe_points = tuple(
            point
            for point in result.protection_points
            if point.feedback_ratio > 1.0
        )
        self.assertTrue(
            all(
                point.certified_core_retention == 1.0
                for point in safe_points
            )
        )
        self.assertTrue(
            all(
                point.opposing_core_retention < 0.5
                for point in unsafe_points
            )
        )

    def test_study_is_deterministic(self) -> None:
        arguments = {
            "trial_seeds": (19, 23),
            "analytical_recurrent_gains": (1.5,),
            "feedback_ratios": (0.5, 1.5),
            "adaptation_candidates": ((1.3, 1.0),),
            "calibration_trials": 8,
            "evaluation_trials": 12,
            "analytical_steps": 500,
            "evaluation_steps": 200,
            "bootstrap_resamples": 50,
        }

        first = run_bistable_core_protection_study(**arguments)
        second = run_bistable_core_protection_study(**arguments)

        self.assertEqual(first, second)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial_seeds"):
            run_bistable_core_protection_study(trial_seeds=(11,))

        with self.assertRaisesRegex(ValueError, "feedback_ratios"):
            run_bistable_core_protection_study(
                feedback_ratios=(0.5, 0.9),
            )

        with self.assertRaisesRegex(ValueError, "adaptation_candidates"):
            run_bistable_core_protection_study(
                adaptation_candidates=((0.9, 1.0),),
            )


if __name__ == "__main__":
    unittest.main()
