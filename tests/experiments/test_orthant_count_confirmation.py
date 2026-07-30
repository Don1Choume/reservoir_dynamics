import unittest

from reservoir_dynamics.experiments.orthant_count_confirmation import (
    EXPERIMENT_ID,
    run_orthant_count_confirmation_study,
)


class OrthantCountConfirmationTest(unittest.TestCase):
    def test_new_seeds_confirm_count_matched_robust_separation(self) -> None:
        result = run_orthant_count_confirmation_study(
            trial_seeds=(503, 509),
            safe_trials=2,
            simulation_steps=40,
            autonomous_steps=300,
            bootstrap_resamples=50,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(result.coupling_gains, (0.04, 0.07))
        self.assertTrue(result.decisions.autonomous_repertoire_preserved)
        self.assertTrue(result.decisions.robust_repertoire_separated)
        self.assertTrue(result.decisions.safe_box_invariance)
        self.assertTrue(result.decisions.adversarial_boundary_witness)

    def test_rejects_discovery_seed_reuse(self) -> None:
        with self.assertRaisesRegex(ValueError, "発見用seed"):
            run_orthant_count_confirmation_study(
                trial_seeds=(401, 503),
            )

    def test_confirmation_is_deterministic(self) -> None:
        arguments = {
            "trial_seeds": (521, 523),
            "safe_trials": 1,
            "simulation_steps": 20,
            "autonomous_steps": 300,
            "bootstrap_resamples": 20,
        }

        first = run_orthant_count_confirmation_study(**arguments)
        second = run_orthant_count_confirmation_study(**arguments)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
