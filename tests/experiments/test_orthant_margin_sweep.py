import unittest

from reservoir_dynamics.experiments.orthant_margin_sweep import (
    EXPERIMENT_ID,
    run_orthant_margin_sweep,
)


class OrthantMarginSweepTest(unittest.TestCase):
    def test_same_attractor_count_has_different_robust_repertoire(self) -> None:
        result = run_orthant_margin_sweep(
            trial_seeds=(31, 37),
            dimension=4,
            diagonal_gain=1.5,
            coupling_gains=(0.04, 0.08),
            safe_trials=4,
            simulation_steps=60,
            bootstrap_resamples=100,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.points), 64)
        self.assertTrue(result.decisions.autonomous_repertoire_preserved)
        self.assertTrue(result.decisions.robust_repertoire_separated)
        self.assertTrue(result.decisions.safe_box_invariance)
        self.assertTrue(result.decisions.adversarial_boundary_witness)

        weak, strong = result.coupling_summaries
        self.assertEqual(weak.autonomous_repertoire_count.estimate, 16.0)
        self.assertEqual(strong.autonomous_repertoire_count.estimate, 16.0)
        self.assertEqual(
            weak.certified_robust_repertoire_count.estimate,
            16.0,
        )
        self.assertLess(
            strong.certified_robust_repertoire_count.estimate,
            16.0,
        )

    def test_study_is_deterministic(self) -> None:
        arguments = {
            "trial_seeds": (41, 43),
            "dimension": 3,
            "coupling_gains": (0.04, 0.08),
            "safe_trials": 2,
            "simulation_steps": 30,
            "bootstrap_resamples": 50,
        }

        first = run_orthant_margin_sweep(**arguments)
        second = run_orthant_margin_sweep(**arguments)

        self.assertEqual(first, second)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial_seeds"):
            run_orthant_margin_sweep(trial_seeds=(31,))
        with self.assertRaisesRegex(ValueError, "coupling_gains"):
            run_orthant_margin_sweep(coupling_gains=(0.08, 0.04))
        with self.assertRaisesRegex(ValueError, "dimension"):
            run_orthant_margin_sweep(dimension=1)


if __name__ == "__main__":
    unittest.main()
