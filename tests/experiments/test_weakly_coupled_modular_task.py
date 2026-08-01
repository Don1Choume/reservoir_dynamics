import unittest

from reservoir_dynamics.experiments.weakly_coupled_modular_task import (
    audit_weak_coupling_structure,
    build_weakly_coupled_modular_weights,
    run_weak_coupling_factorization,
)
from reservoir_dynamics.experiments.weakly_coupled_modular_cli import (
    summary_payload,
)
from reservoir_dynamics.metrics.structural_equivalence import (
    weakly_connected_components,
)
from reservoir_dynamics.theory.orthant_rectangle import (
    matrix_infinity_norm_difference,
)


class WeaklyCoupledModularTaskTest(unittest.TestCase):
    def test_cross_coupling_has_declared_norm_and_connects_modules(self) -> None:
        base = build_weakly_coupled_modular_weights(
            trial_seed=1501,
            internal_coupling_gain=0.05,
            cross_coupling_strength=0.0,
        )
        coupled = build_weakly_coupled_modular_weights(
            trial_seed=1501,
            internal_coupling_gain=0.05,
            cross_coupling_strength=0.02,
        )

        self.assertAlmostEqual(
            matrix_infinity_norm_difference(base, coupled),
            0.02,
        )
        self.assertEqual(
            weakly_connected_components(base),
            ((0, 1), (2, 3)),
        )
        self.assertEqual(
            weakly_connected_components(coupled),
            ((0, 1, 2, 3),),
        )

    def test_structure_gate_and_small_factorization_run(self) -> None:
        arguments = {
            "trial_seeds": (1501, 1502),
            "internal_coupling_gains": (0.05,),
            "cross_coupling_strengths": (0.0, 0.02),
        }
        gate = audit_weak_coupling_structure(**arguments)

        self.assertTrue(gate.passed)
        self.assertEqual(gate.raw_network_counts, (2, 2))
        self.assertEqual(gate.effective_class_counts, (2, 2))

        result = run_weak_coupling_factorization(
            phase="pilot",
            disturbance_bounds=(0.12,),
            task_steps=20,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
            **arguments,
        )

        self.assertEqual(len(result.points), 4)
        self.assertTrue(result.decisions.structure_gate)
        self.assertTrue(result.decisions.zero_coupling_recovers_product)
        self.assertTrue(result.decisions.transported_certificate_lower_bound)
        self.assertTrue(result.decisions.norm_shifted_certificate_lower_bound)
        self.assertTrue(result.decisions.transported_dominates_norm_shifted)
        self.assertIsInstance(
            result.decisions.maximum_strength_mean_absolute_residual,
            bool,
        )
        self.assertIsInstance(
            result.decisions.maximum_strength_nonzero_residual_prevalence,
            bool,
        )
        self.assertIsInstance(
            result.decisions.mean_absolute_residual_non_decreasing,
            bool,
        )
        zero_points = tuple(
            point for point in result.points
            if point.cross_coupling_strength == 0.0
        )
        self.assertTrue(
            all(point.task_product_residual == 0.0 for point in zero_points)
        )
        summary = summary_payload(result)
        self.assertEqual(summary["point_count"], 4)
        self.assertEqual(summary["challenge_count"], 1_024)
        self.assertEqual(len(summary["strength_summaries"]), 2)

    def test_rejects_nonzero_first_strength(self) -> None:
        with self.assertRaisesRegex(ValueError, "0から"):
            audit_weak_coupling_structure(
                trial_seeds=(1501, 1502),
                internal_coupling_gains=(0.05,),
                cross_coupling_strengths=(0.01, 0.02),
            )


if __name__ == "__main__":
    unittest.main()
