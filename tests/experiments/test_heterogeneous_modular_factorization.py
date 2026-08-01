import unittest

from reservoir_dynamics.experiments.heterogeneous_modular_factorization import (
    audit_heterogeneous_modular_structure,
    run_heterogeneous_modular_factorization,
)


class HeterogeneousModularFactorizationTest(unittest.TestCase):
    def test_preregistered_structure_gate_has_thirty_classes_per_gain(
        self,
    ) -> None:
        gate = audit_heterogeneous_modular_structure(
            trial_seeds=tuple(range(1401, 1431)),
            coupling_gains=(0.05, 0.07),
        )

        self.assertTrue(gate.passed)
        self.assertEqual(gate.raw_network_counts, (30, 30))
        self.assertEqual(gate.effective_class_counts, (30, 30))
        self.assertTrue(gate.components_valid)
        self.assertTrue(gate.magnitude_pairs_unique)

    def test_small_independent_module_run_satisfies_product_decisions(
        self,
    ) -> None:
        result = run_heterogeneous_modular_factorization(
            trial_seeds=(1501, 1502, 1503, 1504),
            coupling_gains=(0.05,),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )

        self.assertEqual(len(result.points), 4)
        self.assertTrue(result.decisions.effective_structure_gate)
        self.assertTrue(result.decisions.fixed_point_product)
        self.assertTrue(result.decisions.component_margin_product)
        self.assertTrue(result.decisions.global_certificate_is_conservative)
        self.assertTrue(result.decisions.task_retention_product)
        self.assertTrue(result.decisions.certificate_lower_bound_valid)
        self.assertLessEqual(result.maximum_task_product_residual, 1e-12)

    def test_rejects_odd_dimension_before_structure_audit(self) -> None:
        with self.assertRaisesRegex(ValueError, "偶数"):
            audit_heterogeneous_modular_structure(
                trial_seeds=(1401, 1402),
                coupling_gains=(0.05,),
                dimension=3,
            )


if __name__ == "__main__":
    unittest.main()
