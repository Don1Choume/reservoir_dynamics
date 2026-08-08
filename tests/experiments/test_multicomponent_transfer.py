import unittest
from pathlib import Path

from reservoir_dynamics.experiments.multicomponent_transfer import (
    PREREGISTERED_DEVELOPMENT_SEEDS,
    run_multicomponent_confirmation,
    run_multicomponent_development,
)
from reservoir_dynamics.experiments.multicomponent_transfer_cli import (
    confirmation_summary_payload,
    load_frozen_models,
    structure_gate_payload,
)


PILOT_ARTIFACT = Path(
    "docs/research/artifacts/EXP-2026-016-pilot-summary.json"
)
EXPECTED_MODEL_SHA256 = (
    "db0b50a648fb085ca687922a531fab5482af2a134bd01eefc3efe3dd85675a01"
)


class MultiComponentTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models, cls.model_sha256 = load_frozen_models(
            PILOT_ARTIFACT,
            expected_sha256=EXPECTED_MODEL_SHA256,
        )

    def test_small_development_run_preserves_theory_and_fixed_models(self) -> None:
        result = run_multicomponent_development(
            fitted_models=self.models,
            trial_seeds=(2201, 2202),
            module_sizes=(2, 2, 3),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
            bootstrap_resamples=100,
        )

        self.assertEqual(result.phase, "development")
        self.assertEqual(len(result.points), 4)
        self.assertTrue(result.structure_gate.passed)
        self.assertTrue(result.theory_decisions.partition_recovery)
        self.assertTrue(result.theory_decisions.oracle_inferred_equivalence)
        self.assertTrue(result.theory_decisions.zero_coupling_factorization)
        self.assertTrue(result.theory_decisions.factorized_certificate_exactness)
        self.assertTrue(result.theory_decisions.transported_lower_bound)
        self.assertTrue(result.theory_decisions.certificate_chain)
        self.assertTrue(result.theory_decisions.feature_finiteness)
        self.assertTrue(result.theory_decisions.complexity_reduction)
        self.assertEqual(
            tuple(value.name for value in result.evaluations),
            ("component_aware", "global_profile", "product_only"),
        )

    def test_confirmation_rejects_every_development_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "development"):
            run_multicomponent_confirmation(
                fitted_models=self.models,
                trial_seeds=(PREREGISTERED_DEVELOPMENT_SEEDS[0], 2301),
                internal_gains=(0.025,),
                maximum_bridge_strengths=(0.0,),
                disturbance_bounds=(0.12,),
                task_steps=5,
                autonomous_steps=500,
                bootstrap_resamples=100,
            )

    def test_payloads_preserve_task_free_gate_hash_and_challenge_count(self) -> None:
        gate_payload = structure_gate_payload(
            trial_seeds=(2201, 2202),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
        )
        result = run_multicomponent_development(
            fitted_models=self.models,
            trial_seeds=(2201, 2202),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0,),
            disturbance_bounds=(0.12,),
            task_steps=5,
            autonomous_steps=500,
            bootstrap_resamples=100,
        )
        payload = confirmation_summary_payload(
            result,
            fixed_model_sha256=self.model_sha256,
        )

        self.assertFalse(gate_payload["task_values_generated"])
        self.assertTrue(gate_payload["structure_gate"]["passed"])
        self.assertEqual(payload["fixed_model_sha256"], EXPECTED_MODEL_SHA256)
        self.assertEqual(payload["point_count"], 2)
        self.assertEqual(payload["challenge_count"], 2 * 4 * 128)


if __name__ == "__main__":
    unittest.main()
