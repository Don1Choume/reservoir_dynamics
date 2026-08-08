import json
import pathlib
from dataclasses import replace
import unittest

from reservoir_dynamics.experiments.component_predictor import (
    predictors_from_payload,
)
from reservoir_dynamics.experiments.partition_functional_impact import (
    PREREGISTERED_DEVELOPMENT_SEEDS,
    run_partition_functional_confirmation,
    run_partition_functional_development,
)
from reservoir_dynamics.experiments.partition_functional_impact_cli import (
    result_payload,
)


PILOT_ARTIFACT = pathlib.Path(
    "docs/research/artifacts/EXP-2026-016-pilot-summary.json"
)


class PartitionFunctionalImpactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = json.loads(PILOT_ARTIFACT.read_text(encoding="utf-8"))
        cls.models = predictors_from_payload(payload["fitted_models"])

    def test_small_grid_separates_structure_and_partition_conditioned_function(self) -> None:
        result = run_partition_functional_development(
            fitted_models=self.models,
            trial_seeds=(2601, 2602),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.02,),
            relative_amplitudes=(0.9, 4.0),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
        )

        self.assertEqual(result.experiment_id, "EXP-2026-019")
        self.assertEqual(result.phase, "development")
        self.assertEqual(len(result.perturbations), 4)
        self.assertGreaterEqual(len(result.points), 2)
        subradius = tuple(
            point
            for point in result.perturbations
            if point.relative_amplitude < 1.0
        )
        self.assertTrue(all(point.partition_recovered for point in subradius))
        self.assertTrue(result.decisions.fixed_model)
        self.assertTrue(result.decisions.subradius_recovery)
        self.assertTrue(result.decisions.subradius_profile_identity)
        self.assertTrue(result.decisions.shared_target_identity)
        self.assertTrue(result.decisions.prediction_shift_bound)
        self.assertTrue(result.decisions.factorized_exactness)
        self.assertTrue(result.decisions.certificate_chain)
        self.assertTrue(result.decisions.seed_independence)
        self.assertTrue(result.decisions.task_free_inference)
        self.assertTrue(result.decisions.all_passed)
        self.assertTrue(
            all(
                point.component_prediction_shift
                <= point.component_prediction_shift_bound + 1e-12
                for point in result.points
            )
        )
        self.assertTrue(
            all(point.shared_target_difference == 0.0 for point in result.points)
        )

        payload = result_payload(result)
        self.assertEqual(payload["experiment_id"], "EXP-2026-019")
        self.assertEqual(payload["perturbation_count"], 4)
        self.assertEqual(len(payload["amplitude_summaries"]), 2)
        self.assertEqual(len(payload["perturbations"]), 4)
        self.assertEqual(len(payload["points"]), len(result.points))
        self.assertTrue(payload["decisions"]["all_passed"])

    def test_confirmation_rejects_development_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "development"):
            run_partition_functional_confirmation(
                fitted_models=self.models,
                trial_seeds=(PREREGISTERED_DEVELOPMENT_SEEDS[0], 2701),
                internal_gains=(0.025,),
                maximum_bridge_strengths=(0.02,),
                relative_amplitudes=(0.9, 4.0),
                disturbance_bounds=(0.12,),
                task_steps=10,
                autonomous_steps=500,
            )

    def test_rejects_modified_fixed_model_before_task(self) -> None:
        modified_first = replace(
            self.models[0],
            model=replace(
                self.models[0].model,
                intercept=self.models[0].model.intercept + 0.001,
            ),
        )
        with self.assertRaisesRegex(ValueError, "model hash"):
            run_partition_functional_development(
                fitted_models=(modified_first, *self.models[1:]),
                trial_seeds=(2601, 2602),
                internal_gains=(0.025,),
                maximum_bridge_strengths=(0.02,),
                relative_amplitudes=(0.9, 4.0),
                disturbance_bounds=(0.12,),
                task_steps=10,
                autonomous_steps=500,
            )


if __name__ == "__main__":
    unittest.main()
