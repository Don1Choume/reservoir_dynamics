from types import SimpleNamespace
import unittest

from reservoir_dynamics.experiments.component_predictor import (
    NamedComponentPredictor,
    fixed_predictor_feature_shift_bound,
    predictors_from_payload,
    predictors_to_payload,
)
from reservoir_dynamics.experiments.component_profile import (
    COMPONENT_FEATURE_NAMES,
)
from reservoir_dynamics.experiments.component_size_experiment import (
    run_component_size_confirmation,
    run_component_size_pilot,
)
from reservoir_dynamics.experiments.component_size_extrapolation_cli import (
    confirmation_summary_payload,
    pilot_summary_payload,
)
from reservoir_dynamics.metrics.standardized_ridge import StandardizedRidgeModel


class ComponentSizeExperimentTest(unittest.TestCase):
    def test_fixed_predictor_feature_shift_bound_controls_prediction_change(self) -> None:
        feature_count = len(COMPONENT_FEATURE_NAMES)
        model = NamedComponentPredictor(
            name="component_aware",
            feature_names=COMPONENT_FEATURE_NAMES,
            model=StandardizedRidgeModel(
                intercept=0.5,
                coefficients=tuple(0.2 for _ in range(feature_count)),
                feature_means=tuple(0.0 for _ in range(feature_count)),
                feature_scales=tuple(1.0 for _ in range(feature_count)),
                penalty=0.001,
            ),
        )
        first_row = tuple(0.0 for _ in range(feature_count))
        second_row = (0.1,) + tuple(0.0 for _ in range(feature_count - 1))
        first = SimpleNamespace(component_feature_row=first_row)
        second = SimpleNamespace(component_feature_row=second_row)

        bound = fixed_predictor_feature_shift_bound(model, first, second)
        realized = abs(model.predict(first) - model.predict(second))

        self.assertAlmostEqual(bound, 0.02)
        self.assertLessEqual(realized, bound + 1e-12)

    def test_small_pilot_fixes_models_after_seed_holdout_evaluation(self) -> None:
        result = run_component_size_pilot(
            trial_seeds=(2001, 2002, 2003),
            module_size_pairs=((2, 2), (2, 3)),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )

        self.assertEqual(result.phase, "pilot")
        self.assertEqual(len(result.points), 12)
        self.assertTrue(result.structure_gate.passed)
        self.assertTrue(result.decisions.zero_coupling_factorization)
        self.assertTrue(result.decisions.transported_lower_bound)
        self.assertTrue(result.decisions.certificate_chain)
        self.assertTrue(result.decisions.feature_finiteness)
        self.assertEqual(len(result.fitted_models), 3)
        self.assertEqual(len(result.cross_validated_evaluations), 3)

    def test_model_payload_roundtrip_and_fixed_confirmation(self) -> None:
        pilot = run_component_size_pilot(
            trial_seeds=(2001, 2002, 2003),
            module_size_pairs=((2, 2), (2, 3)),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )
        payload = predictors_to_payload(pilot.fitted_models)
        restored_models = predictors_from_payload(payload)

        confirmation = run_component_size_confirmation(
            fitted_models=restored_models,
            trial_seeds=(2101, 2102),
            module_sizes=(2, 3),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
            bootstrap_resamples=100,
        )

        self.assertEqual(confirmation.phase, "confirmation")
        self.assertEqual(len(confirmation.points), 4)
        self.assertTrue(confirmation.structure_gate.passed)
        self.assertEqual(
            tuple(value.name for value in confirmation.evaluations),
            ("component_aware", "global_profile", "product_only"),
        )
        self.assertEqual(
            tuple(value.baseline_name for value in confirmation.error_intervals),
            ("global_profile", "product_only"),
        )
        self.assertTrue(confirmation.decisions.transported_lower_bound)
        self.assertTrue(confirmation.decisions.certificate_chain)
        self.assertIsInstance(
            confirmation.decisions.component_mae_within_threshold,
            bool,
        )
        self.assertIsInstance(
            confirmation.decisions.component_spearman_above_threshold,
            bool,
        )
        self.assertIsInstance(
            confirmation.decisions.component_beats_global,
            bool,
        )
        self.assertIsInstance(
            confirmation.decisions.component_beats_product,
            bool,
        )

    def test_confirmation_rejects_seed_overlap_with_preregistered_pilot(self) -> None:
        pilot = run_component_size_pilot(
            trial_seeds=(2001, 2002, 2003),
            module_size_pairs=((2, 2), (2, 3)),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0,),
            disturbance_bounds=(0.12,),
            task_steps=5,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )

        with self.assertRaisesRegex(ValueError, "pilot"):
            run_component_size_confirmation(
                fitted_models=pilot.fitted_models,
                trial_seeds=(2001, 2101),
                module_sizes=(2, 3),
                internal_gains=(0.025,),
                maximum_bridge_strengths=(0.0,),
                disturbance_bounds=(0.12,),
                task_steps=5,
                autonomous_steps=500,
                convergence_tolerance=1e-9,
                bootstrap_resamples=100,
            )

    def test_summary_payload_preserves_frozen_models_and_points(self) -> None:
        pilot = run_component_size_pilot(
            trial_seeds=(2001, 2002, 2003),
            module_size_pairs=((2, 2), (2, 3)),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0, 0.02),
            disturbance_bounds=(0.12,),
            task_steps=5,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )
        pilot_payload = pilot_summary_payload(pilot)
        restored = predictors_from_payload(pilot_payload["fitted_models"])
        confirmation = run_component_size_confirmation(
            fitted_models=restored,
            trial_seeds=(2101, 2102),
            module_sizes=(2, 3),
            internal_gains=(0.025,),
            maximum_bridge_strengths=(0.0,),
            disturbance_bounds=(0.12,),
            task_steps=5,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
            bootstrap_resamples=100,
        )
        confirmation_payload = confirmation_summary_payload(
            confirmation,
            pilot_model_sha256=pilot_payload["model_sha256"],
        )

        self.assertEqual(pilot_payload["point_count"], 12)
        self.assertEqual(len(pilot_payload["model_sha256"]), 64)
        self.assertEqual(confirmation_payload["point_count"], 2)
        self.assertEqual(
            confirmation_payload["pilot_model_sha256"],
            pilot_payload["model_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
