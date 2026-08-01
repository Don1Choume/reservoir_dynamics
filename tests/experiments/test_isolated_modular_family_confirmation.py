import unittest
from types import SimpleNamespace
from unittest.mock import patch

from reservoir_dynamics.experiments.cross_family_robust_task_confirmation import (
    FamilyTaskSpecification,
)
from reservoir_dynamics.experiments.isolated_modular_family_confirmation import (
    run_isolated_modular_family_confirmation,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
)


def _point(
    *,
    trial_seed: int,
    network_family: str,
) -> RobustRepertoireTaskPoint:
    return RobustRepertoireTaskPoint(
        trial_seed=trial_seed,
        network_family=network_family,  # type: ignore[arg-type]
        coupling_gain=0.04,
        disturbance_bound=0.16,
        raw_attractor_count=16,
        certified_robust_count=8,
        certified_robust_fraction=0.5,
        mean_uniform_disturbance_margin=0.08,
        task_retention=0.6,
        guarantee_gap=0.1,
        off_diagonal_infinity_norm=0.04,
        maximum_local_jacobian_infinity_norm=0.4,
        minimum_fixed_point_coordinate=0.7,
        nonnormality_commutator_norm=0.0,
    )


class IsolatedModularFamilyConfirmationTest(unittest.TestCase):
    @patch(
        "reservoir_dynamics.experiments."
        "isolated_modular_family_confirmation."
        "evaluate_external_family_confirmation"
    )
    @patch(
        "reservoir_dynamics.experiments."
        "isolated_modular_family_confirmation."
        "run_robust_repertoire_task_diagnostics"
    )
    def test_generates_known_and_external_families_in_separate_calls(
        self,
        run_diagnostics,
        evaluate_confirmation,
    ) -> None:
        training_seeds = (1, 2, 3, 4)
        confirmation_seeds = (11, 12, 13, 14)
        known_specifications = (
            FamilyTaskSpecification(
                network_family="dense_symmetric",
                coupling_gains=(0.04, 0.05),
                disturbance_bound=0.16,
            ),
            FamilyTaskSpecification(
                network_family="asymmetric_dense",
                coupling_gains=(0.04, 0.05),
                disturbance_bound=0.16,
            ),
        )
        external_specification = FamilyTaskSpecification(
            network_family="modular_paired",
            coupling_gains=(0.04, 0.05),
            disturbance_bound=0.16,
        )

        def diagnostics_side_effect(**arguments):
            seeds = arguments["trial_seeds"]
            family = arguments["network_family"]
            return SimpleNamespace(
                points=tuple(
                    _point(trial_seed=seed, network_family=family)
                    for seed in seeds
                )
            )

        run_diagnostics.side_effect = diagnostics_side_effect
        sentinel_evaluation = SimpleNamespace(experiment_id="EXP-2026-012")
        evaluate_confirmation.return_value = sentinel_evaluation

        result = run_isolated_modular_family_confirmation(
            training_seeds=training_seeds,
            confirmation_seeds=confirmation_seeds,
            training_family_specifications=known_specifications,
            external_family_specification=external_specification,
            bootstrap_resamples=20,
        )

        self.assertEqual(run_diagnostics.call_count, 3)
        self.assertEqual(
            tuple(
                call.kwargs["network_family"]
                for call in run_diagnostics.call_args_list
            ),
            (
                "dense_symmetric",
                "asymmetric_dense",
                "modular_paired",
            ),
        )
        self.assertTrue(
            all(
                call.kwargs["trial_seeds"] == training_seeds
                for call in run_diagnostics.call_args_list[:2]
            )
        )
        self.assertEqual(
            run_diagnostics.call_args_list[2].kwargs["trial_seeds"],
            confirmation_seeds,
        )
        evaluation_call = evaluate_confirmation.call_args.kwargs
        self.assertEqual(
            {
                point.network_family
                for point in evaluation_call["training_points"]
            },
            {"dense_symmetric", "asymmetric_dense"},
        )
        self.assertEqual(
            {
                point.network_family
                for point in evaluation_call["confirmation_points"]
            },
            {"modular_paired"},
        )
        self.assertEqual(result.evaluation, sentinel_evaluation)

    def test_rejects_external_family_in_training_specifications(
        self,
    ) -> None:
        modular_specification = FamilyTaskSpecification(
            network_family="modular_paired",
            coupling_gains=(0.04, 0.05),
            disturbance_bound=0.16,
        )

        with self.assertRaisesRegex(ValueError, "family"):
            run_isolated_modular_family_confirmation(
                training_seeds=(1, 2, 3, 4),
                confirmation_seeds=(11, 12, 13, 14),
                training_family_specifications=(
                    modular_specification,
                    FamilyTaskSpecification(
                        network_family="dense_symmetric",
                        coupling_gains=(0.04, 0.05),
                        disturbance_bound=0.16,
                    ),
                ),
                external_family_specification=modular_specification,
                bootstrap_resamples=20,
            )


if __name__ == "__main__":
    unittest.main()
