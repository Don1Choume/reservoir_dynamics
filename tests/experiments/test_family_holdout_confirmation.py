import unittest

from reservoir_dynamics.experiments.family_holdout_confirmation import (
    evaluate_family_holdout_confirmation,
)
from reservoir_dynamics.experiments.family_holdout_robust_task import (
    FamilyHoldoutCandidate,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
)


def _point(
    *,
    trial_seed: int,
    network_family: str,
    normalized_margin: float,
) -> RobustRepertoireTaskPoint:
    disturbance_bound = (
        0.1 if network_family == "sparse_symmetric" else 0.2
    )
    task_retention = 0.2 + 0.5 * normalized_margin
    return RobustRepertoireTaskPoint(
        trial_seed=trial_seed,
        network_family=network_family,  # type: ignore[arg-type]
        coupling_gain=0.01,
        disturbance_bound=disturbance_bound,
        raw_attractor_count=8,
        certified_robust_count=round(task_retention * 8),
        certified_robust_fraction=task_retention,
        mean_uniform_disturbance_margin=(
            normalized_margin * disturbance_bound
        ),
        task_retention=task_retention,
        guarantee_gap=0.01,
        off_diagonal_infinity_norm=0.02,
        maximum_local_jacobian_infinity_norm=0.4,
        minimum_fixed_point_coordinate=0.7,
        nonnormality_commutator_norm=0.2,
    )


class FamilyHoldoutConfirmationTest(unittest.TestCase):
    def test_evaluates_preregistered_decisions_by_seed(self) -> None:
        families = (
            "dense_symmetric",
            "sparse_symmetric",
            "asymmetric_dense",
        )
        points = tuple(
            _point(
                trial_seed=seed,
                network_family=family,
                normalized_margin=margin,
            )
            for family in families
            for seed, margin in (
                (1, 0.1),
                (2, 0.9),
                (11, 0.2),
                (12, 0.8),
            )
        )
        result = evaluate_family_holdout_confirmation(
            points=points,
            training_seeds=(1, 2),
            confirmation_seeds=(11, 12),
            selected_candidate=FamilyHoldoutCandidate(
                name="robust_pair",
                feature_names=(
                    "normalized_mean_margin",
                    "certified_robust_fraction",
                ),
                penalty=1e-3,
            ),
            baseline_candidates=(
                FamilyHoldoutCandidate(
                    name="raw_count",
                    feature_names=("raw_attractor_count",),
                    penalty=1e-3,
                ),
                FamilyHoldoutCandidate(
                    name="structural",
                    feature_names=(
                        "maximum_local_jacobian_infinity_norm",
                    ),
                    penalty=1e-3,
                ),
            ),
            expected_raw_attractor_count=8,
            association_threshold=0.75,
            bootstrap_resamples=20,
        )

        self.assertTrue(result.decisions.raw_count_matched)
        self.assertTrue(result.decisions.certificate_lower_bound_valid)
        self.assertTrue(result.decisions.all_family_rank_association)
        self.assertTrue(result.decisions.selected_beats_raw_count)
        self.assertTrue(result.decisions.selected_beats_structural)
        self.assertGreater(
            result.baseline_minus_selected_intervals[0].lower,
            0.0,
        )

    def test_requires_named_primary_baselines(self) -> None:
        points = tuple(
            _point(
                trial_seed=seed,
                network_family=family,
                normalized_margin=0.5,
            )
            for family in ("dense_symmetric", "sparse_symmetric")
            for seed in (1, 2, 11, 12)
        )
        with self.assertRaisesRegex(ValueError, "raw_count"):
            evaluate_family_holdout_confirmation(
                points=points,
                training_seeds=(1, 2),
                confirmation_seeds=(11, 12),
                selected_candidate=FamilyHoldoutCandidate(
                    name="robust_pair",
                    feature_names=("normalized_mean_margin",),
                    penalty=1e-3,
                ),
                baseline_candidates=(
                    FamilyHoldoutCandidate(
                        name="structural",
                        feature_names=(
                            "maximum_local_jacobian_infinity_norm",
                        ),
                        penalty=1e-3,
                    ),
                ),
                expected_raw_attractor_count=8,
                bootstrap_resamples=20,
            )


if __name__ == "__main__":
    unittest.main()
