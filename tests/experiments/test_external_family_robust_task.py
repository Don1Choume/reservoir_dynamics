import unittest

from reservoir_dynamics.experiments.external_family_robust_task import (
    evaluate_external_family_candidate,
    evaluate_external_family_confirmation,
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
    task_retention: float,
) -> RobustRepertoireTaskPoint:
    disturbance_bound = 0.2
    return RobustRepertoireTaskPoint(
        trial_seed=trial_seed,
        network_family=network_family,  # type: ignore[arg-type]
        coupling_gain=0.01,
        disturbance_bound=disturbance_bound,
        raw_attractor_count=8,
        certified_robust_count=round(task_retention * 8),
        certified_robust_fraction=task_retention,
        mean_uniform_disturbance_margin=(
            task_retention * disturbance_bound
        ),
        task_retention=task_retention,
        guarantee_gap=0.01,
        off_diagonal_infinity_norm=0.02,
        maximum_local_jacobian_infinity_norm=0.4,
        minimum_fixed_point_coordinate=0.7,
        nonnormality_commutator_norm=0.0,
    )


ROBUST_PAIR = FamilyHoldoutCandidate(
    name="robust_pair",
    feature_names=(
        "normalized_mean_margin",
        "certified_robust_fraction",
    ),
    penalty=1e-6,
)

BASELINES = (
    FamilyHoldoutCandidate(
        name="raw_count",
        feature_names=("raw_attractor_count",),
        penalty=1e-6,
    ),
    FamilyHoldoutCandidate(
        name="structural",
        feature_names=(
            "maximum_local_jacobian_infinity_norm",
        ),
        penalty=1e-6,
    ),
)


class ExternalFamilyRobustTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.training_points = tuple(
            _point(
                trial_seed=seed,
                network_family=family,
                task_retention=retention,
            )
            for family in ("dense_symmetric", "asymmetric_dense")
            for seed, retention in ((1, 0.2), (2, 0.8))
        )
        self.testing_points = tuple(
            _point(
                trial_seed=seed,
                network_family="modular_paired",
                task_retention=retention,
            )
            for seed, retention in ((11, 0.3), (12, 0.7))
        )

    def test_fits_only_known_families_and_predicts_one_external_family(
        self,
    ) -> None:
        result = evaluate_external_family_candidate(
            training_points=self.training_points,
            testing_points=self.testing_points,
            candidate=ROBUST_PAIR,
        )

        self.assertEqual(
            result.training_families,
            ("asymmetric_dense", "dense_symmetric"),
        )
        self.assertEqual(result.testing_family, "modular_paired")
        self.assertEqual(result.training_seeds, (1, 2))
        self.assertEqual(result.testing_seeds, (11, 12))
        self.assertEqual(result.training_point_count, 4)
        self.assertEqual(result.testing_point_count, 2)
        self.assertLess(result.test_mae, 1e-5)
        self.assertGreater(result.test_spearman, 0.99)

    def test_rejects_family_or_seed_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "family"):
            evaluate_external_family_candidate(
                training_points=self.training_points
                + (
                    _point(
                        trial_seed=3,
                        network_family="modular_paired",
                        task_retention=0.5,
                    ),
                ),
                testing_points=self.testing_points,
                candidate=ROBUST_PAIR,
            )

        with self.assertRaisesRegex(ValueError, "seed"):
            evaluate_external_family_candidate(
                training_points=self.training_points,
                testing_points=self.testing_points
                + (
                    _point(
                        trial_seed=1,
                        network_family="modular_paired",
                        task_retention=0.5,
                    ),
                ),
                candidate=ROBUST_PAIR,
            )

    def test_evaluates_preregistered_external_family_decisions(
        self,
    ) -> None:
        result = evaluate_external_family_confirmation(
            training_points=self.training_points,
            confirmation_points=self.testing_points,
            selected_candidate=ROBUST_PAIR,
            baseline_candidates=BASELINES,
            expected_raw_attractor_count=8,
            association_threshold=0.75,
            bootstrap_resamples=20,
        )

        self.assertTrue(result.decisions.raw_count_matched)
        self.assertTrue(result.decisions.certificate_lower_bound_valid)
        self.assertTrue(result.decisions.rank_association)
        self.assertTrue(result.decisions.selected_beats_raw_count)
        self.assertTrue(result.decisions.selected_beats_structural)
        self.assertGreater(
            result.baseline_minus_selected_intervals[0].lower,
            0.0,
        )

    def test_confirmation_requires_primary_baselines(self) -> None:
        with self.assertRaisesRegex(ValueError, "structural"):
            evaluate_external_family_confirmation(
                training_points=self.training_points,
                confirmation_points=self.testing_points,
                selected_candidate=ROBUST_PAIR,
                baseline_candidates=BASELINES[:1],
                expected_raw_attractor_count=8,
                bootstrap_resamples=20,
            )


if __name__ == "__main__":
    unittest.main()
