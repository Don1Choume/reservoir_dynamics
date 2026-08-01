import unittest

from reservoir_dynamics.experiments.family_holdout_robust_task import (
    FamilyHoldoutCandidate,
    evaluate_family_holdout_candidate,
    select_family_holdout_candidate,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
)


def _point(
    *,
    trial_seed: int,
    network_family: str,
    normalized_margin: float,
    task_retention: float,
    disturbance_bound: float = 0.2,
) -> RobustRepertoireTaskPoint:
    return RobustRepertoireTaskPoint(
        trial_seed=trial_seed,
        network_family=network_family,  # type: ignore[arg-type]
        coupling_gain=0.05 * disturbance_bound,
        disturbance_bound=disturbance_bound,
        raw_attractor_count=8,
        certified_robust_count=round(task_retention * 8),
        certified_robust_fraction=task_retention,
        mean_uniform_disturbance_margin=(
            normalized_margin * disturbance_bound
        ),
        task_retention=task_retention,
        guarantee_gap=0.0,
        off_diagonal_infinity_norm=0.1 * disturbance_bound,
        maximum_local_jacobian_infinity_norm=0.4,
        minimum_fixed_point_coordinate=0.7,
        nonnormality_commutator_norm=0.2,
    )


class FamilyHoldoutRobustTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        families = (
            "dense_symmetric",
            "sparse_symmetric",
            "asymmetric_dense",
        )
        self.points = tuple(
            _point(
                trial_seed=seed,
                network_family=family,
                normalized_margin=normalized_margin,
                task_retention=0.1 + 0.4 * normalized_margin,
                disturbance_bound=(
                    0.1 if family == "sparse_symmetric" else 0.2
                ),
            )
            for family in families
            for seed, normalized_margin in (
                (1, 0.2),
                (2, 0.4),
                (11, 0.3),
                (12, 0.5),
            )
        )

    def test_leaves_family_and_seed_out_of_training(self) -> None:
        result = evaluate_family_holdout_candidate(
            points=self.points,
            training_seeds=(1, 2),
            testing_seeds=(11, 12),
            candidate=FamilyHoldoutCandidate(
                name="normalized_margin",
                feature_names=("normalized_mean_margin",),
                penalty=1e-6,
            ),
        )

        self.assertEqual(len(result.folds), 3)
        self.assertLess(result.pooled_test_mae, 1e-5)
        for fold in result.folds:
            self.assertEqual(fold.training_point_count, 4)
            self.assertEqual(fold.testing_point_count, 2)
            self.assertNotIn(
                fold.held_out_family,
                fold.training_families,
            )
            self.assertTrue(
                all(
                    prediction.trial_seed in (11, 12)
                    for prediction in fold.predictions
                )
            )

    def test_normalizes_margin_by_disturbance_bound(self) -> None:
        result = evaluate_family_holdout_candidate(
            points=self.points,
            training_seeds=(1, 2),
            testing_seeds=(11, 12),
            candidate=FamilyHoldoutCandidate(
                name="normalized_margin",
                feature_names=("normalized_mean_margin",),
                penalty=1e-6,
            ),
        )

        sparse_fold = next(
            fold
            for fold in result.folds
            if fold.held_out_family == "sparse_symmetric"
        )
        self.assertLess(sparse_fold.test_mae, 1e-5)

    def test_selects_candidate_by_pooled_holdout_error(self) -> None:
        selection = select_family_holdout_candidate(
            points=self.points,
            training_seeds=(1, 2),
            testing_seeds=(11, 12),
            candidates=(
                FamilyHoldoutCandidate(
                    name="raw_count",
                    feature_names=("raw_attractor_count",),
                    penalty=1e-6,
                ),
                FamilyHoldoutCandidate(
                    name="normalized_margin",
                    feature_names=("normalized_mean_margin",),
                    penalty=1e-6,
                ),
            ),
        )

        self.assertEqual(
            selection.selected_candidate.name,
            "normalized_margin",
        )
        self.assertEqual(len(selection.evaluations), 2)

    def test_rejects_overlapping_seeds_and_invalid_features(self) -> None:
        candidate = FamilyHoldoutCandidate(
            name="margin",
            feature_names=("normalized_mean_margin",),
            penalty=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "重複"):
            evaluate_family_holdout_candidate(
                points=self.points,
                training_seeds=(1, 2),
                testing_seeds=(2, 11),
                candidate=candidate,
            )
        with self.assertRaisesRegex(ValueError, "特徴量"):
            evaluate_family_holdout_candidate(
                points=self.points,
                training_seeds=(1, 2),
                testing_seeds=(11, 12),
                candidate=FamilyHoldoutCandidate(
                    name="unknown",
                    feature_names=("unknown",),  # type: ignore[arg-type]
                    penalty=1e-6,
                ),
            )


if __name__ == "__main__":
    unittest.main()
