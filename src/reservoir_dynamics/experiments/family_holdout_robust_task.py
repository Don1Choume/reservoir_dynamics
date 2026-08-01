"""未知network familyへのrobust task予測を評価する。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from reservoir_dynamics.experiments.recurrent_weight_families import (
    RecurrentWeightFamily,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
    _spearman,
)
from reservoir_dynamics.metrics.standardized_ridge import (
    StandardizedRidgeModel,
    fit_standardized_ridge,
)

EXPERIMENT_ID = "EXP-2026-011"

FamilyHoldoutFeatureName = Literal[
    "normalized_mean_margin",
    "certified_robust_fraction",
    "normalized_coupling_gain",
    "normalized_off_diagonal_norm",
    "maximum_local_jacobian_infinity_norm",
    "minimum_fixed_point_coordinate",
    "nonnormality_commutator_norm",
    "raw_attractor_count",
]

SUPPORTED_FEATURE_NAMES: tuple[FamilyHoldoutFeatureName, ...] = (
    "normalized_mean_margin",
    "certified_robust_fraction",
    "normalized_coupling_gain",
    "normalized_off_diagonal_norm",
    "maximum_local_jacobian_infinity_norm",
    "minimum_fixed_point_coordinate",
    "nonnormality_commutator_norm",
    "raw_attractor_count",
)

@dataclass(frozen=True, slots=True)
class FamilyHoldoutCandidate:
    """探索段階で比較する固定特徴量・ridge強度の候補。"""

    name: str
    feature_names: tuple[FamilyHoldoutFeatureName, ...]
    penalty: float


DEFAULT_PILOT_CANDIDATES = (
    FamilyHoldoutCandidate(
        name="raw_count",
        feature_names=("raw_attractor_count",),
        penalty=1e-3,
    ),
    FamilyHoldoutCandidate(
        name="normalized_margin",
        feature_names=("normalized_mean_margin",),
        penalty=1e-3,
    ),
    FamilyHoldoutCandidate(
        name="certified_fraction",
        feature_names=("certified_robust_fraction",),
        penalty=1e-3,
    ),
    FamilyHoldoutCandidate(
        name="robust_pair",
        feature_names=(
            "normalized_mean_margin",
            "certified_robust_fraction",
        ),
        penalty=1e-3,
    ),
    FamilyHoldoutCandidate(
        name="structural",
        feature_names=(
            "normalized_coupling_gain",
            "normalized_off_diagonal_norm",
            "maximum_local_jacobian_infinity_norm",
            "minimum_fixed_point_coordinate",
            "nonnormality_commutator_norm",
        ),
        penalty=1e-3,
    ),
    FamilyHoldoutCandidate(
        name="hybrid",
        feature_names=(
            "normalized_mean_margin",
            "certified_robust_fraction",
            "normalized_coupling_gain",
            "normalized_off_diagonal_norm",
            "maximum_local_jacobian_infinity_norm",
            "minimum_fixed_point_coordinate",
            "nonnormality_commutator_norm",
        ),
        penalty=1e-3,
    ),
)


@dataclass(frozen=True, slots=True)
class FamilyHoldoutPrediction:
    """未知familyに属する一条件の予測と観測。"""

    trial_seed: int
    network_family: RecurrentWeightFamily
    coupling_gain: float
    disturbance_bound: float
    observed_task_retention: float
    predicted_task_retention: float
    absolute_error: float


@dataclass(frozen=True, slots=True)
class FamilyHoldoutFoldResult:
    """一つの未知familyに対する学習条件と予測性能。"""

    held_out_family: RecurrentWeightFamily
    training_families: tuple[RecurrentWeightFamily, ...]
    training_point_count: int
    testing_point_count: int
    model: StandardizedRidgeModel
    test_mae: float
    test_spearman: float
    predictions: tuple[FamilyHoldoutPrediction, ...]


@dataclass(frozen=True, slots=True)
class FamilyHoldoutCandidateEvaluation:
    """全familyを一度ずつ未知とした候補評価。"""

    experiment_id: str
    training_seeds: tuple[int, ...]
    testing_seeds: tuple[int, ...]
    candidate: FamilyHoldoutCandidate
    folds: tuple[FamilyHoldoutFoldResult, ...]
    pooled_test_mae: float
    pooled_test_spearman: float


@dataclass(frozen=True, slots=True)
class FamilyHoldoutSelection:
    """既観測データだけで行った候補選択結果。"""

    selected_candidate: FamilyHoldoutCandidate
    evaluations: tuple[FamilyHoldoutCandidateEvaluation, ...]


def evaluate_family_holdout_candidate(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    testing_seeds: tuple[int, ...],
    candidate: FamilyHoldoutCandidate,
) -> FamilyHoldoutCandidateEvaluation:
    """familyとseedを同時にholdoutしてridge予測を評価する。"""

    _validate_evaluation_inputs(
        points=points,
        training_seeds=training_seeds,
        testing_seeds=testing_seeds,
        candidate=candidate,
    )
    families = tuple(sorted({point.network_family for point in points}))
    folds = tuple(
        _evaluate_fold(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
            candidate=candidate,
            held_out_family=held_out_family,
            families=families,
        )
        for held_out_family in families
    )
    predictions = tuple(
        prediction
        for fold in folds
        for prediction in fold.predictions
    )
    return FamilyHoldoutCandidateEvaluation(
        experiment_id=EXPERIMENT_ID,
        training_seeds=training_seeds,
        testing_seeds=testing_seeds,
        candidate=candidate,
        folds=folds,
        pooled_test_mae=math.fsum(
            prediction.absolute_error for prediction in predictions
        )
        / len(predictions),
        pooled_test_spearman=_spearman(
            tuple(
                prediction.predicted_task_retention
                for prediction in predictions
            ),
            tuple(
                prediction.observed_task_retention
                for prediction in predictions
            ),
        ),
    )


def select_family_holdout_candidate(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    testing_seeds: tuple[int, ...],
    candidates: tuple[FamilyHoldoutCandidate, ...],
) -> FamilyHoldoutSelection:
    """pooled未知family MAEを事前登録用の単一候補へ縮約する。"""

    if not candidates:
        raise ValueError("candidateは1件以上必要です")
    candidate_names = tuple(candidate.name for candidate in candidates)
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate名は重複禁止です")
    evaluations = tuple(
        evaluate_family_holdout_candidate(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=testing_seeds,
            candidate=candidate,
        )
        for candidate in candidates
    )
    selected_evaluation = min(
        evaluations,
        key=lambda evaluation: (
            evaluation.pooled_test_mae,
            evaluation.candidate.name,
        ),
    )
    return FamilyHoldoutSelection(
        selected_candidate=selected_evaluation.candidate,
        evaluations=evaluations,
    )


def _evaluate_fold(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    testing_seeds: tuple[int, ...],
    candidate: FamilyHoldoutCandidate,
    held_out_family: RecurrentWeightFamily,
    families: tuple[RecurrentWeightFamily, ...],
) -> FamilyHoldoutFoldResult:
    """未知familyのラベルがfitへ混入しない境界を一か所に固定する。"""

    training_points = tuple(
        point
        for point in points
        if point.trial_seed in training_seeds
        and point.network_family != held_out_family
    )
    testing_points = tuple(
        point
        for point in points
        if point.trial_seed in testing_seeds
        and point.network_family == held_out_family
    )
    if not training_points or not testing_points:
        raise ValueError(
            f"{held_out_family}の学習点または評価点がありません"
        )
    model = fit_standardized_ridge(
        tuple(
            build_family_holdout_feature_row(
                point,
                candidate.feature_names,
            )
            for point in training_points
        ),
        tuple(point.task_retention for point in training_points),
        penalty=candidate.penalty,
    )
    predictions = tuple(
        _predict_point(
            point=point,
            candidate=candidate,
            model=model,
        )
        for point in testing_points
    )
    return FamilyHoldoutFoldResult(
        held_out_family=held_out_family,
        training_families=tuple(
            family for family in families if family != held_out_family
        ),
        training_point_count=len(training_points),
        testing_point_count=len(testing_points),
        model=model,
        test_mae=math.fsum(
            prediction.absolute_error for prediction in predictions
        )
        / len(predictions),
        test_spearman=_spearman(
            tuple(
                prediction.predicted_task_retention
                for prediction in predictions
            ),
            tuple(
                prediction.observed_task_retention
                for prediction in predictions
            ),
        ),
        predictions=predictions,
    )


def _predict_point(
    *,
    point: RobustRepertoireTaskPoint,
    candidate: FamilyHoldoutCandidate,
    model: StandardizedRidgeModel,
) -> FamilyHoldoutPrediction:
    predicted_task_retention = model.predict(
        build_family_holdout_feature_row(
            point,
            candidate.feature_names,
        ),
        clip_to_unit_interval=True,
    )
    return FamilyHoldoutPrediction(
        trial_seed=point.trial_seed,
        network_family=point.network_family,
        coupling_gain=point.coupling_gain,
        disturbance_bound=point.disturbance_bound,
        observed_task_retention=point.task_retention,
        predicted_task_retention=predicted_task_retention,
        absolute_error=abs(
            point.task_retention - predicted_task_retention
        ),
    )


def build_family_holdout_feature_row(
    point: RobustRepertoireTaskPoint,
    feature_names: tuple[FamilyHoldoutFeatureName, ...],
) -> tuple[float, ...]:
    """候補定義に従って無次元化済みの予測特徴量を構成する。"""

    if point.disturbance_bound <= 0.0:
        raise ValueError("無次元化には正のdisturbance_boundが必要です")
    values = {
        "normalized_mean_margin": (
            point.mean_uniform_disturbance_margin
            / point.disturbance_bound
        ),
        "certified_robust_fraction": point.certified_robust_fraction,
        "normalized_coupling_gain": (
            point.coupling_gain / point.disturbance_bound
        ),
        "normalized_off_diagonal_norm": (
            point.off_diagonal_infinity_norm
            / point.disturbance_bound
        ),
        "maximum_local_jacobian_infinity_norm": (
            point.maximum_local_jacobian_infinity_norm
        ),
        "minimum_fixed_point_coordinate": (
            point.minimum_fixed_point_coordinate
        ),
        "nonnormality_commutator_norm": (
            point.nonnormality_commutator_norm
        ),
        "raw_attractor_count": float(point.raw_attractor_count),
    }
    row: list[float] = []
    for feature_name in feature_names:
        value = values.get(feature_name)
        if value is None:
            raise ValueError(
                f"特徴量{feature_name}はこのpointでは定義されません"
            )
        row.append(float(value))
    return tuple(row)


def _validate_evaluation_inputs(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    testing_seeds: tuple[int, ...],
    candidate: FamilyHoldoutCandidate,
) -> None:
    if not points:
        raise ValueError("pointsは1件以上必要です")
    if not training_seeds or not testing_seeds:
        raise ValueError("training/testing seedは各1件以上必要です")
    if set(training_seeds).intersection(testing_seeds):
        raise ValueError("training_seedsとtesting_seedsは重複禁止です")
    if len({point.network_family for point in points}) < 2:
        raise ValueError("familyは2種類以上必要です")
    if not candidate.name.strip():
        raise ValueError("candidate名は空にできません")
    if not candidate.feature_names:
        raise ValueError("特徴量は1件以上必要です")
    unsupported_features = set(candidate.feature_names).difference(
        SUPPORTED_FEATURE_NAMES
    )
    if unsupported_features:
        raise ValueError(
            f"未対応の特徴量です: {sorted(unsupported_features)}"
        )
    if (
        not math.isfinite(candidate.penalty)
        or candidate.penalty <= 0.0
    ):
        raise ValueError("ridge penaltyは正の有限値にしてください")
