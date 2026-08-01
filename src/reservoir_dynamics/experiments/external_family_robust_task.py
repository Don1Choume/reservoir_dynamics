"""既知familyだけでfitし、完全に未観測の外部familyを評価する。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.family_holdout_confirmation import (
    NamedBootstrapMeanInterval,
)
from reservoir_dynamics.experiments.family_holdout_robust_task import (
    SUPPORTED_FEATURE_NAMES,
    FamilyHoldoutCandidate,
    FamilyHoldoutPrediction,
    build_family_holdout_feature_row,
)
from reservoir_dynamics.experiments.recurrent_weight_families import (
    RecurrentWeightFamily,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
    _spearman,
)
from reservoir_dynamics.metrics.bootstrap import bootstrap_mean_interval
from reservoir_dynamics.metrics.standardized_ridge import (
    StandardizedRidgeModel,
    fit_standardized_ridge,
)

EXPERIMENT_ID = "EXP-2026-012"


@dataclass(frozen=True, slots=True)
class ExternalFamilyCandidateEvaluation:
    """既知familyでfitした一候補の外部family予測結果。"""

    experiment_id: str
    candidate: FamilyHoldoutCandidate
    training_families: tuple[RecurrentWeightFamily, ...]
    testing_family: RecurrentWeightFamily
    training_seeds: tuple[int, ...]
    testing_seeds: tuple[int, ...]
    training_point_count: int
    testing_point_count: int
    model: StandardizedRidgeModel
    test_mae: float
    test_spearman: float
    predictions: tuple[FamilyHoldoutPrediction, ...]


@dataclass(frozen=True, slots=True)
class ExternalFamilyConfirmationDecisions:
    """外部familyの結果を見る前に固定する主要判定。"""

    raw_count_matched: bool
    certificate_lower_bound_valid: bool
    rank_association: bool
    selected_beats_raw_count: bool
    selected_beats_structural: bool


@dataclass(frozen=True, slots=True)
class ExternalFamilyConfirmationResult:
    """候補選択から隔離した一つの外部familyの確認結果。"""

    experiment_id: str
    phase: str
    selected_evaluation: ExternalFamilyCandidateEvaluation
    baseline_evaluations: tuple[
        ExternalFamilyCandidateEvaluation, ...
    ]
    baseline_minus_selected_intervals: tuple[
        NamedBootstrapMeanInterval, ...
    ]
    confirmation_point_count: int
    guarantee_violation_count: int
    decisions: ExternalFamilyConfirmationDecisions


def evaluate_external_family_candidate(
    *,
    training_points: tuple[RobustRepertoireTaskPoint, ...],
    testing_points: tuple[RobustRepertoireTaskPoint, ...],
    candidate: FamilyHoldoutCandidate,
) -> ExternalFamilyCandidateEvaluation:
    """family集合とseed集合が交わらない外部予測を評価する。"""

    _validate_candidate_evaluation(
        training_points=training_points,
        testing_points=testing_points,
        candidate=candidate,
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
    return ExternalFamilyCandidateEvaluation(
        experiment_id=EXPERIMENT_ID,
        candidate=candidate,
        training_families=tuple(
            sorted({point.network_family for point in training_points})
        ),
        testing_family=testing_points[0].network_family,
        training_seeds=tuple(
            sorted({point.trial_seed for point in training_points})
        ),
        testing_seeds=tuple(
            sorted({point.trial_seed for point in testing_points})
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


def evaluate_external_family_confirmation(
    *,
    training_points: tuple[RobustRepertoireTaskPoint, ...],
    confirmation_points: tuple[RobustRepertoireTaskPoint, ...],
    selected_candidate: FamilyHoldoutCandidate,
    baseline_candidates: tuple[FamilyHoldoutCandidate, ...],
    expected_raw_attractor_count: int,
    association_threshold: float = 0.75,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_812,
) -> ExternalFamilyConfirmationResult:
    """事前固定した候補を隔離済み外部familyへ一度だけ適用する。"""

    _validate_confirmation_configuration(
        selected_candidate=selected_candidate,
        baseline_candidates=baseline_candidates,
        expected_raw_attractor_count=expected_raw_attractor_count,
        association_threshold=association_threshold,
    )
    selected_evaluation = evaluate_external_family_candidate(
        training_points=training_points,
        testing_points=confirmation_points,
        candidate=selected_candidate,
    )
    baseline_evaluations = tuple(
        evaluate_external_family_candidate(
            training_points=training_points,
            testing_points=confirmation_points,
            candidate=baseline,
        )
        for baseline in baseline_candidates
    )
    intervals = tuple(
        _build_baseline_interval(
            selected_evaluation=selected_evaluation,
            baseline_evaluation=baseline_evaluation,
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed + baseline_index,
        )
        for baseline_index, baseline_evaluation in enumerate(
            baseline_evaluations
        )
    )
    interval_by_name = {
        interval.baseline_name: interval for interval in intervals
    }
    guarantee_violation_count = sum(
        point.guarantee_gap < -1e-12
        for point in confirmation_points
    )
    return ExternalFamilyConfirmationResult(
        experiment_id=EXPERIMENT_ID,
        phase="confirmation",
        selected_evaluation=selected_evaluation,
        baseline_evaluations=baseline_evaluations,
        baseline_minus_selected_intervals=intervals,
        confirmation_point_count=len(confirmation_points),
        guarantee_violation_count=guarantee_violation_count,
        decisions=ExternalFamilyConfirmationDecisions(
            raw_count_matched=all(
                point.raw_attractor_count
                == expected_raw_attractor_count
                for point in confirmation_points
            ),
            certificate_lower_bound_valid=(
                guarantee_violation_count == 0
            ),
            rank_association=(
                selected_evaluation.test_spearman
                > association_threshold
            ),
            selected_beats_raw_count=(
                interval_by_name["raw_count"].lower > 0.0
            ),
            selected_beats_structural=(
                interval_by_name["structural"].lower > 0.0
            ),
        ),
    )


def _predict_point(
    *,
    point: RobustRepertoireTaskPoint,
    candidate: FamilyHoldoutCandidate,
    model: StandardizedRidgeModel,
) -> FamilyHoldoutPrediction:
    predicted_retention = model.predict(
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
        predicted_task_retention=predicted_retention,
        absolute_error=abs(point.task_retention - predicted_retention),
    )


def _build_baseline_interval(
    *,
    selected_evaluation: ExternalFamilyCandidateEvaluation,
    baseline_evaluation: ExternalFamilyCandidateEvaluation,
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> NamedBootstrapMeanInterval:
    selected_predictions = _prediction_map(selected_evaluation)
    baseline_predictions = _prediction_map(baseline_evaluation)
    if selected_predictions.keys() != baseline_predictions.keys():
        raise RuntimeError("selectedとbaselineの評価条件が一致しません")
    seed_differences = tuple(
        _mean_seed_error_difference(
            trial_seed=trial_seed,
            selected_predictions=selected_predictions,
            baseline_predictions=baseline_predictions,
        )
        for trial_seed in selected_evaluation.testing_seeds
    )
    interval = bootstrap_mean_interval(
        seed_differences,
        confidence_level=confidence_level,
        resamples=resamples,
        random_seed=random_seed,
    )
    return NamedBootstrapMeanInterval(
        baseline_name=baseline_evaluation.candidate.name,
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        confidence_level=interval.confidence_level,
        resamples=interval.resamples,
    )


def _mean_seed_error_difference(
    *,
    trial_seed: int,
    selected_predictions: dict[
        tuple[str, int, float, float],
        FamilyHoldoutPrediction,
    ],
    baseline_predictions: dict[
        tuple[str, int, float, float],
        FamilyHoldoutPrediction,
    ],
) -> float:
    seed_keys = tuple(
        key for key in selected_predictions if key[1] == trial_seed
    )
    if not seed_keys:
        raise RuntimeError(f"confirmation seed {trial_seed}がありません")
    return math.fsum(
        baseline_predictions[key].absolute_error
        - selected_predictions[key].absolute_error
        for key in seed_keys
    ) / len(seed_keys)


def _prediction_map(
    evaluation: ExternalFamilyCandidateEvaluation,
) -> dict[
    tuple[str, int, float, float],
    FamilyHoldoutPrediction,
]:
    return {
        (
            prediction.network_family,
            prediction.trial_seed,
            prediction.coupling_gain,
            prediction.disturbance_bound,
        ): prediction
        for prediction in evaluation.predictions
    }


def _validate_candidate_evaluation(
    *,
    training_points: tuple[RobustRepertoireTaskPoint, ...],
    testing_points: tuple[RobustRepertoireTaskPoint, ...],
    candidate: FamilyHoldoutCandidate,
) -> None:
    if not training_points or not testing_points:
        raise ValueError("training/testing pointは各1件以上必要です")
    training_families = {
        point.network_family for point in training_points
    }
    testing_families = {
        point.network_family for point in testing_points
    }
    if len(training_families) < 2:
        raise ValueError("training familyは2種類以上必要です")
    if len(testing_families) != 1:
        raise ValueError("testing familyは1種類に固定してください")
    if training_families.intersection(testing_families):
        raise ValueError("trainingとtestingのfamilyは分離してください")
    training_seeds = {point.trial_seed for point in training_points}
    testing_seeds = {point.trial_seed for point in testing_points}
    if training_seeds.intersection(testing_seeds):
        raise ValueError("trainingとtestingのseedは分離してください")
    if not candidate.name.strip() or not candidate.feature_names:
        raise ValueError("candidate名と特徴量は空にできません")
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


def _validate_confirmation_configuration(
    *,
    selected_candidate: FamilyHoldoutCandidate,
    baseline_candidates: tuple[FamilyHoldoutCandidate, ...],
    expected_raw_attractor_count: int,
    association_threshold: float,
) -> None:
    baseline_names = tuple(
        candidate.name for candidate in baseline_candidates
    )
    if len(set(baseline_names)) != len(baseline_names):
        raise ValueError("baseline名は重複禁止です")
    for required_name in ("raw_count", "structural"):
        if required_name not in baseline_names:
            raise ValueError(f"{required_name} baselineが必要です")
    if selected_candidate.name in baseline_names:
        raise ValueError("selected candidateをbaselineへ含めないでください")
    if (
        not isinstance(expected_raw_attractor_count, int)
        or isinstance(expected_raw_attractor_count, bool)
        or expected_raw_attractor_count < 1
    ):
        raise ValueError("expected raw countは1以上の整数にしてください")
    if (
        not math.isfinite(association_threshold)
        or not 0.0 < association_threshold < 1.0
    ):
        raise ValueError("association thresholdは0と1の間にしてください")
