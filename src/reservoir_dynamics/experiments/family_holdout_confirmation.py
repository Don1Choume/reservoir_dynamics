"""事前登録した未知family・未知seed予測を確認する。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.cross_family_robust_task_confirmation import (
    DEFAULT_CONFIRMATION_SEEDS,
    DEFAULT_DISCOVERY_SEEDS,
    DEFAULT_FAMILY_SPECIFICATIONS,
    FamilyTaskSpecification,
    run_cross_family_robust_task_confirmation,
)
from reservoir_dynamics.experiments.family_holdout_robust_task import (
    EXPERIMENT_ID,
    FamilyHoldoutCandidate,
    FamilyHoldoutCandidateEvaluation,
    FamilyHoldoutPrediction,
    evaluate_family_holdout_candidate,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
)
from reservoir_dynamics.metrics.bootstrap import bootstrap_mean_interval

PREREGISTERED_TRAINING_SEEDS = (
    DEFAULT_DISCOVERY_SEEDS + DEFAULT_CONFIRMATION_SEEDS
)
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(1201, 1231))
PREREGISTERED_SELECTED_CANDIDATE = FamilyHoldoutCandidate(
    name="robust_pair",
    feature_names=(
        "normalized_mean_margin",
        "certified_robust_fraction",
    ),
    penalty=1e-3,
)
PREREGISTERED_BASELINE_CANDIDATES = (
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
)


@dataclass(frozen=True, slots=True)
class NamedBootstrapMeanInterval:
    """比較対象名を伴うseed単位のpaired bootstrap区間。"""

    baseline_name: str
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int


@dataclass(frozen=True, slots=True)
class FamilyHoldoutConfirmationDecisions:
    """未使用seedを見る前に固定する主要判定。"""

    raw_count_matched: bool
    certificate_lower_bound_valid: bool
    all_family_rank_association: bool
    selected_beats_raw_count: bool
    selected_beats_structural: bool


@dataclass(frozen=True, slots=True)
class FamilyHoldoutConfirmationResult:
    """未知family・未知seedの二重holdout確認結果。"""

    experiment_id: str
    phase: str
    training_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    selected_evaluation: FamilyHoldoutCandidateEvaluation
    baseline_evaluations: tuple[
        FamilyHoldoutCandidateEvaluation, ...
    ]
    baseline_minus_selected_intervals: tuple[
        NamedBootstrapMeanInterval, ...
    ]
    confirmation_point_count: int
    guarantee_violation_count: int
    decisions: FamilyHoldoutConfirmationDecisions


def run_preregistered_family_holdout_confirmation(
    *,
    training_seeds: tuple[int, ...] = PREREGISTERED_TRAINING_SEEDS,
    confirmation_seeds: tuple[
        int, ...
    ] = PREREGISTERED_CONFIRMATION_SEEDS,
    family_specifications: tuple[
        FamilyTaskSpecification, ...
    ] = DEFAULT_FAMILY_SPECIFICATIONS,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    task_steps: int = 100,
    autonomous_steps: int = 500,
    bootstrap_resamples: int = 2_000,
) -> FamilyHoldoutConfirmationResult:
    """固定済み設定で未知family・未知seed確認を再現する。"""

    source_result = run_cross_family_robust_task_confirmation(
        discovery_seeds=training_seeds,
        confirmation_seeds=confirmation_seeds,
        family_specifications=family_specifications,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        bootstrap_resamples=20,
    )
    return evaluate_family_holdout_confirmation(
        points=source_result.points,
        training_seeds=training_seeds,
        confirmation_seeds=confirmation_seeds,
        selected_candidate=PREREGISTERED_SELECTED_CANDIDATE,
        baseline_candidates=PREREGISTERED_BASELINE_CANDIDATES,
        expected_raw_attractor_count=2**dimension,
        association_threshold=0.75,
        bootstrap_resamples=bootstrap_resamples,
    )


def evaluate_family_holdout_confirmation(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    training_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
    selected_candidate: FamilyHoldoutCandidate,
    baseline_candidates: tuple[FamilyHoldoutCandidate, ...],
    expected_raw_attractor_count: int,
    association_threshold: float = 0.75,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_811,
) -> FamilyHoldoutConfirmationResult:
    """固定candidateを未知familyの未使用seedへ一度だけ適用する。"""

    _validate_confirmation_configuration(
        selected_candidate=selected_candidate,
        baseline_candidates=baseline_candidates,
        expected_raw_attractor_count=expected_raw_attractor_count,
        association_threshold=association_threshold,
    )
    selected_evaluation = evaluate_family_holdout_candidate(
        points=points,
        training_seeds=training_seeds,
        testing_seeds=confirmation_seeds,
        candidate=selected_candidate,
    )
    baseline_evaluations = tuple(
        evaluate_family_holdout_candidate(
            points=points,
            training_seeds=training_seeds,
            testing_seeds=confirmation_seeds,
            candidate=baseline,
        )
        for baseline in baseline_candidates
    )
    intervals = tuple(
        _build_baseline_interval(
            selected_evaluation=selected_evaluation,
            baseline_evaluation=baseline_evaluation,
            confirmation_seeds=confirmation_seeds,
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
    confirmation_points = tuple(
        point
        for point in points
        if point.trial_seed in confirmation_seeds
    )
    guarantee_violation_count = sum(
        point.guarantee_gap < -1e-12 for point in confirmation_points
    )
    return FamilyHoldoutConfirmationResult(
        experiment_id=EXPERIMENT_ID,
        phase="confirmation",
        training_seeds=training_seeds,
        confirmation_seeds=confirmation_seeds,
        selected_evaluation=selected_evaluation,
        baseline_evaluations=baseline_evaluations,
        baseline_minus_selected_intervals=intervals,
        confirmation_point_count=len(confirmation_points),
        guarantee_violation_count=guarantee_violation_count,
        decisions=FamilyHoldoutConfirmationDecisions(
            raw_count_matched=all(
                point.raw_attractor_count
                == expected_raw_attractor_count
                for point in confirmation_points
            ),
            certificate_lower_bound_valid=(
                guarantee_violation_count == 0
            ),
            all_family_rank_association=all(
                fold.test_spearman > association_threshold
                for fold in selected_evaluation.folds
            ),
            selected_beats_raw_count=(
                interval_by_name["raw_count"].lower > 0.0
            ),
            selected_beats_structural=(
                interval_by_name["structural"].lower > 0.0
            ),
        ),
    )


def _build_baseline_interval(
    *,
    selected_evaluation: FamilyHoldoutCandidateEvaluation,
    baseline_evaluation: FamilyHoldoutCandidateEvaluation,
    confirmation_seeds: tuple[int, ...],
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> NamedBootstrapMeanInterval:
    differences = _paired_seed_error_differences(
        selected_evaluation=selected_evaluation,
        baseline_evaluation=baseline_evaluation,
        confirmation_seeds=confirmation_seeds,
    )
    interval = bootstrap_mean_interval(
        differences,
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


def _paired_seed_error_differences(
    *,
    selected_evaluation: FamilyHoldoutCandidateEvaluation,
    baseline_evaluation: FamilyHoldoutCandidateEvaluation,
    confirmation_seeds: tuple[int, ...],
) -> tuple[float, ...]:
    selected_predictions = _prediction_map(selected_evaluation)
    baseline_predictions = _prediction_map(baseline_evaluation)
    if selected_predictions.keys() != baseline_predictions.keys():
        raise RuntimeError("selectedとbaselineの評価条件が一致しません")
    differences: list[float] = []
    for trial_seed in confirmation_seeds:
        seed_keys = tuple(
            key for key in selected_predictions if key[1] == trial_seed
        )
        if not seed_keys:
            raise RuntimeError(
                f"confirmation seed {trial_seed}の予測がありません"
            )
        differences.append(
            math.fsum(
                baseline_predictions[key].absolute_error
                - selected_predictions[key].absolute_error
                for key in seed_keys
            )
            / len(seed_keys)
        )
    return tuple(differences)


def _prediction_map(
    evaluation: FamilyHoldoutCandidateEvaluation,
) -> dict[
    tuple[str, int, float, float],
    FamilyHoldoutPrediction,
]:
    predictions = tuple(
        prediction
        for fold in evaluation.folds
        for prediction in fold.predictions
    )
    return {
        (
            prediction.network_family,
            prediction.trial_seed,
            prediction.coupling_gain,
            prediction.disturbance_bound,
        ): prediction
        for prediction in predictions
    }


def _validate_confirmation_configuration(
    *,
    selected_candidate: FamilyHoldoutCandidate,
    baseline_candidates: tuple[FamilyHoldoutCandidate, ...],
    expected_raw_attractor_count: int,
    association_threshold: float,
) -> None:
    baseline_names = tuple(
        baseline.name for baseline in baseline_candidates
    )
    if len(set(baseline_names)) != len(baseline_names):
        raise ValueError("baseline名は重複禁止です")
    for required_name in ("raw_count", "structural"):
        if required_name not in baseline_names:
            raise ValueError(
                f"{required_name} baselineが必要です"
            )
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


def _summary_payload(
    result: FamilyHoldoutConfirmationResult,
) -> dict[str, object]:
    """巨大なpoint列を除き、論文追跡に必要な導出値だけを残す。"""

    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "training_seeds": result.training_seeds,
        "confirmation_seeds": result.confirmation_seeds,
        "confirmation_point_count": result.confirmation_point_count,
        "guarantee_violation_count": result.guarantee_violation_count,
        "selected_candidate": asdict(
            result.selected_evaluation.candidate
        ),
        "selected_pooled_test_mae": (
            result.selected_evaluation.pooled_test_mae
        ),
        "selected_pooled_test_spearman": (
            result.selected_evaluation.pooled_test_spearman
        ),
        "selected_folds": tuple(
            {
                "held_out_family": fold.held_out_family,
                "test_mae": fold.test_mae,
                "test_spearman": fold.test_spearman,
            }
            for fold in result.selected_evaluation.folds
        ),
        "baselines": tuple(
            {
                "name": evaluation.candidate.name,
                "pooled_test_mae": evaluation.pooled_test_mae,
                "pooled_test_spearman": (
                    evaluation.pooled_test_spearman
                ),
            }
            for evaluation in result.baseline_evaluations
        ),
        "baseline_minus_selected_intervals": tuple(
            asdict(interval)
            for interval in result.baseline_minus_selected_intervals
        ),
        "decisions": asdict(result.decisions),
    }


def main() -> None:
    """事前登録確認を実行し、導出済み要約を標準出力へ書く。"""

    print(
        json.dumps(
            _summary_payload(
                run_preregistered_family_holdout_confirmation()
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
