"""pilotで固定したrobust task予測を未使用seedで確認する。"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
    _spearman,
    run_robust_repertoire_task_diagnostics,
)
from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)

EXPERIMENT_ID = "EXP-2026-009"
DEFAULT_DISCOVERY_SEEDS = tuple(range(401, 431))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(601, 631))


@dataclass(frozen=True, slots=True)
class LinearPredictor:
    """discovery条件だけで固定した単回帰predictor。"""

    slope: float
    intercept: float

    def predict(self, feature_value: float) -> float:
        return self.intercept + self.slope * feature_value


@dataclass(frozen=True, slots=True)
class RobustTaskConfirmationComparison:
    """一外乱強度のconfirmation関連と予測誤差差。"""

    disturbance_bound: float
    confirmation_condition_count: int
    testing_spearman_certified_fraction: float
    testing_spearman_mean_margin: float
    certified_fraction_test_mae: float
    mean_margin_test_mae: float
    coupling_gain_test_mae: float
    raw_count_test_mae: float
    coupling_minus_certified_error: BootstrapMeanInterval
    raw_minus_certified_error: BootstrapMeanInterval
    coupling_minus_margin_error: BootstrapMeanInterval
    raw_minus_margin_error: BootstrapMeanInterval
    guarantee_violation_count: int
    mean_guarantee_gap: float


@dataclass(frozen=True, slots=True)
class RobustTaskConfirmationDecisions:
    """pilot後、confirmation前に固定する判定。"""

    raw_count_matched: bool
    certificate_lower_bound_valid: bool
    robust_count_predicts_low_disturbance: bool
    mean_margin_predicts_high_disturbance: bool


@dataclass(frozen=True, slots=True)
class RobustTaskConfirmationResult:
    """discovery fitと未使用seed評価の全結果。"""

    experiment_id: str
    discovery_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    dimension: int
    coupling_gains: tuple[float, ...]
    disturbance_bounds: tuple[float, ...]
    task_steps: int
    points: tuple[RobustRepertoireTaskPoint, ...]
    comparisons: tuple[RobustTaskConfirmationComparison, ...]
    decisions: RobustTaskConfirmationDecisions


def run_robust_repertoire_task_confirmation(
    *,
    discovery_seeds: tuple[int, ...] = DEFAULT_DISCOVERY_SEEDS,
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
    dimension: int = 4,
    coupling_gains: tuple[float, ...] = (0.04, 0.05, 0.06, 0.07),
    disturbance_bounds: tuple[float, ...] = (0.08, 0.16),
    task_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    association_threshold: float = 0.75,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_735,
) -> RobustTaskConfirmationResult:
    """discovery回帰をfitし、disjoint confirmation seedだけで評価する。"""

    if set(discovery_seeds).intersection(confirmation_seeds):
        raise ValueError("discovery_seedsとconfirmation_seedsは重複禁止です")
    if (
        not math.isfinite(association_threshold)
        or association_threshold <= 0.0
        or association_threshold >= 1.0
    ):
        raise ValueError("association_thresholdは0と1の間にしてください")
    combined_seeds = discovery_seeds + confirmation_seeds
    diagnostics = run_robust_repertoire_task_diagnostics(
        trial_seeds=combined_seeds,
        dimension=dimension,
        diagonal_gain=1.5,
        coupling_gains=coupling_gains,
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        training_seed_count=len(discovery_seeds),
    )
    comparisons = tuple(
        _build_comparison(
            points=tuple(
                point
                for point in diagnostics.points
                if point.disturbance_bound == disturbance_bound
            ),
            discovery_seeds=discovery_seeds,
            confirmation_seeds=confirmation_seeds,
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + index * 100,
        )
        for index, disturbance_bound in enumerate(disturbance_bounds)
    )
    expected_raw_count = 2**dimension
    confirmation_points = tuple(
        point
        for point in diagnostics.points
        if point.trial_seed in confirmation_seeds
    )
    low_disturbance = comparisons[0]
    high_disturbance = comparisons[-1]
    return RobustTaskConfirmationResult(
        experiment_id=EXPERIMENT_ID,
        discovery_seeds=discovery_seeds,
        confirmation_seeds=confirmation_seeds,
        dimension=dimension,
        coupling_gains=coupling_gains,
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        points=diagnostics.points,
        comparisons=comparisons,
        decisions=RobustTaskConfirmationDecisions(
            raw_count_matched=all(
                point.raw_attractor_count == expected_raw_count
                for point in confirmation_points
            ),
            certificate_lower_bound_valid=all(
                point.guarantee_gap >= -1e-12
                for point in confirmation_points
            ),
            robust_count_predicts_low_disturbance=(
                low_disturbance.testing_spearman_certified_fraction
                > association_threshold
                and low_disturbance.raw_minus_certified_error.lower > 0.0
            ),
            mean_margin_predicts_high_disturbance=(
                high_disturbance.testing_spearman_mean_margin
                > association_threshold
                and high_disturbance.raw_minus_margin_error.lower > 0.0
            ),
        ),
    )


def _build_comparison(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    discovery_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> RobustTaskConfirmationComparison:
    discovery_points = tuple(
        point for point in points if point.trial_seed in discovery_seeds
    )
    confirmation_points = tuple(
        point for point in points if point.trial_seed in confirmation_seeds
    )
    features = (
        lambda point: point.certified_robust_fraction,
        lambda point: point.mean_uniform_disturbance_margin,
        lambda point: point.coupling_gain,
        lambda point: float(point.raw_attractor_count),
    )
    predictors = tuple(
        _fit_predictor(discovery_points, feature)
        for feature in features
    )
    seed_error_differences = tuple(
        _seed_error_differences(
            points=tuple(
                point
                for point in confirmation_points
                if point.trial_seed == trial_seed
            ),
            predictors=predictors,
            features=features,
        )
        for trial_seed in confirmation_seeds
    )
    intervals = tuple(
        bootstrap_mean_interval(
            tuple(values[index] for values in seed_error_differences),
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + index,
        )
        for index in range(4)
    )
    return RobustTaskConfirmationComparison(
        disturbance_bound=points[0].disturbance_bound,
        confirmation_condition_count=len(confirmation_points),
        testing_spearman_certified_fraction=_spearman(
            tuple(
                point.certified_robust_fraction
                for point in confirmation_points
            ),
            tuple(point.task_retention for point in confirmation_points),
        ),
        testing_spearman_mean_margin=_spearman(
            tuple(
                point.mean_uniform_disturbance_margin
                for point in confirmation_points
            ),
            tuple(point.task_retention for point in confirmation_points),
        ),
        certified_fraction_test_mae=_mean_absolute_error(
            confirmation_points,
            predictors[0],
            features[0],
        ),
        mean_margin_test_mae=_mean_absolute_error(
            confirmation_points,
            predictors[1],
            features[1],
        ),
        coupling_gain_test_mae=_mean_absolute_error(
            confirmation_points,
            predictors[2],
            features[2],
        ),
        raw_count_test_mae=_mean_absolute_error(
            confirmation_points,
            predictors[3],
            features[3],
        ),
        coupling_minus_certified_error=intervals[0],
        raw_minus_certified_error=intervals[1],
        coupling_minus_margin_error=intervals[2],
        raw_minus_margin_error=intervals[3],
        guarantee_violation_count=sum(
            point.guarantee_gap < -1e-12
            for point in confirmation_points
        ),
        mean_guarantee_gap=math.fsum(
            point.guarantee_gap for point in confirmation_points
        )
        / len(confirmation_points),
    )


def _fit_predictor(
    points: tuple[RobustRepertoireTaskPoint, ...],
    feature: Callable[[RobustRepertoireTaskPoint], float],
) -> LinearPredictor:
    feature_values = tuple(feature(point) for point in points)
    targets = tuple(point.task_retention for point in points)
    mean_feature = math.fsum(feature_values) / len(feature_values)
    mean_target = math.fsum(targets) / len(targets)
    feature_variance = math.fsum(
        (value - mean_feature) ** 2 for value in feature_values
    )
    if feature_variance <= 1e-15:
        return LinearPredictor(slope=0.0, intercept=mean_target)
    slope = math.fsum(
        (feature_value - mean_feature) * (target - mean_target)
        for feature_value, target in zip(
            feature_values,
            targets,
            strict=True,
        )
    ) / feature_variance
    return LinearPredictor(
        slope=slope,
        intercept=mean_target - slope * mean_feature,
    )


def _seed_error_differences(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    predictors: tuple[LinearPredictor, ...],
    features: tuple[
        Callable[[RobustRepertoireTaskPoint], float], ...
    ],
) -> tuple[float, float, float, float]:
    errors = tuple(
        _mean_absolute_error(points, predictor, feature)
        for predictor, feature in zip(
            predictors,
            features,
            strict=True,
        )
    )
    certified_error, margin_error, coupling_error, raw_error = errors
    return (
        coupling_error - certified_error,
        raw_error - certified_error,
        coupling_error - margin_error,
        raw_error - margin_error,
    )


def _mean_absolute_error(
    points: tuple[RobustRepertoireTaskPoint, ...],
    predictor: LinearPredictor,
    feature: Callable[[RobustRepertoireTaskPoint], float],
) -> float:
    return math.fsum(
        abs(
            point.task_retention
            - predictor.predict(feature(point))
        )
        for point in points
    ) / len(points)


def main() -> None:
    """既定discovery/confirmation実験をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_robust_repertoire_task_confirmation()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
