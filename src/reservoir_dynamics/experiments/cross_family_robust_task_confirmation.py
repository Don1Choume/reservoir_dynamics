"""robust repertoire task予測を複数network familyで確認する。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.recurrent_weight_families import (
    RecurrentWeightFamily,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
    run_robust_repertoire_task_diagnostics,
)
from reservoir_dynamics.experiments.robust_repertoire_task_confirmation import (
    LinearPredictor,
    RobustTaskConfirmationComparison,
    _build_comparison,
    _fit_predictor,
    _mean_absolute_error,
)
from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)

EXPERIMENT_ID = "EXP-2026-010"
DEFAULT_DISCOVERY_SEEDS = tuple(range(801, 831))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(901, 931))


@dataclass(frozen=True, slots=True)
class FamilyTaskSpecification:
    """familyごとにpilotで固定したcount-matched task条件。"""

    network_family: RecurrentWeightFamily
    coupling_gains: tuple[float, ...]
    disturbance_bound: float


DEFAULT_FAMILY_SPECIFICATIONS = (
    FamilyTaskSpecification(
        network_family="dense_symmetric",
        coupling_gains=(0.04, 0.05, 0.06, 0.07),
        disturbance_bound=0.16,
    ),
    FamilyTaskSpecification(
        network_family="sparse_symmetric",
        coupling_gains=(0.04, 0.06, 0.08, 0.10),
        disturbance_bound=0.12,
    ),
    FamilyTaskSpecification(
        network_family="asymmetric_dense",
        coupling_gains=(0.04, 0.05, 0.06, 0.07),
        disturbance_bound=0.16,
    ),
    FamilyTaskSpecification(
        network_family="feedforward_nonnormal",
        coupling_gains=(0.04, 0.05, 0.06, 0.07),
        disturbance_bound=0.16,
    ),
)


@dataclass(frozen=True, slots=True)
class FamilyTaskComparison:
    """network family名を保持したconfirmation比較。"""

    network_family: RecurrentWeightFamily
    comparison: RobustTaskConfirmationComparison
    local_jacobian_test_mae: float
    off_diagonal_norm_test_mae: float
    minimum_coordinate_test_mae: float
    nonnormality_test_mae: float


@dataclass(frozen=True, slots=True)
class CrossFamilyRobustTaskDecisions:
    """confirmation前に固定するfamily横断判定。"""

    raw_count_matched: bool
    certificate_lower_bound_valid: bool
    all_family_margin_association: bool
    pooled_margin_beats_raw_count: bool


@dataclass(frozen=True, slots=True)
class CrossFamilyRobustTaskResult:
    """4 familyのdiscovery fitと未知seed評価。"""

    experiment_id: str
    discovery_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    family_specifications: tuple[FamilyTaskSpecification, ...]
    dimension: int
    task_steps: int
    points: tuple[RobustRepertoireTaskPoint, ...]
    family_comparisons: tuple[FamilyTaskComparison, ...]
    pooled_coupling_minus_margin_error: BootstrapMeanInterval
    pooled_local_jacobian_minus_margin_error: BootstrapMeanInterval
    pooled_raw_minus_margin_error: BootstrapMeanInterval
    decisions: CrossFamilyRobustTaskDecisions


@dataclass(frozen=True, slots=True)
class _FamilyPredictors:
    network_family: RecurrentWeightFamily
    margin: LinearPredictor
    coupling: LinearPredictor
    raw_count: LinearPredictor
    local_jacobian: LinearPredictor
    off_diagonal_norm: LinearPredictor
    minimum_coordinate: LinearPredictor
    nonnormality: LinearPredictor


def run_cross_family_robust_task_confirmation(
    *,
    discovery_seeds: tuple[int, ...] = DEFAULT_DISCOVERY_SEEDS,
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
    family_specifications: tuple[
        FamilyTaskSpecification, ...
    ] = DEFAULT_FAMILY_SPECIFICATIONS,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    task_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    association_threshold: float = 0.75,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_740,
) -> CrossFamilyRobustTaskResult:
    """family別fitを固定し、未知seedで横断的再現性を判定する。"""

    _validate_configuration(
        discovery_seeds=discovery_seeds,
        confirmation_seeds=confirmation_seeds,
        family_specifications=family_specifications,
        association_threshold=association_threshold,
        bootstrap_resamples=bootstrap_resamples,
    )
    combined_seeds = discovery_seeds + confirmation_seeds
    all_points: list[RobustRepertoireTaskPoint] = []
    family_comparisons: list[FamilyTaskComparison] = []
    family_predictors: list[_FamilyPredictors] = []
    for family_index, specification in enumerate(family_specifications):
        diagnostics = run_robust_repertoire_task_diagnostics(
            trial_seeds=combined_seeds,
            network_family=specification.network_family,
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gains=specification.coupling_gains,
            disturbance_bounds=(specification.disturbance_bound,),
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
            training_seed_count=len(discovery_seeds),
        )
        points = diagnostics.points
        all_points.extend(points)
        discovery_points = tuple(
            point
            for point in points
            if point.trial_seed in discovery_seeds
        )
        confirmation_points = tuple(
            point
            for point in points
            if point.trial_seed in confirmation_seeds
        )
        predictors = _fit_family_predictors(
            network_family=specification.network_family,
            discovery_points=discovery_points,
        )
        family_predictors.append(predictors)
        family_comparisons.append(
            FamilyTaskComparison(
                network_family=specification.network_family,
                comparison=_build_comparison(
                    points=points,
                    discovery_seeds=discovery_seeds,
                    confirmation_seeds=confirmation_seeds,
                    confidence_level=bootstrap_confidence_level,
                    resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed + family_index * 10,
                ),
                local_jacobian_test_mae=_mean_absolute_error(
                    confirmation_points,
                    predictors.local_jacobian,
                    _local_jacobian_feature,
                ),
                off_diagonal_norm_test_mae=_mean_absolute_error(
                    confirmation_points,
                    predictors.off_diagonal_norm,
                    lambda point: point.off_diagonal_infinity_norm,
                ),
                minimum_coordinate_test_mae=_mean_absolute_error(
                    confirmation_points,
                    predictors.minimum_coordinate,
                    _minimum_coordinate_feature,
                ),
                nonnormality_test_mae=_mean_absolute_error(
                    confirmation_points,
                    predictors.nonnormality,
                    lambda point: point.nonnormality_commutator_norm,
                ),
            )
        )
    point_tuple = tuple(all_points)
    pooled_differences = tuple(
        _pooled_seed_error_differences(
            points=tuple(
                point
                for point in point_tuple
                if point.trial_seed == trial_seed
            ),
            family_predictors=tuple(family_predictors),
        )
        for trial_seed in confirmation_seeds
    )
    pooled_intervals = tuple(
        bootstrap_mean_interval(
            tuple(values[index] for values in pooled_differences),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed + 100 + index,
        )
        for index in range(3)
    )
    confirmation_points = tuple(
        point
        for point in point_tuple
        if point.trial_seed in confirmation_seeds
    )
    comparisons = tuple(family_comparisons)
    expected_raw_count = 2**dimension
    return CrossFamilyRobustTaskResult(
        experiment_id=EXPERIMENT_ID,
        discovery_seeds=discovery_seeds,
        confirmation_seeds=confirmation_seeds,
        family_specifications=family_specifications,
        dimension=dimension,
        task_steps=task_steps,
        points=point_tuple,
        family_comparisons=comparisons,
        pooled_coupling_minus_margin_error=pooled_intervals[0],
        pooled_local_jacobian_minus_margin_error=pooled_intervals[1],
        pooled_raw_minus_margin_error=pooled_intervals[2],
        decisions=CrossFamilyRobustTaskDecisions(
            raw_count_matched=all(
                point.raw_attractor_count == expected_raw_count
                for point in confirmation_points
            ),
            certificate_lower_bound_valid=all(
                point.guarantee_gap >= -1e-12
                for point in confirmation_points
            ),
            all_family_margin_association=all(
                comparison.comparison.testing_spearman_mean_margin
                > association_threshold
                for comparison in comparisons
            ),
            pooled_margin_beats_raw_count=(
                pooled_intervals[2].lower > 0.0
            ),
        ),
    )


def _fit_family_predictors(
    *,
    network_family: RecurrentWeightFamily,
    discovery_points: tuple[RobustRepertoireTaskPoint, ...],
) -> _FamilyPredictors:
    return _FamilyPredictors(
        network_family=network_family,
        margin=_fit_predictor(
            discovery_points,
            lambda point: point.mean_uniform_disturbance_margin,
        ),
        coupling=_fit_predictor(
            discovery_points,
            lambda point: point.coupling_gain,
        ),
        raw_count=_fit_predictor(
            discovery_points,
            lambda point: float(point.raw_attractor_count),
        ),
        local_jacobian=_fit_predictor(
            discovery_points,
            _local_jacobian_feature,
        ),
        off_diagonal_norm=_fit_predictor(
            discovery_points,
            lambda point: point.off_diagonal_infinity_norm,
        ),
        minimum_coordinate=_fit_predictor(
            discovery_points,
            _minimum_coordinate_feature,
        ),
        nonnormality=_fit_predictor(
            discovery_points,
            lambda point: point.nonnormality_commutator_norm,
        ),
    )


def _pooled_seed_error_differences(
    *,
    points: tuple[RobustRepertoireTaskPoint, ...],
    family_predictors: tuple[_FamilyPredictors, ...],
) -> tuple[float, float, float]:
    absolute_errors: list[tuple[float, float, float, float]] = []
    for predictors in family_predictors:
        family_points = tuple(
            point
            for point in points
            if point.network_family == predictors.network_family
        )
        absolute_errors.extend(
            (
                abs(
                    point.task_retention
                    - predictors.margin.predict(
                        point.mean_uniform_disturbance_margin
                    )
                ),
                abs(
                    point.task_retention
                    - predictors.coupling.predict(point.coupling_gain)
                ),
                abs(
                    point.task_retention
                    - predictors.raw_count.predict(
                        float(point.raw_attractor_count)
                    )
                ),
                abs(
                    point.task_retention
                    - predictors.local_jacobian.predict(
                        _local_jacobian_feature(point)
                    )
                ),
            )
            for point in family_points
        )
    if not absolute_errors:
        raise RuntimeError("confirmation誤差を計算するpointがありません")
    margin_error, coupling_error, raw_error, local_jacobian_error = (
        math.fsum(values[index] for values in absolute_errors)
        / len(absolute_errors)
        for index in range(4)
    )
    return (
        coupling_error - margin_error,
        local_jacobian_error - margin_error,
        raw_error - margin_error,
    )


def _local_jacobian_feature(point: RobustRepertoireTaskPoint) -> float:
    value = point.maximum_local_jacobian_infinity_norm
    if value is None:
        raise RuntimeError("固定点がないnetworkの局所Jacobianは比較できません")
    return value


def _minimum_coordinate_feature(point: RobustRepertoireTaskPoint) -> float:
    value = point.minimum_fixed_point_coordinate
    if value is None:
        raise RuntimeError("固定点がないnetworkの最小座標は比較できません")
    return value


def _validate_configuration(
    *,
    discovery_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
    family_specifications: tuple[FamilyTaskSpecification, ...],
    association_threshold: float,
    bootstrap_resamples: int,
) -> None:
    if set(discovery_seeds).intersection(confirmation_seeds):
        raise ValueError("discovery_seedsとconfirmation_seedsは重複禁止です")
    if len(discovery_seeds) < 4 or len(confirmation_seeds) < 4:
        raise ValueError("discoveryとconfirmationは各4 seed以上にしてください")
    families = tuple(
        specification.network_family
        for specification in family_specifications
    )
    if len(families) < 2 or len(set(families)) != len(families):
        raise ValueError("familyは重複しない2種類以上にしてください")
    if (
        not math.isfinite(association_threshold)
        or not 0.0 < association_threshold < 1.0
    ):
        raise ValueError("association_thresholdは0と1の間にしてください")
    if (
        not isinstance(bootstrap_resamples, int)
        or isinstance(bootstrap_resamples, bool)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamplesは1以上の整数にしてください")


def main() -> None:
    """既定のfamily横断confirmationをJSONで出力する。"""

    print(
        json.dumps(
            asdict(run_cross_family_robust_task_confirmation()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
