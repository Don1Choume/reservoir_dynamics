"""局所記憶と初期状態を跨いで利用できる共有readout記憶を検証する。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.multidimensional_memory_sweep import (
    create_seeded_orthogonal_reservoir,
    generate_trial_inputs,
    standard_replica_initial_states,
)
from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)
from reservoir_dynamics.metrics.linear_memory import (
    linear_memory_curve,
    shared_readout_memory_capacity,
)
from reservoir_dynamics.metrics.replica import pairwise_replica_distance_curve
from reservoir_dynamics.metrics.top_conditional_lyapunov import (
    top_conditional_lyapunov_exponent,
)
from reservoir_dynamics.simulation.discrete import simulate_discrete_replicas

EXPERIMENT_ID = "EXP-2026-004"
FOCUSED_CONDITIONS = (
    (0.6, 0.1),
    (0.9, 0.5),
    (1.2, 0.5),
    (1.2, 1.5),
    (1.5, 0.1),
    (1.5, 1.5),
)
DEFAULT_TRIAL_SEEDS = tuple(range(101, 131))


@dataclass(frozen=True, slots=True)
class ReplicaReadoutPoint:
    """一つの条件とseedに対する局所・共有readout評価。"""

    recurrent_gain: float
    input_gain: float
    trial_seed: int
    top_conditional_lyapunov_exponent: float
    tail_replica_rms_distance: float
    replica_synchronized: bool
    local_memory_capacity: float
    shared_mean_capacity: float
    shared_worst_capacity: float
    shared_worst_to_reference_ratio: float


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """同じgain条件に属するseed標本の区間推定。"""

    recurrent_gain: float
    input_gain: float
    seed_count: int
    top_exponent_mean: BootstrapMeanInterval
    synchronization_rate: BootstrapMeanInterval
    negative_exponent_without_sync_rate: BootstrapMeanInterval
    local_memory_mean: BootstrapMeanInterval
    shared_mean_capacity_mean: BootstrapMeanInterval
    shared_worst_capacity_mean: BootstrapMeanInterval
    shared_retention_mean: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class PairedContrast:
    """同じseedで対応付けた条件差または指標差。"""

    contrast_id: str
    baseline_condition: tuple[float, float]
    comparison_condition: tuple[float, float]
    metric: str
    mean_difference: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class ValidationDecisions:
    """事前規定したpilot所見の再現判定。"""

    negative_local_stability_without_sync: bool | None
    input_driven_sync_recovery: bool | None
    strong_contraction_memory_advantage: bool | None
    multistability_shared_readout_penalty: bool | None


@dataclass(frozen=True, slots=True)
class ReplicaReadoutValidationResult:
    """30 seed焦点化検証の全結果。"""

    experiment_id: str
    state_dimension: int
    condition_pairs: tuple[tuple[float, float], ...]
    trial_seeds: tuple[int, ...]
    washout: int
    training_steps: int
    testing_steps: int
    max_delay: int
    tail_window: int
    synchronization_tolerance: float
    ridge: float
    bootstrap_confidence_level: float
    bootstrap_resamples: int
    points: tuple[ReplicaReadoutPoint, ...]
    condition_summaries: tuple[ConditionSummary, ...]
    paired_contrasts: tuple[PairedContrast, ...]
    decisions: ValidationDecisions


def run_replica_readout_validation(
    *,
    state_dimension: int = 8,
    condition_pairs: tuple[tuple[float, float], ...] = FOCUSED_CONDITIONS,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    washout: int = 200,
    training_steps: int = 800,
    testing_steps: int = 400,
    max_delay: int = 12,
    tail_window: int = 100,
    synchronization_tolerance: float = 1e-7,
    ridge: float = 1e-8,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_730,
) -> ReplicaReadoutValidationResult:
    """焦点化したgain条件を同じ30 seedで比較する。"""

    _validate_configuration(
        state_dimension=state_dimension,
        condition_pairs=condition_pairs,
        trial_seeds=trial_seeds,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        max_delay=max_delay,
        tail_window=tail_window,
        synchronization_tolerance=synchronization_tolerance,
        ridge=ridge,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    steps = max(washout, max_delay) + training_steps + testing_steps
    points = tuple(
        _evaluate_point(
            state_dimension=state_dimension,
            recurrent_gain=recurrent_gain,
            input_gain=input_gain,
            trial_seed=trial_seed,
            steps=steps,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            max_delay=max_delay,
            tail_window=tail_window,
            synchronization_tolerance=synchronization_tolerance,
            ridge=ridge,
        )
        for recurrent_gain, input_gain in condition_pairs
        for trial_seed in trial_seeds
    )
    condition_summaries = tuple(
        _summarize_condition(
            points=tuple(
                point
                for point in points
                if (
                    point.recurrent_gain,
                    point.input_gain,
                )
                == condition
            ),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + condition_index * 100,
        )
        for condition_index, condition in enumerate(condition_pairs)
    )
    paired_contrasts = _build_paired_contrasts(
        points=points,
        condition_pairs=condition_pairs,
        trial_seeds=trial_seeds,
        confidence_level=bootstrap_confidence_level,
        resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 10_000,
    )
    return ReplicaReadoutValidationResult(
        experiment_id=EXPERIMENT_ID,
        state_dimension=state_dimension,
        condition_pairs=condition_pairs,
        trial_seeds=trial_seeds,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        max_delay=max_delay,
        tail_window=tail_window,
        synchronization_tolerance=synchronization_tolerance,
        ridge=ridge,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        points=points,
        condition_summaries=condition_summaries,
        paired_contrasts=paired_contrasts,
        decisions=_make_decisions(
            summaries=condition_summaries,
            contrasts=paired_contrasts,
        ),
    )


def _evaluate_point(
    *,
    state_dimension: int,
    recurrent_gain: float,
    input_gain: float,
    trial_seed: int,
    steps: int,
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    tail_window: int,
    synchronization_tolerance: float,
    ridge: float,
) -> ReplicaReadoutPoint:
    system = create_seeded_orthogonal_reservoir(
        state_dimension=state_dimension,
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        random_seed=trial_seed,
    )
    inputs = generate_trial_inputs(random_seed=trial_seed, steps=steps)
    simulation = simulate_discrete_replicas(
        system=system,
        initial_states=standard_replica_initial_states(state_dimension),
        inputs=inputs,
    )
    reference_trajectory = simulation.trajectories[0]
    scalar_inputs = tuple(input_value[0] for input_value in inputs)
    top_exponent = top_conditional_lyapunov_exponent(
        system=system,
        trajectory=reference_trajectory,
        inputs=inputs,
        washout=washout,
    )
    replica_distances = pairwise_replica_distance_curve(
        simulation.trajectories
    )
    tail_distances = replica_distances[-tail_window:]
    tail_rms_distance = math.sqrt(
        math.fsum(distance * distance for distance in tail_distances)
        / len(tail_distances)
    )
    local_memory = linear_memory_curve(
        states=reference_trajectory,
        inputs=scalar_inputs,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    shared_memory = shared_readout_memory_capacity(
        trajectories=simulation.trajectories,
        inputs=scalar_inputs,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    return ReplicaReadoutPoint(
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        trial_seed=trial_seed,
        top_conditional_lyapunov_exponent=top_exponent,
        tail_replica_rms_distance=tail_rms_distance,
        replica_synchronized=tail_rms_distance <= synchronization_tolerance,
        local_memory_capacity=local_memory.total_capacity,
        shared_mean_capacity=shared_memory.mean_total_capacity,
        shared_worst_capacity=shared_memory.worst_total_capacity,
        shared_worst_to_reference_ratio=(
            shared_memory.worst_to_reference_ratio
        ),
    )


def _summarize_condition(
    *,
    points: tuple[ReplicaReadoutPoint, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> ConditionSummary:
    first_point = points[0]
    value_extractors = (
        lambda point: point.top_conditional_lyapunov_exponent,
        lambda point: float(point.replica_synchronized),
        lambda point: float(
            point.top_conditional_lyapunov_exponent < 0.0
            and not point.replica_synchronized
        ),
        lambda point: point.local_memory_capacity,
        lambda point: point.shared_mean_capacity,
        lambda point: point.shared_worst_capacity,
        lambda point: point.shared_worst_to_reference_ratio,
    )
    intervals = tuple(
        bootstrap_mean_interval(
            tuple(extractor(point) for point in points),
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + metric_index,
        )
        for metric_index, extractor in enumerate(value_extractors)
    )
    return ConditionSummary(
        recurrent_gain=first_point.recurrent_gain,
        input_gain=first_point.input_gain,
        seed_count=len(points),
        top_exponent_mean=intervals[0],
        synchronization_rate=intervals[1],
        negative_exponent_without_sync_rate=intervals[2],
        local_memory_mean=intervals[3],
        shared_mean_capacity_mean=intervals[4],
        shared_worst_capacity_mean=intervals[5],
        shared_retention_mean=intervals[6],
    )


def _build_paired_contrasts(
    *,
    points: tuple[ReplicaReadoutPoint, ...],
    condition_pairs: tuple[tuple[float, float], ...],
    trial_seeds: tuple[int, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> tuple[PairedContrast, ...]:
    condition_set = frozenset(condition_pairs)
    point_lookup = {
        ((point.recurrent_gain, point.input_gain), point.trial_seed): point
        for point in points
    }
    contrast_specs = (
        (
            "strong_contraction_minus_edge_local_memory",
            (0.9, 0.5),
            (0.6, 0.1),
            "local_memory_capacity",
        ),
        (
            "input_recovery_a1_2_sync",
            (1.2, 0.5),
            (1.2, 1.5),
            "sync_indicator",
        ),
        (
            "input_recovery_a1_2_worst_shared",
            (1.2, 0.5),
            (1.2, 1.5),
            "shared_worst_capacity",
        ),
        (
            "input_recovery_a1_5_sync",
            (1.5, 0.1),
            (1.5, 1.5),
            "sync_indicator",
        ),
        (
            "input_recovery_a1_5_worst_shared",
            (1.5, 0.1),
            (1.5, 1.5),
            "shared_worst_capacity",
        ),
        (
            "multistability_penalty_a1_2_b0_5",
            (1.2, 0.5),
            (1.2, 0.5),
            "local_minus_worst_shared",
        ),
        (
            "multistability_penalty_a1_5_b0_1",
            (1.5, 0.1),
            (1.5, 0.1),
            "local_minus_worst_shared",
        ),
    )
    contrasts: list[PairedContrast] = []
    for contrast_index, (
        contrast_id,
        baseline_condition,
        comparison_condition,
        metric,
    ) in enumerate(contrast_specs):
        if (
            baseline_condition not in condition_set
            or comparison_condition not in condition_set
        ):
            continue
        differences = tuple(
            _paired_metric_difference(
                baseline=point_lookup[(baseline_condition, seed)],
                comparison=point_lookup[(comparison_condition, seed)],
                metric=metric,
            )
            for seed in trial_seeds
        )
        contrasts.append(
            PairedContrast(
                contrast_id=contrast_id,
                baseline_condition=baseline_condition,
                comparison_condition=comparison_condition,
                metric=metric,
                mean_difference=bootstrap_mean_interval(
                    differences,
                    confidence_level=confidence_level,
                    resamples=resamples,
                    random_seed=bootstrap_seed + contrast_index,
                ),
            )
        )
    return tuple(contrasts)


def _paired_metric_difference(
    *,
    baseline: ReplicaReadoutPoint,
    comparison: ReplicaReadoutPoint,
    metric: str,
) -> float:
    if metric == "sync_indicator":
        return float(comparison.replica_synchronized) - float(
            baseline.replica_synchronized
        )
    if metric == "local_minus_worst_shared":
        return (
            baseline.local_memory_capacity
            - baseline.shared_worst_capacity
        )
    return float(getattr(comparison, metric)) - float(
        getattr(baseline, metric)
    )


def _make_decisions(
    *,
    summaries: tuple[ConditionSummary, ...],
    contrasts: tuple[PairedContrast, ...],
) -> ValidationDecisions:
    summary_lookup = {
        (summary.recurrent_gain, summary.input_gain): summary
        for summary in summaries
    }
    contrast_lookup = {
        contrast.contrast_id: contrast for contrast in contrasts
    }
    negative_summary = summary_lookup.get((1.2, 0.5))
    sync_contrasts = tuple(
        contrast_lookup.get(contrast_id)
        for contrast_id in (
            "input_recovery_a1_2_sync",
            "input_recovery_a1_5_sync",
        )
    )
    contraction_contrast = contrast_lookup.get(
        "strong_contraction_minus_edge_local_memory"
    )
    penalty_contrasts = tuple(
        contrast_lookup.get(contrast_id)
        for contrast_id in (
            "multistability_penalty_a1_2_b0_5",
            "multistability_penalty_a1_5_b0_1",
        )
    )
    return ValidationDecisions(
        negative_local_stability_without_sync=(
            negative_summary.negative_exponent_without_sync_rate.estimate
            >= 0.8
            if negative_summary is not None
            else None
        ),
        input_driven_sync_recovery=(
            all(
                contrast.mean_difference.lower > 0.5
                for contrast in sync_contrasts
                if contrast is not None
            )
            if all(contrast is not None for contrast in sync_contrasts)
            else None
        ),
        strong_contraction_memory_advantage=(
            contraction_contrast.mean_difference.lower > 0.0
            if contraction_contrast is not None
            else None
        ),
        multistability_shared_readout_penalty=(
            any(
                contrast.mean_difference.lower > 0.5
                for contrast in penalty_contrasts
                if contrast is not None
            )
            if any(contrast is not None for contrast in penalty_contrasts)
            else None
        ),
    )


def _validate_configuration(
    *,
    state_dimension: int,
    condition_pairs: tuple[tuple[float, float], ...],
    trial_seeds: tuple[int, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    tail_window: int,
    synchronization_tolerance: float,
    ridge: float,
    bootstrap_confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> None:
    _validate_positive_integer(state_dimension, "state_dimension")
    if not condition_pairs:
        raise ValueError("condition_pairsは1要素以上必要です")
    if len(set(condition_pairs)) != len(condition_pairs):
        raise ValueError("condition_pairsを重複させないでください")
    if any(
        len(condition) != 2
        or any(not math.isfinite(value) or value < 0.0 for value in condition)
        for condition in condition_pairs
    ):
        raise ValueError("condition_pairsは有限の非負gain対にしてください")
    if len(trial_seeds) < 2:
        raise ValueError("trial_seedsは2要素以上必要です")
    if len(set(trial_seeds)) != len(trial_seeds) or any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in trial_seeds
    ):
        raise ValueError("trial_seedsは重複しない整数にしてください")
    _validate_non_negative_integer(washout, "washout")
    _validate_positive_integer(training_steps, "training_steps")
    _validate_positive_integer(testing_steps, "testing_steps")
    _validate_positive_integer(max_delay, "max_delay")
    _validate_positive_integer(tail_window, "tail_window")
    steps = max(washout, max_delay) + training_steps + testing_steps
    if tail_window > steps + 1:
        raise ValueError("tail_windowは軌道長以下にしてください")
    if (
        not math.isfinite(synchronization_tolerance)
        or synchronization_tolerance <= 0.0
    ):
        raise ValueError(
            "synchronization_toleranceは有限の正数にしてください"
        )
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridgeは有限の非負値にしてください")
    if (
        not math.isfinite(bootstrap_confidence_level)
        or bootstrap_confidence_level <= 0.0
        or bootstrap_confidence_level >= 1.0
    ):
        raise ValueError(
            "bootstrap_confidence_levelは0と1の間にしてください"
        )
    _validate_positive_integer(bootstrap_resamples, "bootstrap_resamples")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise ValueError("bootstrap_seedは整数にしてください")


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数にしてください")


def _validate_non_negative_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{value_name}は0以上の整数にしてください")


def main() -> None:
    """既定30 seed検証の全結果をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_replica_readout_validation()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
