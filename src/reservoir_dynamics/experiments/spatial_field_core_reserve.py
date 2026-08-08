"""EXP-2026-015の構造gate、集計、事前登録判定。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from reservoir_dynamics.experiments.spatial_field_core_reserve_trial import (
    SpatialFieldConditionEvaluation,
    evaluate_spatial_field_condition,
)
from reservoir_dynamics.experiments.spatial_field_network import (
    build_spatial_core_reserve_network,
)
from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)
from reservoir_dynamics.systems.tanh_rnn import Matrix
from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
)


EXPERIMENT_ID = "EXP-2026-015"
DEFAULT_PILOT_SEEDS = tuple(range(1801, 1811))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(1901, 1931))
DEFAULT_FEEDBACK_GAINS = (0.48, 0.64, 0.80)
DEFAULT_DISTURBANCE_BOUNDS = (0.0, 0.04, 0.08)


@dataclass(frozen=True, slots=True)
class SpatialFieldStructureGate:
    """task結果を見る前に確認する有効構造多様性。"""

    raw_network_count: int
    effective_magnitude_class_count: int
    magnitude_classes_by_feedback_gain: tuple[tuple[float, int], ...]
    all_reserve_blocks_asymmetric: bool
    all_bridges_bidirectional_nonzero: bool
    unequal_module_sizes: bool


@dataclass(frozen=True, slots=True)
class SpatialFieldConditionSummary:
    """feedback gain x noise群のseed平均。"""

    feedback_gain: float
    disturbance_bound: float
    seed_count: int
    local_safe_box_retention: float
    ungated_safe_box_retention: float
    global_safe_box_retention: float
    local_reserve_capacity: float
    ungated_reserve_capacity: float
    global_reserve_capacity: float
    certified_challenge_fraction: float
    mean_gated_load_ratio: float


@dataclass(frozen=True, slots=True)
class SpatialFieldContrasts:
    """seedを推論単位とするpaired policy差。"""

    local_minus_ungated_core_retention: BootstrapMeanInterval
    local_minus_global_core_retention: BootstrapMeanInterval
    local_minus_global_reserve_capacity: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class SpatialFieldDecisions:
    """pilot前に固定した理論・構造sanity判定。"""

    field_hypercube_invariant: bool
    intervention_energy_matched: bool
    certificate_lower_bound_valid: bool
    structures_effectively_distinct: bool
    reserve_blocks_asymmetric: bool
    bridges_bidirectional_nonzero: bool
    local_core_fully_protected: bool
    maximum_feedback_advantage: bool
    local_beats_global_core: bool
    local_beats_global_reserve: bool


@dataclass(frozen=True, slots=True)
class SpatialFieldStudyResult:
    """EXP-2026-015の全結果。"""

    experiment_id: str
    trial_seeds: tuple[int, ...]
    feedback_gains: tuple[float, ...]
    disturbance_bounds: tuple[float, ...]
    structure_gate: SpatialFieldStructureGate
    points: tuple[SpatialFieldConditionEvaluation, ...]
    summaries: tuple[SpatialFieldConditionSummary, ...]
    contrasts: SpatialFieldContrasts
    decisions: SpatialFieldDecisions


def audit_spatial_field_structures(
    *,
    trial_seeds: tuple[int, ...],
    feedback_gains: tuple[float, ...] = DEFAULT_FEEDBACK_GAINS,
) -> SpatialFieldStructureGate:
    """絶対値fingerprint、非対称性、bridge方向を監査する。"""

    _validate_trial_seeds(trial_seeds)
    _validate_positive_grid(feedback_gains, "feedback_gains")
    networks = tuple(
        (
            feedback_gain,
            build_spatial_core_reserve_network(
                trial_seed=trial_seed,
                feedback_gain=feedback_gain,
            ),
        )
        for feedback_gain in feedback_gains
        for trial_seed in trial_seeds
    )
    fingerprints = tuple(
        network.magnitude_fingerprint for _, network in networks
    )
    class_counts = tuple(
        (
            feedback_gain,
            len(
                {
                    network.magnitude_fingerprint
                    for group_gain, network in networks
                    if group_gain == feedback_gain
                }
            ),
        )
        for feedback_gain in feedback_gains
    )
    return SpatialFieldStructureGate(
        raw_network_count=len(networks),
        effective_magnitude_class_count=len(set(fingerprints)),
        magnitude_classes_by_feedback_gain=class_counts,
        all_reserve_blocks_asymmetric=all(
            _is_asymmetric(network.reserve_recurrent_weights)
            for _, network in networks
        ),
        all_bridges_bidirectional_nonzero=all(
            _has_nonzero(network.reserve_to_core_weights)
            and _has_nonzero(network.core_to_reserve_weights)
            for _, network in networks
        ),
        unequal_module_sizes=all(
            network.core_dimension != network.reserve_dimension
            for _, network in networks
        ),
    )


def run_spatial_field_study(
    *,
    trial_seeds: tuple[int, ...] = DEFAULT_PILOT_SEEDS,
    feedback_gains: tuple[float, ...] = DEFAULT_FEEDBACK_GAINS,
    disturbance_bounds: tuple[float, ...] = DEFAULT_DISTURBANCE_BOUNDS,
    washout: int = 80,
    training_steps: int = 240,
    testing_steps: int = 120,
    max_delay: int = 6,
    ridge: float = 1e-8,
    diffusion_rate: float = 0.20,
    source_rate: float = 0.55,
    minimum_gate: float = 0.05,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_802,
    minimum_local_safe_box_retention: float = 1.0,
    minimum_max_feedback_core_advantage: float = 0.05,
    minimum_core_advantage_lower: float = 0.05,
    minimum_reserve_advantage_lower: float = 0.02,
) -> SpatialFieldStudyResult:
    """全条件を実行し、seed単位のpaired差と理論gateを返す。"""

    _validate_configuration(
        trial_seeds=trial_seeds,
        feedback_gains=feedback_gains,
        disturbance_bounds=disturbance_bounds,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        max_delay=max_delay,
        ridge=ridge,
        diffusion_rate=diffusion_rate,
        source_rate=source_rate,
        minimum_gate=minimum_gate,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        minimum_local_safe_box_retention=(
            minimum_local_safe_box_retention
        ),
        minimum_max_feedback_core_advantage=(
            minimum_max_feedback_core_advantage
        ),
        minimum_core_advantage_lower=minimum_core_advantage_lower,
        minimum_reserve_advantage_lower=minimum_reserve_advantage_lower,
    )
    structure_gate = audit_spatial_field_structures(
        trial_seeds=trial_seeds,
        feedback_gains=feedback_gains,
    )
    points = tuple(
        evaluate_spatial_field_condition(
            trial_seed=trial_seed,
            feedback_gain=feedback_gain,
            disturbance_bound=disturbance_bound,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            max_delay=max_delay,
            ridge=ridge,
            diffusion_rate=diffusion_rate,
            source_rate=source_rate,
            minimum_gate=minimum_gate,
        )
        for trial_seed in trial_seeds
        for feedback_gain in feedback_gains
        for disturbance_bound in disturbance_bounds
    )
    summaries = tuple(
        _summarize_condition(
            tuple(
                point
                for point in points
                if point.feedback_gain == feedback_gain
                and point.disturbance_bound == disturbance_bound
            )
        )
        for feedback_gain in feedback_gains
        for disturbance_bound in disturbance_bounds
    )
    contrasts = SpatialFieldContrasts(
        local_minus_ungated_core_retention=_contrast_interval(
            points=points,
            trial_seeds=trial_seeds,
            extractor=lambda point: (
                point.local_safe_box_retention
                - point.ungated_safe_box_retention
            ),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed,
        ),
        local_minus_global_core_retention=_contrast_interval(
            points=points,
            trial_seeds=trial_seeds,
            extractor=lambda point: (
                point.local_safe_box_retention
                - point.global_safe_box_retention
            ),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed + 1,
        ),
        local_minus_global_reserve_capacity=_contrast_interval(
            points=points,
            trial_seeds=trial_seeds,
            extractor=lambda point: (
                point.local_reserve_capacity
                - point.global_reserve_capacity
            ),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed + 2,
        ),
    )
    decisions = SpatialFieldDecisions(
        field_hypercube_invariant=all(
            point.minimum_field_value >= -1e-12
            and point.maximum_field_value <= 1.0 + 1e-12
            for point in points
        ),
        intervention_energy_matched=all(
            point.maximum_energy_match_error <= 1e-12
            for point in points
        ),
        certificate_lower_bound_valid=all(
            point.certificate_violation_count == 0 for point in points
        ),
        structures_effectively_distinct=(
            structure_gate.effective_magnitude_class_count
            == structure_gate.raw_network_count
        ),
        reserve_blocks_asymmetric=(
            structure_gate.all_reserve_blocks_asymmetric
        ),
        bridges_bidirectional_nonzero=(
            structure_gate.all_bridges_bidirectional_nonzero
        ),
        local_core_fully_protected=all(
            point.local_safe_box_retention
            >= minimum_local_safe_box_retention
            for point in points
        ),
        maximum_feedback_advantage=(
            _mean(
                tuple(
                    point.local_safe_box_retention
                    - point.ungated_safe_box_retention
                    for point in points
                    if point.feedback_gain == max(feedback_gains)
                )
            )
            >= minimum_max_feedback_core_advantage
        ),
        local_beats_global_core=(
            contrasts.local_minus_global_core_retention.lower
            >= minimum_core_advantage_lower
        ),
        local_beats_global_reserve=(
            contrasts.local_minus_global_reserve_capacity.lower
            >= minimum_reserve_advantage_lower
        ),
    )
    return SpatialFieldStudyResult(
        experiment_id=EXPERIMENT_ID,
        trial_seeds=trial_seeds,
        feedback_gains=feedback_gains,
        disturbance_bounds=disturbance_bounds,
        structure_gate=structure_gate,
        points=points,
        summaries=summaries,
        contrasts=contrasts,
        decisions=decisions,
    )


def _summarize_condition(
    points: tuple[SpatialFieldConditionEvaluation, ...],
) -> SpatialFieldConditionSummary:
    first = points[0]
    return SpatialFieldConditionSummary(
        feedback_gain=first.feedback_gain,
        disturbance_bound=first.disturbance_bound,
        seed_count=len(points),
        local_safe_box_retention=_mean(
            tuple(point.local_safe_box_retention for point in points)
        ),
        ungated_safe_box_retention=_mean(
            tuple(point.ungated_safe_box_retention for point in points)
        ),
        global_safe_box_retention=_mean(
            tuple(point.global_safe_box_retention for point in points)
        ),
        local_reserve_capacity=_mean(
            tuple(point.local_reserve_capacity for point in points)
        ),
        ungated_reserve_capacity=_mean(
            tuple(point.ungated_reserve_capacity for point in points)
        ),
        global_reserve_capacity=_mean(
            tuple(point.global_reserve_capacity for point in points)
        ),
        certified_challenge_fraction=_mean(
            tuple(point.certified_challenge_fraction for point in points)
        ),
        mean_gated_load_ratio=_mean(
            tuple(
                point.mean_gated_feedback_load
                / point.mean_raw_feedback_load
                if point.mean_raw_feedback_load > 0.0
                else 0.0
                for point in points
            )
        ),
    )


def _contrast_interval(
    *,
    points: tuple[SpatialFieldConditionEvaluation, ...],
    trial_seeds: tuple[int, ...],
    extractor: Callable[[SpatialFieldConditionEvaluation], float],
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> BootstrapMeanInterval:
    seed_means = tuple(
        _mean(
            tuple(
                extractor(point)
                for point in points
                if point.trial_seed == trial_seed
            )
        )
        for trial_seed in trial_seeds
    )
    return bootstrap_mean_interval(
        seed_means,
        confidence_level=confidence_level,
        resamples=resamples,
        random_seed=random_seed,
    )


def _is_asymmetric(matrix: Matrix) -> bool:
    return any(
        not math.isclose(
            matrix[row][column],
            matrix[column][row],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for row in range(len(matrix))
        for column in range(row + 1, len(matrix))
    )


def _has_nonzero(matrix: Matrix) -> bool:
    return any(value != 0.0 for row in matrix for value in row)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _validate_configuration(
    *,
    trial_seeds: tuple[int, ...],
    feedback_gains: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
    diffusion_rate: float,
    source_rate: float,
    minimum_gate: float,
    bootstrap_confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    minimum_local_safe_box_retention: float,
    minimum_max_feedback_core_advantage: float,
    minimum_core_advantage_lower: float,
    minimum_reserve_advantage_lower: float,
) -> None:
    _validate_trial_seeds(trial_seeds)
    _validate_positive_grid(feedback_gains, "feedback_gains")
    if (
        not disturbance_bounds
        or len(set(disturbance_bounds)) != len(disturbance_bounds)
        or tuple(sorted(disturbance_bounds)) != disturbance_bounds
        or any(
            not math.isfinite(value) or value < 0.0
            for value in disturbance_bounds
        )
    ):
        raise ValueError(
            "disturbance_boundsは昇順・重複なしの有限非負値にしてください"
        )
    critical_forcing = bistable_tanh_certificate(1.5).critical_forcing
    if max(disturbance_bounds) >= critical_forcing:
        raise ValueError(
            "disturbance_boundsはcritical forcing未満にしてください"
        )
    _validate_non_negative_integer(washout, "washout")
    _validate_positive_integer(training_steps, "training_steps")
    _validate_positive_integer(testing_steps, "testing_steps")
    _validate_positive_integer(max_delay, "max_delay")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridgeは有限の非負値にしてください")
    for value, name in (
        (diffusion_rate, "diffusion_rate"),
        (source_rate, "source_rate"),
        (minimum_gate, "minimum_gate"),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name}は[0, 1]内にしてください")
    if diffusion_rate + source_rate > 1.0:
        raise ValueError("diffusion_rateとsource_rateの合計は1以下にしてください")
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
    if (
        not math.isfinite(minimum_local_safe_box_retention)
        or minimum_local_safe_box_retention < 0.0
        or minimum_local_safe_box_retention > 1.0
    ):
        raise ValueError(
            "minimum_local_safe_box_retentionは[0, 1]内にしてください"
        )
    for threshold, name in (
        (
            minimum_max_feedback_core_advantage,
            "minimum_max_feedback_core_advantage",
        ),
        (minimum_core_advantage_lower, "minimum_core_advantage_lower"),
    ):
        if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
            raise ValueError(f"{name}は[-1, 1]内にしてください")
    if not math.isfinite(minimum_reserve_advantage_lower):
        raise ValueError(
            "minimum_reserve_advantage_lowerは有限値にしてください"
        )


def _validate_trial_seeds(trial_seeds: tuple[int, ...]) -> None:
    if (
        len(trial_seeds) < 2
        or len(set(trial_seeds)) != len(trial_seeds)
        or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in trial_seeds
        )
    ):
        raise ValueError(
            "trial_seedsは重複しない2個以上の整数にしてください"
        )


def _validate_positive_grid(values: tuple[float, ...], name: str) -> None:
    if (
        not values
        or len(set(values)) != len(values)
        or tuple(sorted(values)) != values
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
    ):
        raise ValueError(f"{name}は昇順・重複なしの有限正値にしてください")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name}は1以上の整数にしてください")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name}は0以上の整数にしてください")
