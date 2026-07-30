"""機能coreを保護したreserve学習の十分条件を検証する。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.core_reserve_trial import (
    EntangledUpdatePoint,
    ProtectedUpdatePoint,
    evaluate_entangled_update,
    evaluate_protected_update,
    generate_trial_inputs,
    seeded_directions,
    select_adaptation,
    select_entangled_adaptation,
)
from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)

EXPERIMENT_ID = "EXP-2026-005"
DEFAULT_TRIAL_SEEDS = tuple(range(201, 231))
DEFAULT_FEEDBACK_GAINS = (0.0, 0.02, 0.05, 0.1)
DEFAULT_ADAPTATION_CANDIDATES = (
    (0.3, 0.25),
    (0.3, 0.75),
    (0.6, 0.25),
    (0.6, 0.75),
    (0.9, 0.25),
    (0.9, 0.75),
)


@dataclass(frozen=True, slots=True)
class FeedbackSummary:
    """feedback gainごとのseed集約。"""

    feedback_gain: float
    seed_count: int
    novel_capacity_gain: BootstrapMeanInterval
    core_retention: BootstrapMeanInterval
    max_core_deviation: BootstrapMeanInterval
    max_core_deviation_bound: BootstrapMeanInterval
    bound_satisfaction_rate: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class EntangledSummary:
    """core更新対照のseed集約。"""

    seed_count: int
    post_novel_capacity: BootstrapMeanInterval
    core_retention: BootstrapMeanInterval
    max_core_deviation: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class CoreReserveContrasts:
    """zero-feedback保護更新とcore更新の対応差。"""

    protected_novel_gain: BootstrapMeanInterval
    protected_minus_entangled_novel_capacity: BootstrapMeanInterval
    protected_minus_entangled_core_retention: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class ProtectionDecisions:
    """事前規定した十分条件と構成例の判定。"""

    exact_zero_feedback_protection: bool
    reserve_acquisition: bool
    finite_feedback_bound: bool
    protected_pareto_advantage: bool


@dataclass(frozen=True, slots=True)
class CoreReserveProtectionResult:
    """core–reserve ground-truth実験の全結果。"""

    experiment_id: str
    trial_seeds: tuple[int, ...]
    core_dimension: int
    reserve_dimension: int
    feedback_gains: tuple[float, ...]
    adaptation_candidates: tuple[tuple[float, float], ...]
    core_recurrent_gain: float
    core_input_gain: float
    max_delay: int
    protected_points: tuple[ProtectedUpdatePoint, ...]
    entangled_points: tuple[EntangledUpdatePoint, ...]
    feedback_summaries: tuple[FeedbackSummary, ...]
    entangled_summary: EntangledSummary
    contrasts: CoreReserveContrasts
    decisions: ProtectionDecisions


def run_core_reserve_protection_study(
    *,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    core_dimension: int = 4,
    reserve_dimension: int = 4,
    feedback_gains: tuple[float, ...] = DEFAULT_FEEDBACK_GAINS,
    adaptation_candidates: tuple[
        tuple[float, float], ...
    ] = DEFAULT_ADAPTATION_CANDIDATES,
    core_recurrent_gain: float = 0.6,
    core_input_gain: float = 0.75,
    calibration_washout: int = 50,
    calibration_training_steps: int = 200,
    calibration_testing_steps: int = 100,
    evaluation_washout: int = 100,
    evaluation_training_steps: int = 400,
    evaluation_testing_steps: int = 200,
    max_delay: int = 6,
    ridge: float = 1e-8,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_731,
) -> CoreReserveProtectionResult:
    """30 seedで保護更新、feedback漏洩、core更新対照を比較する。"""

    _validate_configuration(
        trial_seeds=trial_seeds,
        core_dimension=core_dimension,
        reserve_dimension=reserve_dimension,
        feedback_gains=feedback_gains,
        adaptation_candidates=adaptation_candidates,
        core_recurrent_gain=core_recurrent_gain,
        core_input_gain=core_input_gain,
        calibration_washout=calibration_washout,
        calibration_training_steps=calibration_training_steps,
        calibration_testing_steps=calibration_testing_steps,
        evaluation_washout=evaluation_washout,
        evaluation_training_steps=evaluation_training_steps,
        evaluation_testing_steps=evaluation_testing_steps,
        max_delay=max_delay,
        ridge=ridge,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    calibration_steps = (
        max(calibration_washout, max_delay)
        + calibration_training_steps
        + calibration_testing_steps
    )
    evaluation_steps = (
        max(evaluation_washout, max_delay)
        + evaluation_training_steps
        + evaluation_testing_steps
    )
    protected_points: list[ProtectedUpdatePoint] = []
    entangled_points: list[EntangledUpdatePoint] = []
    for trial_seed in trial_seeds:
        core_direction, reserve_direction = seeded_directions(
            dimension=core_dimension,
            trial_seed=trial_seed,
        )
        calibration_inputs = generate_trial_inputs(
            trial_seed=trial_seed + 100_000,
            steps=calibration_steps,
        )
        protected_adaptation = select_adaptation(
            core_dimension=core_dimension,
            core_direction=core_direction,
            reserve_direction=reserve_direction,
            core_recurrent_gain=core_recurrent_gain,
            core_input_gain=core_input_gain,
            candidates=adaptation_candidates,
            inputs=calibration_inputs,
            washout=calibration_washout,
            training_steps=calibration_training_steps,
            testing_steps=calibration_testing_steps,
            max_delay=max_delay,
            ridge=ridge,
        )
        entangled_adaptation = select_entangled_adaptation(
            core_dimension=core_dimension,
            core_direction=core_direction,
            reserve_direction=reserve_direction,
            core_recurrent_gain=core_recurrent_gain,
            core_input_gain=core_input_gain,
            candidates=adaptation_candidates,
            inputs=calibration_inputs,
            washout=calibration_washout,
            training_steps=calibration_training_steps,
            testing_steps=calibration_testing_steps,
            max_delay=max_delay,
            ridge=ridge,
        )
        evaluation_inputs = generate_trial_inputs(
            trial_seed=trial_seed + 200_000,
            steps=evaluation_steps,
        )
        protected_points.extend(
            evaluate_protected_update(
                trial_seed=trial_seed,
                dimension=core_dimension,
                core_direction=core_direction,
                reserve_direction=reserve_direction,
                core_recurrent_gain=core_recurrent_gain,
                core_input_gain=core_input_gain,
                feedback_gain=feedback_gain,
                adaptation=protected_adaptation,
                inputs=evaluation_inputs,
                washout=evaluation_washout,
                training_steps=evaluation_training_steps,
                testing_steps=evaluation_testing_steps,
                max_delay=max_delay,
                ridge=ridge,
            )
            for feedback_gain in feedback_gains
        )
        entangled_points.append(
            evaluate_entangled_update(
                trial_seed=trial_seed,
                dimension=core_dimension,
                core_direction=core_direction,
                reserve_direction=reserve_direction,
                core_recurrent_gain=core_recurrent_gain,
                core_input_gain=core_input_gain,
                adaptation=entangled_adaptation,
                inputs=evaluation_inputs,
                washout=evaluation_washout,
                training_steps=evaluation_training_steps,
                testing_steps=evaluation_testing_steps,
                max_delay=max_delay,
                ridge=ridge,
            )
        )

    protected_tuple = tuple(protected_points)
    entangled_tuple = tuple(entangled_points)
    feedback_summaries = tuple(
        _summarize_feedback(
            points=tuple(
                point
                for point in protected_tuple
                if point.feedback_gain == feedback_gain
            ),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + index * 100,
        )
        for index, feedback_gain in enumerate(feedback_gains)
    )
    contrasts = _build_contrasts(
        protected_points=protected_tuple,
        entangled_points=entangled_tuple,
        trial_seeds=trial_seeds,
        confidence_level=bootstrap_confidence_level,
        resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 20_000,
    )
    return CoreReserveProtectionResult(
        experiment_id=EXPERIMENT_ID,
        trial_seeds=trial_seeds,
        core_dimension=core_dimension,
        reserve_dimension=reserve_dimension,
        feedback_gains=feedback_gains,
        adaptation_candidates=adaptation_candidates,
        core_recurrent_gain=core_recurrent_gain,
        core_input_gain=core_input_gain,
        max_delay=max_delay,
        protected_points=protected_tuple,
        entangled_points=entangled_tuple,
        feedback_summaries=feedback_summaries,
        entangled_summary=_summarize_entangled(
            points=entangled_tuple,
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + 10_000,
        ),
        contrasts=contrasts,
        decisions=_make_decisions(
            protected_points=protected_tuple,
            contrasts=contrasts,
        ),
    )


def _summarize_feedback(
    *,
    points: tuple[ProtectedUpdatePoint, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> FeedbackSummary:
    extractors = (
        lambda point: point.post_novel_capacity - point.pre_novel_capacity,
        lambda point: point.core_retention,
        lambda point: point.max_core_deviation,
        lambda point: point.max_core_deviation_bound,
        lambda point: float(point.bound_satisfied),
    )
    intervals = tuple(
        bootstrap_mean_interval(
            tuple(extractor(point) for point in points),
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + index,
        )
        for index, extractor in enumerate(extractors)
    )
    return FeedbackSummary(
        feedback_gain=points[0].feedback_gain,
        seed_count=len(points),
        novel_capacity_gain=intervals[0],
        core_retention=intervals[1],
        max_core_deviation=intervals[2],
        max_core_deviation_bound=intervals[3],
        bound_satisfaction_rate=intervals[4],
    )


def _summarize_entangled(
    *,
    points: tuple[EntangledUpdatePoint, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> EntangledSummary:
    metric_values = (
        tuple(point.post_novel_capacity for point in points),
        tuple(point.core_retention for point in points),
        tuple(point.max_core_deviation for point in points),
    )
    intervals = tuple(
        bootstrap_mean_interval(
            values,
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + index,
        )
        for index, values in enumerate(metric_values)
    )
    return EntangledSummary(
        seed_count=len(points),
        post_novel_capacity=intervals[0],
        core_retention=intervals[1],
        max_core_deviation=intervals[2],
    )


def _build_contrasts(
    *,
    protected_points: tuple[ProtectedUpdatePoint, ...],
    entangled_points: tuple[EntangledUpdatePoint, ...],
    trial_seeds: tuple[int, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> CoreReserveContrasts:
    protected_lookup = {
        point.trial_seed: point
        for point in protected_points
        if point.feedback_gain == 0.0
    }
    entangled_lookup = {point.trial_seed: point for point in entangled_points}
    interval_values = (
        tuple(
            protected_lookup[seed].post_novel_capacity
            - protected_lookup[seed].pre_novel_capacity
            for seed in trial_seeds
        ),
        tuple(
            protected_lookup[seed].post_novel_capacity
            - entangled_lookup[seed].post_novel_capacity
            for seed in trial_seeds
        ),
        tuple(
            protected_lookup[seed].core_retention
            - entangled_lookup[seed].core_retention
            for seed in trial_seeds
        ),
    )
    intervals = tuple(
        bootstrap_mean_interval(
            values,
            confidence_level=confidence_level,
            resamples=resamples,
            random_seed=bootstrap_seed + index,
        )
        for index, values in enumerate(interval_values)
    )
    return CoreReserveContrasts(
        protected_novel_gain=intervals[0],
        protected_minus_entangled_novel_capacity=intervals[1],
        protected_minus_entangled_core_retention=intervals[2],
    )


def _make_decisions(
    *,
    protected_points: tuple[ProtectedUpdatePoint, ...],
    contrasts: CoreReserveContrasts,
) -> ProtectionDecisions:
    zero_feedback_points = tuple(
        point
        for point in protected_points
        if point.feedback_gain == 0.0
    )
    return ProtectionDecisions(
        exact_zero_feedback_protection=all(
            point.max_core_deviation <= 1e-12
            and point.core_retention >= 1.0 - 1e-12
            for point in zero_feedback_points
        ),
        reserve_acquisition=contrasts.protected_novel_gain.lower > 1.0,
        finite_feedback_bound=all(
            point.bound_satisfied for point in protected_points
        ),
        protected_pareto_advantage=(
            contrasts.protected_minus_entangled_core_retention.lower > 0.0
            and (
                contrasts.protected_minus_entangled_novel_capacity.lower
                > 0.0
            )
        ),
    )


def _validate_configuration(
    *,
    trial_seeds: tuple[int, ...],
    core_dimension: int,
    reserve_dimension: int,
    feedback_gains: tuple[float, ...],
    adaptation_candidates: tuple[tuple[float, float], ...],
    core_recurrent_gain: float,
    core_input_gain: float,
    calibration_washout: int,
    calibration_training_steps: int,
    calibration_testing_steps: int,
    evaluation_washout: int,
    evaluation_training_steps: int,
    evaluation_testing_steps: int,
    max_delay: int,
    ridge: float,
    bootstrap_confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> None:
    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in trial_seeds
    ):
        raise ValueError("trial_seedsは整数にしてください")
    _validate_positive_integer(core_dimension, "core_dimension")
    _validate_positive_integer(reserve_dimension, "reserve_dimension")
    if core_dimension != reserve_dimension:
        raise ValueError(
            "公平なparameter対照のためcoreとreserve次元を同じにしてください"
        )
    if (
        not feedback_gains
        or len(set(feedback_gains)) != len(feedback_gains)
        or 0.0 not in feedback_gains
        or any(
            not math.isfinite(gain) or gain < 0.0
            for gain in feedback_gains
        )
    ):
        raise ValueError(
            "feedback_gainsは0を含む重複なしの有限非負値にしてください"
        )
    if (
        not adaptation_candidates
        or len(set(adaptation_candidates)) != len(adaptation_candidates)
        or any(
            len(candidate) != 2
            or not math.isfinite(candidate[0])
            or candidate[0] < 0.0
            or candidate[0] >= 1.0
            or not math.isfinite(candidate[1])
            or candidate[1] <= 0.0
            for candidate in adaptation_candidates
        )
    ):
        raise ValueError(
            "adaptation_candidatesはrecurrent [0,1)とinput正値の対にしてください"
        )
    if (
        not math.isfinite(core_recurrent_gain)
        or core_recurrent_gain < 0.0
        or core_recurrent_gain >= 1.0
    ):
        raise ValueError("core_recurrent_gainは0以上1未満にしてください")
    if not math.isfinite(core_input_gain) or core_input_gain <= 0.0:
        raise ValueError("core_input_gainは有限の正値にしてください")
    _validate_non_negative_integer(calibration_washout, "calibration_washout")
    _validate_positive_integer(
        calibration_training_steps,
        "calibration_training_steps",
    )
    _validate_positive_integer(
        calibration_testing_steps,
        "calibration_testing_steps",
    )
    _validate_non_negative_integer(evaluation_washout, "evaluation_washout")
    _validate_positive_integer(
        evaluation_training_steps,
        "evaluation_training_steps",
    )
    _validate_positive_integer(
        evaluation_testing_steps,
        "evaluation_testing_steps",
    )
    _validate_positive_integer(max_delay, "max_delay")
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
    """既定30 seed実験をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_core_reserve_protection_study()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
