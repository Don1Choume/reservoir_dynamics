"""設計済み双安定coreと学習余剰reserveの安全余裕実験。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.bistable_reserve_trial import (
    AdversarialThresholdPoint,
    BistableProtectionPoint,
    evaluate_adversarial_threshold,
    evaluate_bistable_protection,
    select_reserve_attractor_adaptation,
)
from reservoir_dynamics.metrics.bootstrap import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)
from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
)

EXPERIMENT_ID = "EXP-2026-006"
DEFAULT_TRIAL_SEEDS = tuple(range(301, 331))
DEFAULT_ANALYTICAL_RECURRENT_GAINS = (1.2, 1.5, 2.0)
DEFAULT_FEEDBACK_RATIOS = (0.5, 0.9, 1.1, 1.5)
DEFAULT_ADAPTATION_CANDIDATES = (
    (1.1, 0.5),
    (1.3, 1.0),
    (1.5, 1.0),
    (1.8, 2.0),
)


@dataclass(frozen=True, slots=True)
class FeedbackRatioSummary:
    """臨界外力比ごとのseed bootstrap集約。"""

    feedback_ratio: float
    seed_count: int
    reserve_recall_gain: BootstrapMeanInterval
    certified_core_retention: BootstrapMeanInterval
    full_basin_core_retention: BootstrapMeanInterval
    opposing_core_retention: BootstrapMeanInterval
    mean_minimum_certified_margin: BootstrapMeanInterval


@dataclass(frozen=True, slots=True)
class BistableProtectionDecisions:
    """実行前に固定したEXP-006の判定。"""

    analytic_safe_invariance: bool
    analytic_threshold_tipping: bool
    reserve_attractor_acquisition: bool
    certified_core_preservation: bool
    margin_predicts_failure: bool


@dataclass(frozen=True, slots=True)
class BistableCoreProtectionResult:
    """解析閾値、seed別観測、集約、事前規定判定。"""

    experiment_id: str
    trial_seeds: tuple[int, ...]
    analytical_recurrent_gains: tuple[float, ...]
    core_recurrent_gain: float
    feedback_ratios: tuple[float, ...]
    adaptation_candidates: tuple[tuple[float, float], ...]
    core_invariant_boundary: float
    core_critical_forcing: float
    core_certified_uniform_fraction: float
    analytical_points: tuple[AdversarialThresholdPoint, ...]
    protection_points: tuple[BistableProtectionPoint, ...]
    feedback_summaries: tuple[FeedbackRatioSummary, ...]
    decisions: BistableProtectionDecisions


def run_bistable_core_protection_study(
    *,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    analytical_recurrent_gains: tuple[
        float, ...
    ] = DEFAULT_ANALYTICAL_RECURRENT_GAINS,
    core_recurrent_gain: float = 1.5,
    feedback_ratios: tuple[float, ...] = DEFAULT_FEEDBACK_RATIOS,
    adaptation_candidates: tuple[
        tuple[float, float], ...
    ] = DEFAULT_ADAPTATION_CANDIDATES,
    calibration_trials: int = 64,
    evaluation_trials: int = 128,
    analytical_steps: int = 2_000,
    evaluation_steps: int = 300,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_732,
) -> BistableCoreProtectionResult:
    """30 seedで安全域内保持と閾値超過tippingを比較する。"""

    _validate_configuration(
        trial_seeds=trial_seeds,
        analytical_recurrent_gains=analytical_recurrent_gains,
        core_recurrent_gain=core_recurrent_gain,
        feedback_ratios=feedback_ratios,
        adaptation_candidates=adaptation_candidates,
        calibration_trials=calibration_trials,
        evaluation_trials=evaluation_trials,
        analytical_steps=analytical_steps,
        evaluation_steps=evaluation_steps,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    certificate = bistable_tanh_certificate(core_recurrent_gain)
    analytical_points = tuple(
        evaluate_adversarial_threshold(
            recurrent_gain=recurrent_gain,
            feedback_ratio=feedback_ratio,
            steps=analytical_steps,
        )
        for recurrent_gain in analytical_recurrent_gains
        for feedback_ratio in feedback_ratios
    )
    protection_points: list[BistableProtectionPoint] = []
    for trial_seed in trial_seeds:
        adaptation = select_reserve_attractor_adaptation(
            trial_seed=trial_seed + 100_000,
            candidates=adaptation_candidates,
            calibration_trials=calibration_trials,
        )
        protection_points.extend(
            evaluate_bistable_protection(
                trial_seed=trial_seed + 200_000,
                core_recurrent_gain=core_recurrent_gain,
                feedback_ratio=feedback_ratio,
                adaptation=adaptation,
                evaluation_trials=evaluation_trials,
                evaluation_steps=evaluation_steps,
            )
            for feedback_ratio in feedback_ratios
        )
    point_tuple = tuple(protection_points)
    feedback_summaries = tuple(
        _summarize_feedback_ratio(
            points=tuple(
                point
                for point in point_tuple
                if point.feedback_ratio == feedback_ratio
            ),
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + index * 100,
        )
        for index, feedback_ratio in enumerate(feedback_ratios)
    )
    return BistableCoreProtectionResult(
        experiment_id=EXPERIMENT_ID,
        trial_seeds=trial_seeds,
        analytical_recurrent_gains=analytical_recurrent_gains,
        core_recurrent_gain=core_recurrent_gain,
        feedback_ratios=feedback_ratios,
        adaptation_candidates=adaptation_candidates,
        core_invariant_boundary=certificate.invariant_boundary,
        core_critical_forcing=certificate.critical_forcing,
        core_certified_uniform_fraction=(
            certificate.certified_uniform_fraction
        ),
        analytical_points=analytical_points,
        protection_points=point_tuple,
        feedback_summaries=feedback_summaries,
        decisions=_make_decisions(
            analytical_points=analytical_points,
            feedback_summaries=feedback_summaries,
        ),
    )


def _summarize_feedback_ratio(
    *,
    points: tuple[BistableProtectionPoint, ...],
    confidence_level: float,
    resamples: int,
    bootstrap_seed: int,
) -> FeedbackRatioSummary:
    extractors = (
        lambda point: point.reserve_recall_gain,
        lambda point: point.certified_core_retention,
        lambda point: point.full_basin_core_retention,
        lambda point: point.opposing_core_retention,
        lambda point: point.mean_minimum_certified_margin,
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
    return FeedbackRatioSummary(
        feedback_ratio=points[0].feedback_ratio,
        seed_count=len(points),
        reserve_recall_gain=intervals[0],
        certified_core_retention=intervals[1],
        full_basin_core_retention=intervals[2],
        opposing_core_retention=intervals[3],
        mean_minimum_certified_margin=intervals[4],
    )


def _make_decisions(
    *,
    analytical_points: tuple[AdversarialThresholdPoint, ...],
    feedback_summaries: tuple[FeedbackRatioSummary, ...],
) -> BistableProtectionDecisions:
    safe_analytical = tuple(
        point
        for point in analytical_points
        if point.feedback_ratio < 1.0
    )
    unsafe_analytical = tuple(
        point
        for point in analytical_points
        if point.feedback_ratio > 1.0
    )
    safe_summaries = tuple(
        summary
        for summary in feedback_summaries
        if summary.feedback_ratio < 1.0
    )
    maximum_feedback_summary = max(
        feedback_summaries,
        key=lambda summary: summary.feedback_ratio,
    )
    return BistableProtectionDecisions(
        analytic_safe_invariance=all(
            point.invariant_region_survived and not point.sign_tipped
            for point in safe_analytical
        ),
        analytic_threshold_tipping=all(
            point.sign_tipped and point.final_signed_state < 0.0
            for point in unsafe_analytical
        ),
        reserve_attractor_acquisition=all(
            summary.reserve_recall_gain.lower > 0.5
            for summary in feedback_summaries
        ),
        certified_core_preservation=all(
            summary.certified_core_retention.lower >= 1.0 - 1e-12
            for summary in safe_summaries
        ),
        margin_predicts_failure=(
            maximum_feedback_summary.opposing_core_retention.upper < 0.5
        ),
    )


def _validate_configuration(
    *,
    trial_seeds: tuple[int, ...],
    analytical_recurrent_gains: tuple[float, ...],
    core_recurrent_gain: float,
    feedback_ratios: tuple[float, ...],
    adaptation_candidates: tuple[tuple[float, float], ...],
    calibration_trials: int,
    evaluation_trials: int,
    analytical_steps: int,
    evaluation_steps: int,
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
    _validate_bistable_gains(
        analytical_recurrent_gains,
        "analytical_recurrent_gains",
    )
    _validate_bistable_gains(
        (core_recurrent_gain,),
        "core_recurrent_gain",
    )
    if (
        not feedback_ratios
        or len(set(feedback_ratios)) != len(feedback_ratios)
        or not any(ratio < 1.0 for ratio in feedback_ratios)
        or not any(ratio > 1.0 for ratio in feedback_ratios)
        or any(
            not math.isfinite(ratio) or ratio < 0.0
            for ratio in feedback_ratios
        )
    ):
        raise ValueError(
            "feedback_ratiosは1未満と1超を含む有限非負値にしてください"
        )
    if (
        not adaptation_candidates
        or len(set(adaptation_candidates)) != len(adaptation_candidates)
        or any(
            len(candidate) != 2
            or not math.isfinite(candidate[0])
            or candidate[0] <= 1.0
            or not math.isfinite(candidate[1])
            or candidate[1] <= 0.0
            for candidate in adaptation_candidates
        )
    ):
        raise ValueError(
            "adaptation_candidatesはrecurrent > 1とcue正値の対にしてください"
        )
    _validate_positive_integer(calibration_trials, "calibration_trials")
    _validate_positive_integer(evaluation_trials, "evaluation_trials")
    _validate_positive_integer(analytical_steps, "analytical_steps")
    _validate_positive_integer(evaluation_steps, "evaluation_steps")
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


def _validate_bistable_gains(
    gains: tuple[float, ...],
    value_name: str,
) -> None:
    if (
        not gains
        or len(set(gains)) != len(gains)
        or any(not math.isfinite(gain) or gain <= 1.0 for gain in gains)
    ):
        raise ValueError(f"{value_name}は重複なしの1より大きい有限値にしてください")


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数にしてください")


def main() -> None:
    """既定30 seed実験をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_bistable_core_protection_study()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
