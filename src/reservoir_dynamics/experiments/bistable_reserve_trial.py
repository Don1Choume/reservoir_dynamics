"""双安定coreとcue形成reserveの一seed評価。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
    positive_bistable_fixed_point,
)


@dataclass(frozen=True, slots=True)
class ReserveAttractorChoice:
    """独立calibrationで選んだreserve双安定化parameter。"""

    recurrent_gain: float
    cue_gain: float
    calibration_recall: float


@dataclass(frozen=True, slots=True)
class AdversarialThresholdPoint:
    """一定最悪外力に対する解析閾値の数値照合。"""

    recurrent_gain: float
    feedback_ratio: float
    invariant_boundary: float
    critical_forcing: float
    minimum_signed_state: float
    final_signed_state: float
    invariant_region_survived: bool
    sign_tipped: bool


@dataclass(frozen=True, slots=True)
class BistableProtectionPoint:
    """一seedにおけるreserve獲得とcore survivability。"""

    trial_seed: int
    feedback_ratio: float
    feedback_gain: float
    selected_recurrent_gain: float
    selected_cue_gain: float
    pre_reserve_recall: float
    post_reserve_recall: float
    reserve_recall_gain: float
    certified_core_retention: float
    full_basin_core_retention: float
    opposing_core_retention: float
    mean_minimum_certified_margin: float


def evaluate_adversarial_threshold(
    *,
    recurrent_gain: float,
    feedback_ratio: float,
    steps: int,
) -> AdversarialThresholdPoint:
    """一定の最悪外力でsaddle-node閾値を横断する。"""

    certificate = bistable_tanh_certificate(recurrent_gain)
    _validate_non_negative_finite(feedback_ratio, "feedback_ratio")
    _validate_positive_integer(steps, "steps")
    state = positive_bistable_fixed_point(recurrent_gain)
    minimum_state = state
    adverse_forcing = -feedback_ratio * certificate.critical_forcing
    sign_tipped = False
    for _ in range(steps):
        state = math.tanh(recurrent_gain * state + adverse_forcing)
        minimum_state = min(minimum_state, state)
        sign_tipped = sign_tipped or state <= 0.0
    return AdversarialThresholdPoint(
        recurrent_gain=recurrent_gain,
        feedback_ratio=feedback_ratio,
        invariant_boundary=certificate.invariant_boundary,
        critical_forcing=certificate.critical_forcing,
        minimum_signed_state=minimum_state,
        final_signed_state=state,
        invariant_region_survived=(
            minimum_state >= certificate.invariant_boundary - 1e-12
        ),
        sign_tipped=sign_tipped,
    )


def select_reserve_attractor_adaptation(
    *,
    trial_seed: int,
    candidates: tuple[tuple[float, float], ...],
    calibration_trials: int,
    recall_steps: int = 80,
) -> ReserveAttractorChoice:
    """cue後の符号再生marginが最大の双安定reserveを選ぶ。"""

    _validate_candidates(candidates)
    _validate_positive_integer(calibration_trials, "calibration_trials")
    _validate_positive_integer(recall_steps, "recall_steps")
    random_generator = random.Random(trial_seed)
    cue_trials = tuple(
        (
            1.0 if trial_index % 2 == 0 else -1.0,
            random_generator.uniform(0.5, 1.5),
        )
        for trial_index in range(calibration_trials)
    )
    best_choice: ReserveAttractorChoice | None = None
    for recurrent_gain, cue_gain in candidates:
        recall = math.fsum(
            cue_sign
            * _simulate_reserve(
                recurrent_gain=recurrent_gain,
                cue_gain=cue_gain,
                cue=cue_sign * cue_amplitude,
                steps=recall_steps,
            )
            for cue_sign, cue_amplitude in cue_trials
        ) / calibration_trials
        if best_choice is None or recall > best_choice.calibration_recall:
            best_choice = ReserveAttractorChoice(
                recurrent_gain=recurrent_gain,
                cue_gain=cue_gain,
                calibration_recall=recall,
            )
    if best_choice is None:
        raise RuntimeError("reserve adaptation候補の選択に失敗しました")
    return best_choice


def evaluate_bistable_protection(
    *,
    trial_seed: int,
    core_recurrent_gain: float,
    feedback_ratio: float,
    adaptation: ReserveAttractorChoice,
    evaluation_trials: int,
    evaluation_steps: int,
) -> BistableProtectionPoint:
    """独立cueで形成したreserveがcore符号記憶を侵すか評価する。"""

    certificate = bistable_tanh_certificate(core_recurrent_gain)
    _validate_non_negative_finite(feedback_ratio, "feedback_ratio")
    _validate_positive_integer(evaluation_trials, "evaluation_trials")
    _validate_positive_integer(evaluation_steps, "evaluation_steps")
    _validate_candidates(
        ((adaptation.recurrent_gain, adaptation.cue_gain),)
    )
    feedback_gain = feedback_ratio * certificate.critical_forcing
    random_generator = random.Random(trial_seed)
    certified_survivals: list[float] = []
    full_basin_survivals: list[float] = []
    opposing_survivals: list[float] = []
    reserve_recalls: list[float] = []
    minimum_margins: list[float] = []
    for trial_index in range(evaluation_trials):
        core_sign = 1.0 if trial_index % 2 == 0 else -1.0
        cue_sign = 1.0 if (trial_index // 2) % 2 == 0 else -1.0
        cue_amplitude = random_generator.uniform(0.5, 1.5)
        certified_magnitude = random_generator.uniform(
            certificate.invariant_boundary,
            1.0,
        )
        full_basin_magnitude = random_generator.uniform(1e-12, 1.0)
        certified_result = _simulate_coupled_trial(
            core_recurrent_gain=core_recurrent_gain,
            feedback_gain=feedback_gain,
            reserve_recurrent_gain=adaptation.recurrent_gain,
            cue_gain=adaptation.cue_gain,
            core_initial=core_sign * certified_magnitude,
            cue=cue_sign * cue_amplitude,
            intended_core_sign=core_sign,
            intended_cue_sign=cue_sign,
            steps=evaluation_steps,
        )
        full_basin_result = _simulate_coupled_trial(
            core_recurrent_gain=core_recurrent_gain,
            feedback_gain=feedback_gain,
            reserve_recurrent_gain=adaptation.recurrent_gain,
            cue_gain=adaptation.cue_gain,
            core_initial=core_sign * full_basin_magnitude,
            cue=cue_sign * cue_amplitude,
            intended_core_sign=core_sign,
            intended_cue_sign=cue_sign,
            steps=evaluation_steps,
        )
        certified_survivals.append(float(certified_result[0]))
        full_basin_survivals.append(float(full_basin_result[0]))
        reserve_recalls.append(certified_result[1])
        minimum_margins.append(
            certified_result[2] - certificate.invariant_boundary
        )
        if core_sign != cue_sign:
            opposing_survivals.append(float(certified_result[0]))
    if not opposing_survivals:
        raise RuntimeError("opposing cue評価が一件も生成されませんでした")
    post_recall = math.fsum(reserve_recalls) / len(reserve_recalls)
    return BistableProtectionPoint(
        trial_seed=trial_seed,
        feedback_ratio=feedback_ratio,
        feedback_gain=feedback_gain,
        selected_recurrent_gain=adaptation.recurrent_gain,
        selected_cue_gain=adaptation.cue_gain,
        pre_reserve_recall=0.0,
        post_reserve_recall=post_recall,
        reserve_recall_gain=post_recall,
        certified_core_retention=_mean(certified_survivals),
        full_basin_core_retention=_mean(full_basin_survivals),
        opposing_core_retention=_mean(opposing_survivals),
        mean_minimum_certified_margin=_mean(minimum_margins),
    )


def _simulate_reserve(
    *,
    recurrent_gain: float,
    cue_gain: float,
    cue: float,
    steps: int,
) -> float:
    reserve_state = 0.0
    for step_index in range(steps):
        input_value = cue if step_index == 0 else 0.0
        reserve_state = math.tanh(
            recurrent_gain * reserve_state + cue_gain * input_value
        )
    return reserve_state


def _simulate_coupled_trial(
    *,
    core_recurrent_gain: float,
    feedback_gain: float,
    reserve_recurrent_gain: float,
    cue_gain: float,
    core_initial: float,
    cue: float,
    intended_core_sign: float,
    intended_cue_sign: float,
    steps: int,
) -> tuple[bool, float, float]:
    core_state = core_initial
    reserve_state = 0.0
    minimum_signed_core = intended_core_sign * core_state
    survived = minimum_signed_core > 0.0
    for step_index in range(steps):
        input_value = cue if step_index == 0 else 0.0
        next_core = math.tanh(
            core_recurrent_gain * core_state
            + feedback_gain * reserve_state
        )
        next_reserve = math.tanh(
            reserve_recurrent_gain * reserve_state
            + cue_gain * input_value
        )
        core_state = next_core
        reserve_state = next_reserve
        signed_core = intended_core_sign * core_state
        minimum_signed_core = min(minimum_signed_core, signed_core)
        survived = survived and signed_core > 0.0
    return (
        survived,
        intended_cue_sign * reserve_state,
        minimum_signed_core,
    )


def _validate_candidates(
    candidates: tuple[tuple[float, float], ...],
) -> None:
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(
            len(candidate) != 2
            or not math.isfinite(candidate[0])
            or candidate[0] <= 1.0
            or not math.isfinite(candidate[1])
            or candidate[1] <= 0.0
            for candidate in candidates
        )
    ):
        raise ValueError(
            "adaptation_candidatesはrecurrent > 1とcue正値の対にしてください"
        )


def _validate_non_negative_finite(value: float, value_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{value_name}は有限の非負値にしてください")


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数にしてください")


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)
