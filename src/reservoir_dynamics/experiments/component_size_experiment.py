"""EXP-2026-016のpilot fitとmodule-size分離confirmation。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.asymmetric_modular_family import (
    AsymmetricModularStructureGate,
    audit_asymmetric_modular_structure,
    build_asymmetric_modular_network,
)
from reservoir_dynamics.experiments.component_predictor import (
    ComponentPredictorEvaluation,
    NamedComponentPredictor,
    evaluate_fixed_component_models,
    fit_preregistered_component_models,
    leave_one_seed_out_evaluations,
)
from reservoir_dynamics.experiments.component_profile import (
    ComponentProfilePoint,
    evaluate_component_profile,
)
from reservoir_dynamics.metrics.bootstrap import bootstrap_mean_interval

EXPERIMENT_ID = "EXP-2026-016"
PREREGISTERED_PILOT_SEEDS = tuple(range(2001, 2011))
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(2101, 2131))
PREREGISTERED_PILOT_MODULE_SIZES = ((2, 2), (2, 3))
PREREGISTERED_CONFIRMATION_MODULE_SIZES = (3, 5)
PREREGISTERED_INTERNAL_GAINS = (0.025, 0.05)
PREREGISTERED_BRIDGE_STRENGTHS = (0.0, 0.01, 0.02, 0.04)
PREREGISTERED_DISTURBANCE_BOUNDS = (0.08, 0.12, 0.16, 0.20)
CONFIRMATION_COMPONENT_MAE_THRESHOLD = 0.03
CONFIRMATION_COMPONENT_SPEARMAN_THRESHOLD = 0.75
CONFIRMATION_GLOBAL_DIFFERENCE_LOWER_THRESHOLD = 0.01
CONFIRMATION_PRODUCT_DIFFERENCE_LOWER_THRESHOLD = 0.002


@dataclass(frozen=True, slots=True)
class ComponentSizeTheoryDecisions:
    """pilot前に固定した保証・実装sanity判定。"""

    zero_coupling_factorization: bool
    transported_lower_bound: bool
    certificate_chain: bool
    feature_finiteness: bool


@dataclass(frozen=True, slots=True)
class ComponentSizeConfirmationDecisions:
    """理論gateとpilot後に固定した外挿性能判定。"""

    zero_coupling_factorization: bool
    transported_lower_bound: bool
    certificate_chain: bool
    feature_finiteness: bool
    component_mae_within_threshold: bool
    component_spearman_above_threshold: bool
    component_beats_global: bool
    component_beats_product: bool
    all_passed: bool


@dataclass(frozen=True, slots=True)
class PredictorErrorDifferenceInterval:
    """baseline MAE minus component-aware MAEのseed bootstrap区間。"""

    baseline_name: str
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int


@dataclass(frozen=True, slots=True)
class ComponentSizePilotResult:
    """小module上のLOSO評価と全pilot fit model。"""

    experiment_id: str
    phase: str
    structure_gate: AsymmetricModularStructureGate
    points: tuple[ComponentProfilePoint, ...]
    cross_validated_evaluations: tuple[ComponentPredictorEvaluation, ...]
    fitted_models: tuple[NamedComponentPredictor, ...]
    decisions: ComponentSizeTheoryDecisions


@dataclass(frozen=True, slots=True)
class ComponentSizeConfirmationResult:
    """固定modelを未知module sizeへ一度適用した結果。"""

    experiment_id: str
    phase: str
    structure_gate: AsymmetricModularStructureGate
    points: tuple[ComponentProfilePoint, ...]
    evaluations: tuple[ComponentPredictorEvaluation, ...]
    error_intervals: tuple[PredictorErrorDifferenceInterval, ...]
    decisions: ComponentSizeConfirmationDecisions


def run_component_size_pilot(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_PILOT_SEEDS,
    module_size_pairs: tuple[tuple[int, int], ...] = PREREGISTERED_PILOT_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    disturbance_bounds: tuple[float, ...] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 80,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
) -> ComponentSizePilotResult:
    """pilot familyを評価し、LOSO後に固定三modelを全点でfitする。"""

    structure_gate = audit_asymmetric_modular_structure(
        trial_seeds=trial_seeds,
        module_size_pairs=module_size_pairs,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        diagonal_gain=diagonal_gain,
    )
    if not structure_gate.passed:
        raise RuntimeError("構造gateが不成立のためpilot taskを開始しません")
    points = _evaluate_grid(
        trial_seeds=trial_seeds,
        module_size_pairs=module_size_pairs,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        disturbance_bounds=disturbance_bounds,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    return ComponentSizePilotResult(
        experiment_id=EXPERIMENT_ID,
        phase="pilot",
        structure_gate=structure_gate,
        points=points,
        cross_validated_evaluations=leave_one_seed_out_evaluations(points),
        fitted_models=fit_preregistered_component_models(points),
        decisions=_theory_decisions(points),
    )


def run_component_size_confirmation(
    *,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    module_sizes: tuple[int, int] = PREREGISTERED_CONFIRMATION_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    disturbance_bounds: tuple[float, ...] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 80,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_816,
) -> ComponentSizeConfirmationResult:
    """pilot係数を再fitせず、seed・size分離familyへ適用する。"""

    if set(trial_seeds).intersection(PREREGISTERED_PILOT_SEEDS):
        raise ValueError("confirmation seedは登録pilot seedと分離してください")
    structure_gate = audit_asymmetric_modular_structure(
        trial_seeds=trial_seeds,
        module_size_pairs=(module_sizes,),
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        diagonal_gain=diagonal_gain,
    )
    if not structure_gate.passed:
        raise RuntimeError("構造gateが不成立のためconfirmation taskを開始しません")
    points = _evaluate_grid(
        trial_seeds=trial_seeds,
        module_size_pairs=(module_sizes,),
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        disturbance_bounds=disturbance_bounds,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    evaluations = evaluate_fixed_component_models(
        models=fitted_models,
        points=points,
    )
    intervals = tuple(
        predictor_error_difference_interval(
            selected=evaluations[0],
            baseline=baseline,
            confidence_level=bootstrap_confidence_level,
            resamples=bootstrap_resamples,
            random_seed=bootstrap_seed + index,
        )
        for index, baseline in enumerate(evaluations[1:])
    )
    theory_decisions = _theory_decisions(points)
    return ComponentSizeConfirmationResult(
        experiment_id=EXPERIMENT_ID,
        phase="confirmation",
        structure_gate=structure_gate,
        points=points,
        evaluations=evaluations,
        error_intervals=intervals,
        decisions=_confirmation_decisions(
            theory=theory_decisions,
            evaluations=evaluations,
            intervals=intervals,
        ),
    )


def _evaluate_grid(
    *,
    trial_seeds: tuple[int, ...],
    module_size_pairs: tuple[tuple[int, int], ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[ComponentProfilePoint, ...]:
    return tuple(
        point
        for trial_seed in trial_seeds
        for module_sizes in module_size_pairs
        for internal_gain in internal_gains
        for bridge_strength in maximum_bridge_strengths
        for point in evaluate_component_profile(
            network=build_asymmetric_modular_network(
                trial_seed=trial_seed,
                module_sizes=module_sizes,
                internal_gain=internal_gain,
                maximum_bridge_strength=bridge_strength,
                diagonal_gain=diagonal_gain,
            ),
            disturbance_bounds=disturbance_bounds,
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
    )


def _theory_decisions(
    points: tuple[ComponentProfilePoint, ...],
    tolerance: float = 1e-12,
) -> ComponentSizeTheoryDecisions:
    zero_points = tuple(
        point for point in points if point.maximum_bridge_strength == 0.0
    )
    return ComponentSizeTheoryDecisions(
        zero_coupling_factorization=bool(zero_points) and all(
            abs(point.task_product_residual) <= tolerance
            for point in zero_points
        ),
        transported_lower_bound=all(
            point.observed_task_retention
            >= point.transported_certified_fraction - tolerance
            for point in points
        ),
        certificate_chain=all(
            point.transported_certified_fraction + tolerance
            >= point.directional_certified_fraction
            >= point.global_shifted_certified_fraction - tolerance
            for point in points
        ),
        feature_finiteness=all(
            math.isfinite(value)
            for point in points
            for value in point.component_feature_row
            + (point.observed_task_retention,)
        ),
    )


def predictor_error_difference_interval(
    *,
    selected: ComponentPredictorEvaluation,
    baseline: ComponentPredictorEvaluation,
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> PredictorErrorDifferenceInterval:
    if len(selected.predictions) != len(baseline.predictions):
        raise RuntimeError("selectedとbaselineの予測点数が一致しません")
    trial_seeds = tuple(
        sorted({value.trial_seed for value in selected.predictions})
    )
    seed_differences = tuple(
        _mean_seed_error_difference(
            trial_seed=trial_seed,
            selected=selected,
            baseline=baseline,
        )
        for trial_seed in trial_seeds
    )
    interval = bootstrap_mean_interval(
        seed_differences,
        confidence_level=confidence_level,
        resamples=resamples,
        random_seed=random_seed,
    )
    return PredictorErrorDifferenceInterval(
        baseline_name=baseline.name,
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        confidence_level=interval.confidence_level,
        resamples=interval.resamples,
    )


def _confirmation_decisions(
    *,
    theory: ComponentSizeTheoryDecisions,
    evaluations: tuple[ComponentPredictorEvaluation, ...],
    intervals: tuple[PredictorErrorDifferenceInterval, ...],
) -> ComponentSizeConfirmationDecisions:
    component_evaluation = evaluations[0]
    interval_by_name = {
        value.baseline_name: value for value in intervals
    }
    empirical = (
        component_evaluation.mae <= CONFIRMATION_COMPONENT_MAE_THRESHOLD,
        component_evaluation.spearman
        >= CONFIRMATION_COMPONENT_SPEARMAN_THRESHOLD,
        interval_by_name["global_profile"].lower
        >= CONFIRMATION_GLOBAL_DIFFERENCE_LOWER_THRESHOLD,
        interval_by_name["product_only"].lower
        >= CONFIRMATION_PRODUCT_DIFFERENCE_LOWER_THRESHOLD,
    )
    theory_values = (
        theory.zero_coupling_factorization,
        theory.transported_lower_bound,
        theory.certificate_chain,
        theory.feature_finiteness,
    )
    return ComponentSizeConfirmationDecisions(
        zero_coupling_factorization=theory_values[0],
        transported_lower_bound=theory_values[1],
        certificate_chain=theory_values[2],
        feature_finiteness=theory_values[3],
        component_mae_within_threshold=empirical[0],
        component_spearman_above_threshold=empirical[1],
        component_beats_global=empirical[2],
        component_beats_product=empirical[3],
        all_passed=all(theory_values + empirical),
    )


def _mean_seed_error_difference(
    *,
    trial_seed: int,
    selected: ComponentPredictorEvaluation,
    baseline: ComponentPredictorEvaluation,
) -> float:
    paired = tuple(
        (selected_value, baseline_value)
        for selected_value, baseline_value in zip(
            selected.predictions,
            baseline.predictions,
            strict=True,
        )
        if selected_value.trial_seed == trial_seed
        and baseline_value.trial_seed == trial_seed
    )
    if not paired:
        raise RuntimeError(f"seed {trial_seed}のpaired predictionがありません")
    return math.fsum(
        baseline_value.absolute_error - selected_value.absolute_error
        for selected_value, baseline_value in paired
    ) / len(paired)
