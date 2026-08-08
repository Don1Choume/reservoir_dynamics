"""EXP-2026-017の固定model三module移送と事前登録判定。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.component_predictor import (
    ComponentPredictorEvaluation,
    NamedComponentPredictor,
    evaluate_fixed_component_models,
)
from reservoir_dynamics.experiments.component_size_experiment import (
    PredictorErrorDifferenceInterval,
    predictor_error_difference_interval,
)
from reservoir_dynamics.experiments.multicomponent_modular_family import (
    MultiComponentStructureGate,
    audit_multicomponent_structure,
    build_multicomponent_modular_network,
)
from reservoir_dynamics.experiments.multicomponent_profile import (
    MultiComponentProfilePoint,
    evaluate_multicomponent_profile,
)

EXPERIMENT_ID = "EXP-2026-017"
PREREGISTERED_DEVELOPMENT_SEEDS = tuple(range(2201, 2231))
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(2301, 2331))
PREREGISTERED_MODULE_SIZES = (2, 2, 3)
PREREGISTERED_INTERNAL_GAINS = (0.025, 0.05)
PREREGISTERED_BRIDGE_STRENGTHS = (0.0, 0.01, 0.02, 0.04)
PREREGISTERED_DISTURBANCE_BOUNDS = (0.08, 0.12, 0.16, 0.20)
CONFIRMATION_COMPONENT_MAE_THRESHOLD = 0.05
CONFIRMATION_COMPONENT_SPEARMAN_THRESHOLD = 0.60
CONFIRMATION_GLOBAL_DIFFERENCE_LOWER_THRESHOLD = 0.005
CONFIRMATION_PRODUCT_DIFFERENCE_LOWER_THRESHOLD = 0.0


@dataclass(frozen=True, slots=True)
class MultiComponentTheoryDecisions:
    """task前に固定した構造・保証・計算量の八判定。"""

    partition_recovery: bool
    oracle_inferred_equivalence: bool
    zero_coupling_factorization: bool
    factorized_certificate_exactness: bool
    transported_lower_bound: bool
    certificate_chain: bool
    feature_finiteness: bool
    complexity_reduction: bool


@dataclass(frozen=True, slots=True)
class MultiComponentConfirmationDecisions:
    """八理論判定と四経験判定を分けて保持する。"""

    partition_recovery: bool
    oracle_inferred_equivalence: bool
    zero_coupling_factorization: bool
    factorized_certificate_exactness: bool
    transported_lower_bound: bool
    certificate_chain: bool
    feature_finiteness: bool
    complexity_reduction: bool
    component_mae_within_threshold: bool
    component_spearman_above_threshold: bool
    component_beats_global: bool
    component_beats_product: bool
    all_theory_passed: bool
    all_empirical_passed: bool
    all_passed: bool


@dataclass(frozen=True, slots=True)
class MultiComponentTransferResult:
    """固定modelを一つのseed分離familyへ適用した結果。"""

    experiment_id: str
    phase: str
    structure_gate: MultiComponentStructureGate
    points: tuple[MultiComponentProfilePoint, ...]
    evaluations: tuple[ComponentPredictorEvaluation, ...]
    error_intervals: tuple[PredictorErrorDifferenceInterval, ...]
    theory_decisions: MultiComponentTheoryDecisions
    decisions: MultiComponentConfirmationDecisions


def run_multicomponent_development(
    *,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...] = PREREGISTERED_DEVELOPMENT_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    disturbance_bounds: tuple[float, ...] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 80,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_817,
) -> MultiComponentTransferResult:
    """登録閾値を変えず、開発seedだけで実装識別性を確認する。"""

    return _run_transfer(
        phase="development",
        fitted_models=fitted_models,
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        disturbance_bounds=disturbance_bounds,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def run_multicomponent_confirmation(
    *,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    disturbance_bounds: tuple[float, ...] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 80,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_817,
) -> MultiComponentTransferResult:
    """開発seedを拒否し、登録confirmationへ固定modelを一度適用する。"""

    if set(trial_seeds).intersection(PREREGISTERED_DEVELOPMENT_SEEDS):
        raise ValueError("confirmation seedはdevelopment seedと分離してください")
    if set(trial_seeds).intersection(range(2001, 2131)):
        raise ValueError("confirmation seedは過去のpilot・confirmationと分離してください")
    return _run_transfer(
        phase="confirmation",
        fitted_models=fitted_models,
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        disturbance_bounds=disturbance_bounds,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _run_transfer(
    *,
    phase: str,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...],
    module_sizes: tuple[int, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    bootstrap_confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> MultiComponentTransferResult:
    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    structure_gate = audit_multicomponent_structure(
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_total_bridge_strengths=maximum_bridge_strengths,
        diagonal_gain=diagonal_gain,
    )
    if not structure_gate.passed:
        raise RuntimeError("構造gateが不成立のためtaskを開始しません")
    points = _evaluate_grid(
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
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
    theory = _theory_decisions(structure_gate, points)
    decisions = _confirmation_decisions(
        theory=theory,
        evaluations=evaluations,
        intervals=intervals,
    )
    return MultiComponentTransferResult(
        experiment_id=EXPERIMENT_ID,
        phase=phase,
        structure_gate=structure_gate,
        points=points,
        evaluations=evaluations,
        error_intervals=intervals,
        theory_decisions=theory,
        decisions=decisions,
    )


def _evaluate_grid(
    *,
    trial_seeds: tuple[int, ...],
    module_sizes: tuple[int, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[MultiComponentProfilePoint, ...]:
    return tuple(
        point
        for trial_seed in trial_seeds
        for internal_gain in internal_gains
        for bridge_strength in maximum_bridge_strengths
        for network in (
            build_multicomponent_modular_network(
                trial_seed=trial_seed,
                module_sizes=module_sizes,
                internal_gain=internal_gain,
                maximum_total_bridge_strength=bridge_strength,
                diagonal_gain=diagonal_gain,
            ),
        )
        for point in evaluate_multicomponent_profile(
            network=network,
            partition=network.inferred_partition.components,
            disturbance_bounds=disturbance_bounds,
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
    )


def _theory_decisions(
    structure_gate: MultiComponentStructureGate,
    points: tuple[MultiComponentProfilePoint, ...],
    tolerance: float = 1e-12,
) -> MultiComponentTheoryDecisions:
    zero_points = tuple(
        point for point in points if point.maximum_bridge_strength == 0.0
    )
    return MultiComponentTheoryDecisions(
        partition_recovery=structure_gate.partition_recovery_exact,
        oracle_inferred_equivalence=structure_gate.partition_recovery_exact,
        zero_coupling_factorization=bool(zero_points) and all(
            abs(point.task_product_residual) <= tolerance for point in zero_points
        ),
        factorized_certificate_exactness=all(
            abs(
                point.factorized_directional_certified_fraction
                - point.enumerated_directional_certified_fraction
            )
            <= tolerance
            for point in points
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
        complexity_reduction=all(
            point.local_orthant_count == 16
            and point.global_orthant_count == 128
            for point in points
        ),
    )


def _confirmation_decisions(
    *,
    theory: MultiComponentTheoryDecisions,
    evaluations: tuple[ComponentPredictorEvaluation, ...],
    intervals: tuple[PredictorErrorDifferenceInterval, ...],
) -> MultiComponentConfirmationDecisions:
    component = evaluations[0]
    interval_by_name = {value.baseline_name: value for value in intervals}
    theory_values = tuple(getattr(theory, name) for name in theory.__slots__)
    empirical_values = (
        component.mae <= CONFIRMATION_COMPONENT_MAE_THRESHOLD,
        component.spearman >= CONFIRMATION_COMPONENT_SPEARMAN_THRESHOLD,
        interval_by_name["global_profile"].lower
        >= CONFIRMATION_GLOBAL_DIFFERENCE_LOWER_THRESHOLD,
        interval_by_name["product_only"].lower
        >= CONFIRMATION_PRODUCT_DIFFERENCE_LOWER_THRESHOLD,
    )
    return MultiComponentConfirmationDecisions(
        partition_recovery=theory.partition_recovery,
        oracle_inferred_equivalence=theory.oracle_inferred_equivalence,
        zero_coupling_factorization=theory.zero_coupling_factorization,
        factorized_certificate_exactness=(
            theory.factorized_certificate_exactness
        ),
        transported_lower_bound=theory.transported_lower_bound,
        certificate_chain=theory.certificate_chain,
        feature_finiteness=theory.feature_finiteness,
        complexity_reduction=theory.complexity_reduction,
        component_mae_within_threshold=empirical_values[0],
        component_spearman_above_threshold=empirical_values[1],
        component_beats_global=empirical_values[2],
        component_beats_product=empirical_values[3],
        all_theory_passed=all(theory_values),
        all_empirical_passed=all(empirical_values),
        all_passed=all(theory_values + empirical_values),
    )
