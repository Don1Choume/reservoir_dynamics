"""異質な独立moduleに対するcomponent積則の確認。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from reservoir_dynamics.experiments.recurrent_weight_families import (
    build_recurrent_weights,
)
from reservoir_dynamics.experiments.sign_memory_evaluation import (
    SignMemoryNetworkProfile,
    evaluate_sign_memory_network,
)
from reservoir_dynamics.metrics.structural_equivalence import (
    audit_signed_coordinate_conjugacy,
    weakly_connected_components,
)
from reservoir_dynamics.theory.orthant_box import Matrix

EXPERIMENT_ID = "EXP-2026-013"
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(1401, 1431))
PREREGISTERED_COUPLING_GAINS = (0.05, 0.07)
PREREGISTERED_DISTURBANCE_BOUNDS = (0.08, 0.12, 0.16, 0.20)


@dataclass(frozen=True, slots=True)
class HeterogeneousModularStructureGate:
    """task実行前に固定する構造多様性監査。"""

    trial_seeds: tuple[int, ...]
    coupling_gains: tuple[float, ...]
    raw_network_counts: tuple[int, ...]
    effective_class_counts: tuple[int, ...]
    components_valid: bool
    magnitude_pairs_unique: bool
    module_magnitude_range: tuple[float, float]
    passed: bool


@dataclass(frozen=True, slots=True)
class ModularFactorizationPoint:
    """一network・一外乱におけるglobalとcomponentの対応。"""

    trial_seed: int
    coupling_gain: float
    disturbance_bound: float
    module_coupling_magnitudes: tuple[float, float]
    full_raw_attractor_count: int
    module_raw_attractor_counts: tuple[int, int]
    full_common_certified_fraction: float
    component_certified_fraction: float
    full_task_retention: float
    module_task_retentions: tuple[float, float]
    task_product_residual: float
    component_certificate_gap: float
    maximum_common_margin_excess: float
    minimum_component_box_slack: float


@dataclass(frozen=True, slots=True)
class ModularFactorizationDecisions:
    """EXP-2026-013で事前固定した6判定。"""

    effective_structure_gate: bool
    fixed_point_product: bool
    component_margin_product: bool
    global_certificate_is_conservative: bool
    task_retention_product: bool
    certificate_lower_bound_valid: bool


@dataclass(frozen=True, slots=True)
class HeterogeneousModularFactorizationResult:
    """構造gateと全factorization点を含む確認結果。"""

    experiment_id: str
    phase: str
    dimension: int
    diagonal_gain: float
    task_steps: int
    autonomous_steps: int
    convergence_tolerance: float
    structure_gate: HeterogeneousModularStructureGate
    points: tuple[ModularFactorizationPoint, ...]
    decisions: ModularFactorizationDecisions
    maximum_task_product_residual: float
    maximum_common_margin_excess: float
    minimum_component_box_slack: float
    minimum_component_certificate_gap: float


def audit_heterogeneous_modular_structure(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    coupling_gains: tuple[float, ...] = PREREGISTERED_COUPLING_GAINS,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    tolerance: float = 1e-12,
) -> HeterogeneousModularStructureGate:
    """task値を計算せず、共役class、component、絶対値重複を監査する。"""

    _validate_structure_configuration(
        trial_seeds=trial_seeds,
        coupling_gains=coupling_gains,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        tolerance=tolerance,
    )
    expected_components = tuple(
        (index, index + 1) for index in range(0, dimension, 2)
    )
    raw_network_counts: list[int] = []
    effective_class_counts: list[int] = []
    all_magnitude_signatures: list[tuple[float, ...]] = []
    all_magnitudes: list[float] = []
    components_valid = True
    magnitude_pairs_unique = True
    for coupling_gain in coupling_gains:
        matrices = tuple(
            build_recurrent_weights(
                network_family="modular_heterogeneous",
                dimension=dimension,
                diagonal_gain=diagonal_gain,
                coupling_gain=coupling_gain,
                trial_seed=trial_seed,
            )
            for trial_seed in trial_seeds
        )
        audit = audit_signed_coordinate_conjugacy(
            matrices,
            tolerance=tolerance,
        )
        raw_network_counts.append(audit.raw_network_count)
        effective_class_counts.append(audit.effective_class_count)
        signatures = tuple(
            tuple(
                sorted(
                    abs(matrix[index][index + 1])
                    for index in range(0, dimension, 2)
                )
            )
            for matrix in matrices
        )
        all_magnitude_signatures.extend(signatures)
        all_magnitudes.extend(
            magnitude for signature in signatures for magnitude in signature
        )
        magnitude_pairs_unique = magnitude_pairs_unique and (
            len(set(signatures)) == len(signatures)
        )
        components_valid = components_valid and all(
            weakly_connected_components(matrix, tolerance=tolerance)
            == expected_components
            for matrix in matrices
        )

    effective_counts = tuple(effective_class_counts)
    raw_counts = tuple(raw_network_counts)
    passed = (
        components_valid
        and magnitude_pairs_unique
        and all(count == len(trial_seeds) for count in effective_counts)
        and all(count == len(trial_seeds) for count in raw_counts)
    )
    return HeterogeneousModularStructureGate(
        trial_seeds=trial_seeds,
        coupling_gains=coupling_gains,
        raw_network_counts=raw_counts,
        effective_class_counts=effective_counts,
        components_valid=components_valid,
        magnitude_pairs_unique=magnitude_pairs_unique,
        module_magnitude_range=(min(all_magnitudes), max(all_magnitudes)),
        passed=passed,
    )


def run_heterogeneous_modular_factorization(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    coupling_gains: tuple[float, ...] = PREREGISTERED_COUPLING_GAINS,
    disturbance_bounds: tuple[
        float, ...
    ] = PREREGISTERED_DISTURBANCE_BOUNDS,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    task_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    decision_tolerance: float = 1e-12,
) -> HeterogeneousModularFactorizationResult:
    """構造gate通過後だけfull系と二moduleを独立評価する。"""

    if dimension != 4:
        raise ValueError("EXP-2026-013のdimensionは4に固定します")
    structure_gate = audit_heterogeneous_modular_structure(
        trial_seeds=trial_seeds,
        coupling_gains=coupling_gains,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        tolerance=decision_tolerance,
    )
    if not structure_gate.passed:
        raise RuntimeError("構造gateが不成立のためtask評価を開始しません")

    points = tuple(
        point
        for trial_seed in trial_seeds
        for coupling_gain in coupling_gains
        for point in _evaluate_network(
            trial_seed=trial_seed,
            coupling_gain=coupling_gain,
            disturbance_bounds=disturbance_bounds,
            diagonal_gain=diagonal_gain,
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
    )
    decisions = ModularFactorizationDecisions(
        effective_structure_gate=structure_gate.passed,
        fixed_point_product=all(
            point.full_raw_attractor_count
            == math.prod(point.module_raw_attractor_counts)
            for point in points
        ),
        component_margin_product=all(
            point.minimum_component_box_slack >= -decision_tolerance
            for point in points
        ),
        global_certificate_is_conservative=all(
            point.maximum_common_margin_excess <= decision_tolerance
            and point.full_common_certified_fraction
            <= point.component_certified_fraction + decision_tolerance
            for point in points
        ),
        task_retention_product=all(
            abs(point.task_product_residual) <= decision_tolerance
            for point in points
        ),
        certificate_lower_bound_valid=all(
            point.component_certificate_gap >= -decision_tolerance
            for point in points
        ),
    )
    return HeterogeneousModularFactorizationResult(
        experiment_id=EXPERIMENT_ID,
        phase="confirmation",
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        structure_gate=structure_gate,
        points=points,
        decisions=decisions,
        maximum_task_product_residual=max(
            abs(point.task_product_residual) for point in points
        ),
        maximum_common_margin_excess=max(
            point.maximum_common_margin_excess for point in points
        ),
        minimum_component_box_slack=min(
            point.minimum_component_box_slack for point in points
        ),
        minimum_component_certificate_gap=min(
            point.component_certificate_gap for point in points
        ),
    )


def _evaluate_network(
    *,
    trial_seed: int,
    coupling_gain: float,
    disturbance_bounds: tuple[float, ...],
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[ModularFactorizationPoint, ...]:
    recurrent_weights = build_recurrent_weights(
        network_family="modular_heterogeneous",
        dimension=4,
        diagonal_gain=diagonal_gain,
        coupling_gain=coupling_gain,
        trial_seed=trial_seed,
    )
    module_weights = (
        _submatrix(recurrent_weights, (0, 1)),
        _submatrix(recurrent_weights, (2, 3)),
    )
    profile_arguments = {
        "disturbance_bounds": disturbance_bounds,
        "task_steps": task_steps,
        "autonomous_steps": autonomous_steps,
        "convergence_tolerance": convergence_tolerance,
    }
    full_profile = evaluate_sign_memory_network(
        recurrent_weights=recurrent_weights,
        **profile_arguments,
    )
    module_profiles = tuple(
        evaluate_sign_memory_network(
            recurrent_weights=matrix,
            **profile_arguments,
        )
        for matrix in module_weights
    )
    component_margins = _component_margins(module_profiles)
    common_margins = tuple(
        orthant.maximum_uniform_disturbance
        for orthant in full_profile.orthants
    )
    maximum_common_margin_excess = max(
        common_margin - component_margin
        for common_margin, component_margin in zip(
            common_margins,
            component_margins,
            strict=True,
        )
    )
    minimum_box_slack = _minimum_component_box_slack(
        recurrent_weights=recurrent_weights,
        full_profile=full_profile,
        module_profiles=module_profiles,
        component_margins=component_margins,
    )
    return tuple(
        _build_factorization_point(
            trial_seed=trial_seed,
            coupling_gain=coupling_gain,
            disturbance_index=disturbance_index,
            full_profile=full_profile,
            module_profiles=module_profiles,
            module_weights=module_weights,
            maximum_common_margin_excess=maximum_common_margin_excess,
            minimum_component_box_slack=minimum_box_slack,
        )
        for disturbance_index in range(len(disturbance_bounds))
    )


def _build_factorization_point(
    *,
    trial_seed: int,
    coupling_gain: float,
    disturbance_index: int,
    full_profile: SignMemoryNetworkProfile,
    module_profiles: tuple[SignMemoryNetworkProfile, ...],
    module_weights: tuple[Matrix, Matrix],
    maximum_common_margin_excess: float,
    minimum_component_box_slack: float,
) -> ModularFactorizationPoint:
    full_evaluation = full_profile.disturbance_evaluations[
        disturbance_index
    ]
    module_evaluations = tuple(
        profile.disturbance_evaluations[disturbance_index]
        for profile in module_profiles
    )
    component_certified_fraction = math.prod(
        evaluation.certified_robust_fraction
        for evaluation in module_evaluations
    )
    module_task_retentions = tuple(
        evaluation.task_retention for evaluation in module_evaluations
    )
    task_product_residual = (
        full_evaluation.task_retention
        - math.prod(module_task_retentions)
    )
    return ModularFactorizationPoint(
        trial_seed=trial_seed,
        coupling_gain=coupling_gain,
        disturbance_bound=full_evaluation.disturbance_bound,
        module_coupling_magnitudes=(
            abs(module_weights[0][0][1]),
            abs(module_weights[1][0][1]),
        ),
        full_raw_attractor_count=full_profile.raw_attractor_count,
        module_raw_attractor_counts=(
            module_profiles[0].raw_attractor_count,
            module_profiles[1].raw_attractor_count,
        ),
        full_common_certified_fraction=(
            full_evaluation.certified_robust_fraction
        ),
        component_certified_fraction=component_certified_fraction,
        full_task_retention=full_evaluation.task_retention,
        module_task_retentions=(
            module_task_retentions[0],
            module_task_retentions[1],
        ),
        task_product_residual=task_product_residual,
        component_certificate_gap=(
            full_evaluation.task_retention
            - component_certified_fraction
        ),
        maximum_common_margin_excess=maximum_common_margin_excess,
        minimum_component_box_slack=minimum_component_box_slack,
    )


def _component_margins(
    module_profiles: tuple[SignMemoryNetworkProfile, ...],
) -> tuple[float, ...]:
    return tuple(
        min(
            first_orthant.maximum_uniform_disturbance,
            second_orthant.maximum_uniform_disturbance,
        )
        for first_orthant in module_profiles[0].orthants
        for second_orthant in module_profiles[1].orthants
    )


def _minimum_component_box_slack(
    *,
    recurrent_weights: Matrix,
    full_profile: SignMemoryNetworkProfile,
    module_profiles: tuple[SignMemoryNetworkProfile, ...],
    component_margins: tuple[float, ...],
) -> float:
    module_orthant_pairs = tuple(
        (first_orthant, second_orthant)
        for first_orthant in module_profiles[0].orthants
        for second_orthant in module_profiles[1].orthants
    )
    slacks: list[float] = []
    for full_orthant, module_pair, disturbance_margin in zip(
        full_profile.orthants,
        module_orthant_pairs,
        component_margins,
        strict=True,
    ):
        expected_signs = (
            module_pair[0].attractor_signs
            + module_pair[1].attractor_signs
        )
        if full_orthant.attractor_signs != expected_signs:
            return -math.inf
        boundaries = (
            (module_pair[0].invariant_boundary,) * 2
            + (module_pair[1].invariant_boundary,) * 2
        )
        signs = full_orthant.attractor_signs
        transformed_weights = tuple(
            tuple(
                signs[row_index] * weight * signs[column_index]
                for column_index, weight in enumerate(row)
            )
            for row_index, row in enumerate(recurrent_weights)
        )
        slacks.extend(
            math.fsum(
                max(weight, 0.0) * boundaries[column_index]
                + min(weight, 0.0)
                for column_index, weight in enumerate(row)
            )
            - disturbance_margin
            - math.atanh(boundaries[row_index])
            for row_index, row in enumerate(transformed_weights)
        )
    return min(slacks)


def _submatrix(matrix: Matrix, indices: tuple[int, ...]) -> Matrix:
    return tuple(
        tuple(matrix[row_index][column_index] for column_index in indices)
        for row_index in indices
    )


def _validate_structure_configuration(
    *,
    trial_seeds: tuple[int, ...],
    coupling_gains: tuple[float, ...],
    dimension: int,
    diagonal_gain: float,
    tolerance: float,
) -> None:
    if (
        len(trial_seeds) < 2
        or len(set(trial_seeds)) != len(trial_seeds)
        or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in trial_seeds
        )
    ):
        raise ValueError("trial_seedsは重複しない2個以上の整数にしてください")
    if (
        not coupling_gains
        or any(
            not math.isfinite(gain) or gain <= 0.0
            for gain in coupling_gains
        )
        or any(
            first >= second
            for first, second in zip(coupling_gains, coupling_gains[1:])
        )
    ):
        raise ValueError("coupling_gainsは厳密昇順の有限正値にしてください")
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 2
        or dimension > 6
        or dimension % 2 != 0
    ):
        raise ValueError("dimensionは2以上6以下の偶数にしてください")
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")


def _summary_payload(
    result: HeterogeneousModularFactorizationResult,
) -> dict[str, object]:
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "dimension": result.dimension,
        "diagonal_gain": result.diagonal_gain,
        "task_steps": result.task_steps,
        "autonomous_steps": result.autonomous_steps,
        "convergence_tolerance": result.convergence_tolerance,
        "structure_gate": asdict(result.structure_gate),
        "point_count": len(result.points),
        "challenge_count": sum(
            2 ** (2 * result.dimension) for _ in result.points
        ),
        "decisions": asdict(result.decisions),
        "maximum_task_product_residual": (
            result.maximum_task_product_residual
        ),
        "maximum_common_margin_excess": (
            result.maximum_common_margin_excess
        ),
        "minimum_component_box_slack": (
            result.minimum_component_box_slack
        ),
        "minimum_component_certificate_gap": (
            result.minimum_component_certificate_gap
        ),
        "full_raw_attractor_count_values": tuple(
            sorted({point.full_raw_attractor_count for point in result.points})
        ),
        "module_raw_attractor_count_values": tuple(
            sorted(
                {
                    count
                    for point in result.points
                    for count in point.module_raw_attractor_counts
                }
            )
        ),
        "common_minus_component_fraction_range": (
            min(
                point.full_common_certified_fraction
                - point.component_certified_fraction
                for point in result.points
            ),
            max(
                point.full_common_certified_fraction
                - point.component_certified_fraction
                for point in result.points
            ),
        ),
        "task_retention_range": (
            min(point.full_task_retention for point in result.points),
            max(point.full_task_retention for point in result.points),
        ),
    }


def main() -> None:
    """事前登録したEXP-2026-013 confirmationをJSONで出力する。"""

    print(
        json.dumps(
            _summary_payload(run_heterogeneous_modular_factorization()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
