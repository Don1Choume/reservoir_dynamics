"""弱いmodule間結合下での積則残差と移送certificate評価。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

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
from reservoir_dynamics.theory.orthant_rectangle import (
    matrix_infinity_norm_difference,
    orthant_rectangle_certificate,
)

EXPERIMENT_ID = "EXP-2026-014"
PREREGISTERED_PILOT_SEEDS = tuple(range(1501, 1511))
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(1601, 1631))
PREREGISTERED_INTERNAL_COUPLING_GAINS = (0.05, 0.07)
PREREGISTERED_CROSS_COUPLING_STRENGTHS = (
    0.0,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.04,
)
PREREGISTERED_DISTURBANCE_BOUNDS = (0.08, 0.12, 0.16, 0.20)
PREREGISTERED_MAXIMUM_STRENGTH_MEAN_RESIDUAL_THRESHOLD = 0.05
PREREGISTERED_MAXIMUM_STRENGTH_NONZERO_FRACTION_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class WeakCouplingStructureGate:
    """task前に評価する結合norm、連結性、共役classのgate。"""

    trial_seeds: tuple[int, ...]
    internal_coupling_gains: tuple[float, ...]
    cross_coupling_strengths: tuple[float, ...]
    raw_network_counts: tuple[int, ...]
    effective_class_counts: tuple[int, ...]
    zero_components_valid: bool
    positive_connectivity_valid: bool
    cross_norms_valid: bool
    magnitude_pairs_unique: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class WeakCouplingFactorizationPoint:
    """一network・一結合・一外乱における積則と保証の比較。"""

    trial_seed: int
    internal_coupling_gain: float
    cross_coupling_strength: float
    cross_coupling_infinity_norm: float
    disturbance_bound: float
    coupled_raw_attractor_count: int
    isolated_raw_attractor_count_product: int
    coupled_task_retention: float
    isolated_task_retention_product: float
    task_product_residual: float
    absolute_task_product_residual: float
    coupled_common_certified_fraction: float
    isolated_component_certified_fraction: float
    transported_rectangle_certified_fraction: float
    norm_shifted_certified_fraction: float
    coupled_fixed_points_inside_rectangle_fraction: float
    task_minus_transported_certificate: float
    task_minus_norm_shifted_certificate: float
    transported_minus_norm_shifted_certificate: float


@dataclass(frozen=True, slots=True)
class WeakCouplingDecisions:
    """理論判定とpilot後に固定したconfirmation経験判定。"""

    structure_gate: bool
    zero_coupling_recovers_product: bool
    transported_certificate_lower_bound: bool
    norm_shifted_certificate_lower_bound: bool
    transported_dominates_norm_shifted: bool
    maximum_strength_mean_absolute_residual: bool
    maximum_strength_nonzero_residual_prevalence: bool
    mean_absolute_residual_non_decreasing: bool


@dataclass(frozen=True, slots=True)
class WeakCouplingFactorizationResult:
    """EXP-2026-014の構造gateと全評価点。"""

    experiment_id: str
    phase: str
    dimension: int
    diagonal_gain: float
    task_steps: int
    autonomous_steps: int
    convergence_tolerance: float
    structure_gate: WeakCouplingStructureGate
    points: tuple[WeakCouplingFactorizationPoint, ...]
    decisions: WeakCouplingDecisions
    maximum_absolute_task_product_residual: float
    minimum_task_minus_transported_certificate: float
    minimum_task_minus_norm_shifted_certificate: float
    minimum_transported_minus_norm_shifted_certificate: float


def build_weakly_coupled_modular_weights(
    *,
    trial_seed: int,
    internal_coupling_gain: float,
    cross_coupling_strength: float,
    diagonal_gain: float = 1.5,
) -> Matrix:
    """異質block対角系へ、normが既知の二つの対称bridgeを加える。"""

    _validate_seed(trial_seed)
    _validate_positive_finite(
        internal_coupling_gain,
        "internal_coupling_gain",
    )
    if (
        not math.isfinite(cross_coupling_strength)
        or cross_coupling_strength < 0.0
    ):
        raise ValueError(
            "cross_coupling_strengthは有限の非負値にしてください"
        )
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")

    base_weights = build_recurrent_weights(
        network_family="modular_heterogeneous",
        dimension=4,
        diagonal_gain=diagonal_gain,
        coupling_gain=internal_coupling_gain,
        trial_seed=trial_seed,
    )
    random_generator = random.Random(1_000_003 + trial_seed)
    bridge_signs = (
        random_generator.choice((-1.0, 1.0)),
        random_generator.choice((-1.0, 1.0)),
    )
    return tuple(
        tuple(
            base_weights[row_index][column_index]
            + cross_coupling_strength
            * _bridge_entry(
                row_index,
                column_index,
                bridge_signs,
            )
            for column_index in range(4)
        )
        for row_index in range(4)
    )


def audit_weak_coupling_structure(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_PILOT_SEEDS,
    internal_coupling_gains: tuple[
        float, ...
    ] = PREREGISTERED_INTERNAL_COUPLING_GAINS,
    cross_coupling_strengths: tuple[
        float, ...
    ] = PREREGISTERED_CROSS_COUPLING_STRENGTHS,
    diagonal_gain: float = 1.5,
    tolerance: float = 1e-12,
) -> WeakCouplingStructureGate:
    """taskを計算せず、EXP014の構造条件だけを評価する。"""

    _validate_structure_configuration(
        trial_seeds=trial_seeds,
        internal_coupling_gains=internal_coupling_gains,
        cross_coupling_strengths=cross_coupling_strengths,
        diagonal_gain=diagonal_gain,
        tolerance=tolerance,
    )
    raw_network_counts: list[int] = []
    effective_class_counts: list[int] = []
    zero_components_valid = True
    positive_connectivity_valid = True
    cross_norms_valid = True
    magnitude_pairs_unique = True
    for internal_gain in internal_coupling_gains:
        base_matrices = tuple(
            build_weakly_coupled_modular_weights(
                trial_seed=seed,
                internal_coupling_gain=internal_gain,
                cross_coupling_strength=0.0,
                diagonal_gain=diagonal_gain,
            )
            for seed in trial_seeds
        )
        signatures = tuple(
            tuple(sorted((abs(matrix[0][1]), abs(matrix[2][3]))))
            for matrix in base_matrices
        )
        magnitude_pairs_unique = magnitude_pairs_unique and (
            len(set(signatures)) == len(signatures)
        )
        for cross_strength in cross_coupling_strengths:
            matrices = tuple(
                build_weakly_coupled_modular_weights(
                    trial_seed=seed,
                    internal_coupling_gain=internal_gain,
                    cross_coupling_strength=cross_strength,
                    diagonal_gain=diagonal_gain,
                )
                for seed in trial_seeds
            )
            audit = audit_signed_coordinate_conjugacy(
                matrices,
                tolerance=tolerance,
            )
            raw_network_counts.append(audit.raw_network_count)
            effective_class_counts.append(audit.effective_class_count)
            if cross_strength == 0.0:
                zero_components_valid = zero_components_valid and all(
                    weakly_connected_components(
                        matrix,
                        tolerance=tolerance,
                    )
                    == ((0, 1), (2, 3))
                    for matrix in matrices
                )
            else:
                positive_connectivity_valid = (
                    positive_connectivity_valid
                    and all(
                        weakly_connected_components(
                            matrix,
                            tolerance=tolerance,
                        )
                        == ((0, 1, 2, 3),)
                        for matrix in matrices
                    )
                )
            cross_norms_valid = cross_norms_valid and all(
                math.isclose(
                    matrix_infinity_norm_difference(base, matrix),
                    cross_strength,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for base, matrix in zip(
                    base_matrices,
                    matrices,
                    strict=True,
                )
            )

    raw_counts = tuple(raw_network_counts)
    effective_counts = tuple(effective_class_counts)
    passed = (
        zero_components_valid
        and positive_connectivity_valid
        and cross_norms_valid
        and magnitude_pairs_unique
        and all(count == len(trial_seeds) for count in raw_counts)
        and all(count == len(trial_seeds) for count in effective_counts)
    )
    return WeakCouplingStructureGate(
        trial_seeds=trial_seeds,
        internal_coupling_gains=internal_coupling_gains,
        cross_coupling_strengths=cross_coupling_strengths,
        raw_network_counts=raw_counts,
        effective_class_counts=effective_counts,
        zero_components_valid=zero_components_valid,
        positive_connectivity_valid=positive_connectivity_valid,
        cross_norms_valid=cross_norms_valid,
        magnitude_pairs_unique=magnitude_pairs_unique,
        passed=passed,
    )


def run_weak_coupling_factorization(
    *,
    phase: str = "pilot",
    trial_seeds: tuple[int, ...] | None = None,
    internal_coupling_gains: tuple[
        float, ...
    ] = PREREGISTERED_INTERNAL_COUPLING_GAINS,
    cross_coupling_strengths: tuple[
        float, ...
    ] = PREREGISTERED_CROSS_COUPLING_STRENGTHS,
    disturbance_bounds: tuple[
        float, ...
    ] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    decision_tolerance: float = 1e-12,
) -> WeakCouplingFactorizationResult:
    """弱結合full系とisolated module基準を同じchallenge定義で比較する。"""

    resolved_seeds = _resolve_trial_seeds(phase, trial_seeds)
    _validate_task_configuration(
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    structure_gate = audit_weak_coupling_structure(
        trial_seeds=resolved_seeds,
        internal_coupling_gains=internal_coupling_gains,
        cross_coupling_strengths=cross_coupling_strengths,
        diagonal_gain=diagonal_gain,
        tolerance=decision_tolerance,
    )
    if not structure_gate.passed:
        raise RuntimeError("構造gateが不成立のためtask評価を開始しません")

    points = tuple(
        point
        for trial_seed in resolved_seeds
        for internal_gain in internal_coupling_gains
        for point in _evaluate_seed_gain(
            trial_seed=trial_seed,
            internal_coupling_gain=internal_gain,
            cross_coupling_strengths=cross_coupling_strengths,
            disturbance_bounds=disturbance_bounds,
            diagonal_gain=diagonal_gain,
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
    )
    zero_points = tuple(
        point for point in points if point.cross_coupling_strength == 0.0
    )
    mean_absolute_residuals = tuple(
        _mean_absolute_residual(points, strength)
        for strength in cross_coupling_strengths
    )
    maximum_strength_points = tuple(
        point
        for point in points
        if point.cross_coupling_strength
        == cross_coupling_strengths[-1]
    )
    maximum_strength_nonzero_fraction = (
        sum(
            point.absolute_task_product_residual > decision_tolerance
            for point in maximum_strength_points
        )
        / len(maximum_strength_points)
    )
    decisions = WeakCouplingDecisions(
        structure_gate=structure_gate.passed,
        zero_coupling_recovers_product=all(
            point.coupled_raw_attractor_count
            == point.isolated_raw_attractor_count_product
            and abs(point.task_product_residual) <= decision_tolerance
            for point in zero_points
        ),
        transported_certificate_lower_bound=all(
            point.task_minus_transported_certificate
            >= -decision_tolerance
            for point in points
        ),
        norm_shifted_certificate_lower_bound=all(
            point.task_minus_norm_shifted_certificate
            >= -decision_tolerance
            for point in points
        ),
        transported_dominates_norm_shifted=all(
            point.transported_minus_norm_shifted_certificate
            >= -decision_tolerance
            for point in points
        ),
        maximum_strength_mean_absolute_residual=(
            mean_absolute_residuals[-1]
            >= PREREGISTERED_MAXIMUM_STRENGTH_MEAN_RESIDUAL_THRESHOLD
            - decision_tolerance
        ),
        maximum_strength_nonzero_residual_prevalence=(
            maximum_strength_nonzero_fraction
            >= PREREGISTERED_MAXIMUM_STRENGTH_NONZERO_FRACTION_THRESHOLD
            - decision_tolerance
        ),
        mean_absolute_residual_non_decreasing=all(
            first <= second + decision_tolerance
            for first, second in zip(
                mean_absolute_residuals,
                mean_absolute_residuals[1:],
            )
        ),
    )
    return WeakCouplingFactorizationResult(
        experiment_id=EXPERIMENT_ID,
        phase=phase,
        dimension=4,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        structure_gate=structure_gate,
        points=points,
        decisions=decisions,
        maximum_absolute_task_product_residual=max(
            point.absolute_task_product_residual for point in points
        ),
        minimum_task_minus_transported_certificate=min(
            point.task_minus_transported_certificate for point in points
        ),
        minimum_task_minus_norm_shifted_certificate=min(
            point.task_minus_norm_shifted_certificate for point in points
        ),
        minimum_transported_minus_norm_shifted_certificate=min(
            point.transported_minus_norm_shifted_certificate
            for point in points
        ),
    )


def _evaluate_seed_gain(
    *,
    trial_seed: int,
    internal_coupling_gain: float,
    cross_coupling_strengths: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[WeakCouplingFactorizationPoint, ...]:
    base_weights = build_weakly_coupled_modular_weights(
        trial_seed=trial_seed,
        internal_coupling_gain=internal_coupling_gain,
        cross_coupling_strength=0.0,
        diagonal_gain=diagonal_gain,
    )
    module_weights = (
        _submatrix(base_weights, (0, 1)),
        _submatrix(base_weights, (2, 3)),
    )
    profile_arguments = {
        "disturbance_bounds": disturbance_bounds,
        "task_steps": task_steps,
        "autonomous_steps": autonomous_steps,
        "convergence_tolerance": convergence_tolerance,
    }
    module_profiles = tuple(
        evaluate_sign_memory_network(
            recurrent_weights=matrix,
            **profile_arguments,
        )
        for matrix in module_weights
    )
    return tuple(
        point
        for cross_strength in cross_coupling_strengths
        for point in _evaluate_coupled_profile(
            trial_seed=trial_seed,
            internal_coupling_gain=internal_coupling_gain,
            cross_coupling_strength=cross_strength,
            diagonal_gain=diagonal_gain,
            base_weights=base_weights,
            module_profiles=(module_profiles[0], module_profiles[1]),
            profile_arguments=profile_arguments,
        )
    )


def _evaluate_coupled_profile(
    *,
    trial_seed: int,
    internal_coupling_gain: float,
    cross_coupling_strength: float,
    diagonal_gain: float,
    base_weights: Matrix,
    module_profiles: tuple[SignMemoryNetworkProfile, SignMemoryNetworkProfile],
    profile_arguments: dict[str, object],
) -> tuple[WeakCouplingFactorizationPoint, ...]:
    coupled_weights = build_weakly_coupled_modular_weights(
        trial_seed=trial_seed,
        internal_coupling_gain=internal_coupling_gain,
        cross_coupling_strength=cross_coupling_strength,
        diagonal_gain=diagonal_gain,
    )
    coupled_profile = evaluate_sign_memory_network(
        recurrent_weights=coupled_weights,
        disturbance_bounds=profile_arguments["disturbance_bounds"],
        task_steps=profile_arguments["task_steps"],
        autonomous_steps=profile_arguments["autonomous_steps"],
        convergence_tolerance=profile_arguments["convergence_tolerance"],
    )
    cross_norm = matrix_infinity_norm_difference(
        base_weights,
        coupled_weights,
    )
    return tuple(
        _build_point(
            trial_seed=trial_seed,
            internal_coupling_gain=internal_coupling_gain,
            cross_coupling_strength=cross_coupling_strength,
            cross_coupling_infinity_norm=cross_norm,
            disturbance_index=index,
            coupled_weights=coupled_weights,
            coupled_profile=coupled_profile,
            module_profiles=module_profiles,
        )
        for index in range(len(coupled_profile.disturbance_evaluations))
    )


def _build_point(
    *,
    trial_seed: int,
    internal_coupling_gain: float,
    cross_coupling_strength: float,
    cross_coupling_infinity_norm: float,
    disturbance_index: int,
    coupled_weights: Matrix,
    coupled_profile: SignMemoryNetworkProfile,
    module_profiles: tuple[SignMemoryNetworkProfile, SignMemoryNetworkProfile],
) -> WeakCouplingFactorizationPoint:
    coupled_evaluation = coupled_profile.disturbance_evaluations[
        disturbance_index
    ]
    module_evaluations = tuple(
        profile.disturbance_evaluations[disturbance_index]
        for profile in module_profiles
    )
    isolated_task_product = math.prod(
        evaluation.task_retention for evaluation in module_evaluations
    )
    isolated_component_fraction = math.prod(
        evaluation.certified_robust_fraction
        for evaluation in module_evaluations
    )
    transported_fraction, shifted_fraction, inside_fraction = (
        _transported_certificate_fractions(
            coupled_weights=coupled_weights,
            coupled_profile=coupled_profile,
            module_profiles=module_profiles,
            disturbance_bound=coupled_evaluation.disturbance_bound,
            perturbation_norm=cross_coupling_infinity_norm,
        )
    )
    residual = coupled_evaluation.task_retention - isolated_task_product
    return WeakCouplingFactorizationPoint(
        trial_seed=trial_seed,
        internal_coupling_gain=internal_coupling_gain,
        cross_coupling_strength=cross_coupling_strength,
        cross_coupling_infinity_norm=cross_coupling_infinity_norm,
        disturbance_bound=coupled_evaluation.disturbance_bound,
        coupled_raw_attractor_count=coupled_profile.raw_attractor_count,
        isolated_raw_attractor_count_product=math.prod(
            profile.raw_attractor_count for profile in module_profiles
        ),
        coupled_task_retention=coupled_evaluation.task_retention,
        isolated_task_retention_product=isolated_task_product,
        task_product_residual=residual,
        absolute_task_product_residual=abs(residual),
        coupled_common_certified_fraction=(
            coupled_evaluation.certified_robust_fraction
        ),
        isolated_component_certified_fraction=(
            isolated_component_fraction
        ),
        transported_rectangle_certified_fraction=transported_fraction,
        norm_shifted_certified_fraction=shifted_fraction,
        coupled_fixed_points_inside_rectangle_fraction=inside_fraction,
        task_minus_transported_certificate=(
            coupled_evaluation.task_retention - transported_fraction
        ),
        task_minus_norm_shifted_certificate=(
            coupled_evaluation.task_retention - shifted_fraction
        ),
        transported_minus_norm_shifted_certificate=(
            transported_fraction - shifted_fraction
        ),
    )


def _transported_certificate_fractions(
    *,
    coupled_weights: Matrix,
    coupled_profile: SignMemoryNetworkProfile,
    module_profiles: tuple[SignMemoryNetworkProfile, SignMemoryNetworkProfile],
    disturbance_bound: float,
    perturbation_norm: float,
) -> tuple[float, float, float]:
    module_pairs = tuple(
        (first, second)
        for first in module_profiles[0].orthants
        for second in module_profiles[1].orthants
    )
    transported_count = 0
    shifted_count = 0
    inside_count = 0
    for coupled_orthant, module_pair in zip(
        coupled_profile.orthants,
        module_pairs,
        strict=True,
    ):
        boundaries = (
            (module_pair[0].invariant_boundary,) * 2
            + (module_pair[1].invariant_boundary,) * 2
        )
        expected_signs = (
            module_pair[0].attractor_signs
            + module_pair[1].attractor_signs
        )
        if coupled_orthant.attractor_signs != expected_signs:
            raise RuntimeError("orthant列挙順がmodule直積順と一致しません")
        inside = coupled_orthant.fixed_point_retained and all(
            sign * value >= boundary - 1e-12
            for sign, value, boundary in zip(
                expected_signs,
                coupled_orthant.fixed_point,
                boundaries,
                strict=True,
            )
        )
        inside_count += inside
        rectangle = orthant_rectangle_certificate(
            recurrent_weights=coupled_weights,
            attractor_signs=expected_signs,
            lower_boundaries=boundaries,
        )
        transported_count += (
            inside
            and rectangle.maximum_uniform_disturbance
            >= disturbance_bound - 1e-12
        )
        component_margin = min(
            module_pair[0].maximum_uniform_disturbance,
            module_pair[1].maximum_uniform_disturbance,
        )
        component_fixed_points_valid = all(
            orthant.fixed_point_inside_invariant_box
            for orthant in module_pair
        )
        shifted_count += (
            inside
            and component_fixed_points_valid
            and component_margin
            >= disturbance_bound + perturbation_norm - 1e-12
        )
    orthant_count = len(coupled_profile.orthants)
    return (
        transported_count / orthant_count,
        shifted_count / orthant_count,
        inside_count / orthant_count,
    )


def _bridge_entry(
    row_index: int,
    column_index: int,
    bridge_signs: tuple[float, float],
) -> float:
    if (row_index, column_index) in ((0, 2), (2, 0)):
        return bridge_signs[0]
    if (row_index, column_index) in ((1, 3), (3, 1)):
        return bridge_signs[1]
    return 0.0


def _mean_absolute_residual(
    points: tuple[WeakCouplingFactorizationPoint, ...],
    cross_coupling_strength: float,
) -> float:
    selected = tuple(
        point.absolute_task_product_residual
        for point in points
        if point.cross_coupling_strength == cross_coupling_strength
    )
    return math.fsum(selected) / len(selected)


def _submatrix(matrix: Matrix, indices: tuple[int, ...]) -> Matrix:
    return tuple(
        tuple(matrix[row_index][column_index] for column_index in indices)
        for row_index in indices
    )


def _resolve_trial_seeds(
    phase: str,
    trial_seeds: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if phase not in {"pilot", "confirmation"}:
        raise ValueError("phaseはpilotまたはconfirmationにしてください")
    if trial_seeds is not None:
        return trial_seeds
    if phase == "pilot":
        return PREREGISTERED_PILOT_SEEDS
    return PREREGISTERED_CONFIRMATION_SEEDS


def _validate_structure_configuration(
    *,
    trial_seeds: tuple[int, ...],
    internal_coupling_gains: tuple[float, ...],
    cross_coupling_strengths: tuple[float, ...],
    diagonal_gain: float,
    tolerance: float,
) -> None:
    if (
        len(trial_seeds) < 2
        or len(set(trial_seeds)) != len(trial_seeds)
    ):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    for seed in trial_seeds:
        _validate_seed(seed)
    _validate_strictly_increasing_positive(
        internal_coupling_gains,
        "internal_coupling_gains",
    )
    if (
        not cross_coupling_strengths
        or cross_coupling_strengths[0] != 0.0
        or any(
            not math.isfinite(strength) or strength < 0.0
            for strength in cross_coupling_strengths
        )
        or any(
            first >= second
            for first, second in zip(
                cross_coupling_strengths,
                cross_coupling_strengths[1:],
            )
        )
    ):
        raise ValueError(
            "cross_coupling_strengthsは0から始まる厳密昇順の有限非負値にしてください"
        )
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")


def _validate_task_configuration(
    *,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> None:
    _validate_strictly_increasing_positive(
        disturbance_bounds,
        "disturbance_bounds",
    )
    for value, name in (
        (task_steps, "task_steps"),
        (autonomous_steps, "autonomous_steps"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name}は1以上の整数にしてください")
    _validate_positive_finite(
        convergence_tolerance,
        "convergence_tolerance",
    )


def _validate_strictly_increasing_positive(
    values: tuple[float, ...],
    name: str,
) -> None:
    if (
        not values
        or any(not math.isfinite(value) or value <= 0.0 for value in values)
        or any(
            first >= second
            for first, second in zip(values, values[1:])
        )
    ):
        raise ValueError(f"{name}は厳密昇順の有限正値にしてください")


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}は有限の正値にしてください")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("trial_seedは整数にしてください")
