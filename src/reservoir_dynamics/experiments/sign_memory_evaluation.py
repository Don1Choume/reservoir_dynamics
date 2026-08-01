"""任意の小次元tanh RNNに対するorthant符号記憶評価。"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

from reservoir_dynamics.experiments.orthant_margin_sweep import _step
from reservoir_dynamics.theory.orthant_box import (
    Matrix,
    Signs,
    robust_orthant_box_certificate,
)


@dataclass(frozen=True, slots=True)
class OrthantSignMemoryEvaluation:
    """一つのorthantに対する固定点とcommon-boundary certificate。"""

    attractor_signs: Signs
    fixed_point: tuple[float, ...]
    fixed_point_retained: bool
    invariant_boundary: float
    maximum_uniform_disturbance: float
    fixed_point_inside_invariant_box: bool


@dataclass(frozen=True, slots=True)
class DisturbanceSignMemoryEvaluation:
    """一つの外乱強度における認証率と経験符号保持率。"""

    disturbance_bound: float
    certified_robust_count: int
    certified_robust_fraction: float
    task_success_count: int
    total_challenges: int
    task_retention: float
    guarantee_gap: float


@dataclass(frozen=True, slots=True)
class SignMemoryNetworkProfile:
    """一つのnetworkに対する外乱profile。"""

    dimension: int
    raw_attractor_count: int
    orthants: tuple[OrthantSignMemoryEvaluation, ...]
    disturbance_evaluations: tuple[
        DisturbanceSignMemoryEvaluation, ...
    ]


def evaluate_sign_memory_network(
    *,
    recurrent_weights: Matrix,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> SignMemoryNetworkProfile:
    """全orthantと全一定corner外乱を一様列挙して評価する。"""

    dimension = _validate_configuration(
        recurrent_weights=recurrent_weights,
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    sign_patterns = tuple(itertools.product((-1, 1), repeat=dimension))
    orthants = tuple(
        _evaluate_orthant(
            recurrent_weights=recurrent_weights,
            attractor_signs=signs,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
        for signs in sign_patterns
    )
    disturbance_evaluations = tuple(
        _evaluate_disturbance(
            recurrent_weights=recurrent_weights,
            orthants=orthants,
            disturbance_directions=sign_patterns,
            disturbance_bound=disturbance_bound,
            task_steps=task_steps,
        )
        for disturbance_bound in disturbance_bounds
    )
    return SignMemoryNetworkProfile(
        dimension=dimension,
        raw_attractor_count=sum(
            orthant.fixed_point_retained for orthant in orthants
        ),
        orthants=orthants,
        disturbance_evaluations=disturbance_evaluations,
    )


def _evaluate_orthant(
    *,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> OrthantSignMemoryEvaluation:
    fixed_point, fixed_point_retained = _find_fixed_point(
        recurrent_weights=recurrent_weights,
        attractor_signs=attractor_signs,
        steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    certificate = robust_orthant_box_certificate(
        recurrent_weights=recurrent_weights,
        attractor_signs=attractor_signs,
    )
    inside_invariant_box = fixed_point_retained and all(
        sign * value >= certificate.invariant_boundary - 1e-12
        for sign, value in zip(
            attractor_signs,
            fixed_point,
            strict=True,
        )
    )
    return OrthantSignMemoryEvaluation(
        attractor_signs=attractor_signs,
        fixed_point=fixed_point,
        fixed_point_retained=fixed_point_retained,
        invariant_boundary=certificate.invariant_boundary,
        maximum_uniform_disturbance=(
            certificate.maximum_uniform_disturbance
        ),
        fixed_point_inside_invariant_box=inside_invariant_box,
    )


def _evaluate_disturbance(
    *,
    recurrent_weights: Matrix,
    orthants: tuple[OrthantSignMemoryEvaluation, ...],
    disturbance_directions: tuple[Signs, ...],
    disturbance_bound: float,
    task_steps: int,
) -> DisturbanceSignMemoryEvaluation:
    certified_count = sum(
        orthant.fixed_point_inside_invariant_box
        and orthant.maximum_uniform_disturbance
        >= disturbance_bound - 1e-12
        for orthant in orthants
    )
    success_count = sum(
        _retains_sign_memory(
            recurrent_weights=recurrent_weights,
            initial_state=orthant.fixed_point,
            attractor_signs=orthant.attractor_signs,
            disturbance=tuple(
                disturbance_bound * sign for sign in direction
            ),
            steps=task_steps,
        )
        for orthant in orthants
        if orthant.fixed_point_retained
        for direction in disturbance_directions
    )
    total_challenges = len(orthants) * len(disturbance_directions)
    certified_fraction = certified_count / len(orthants)
    task_retention = success_count / total_challenges
    return DisturbanceSignMemoryEvaluation(
        disturbance_bound=disturbance_bound,
        certified_robust_count=certified_count,
        certified_robust_fraction=certified_fraction,
        task_success_count=success_count,
        total_challenges=total_challenges,
        task_retention=task_retention,
        guarantee_gap=task_retention - certified_fraction,
    )


def _find_fixed_point(
    *,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    steps: int,
    convergence_tolerance: float,
) -> tuple[tuple[float, ...], bool]:
    state = tuple(0.9 * sign for sign in attractor_signs)
    retained = True
    residual = math.inf
    for _ in range(steps):
        next_state = _step(
            recurrent_weights,
            state,
            (0.0,) * len(state),
        )
        residual = max(
            abs(next_value - value)
            for next_value, value in zip(next_state, state, strict=True)
        )
        state = next_state
        retained = retained and all(
            sign * value > 0.0
            for sign, value in zip(attractor_signs, state, strict=True)
        )
    return state, retained and residual <= convergence_tolerance


def _retains_sign_memory(
    *,
    recurrent_weights: Matrix,
    initial_state: tuple[float, ...],
    attractor_signs: Signs,
    disturbance: tuple[float, ...],
    steps: int,
) -> bool:
    state = initial_state
    for _ in range(steps):
        state = _step(recurrent_weights, state, disturbance)
        if any(
            sign * value <= 0.0
            for sign, value in zip(attractor_signs, state, strict=True)
        ):
            return False
    return True


def _validate_configuration(
    *,
    recurrent_weights: Matrix,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> int:
    if not recurrent_weights:
        raise ValueError("recurrent_weightsは空にできません")
    dimension = len(recurrent_weights)
    if dimension > 6 or any(
        len(row) != dimension for row in recurrent_weights
    ):
        raise ValueError("recurrent_weightsは6次元以下の正方行列にしてください")
    if any(
        not math.isfinite(value)
        for row in recurrent_weights
        for value in row
    ):
        raise ValueError("recurrent_weightsは有限値にしてください")
    if (
        not disturbance_bounds
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in disturbance_bounds
        )
        or any(
            first >= second
            for first, second in zip(
                disturbance_bounds,
                disturbance_bounds[1:],
            )
        )
    ):
        raise ValueError(
            "disturbance_boundsは厳密昇順の有限正値にしてください"
        )
    _validate_positive_integer(task_steps, "task_steps")
    _validate_positive_integer(autonomous_steps, "autonomous_steps")
    if (
        not math.isfinite(convergence_tolerance)
        or convergence_tolerance <= 0.0
    ):
        raise ValueError("convergence_toleranceは有限の正値にしてください")
    return dimension


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name}は1以上の整数にしてください")
