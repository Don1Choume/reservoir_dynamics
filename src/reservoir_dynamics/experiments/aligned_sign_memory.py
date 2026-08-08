"""module sizeを跨げる固定方向codeによる符号記憶評価。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.sign_memory_evaluation import (
    OrthantSignMemoryEvaluation,
    _evaluate_orthant,
    _retains_sign_memory,
)
from reservoir_dynamics.theory.orthant_box import Matrix

ALIGNED_DIRECTION_CODES = (
    "inward",
    "outward",
    "alternating",
    "reverse_alternating",
)


@dataclass(frozen=True, slots=True)
class AlignedDirectionEvaluation:
    """一つの相対方向codeに対するorthant平均保持率。"""

    code: str
    task_success_count: int
    total_challenges: int
    task_retention: float


@dataclass(frozen=True, slots=True)
class AlignedDisturbanceEvaluation:
    """一外乱強度におけるcertificateと4方向task。"""

    disturbance_bound: float
    certified_robust_fraction: float
    direction_evaluations: tuple[AlignedDirectionEvaluation, ...]
    mean_task_retention: float
    guarantee_gap: float


@dataclass(frozen=True, slots=True)
class AlignedSignMemoryProfile:
    """固定方向数で高次元化できる多重安定network profile。"""

    dimension: int
    coordinate_offset: int
    coordinate_indices: tuple[int, ...]
    raw_attractor_count: int
    mean_uniform_disturbance_margin: float
    orthants: tuple[OrthantSignMemoryEvaluation, ...]
    disturbance_evaluations: tuple[AlignedDisturbanceEvaluation, ...]


def evaluate_aligned_sign_memory_network(
    *,
    recurrent_weights: Matrix,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    coordinate_offset: int = 0,
    coordinate_indices: tuple[int, ...] | None = None,
) -> AlignedSignMemoryProfile:
    """符号相対4方向だけを列挙し、challenge数をO(2^d)へ抑える。"""

    dimension, normalized_coordinate_indices = _validate_configuration(
        recurrent_weights=recurrent_weights,
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        coordinate_offset=coordinate_offset,
        coordinate_indices=coordinate_indices,
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
    mean_margin = math.fsum(
        orthant.maximum_uniform_disturbance for orthant in orthants
    ) / len(orthants)
    evaluations = tuple(
        _evaluate_disturbance(
            recurrent_weights=recurrent_weights,
            orthants=orthants,
            disturbance_bound=disturbance_bound,
            task_steps=task_steps,
            coordinate_indices=normalized_coordinate_indices,
        )
        for disturbance_bound in disturbance_bounds
    )
    return AlignedSignMemoryProfile(
        dimension=dimension,
        coordinate_offset=coordinate_offset,
        coordinate_indices=normalized_coordinate_indices,
        raw_attractor_count=sum(
            orthant.fixed_point_retained for orthant in orthants
        ),
        mean_uniform_disturbance_margin=mean_margin,
        orthants=orthants,
        disturbance_evaluations=evaluations,
    )


def _evaluate_disturbance(
    *,
    recurrent_weights: Matrix,
    orthants: tuple[OrthantSignMemoryEvaluation, ...],
    disturbance_bound: float,
    task_steps: int,
    coordinate_indices: tuple[int, ...],
) -> AlignedDisturbanceEvaluation:
    masks = _direction_masks(coordinate_indices)
    direction_evaluations = tuple(
        _evaluate_direction(
            recurrent_weights=recurrent_weights,
            orthants=orthants,
            disturbance_bound=disturbance_bound,
            task_steps=task_steps,
            code=code,
            relative_mask=mask,
        )
        for code, mask in zip(ALIGNED_DIRECTION_CODES, masks, strict=True)
    )
    certified_fraction = (
        sum(
            orthant.fixed_point_inside_invariant_box
            and orthant.maximum_uniform_disturbance
            >= disturbance_bound - 1e-12
            for orthant in orthants
        )
        / len(orthants)
    )
    mean_retention = math.fsum(
        evaluation.task_retention for evaluation in direction_evaluations
    ) / len(direction_evaluations)
    return AlignedDisturbanceEvaluation(
        disturbance_bound=disturbance_bound,
        certified_robust_fraction=certified_fraction,
        direction_evaluations=direction_evaluations,
        mean_task_retention=mean_retention,
        guarantee_gap=mean_retention - certified_fraction,
    )


def _evaluate_direction(
    *,
    recurrent_weights: Matrix,
    orthants: tuple[OrthantSignMemoryEvaluation, ...],
    disturbance_bound: float,
    task_steps: int,
    code: str,
    relative_mask: tuple[int, ...],
) -> AlignedDirectionEvaluation:
    successes = sum(
        _retains_sign_memory(
            recurrent_weights=recurrent_weights,
            initial_state=orthant.fixed_point,
            attractor_signs=orthant.attractor_signs,
            disturbance=tuple(
                disturbance_bound * sign * mask
                for sign, mask in zip(
                    orthant.attractor_signs,
                    relative_mask,
                    strict=True,
                )
            ),
            steps=task_steps,
        )
        for orthant in orthants
        if orthant.fixed_point_retained
    )
    return AlignedDirectionEvaluation(
        code=code,
        task_success_count=successes,
        total_challenges=len(orthants),
        task_retention=successes / len(orthants),
    )


def _direction_masks(
    coordinate_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    alternating = tuple(
        -1 if coordinate_index % 2 == 0 else 1
        for coordinate_index in coordinate_indices
    )
    dimension = len(coordinate_indices)
    return (
        (-1,) * dimension,
        (1,) * dimension,
        alternating,
        tuple(-value for value in alternating),
    )


def _validate_configuration(
    *,
    recurrent_weights: Matrix,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    coordinate_offset: int,
    coordinate_indices: tuple[int, ...] | None,
) -> tuple[int, tuple[int, ...]]:
    if not recurrent_weights:
        raise ValueError("recurrent_weightsは空にできません")
    dimension = len(recurrent_weights)
    if dimension > 10 or any(len(row) != dimension for row in recurrent_weights):
        raise ValueError("recurrent_weightsは10次元以下の正方行列にしてください")
    if any(not math.isfinite(value) for row in recurrent_weights for value in row):
        raise ValueError("recurrent_weightsは有限値にしてください")
    if (
        not disturbance_bounds
        or any(not math.isfinite(value) or value <= 0.0 for value in disturbance_bounds)
        or any(first >= second for first, second in zip(disturbance_bounds, disturbance_bounds[1:]))
    ):
        raise ValueError("disturbance_boundsは厳密昇順の有限正値にしてください")
    for value, name in ((task_steps, "task_steps"), (autonomous_steps, "autonomous_steps")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name}は1以上の整数にしてください")
    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("convergence_toleranceは有限の正値にしてください")
    if not isinstance(coordinate_offset, int) or isinstance(coordinate_offset, bool) or coordinate_offset < 0:
        raise ValueError("coordinate_offsetは0以上の整数にしてください")
    if coordinate_indices is None:
        normalized_indices = tuple(
            coordinate_offset + index for index in range(dimension)
        )
    else:
        if coordinate_offset != 0:
            raise ValueError(
                "coordinate_indices指定時のcoordinate_offsetは0にしてください"
            )
        normalized_indices = tuple(coordinate_indices)
        if (
            len(normalized_indices) != dimension
            or len(set(normalized_indices)) != dimension
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in normalized_indices
            )
        ):
            raise ValueError(
                "coordinate_indicesは重複しない非負整数を次元数だけ指定してください"
            )
    return dimension, normalized_indices
