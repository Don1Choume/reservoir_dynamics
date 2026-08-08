"""非対称module結合を方向別流入量で評価する純粋関数。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.theory.orthant_box import Matrix


@dataclass(frozen=True, slots=True)
class DirectionalMarginSummary:
    """orthant直積集合における方向別・global norm保証の要約。"""

    inbound_loads: tuple[float, float]
    global_load: float
    directional_certified_fraction: float
    global_certified_fraction: float
    mean_directional_slack: float
    minimum_directional_slack: float


def directional_bridge_norms(
    recurrent_weights: Matrix,
    *,
    split_index: int,
) -> tuple[float, float]:
    """第二moduleから第一、第一から第二への行和normを別々に返す。"""

    dimension = _validate_square_matrix(recurrent_weights)
    if (
        not isinstance(split_index, int)
        or isinstance(split_index, bool)
        or not 0 < split_index < dimension
    ):
        raise ValueError("split_indexはmodule境界を表す内部indexにしてください")

    inbound_first = max(
        math.fsum(abs(row[column]) for column in range(split_index, dimension))
        for row in recurrent_weights[:split_index]
    )
    inbound_second = max(
        math.fsum(abs(row[column]) for column in range(split_index))
        for row in recurrent_weights[split_index:]
    )
    return inbound_first, inbound_second


def summarize_directional_margins(
    *,
    component_margin_pairs: tuple[tuple[float, float], ...],
    disturbance_bound: float,
    inbound_loads: tuple[float, float],
    tolerance: float = 1e-12,
) -> DirectionalMarginSummary:
    """component marginから方向別と単一global budgetの認証率を計算する。"""

    if not component_margin_pairs:
        raise ValueError("component margin pairは1件以上必要です")
    normalized_disturbance = _validate_nonnegative_finite(
        disturbance_bound,
        "disturbance_bound",
    )
    if len(inbound_loads) != 2:
        raise ValueError("inbound_loadsは二module分にしてください")
    normalized_loads = tuple(
        _validate_nonnegative_finite(value, "inbound_load")
        for value in inbound_loads
    )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")

    pairs = tuple(
        tuple(float(value) for value in pair)
        for pair in component_margin_pairs
    )
    if any(len(pair) != 2 for pair in pairs) or any(
        not math.isfinite(value) for pair in pairs for value in pair
    ):
        raise ValueError("component margin pairは有限値二つにしてください")

    directional_slacks = tuple(
        min(
            first - normalized_disturbance - normalized_loads[0],
            second - normalized_disturbance - normalized_loads[1],
        )
        for first, second in pairs
    )
    global_load = max(normalized_loads)
    global_slacks = tuple(
        min(
            first - normalized_disturbance - global_load,
            second - normalized_disturbance - global_load,
        )
        for first, second in pairs
    )
    return DirectionalMarginSummary(
        inbound_loads=(normalized_loads[0], normalized_loads[1]),
        global_load=global_load,
        directional_certified_fraction=(
            sum(slack >= -tolerance for slack in directional_slacks)
            / len(directional_slacks)
        ),
        global_certified_fraction=(
            sum(slack >= -tolerance for slack in global_slacks)
            / len(global_slacks)
        ),
        mean_directional_slack=(
            math.fsum(directional_slacks) / len(directional_slacks)
        ),
        minimum_directional_slack=min(directional_slacks),
    )


def _validate_square_matrix(matrix: Matrix) -> int:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("recurrent_weightsは空でない正方行列にしてください")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("recurrent_weightsは有限値にしてください")
    return len(matrix)


def _validate_nonnegative_finite(value: float, name: str) -> float:
    normalized = float(value)
    if isinstance(value, bool) or not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name}は有限の非負値にしてください")
    return normalized

