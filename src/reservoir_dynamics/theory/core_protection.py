"""可塑reserveから機能coreへ漏れる摂動の決定論的上界。"""

from __future__ import annotations

import math


def core_deviation_bound_curve(
    *,
    core_lipschitz: float,
    feedback_lipschitz: float,
    reserve_difference_bound: float,
    steps: int,
    initial_core_distance: float = 0.0,
) -> tuple[float, ...]:
    """`D[t+1] <= Lc D[t] + Lf R` の反復上界を返す。"""

    if (
        not math.isfinite(core_lipschitz)
        or core_lipschitz < 0.0
        or core_lipschitz >= 1.0
    ):
        raise ValueError(
            "core_lipschitzは0以上1未満の有限値にしてください"
        )
    _validate_non_negative_finite(
        feedback_lipschitz,
        "feedback_lipschitz",
    )
    _validate_non_negative_finite(
        reserve_difference_bound,
        "reserve_difference_bound",
    )
    _validate_non_negative_finite(
        initial_core_distance,
        "initial_core_distance",
    )
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("stepsは0以上の整数にしてください")

    forcing_bound = feedback_lipschitz * reserve_difference_bound
    bounds = [initial_core_distance]
    for _ in range(steps):
        bounds.append(core_lipschitz * bounds[-1] + forcing_bound)
    return tuple(bounds)


def _validate_non_negative_finite(value: float, value_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{value_name}は有限の非負値にしてください")
