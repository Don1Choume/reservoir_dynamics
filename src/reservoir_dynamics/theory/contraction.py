"""Lipschitz写像の反復に対する距離上界。"""

import math


def iterated_lipschitz_distance_bound(
    *,
    initial_distance: float,
    lipschitz_constant: float,
    steps: int,
) -> tuple[float, ...]:
    """`D[t] <= L**t * D[0]` の右辺を時刻0から返す。"""

    if not math.isfinite(initial_distance) or initial_distance < 0.0:
        raise ValueError("初期距離は有限の非負値である必要があります")
    if not math.isfinite(lipschitz_constant) or lipschitz_constant < 0.0:
        raise ValueError("Lipschitz定数は有限の非負値である必要があります")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("stepsは0以上の整数である必要があります")

    return tuple(
        initial_distance * lipschitz_constant**step_index
        for step_index in range(steps + 1)
    )
