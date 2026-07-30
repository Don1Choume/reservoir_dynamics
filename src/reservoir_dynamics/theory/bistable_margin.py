"""scalar tanh双安定coreのロバスト不変集合証明。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BistableTanhCertificate:
    """有界外力に対する二つの符号領域の保護証明。"""

    recurrent_gain: float
    invariant_boundary: float
    critical_forcing: float
    certified_uniform_fraction: float


def bistable_tanh_certificate(
    recurrent_gain: float,
) -> BistableTanhCertificate:
    """`x'=tanh(a*x+eta)` の最大対称外力余裕を返す。"""

    _validate_bistable_gain(recurrent_gain)
    invariant_boundary = math.sqrt(1.0 - 1.0 / recurrent_gain)
    critical_forcing = (
        recurrent_gain * invariant_boundary
        - math.atanh(invariant_boundary)
    )
    return BistableTanhCertificate(
        recurrent_gain=recurrent_gain,
        invariant_boundary=invariant_boundary,
        critical_forcing=critical_forcing,
        # [-1, 1]上一様初期値のうち、正負の証明区間が占める割合である。
        certified_uniform_fraction=1.0 - invariant_boundary,
    )


def positive_bistable_fixed_point(
    recurrent_gain: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> float:
    """非零の正安定固定点を単調な二分法で求める。"""

    certificate = bistable_tanh_certificate(recurrent_gain)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("toleranceは有限の正値にしてください")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterationsは1以上の整数にしてください")

    lower = certificate.invariant_boundary
    upper = 1.0
    for _ in range(max_iterations):
        midpoint = (lower + upper) / 2.0
        residual = math.tanh(recurrent_gain * midpoint) - midpoint
        if abs(residual) <= tolerance or upper - lower <= tolerance:
            return midpoint
        if residual > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    raise RuntimeError("正固定点の計算がmax_iterations内に収束しませんでした")


def _validate_bistable_gain(recurrent_gain: float) -> None:
    if not math.isfinite(recurrent_gain) or recurrent_gain <= 1.0:
        raise ValueError("recurrent_gainは1より大きい有限値にしてください")
