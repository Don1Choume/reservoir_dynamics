"""接ベクトル法で最大条件付きLyapunov指数を推定する。"""

import math
from collections.abc import Sequence
from typing import Protocol

Vector = tuple[float, ...]


class JacobianVectorSystem(Protocol):
    """接ベクトル発展を提供する入力駆動系。"""

    state_dimension: int
    input_dimension: int

    def jacobian_vector_product(
        self,
        state: Vector,
        input_value: Vector,
        tangent: Vector,
    ) -> Vector:
        """状態Jacobianを接ベクトルへ作用させる。"""


def top_conditional_lyapunov_exponent(
    *,
    system: JacobianVectorSystem,
    trajectory: Sequence[Sequence[float]],
    inputs: Sequence[Sequence[float]],
    washout: int = 0,
    initial_tangent: Sequence[float] | None = None,
    derivative_floor: float = 1e-300,
) -> float:
    """一つの入力駆動軌道に沿う最大有限時間指数を返す。"""

    normalized_trajectory = tuple(
        _normalize_vector(
            state,
            expected_dimension=system.state_dimension,
            value_name="状態",
        )
        for state in trajectory
    )
    normalized_inputs = tuple(
        _normalize_vector(
            input_value,
            expected_dimension=system.input_dimension,
            value_name="入力",
        )
        for input_value in inputs
    )
    if len(normalized_trajectory) != len(normalized_inputs) + 1:
        raise ValueError("軌道の時系列長は入力の時系列長より1大きくしてください")
    if not isinstance(washout, int) or isinstance(washout, bool):
        raise ValueError("washoutは0以上の整数である必要があります")
    if washout < 0 or washout >= len(normalized_inputs):
        raise ValueError("washout後の評価区間は1時刻以上必要です")
    if not math.isfinite(derivative_floor) or derivative_floor <= 0.0:
        raise ValueError("derivative_floorは有限の正数である必要があります")

    raw_tangent = (
        tuple(float(value) for value in initial_tangent)
        if initial_tangent is not None
        else (1.0,) * system.state_dimension
    )
    tangent = _normalize_tangent(
        raw_tangent,
        state_dimension=system.state_dimension,
    )
    canonical_tangent = (1.0,) + (0.0,) * (system.state_dimension - 1)
    logarithmic_growth: list[float] = []

    for time_index, input_value in enumerate(normalized_inputs):
        stretched_tangent = system.jacobian_vector_product(
            normalized_trajectory[time_index],
            input_value,
            tangent,
        )
        growth = math.sqrt(
            math.fsum(value * value for value in stretched_tangent)
        )
        if not math.isfinite(growth):
            raise ValueError("接ベクトル成長率は有限である必要があります")

        if growth > derivative_floor:
            tangent = tuple(value / growth for value in stretched_tangent)
        else:
            tangent = canonical_tangent
        if time_index >= washout:
            logarithmic_growth.append(math.log(max(growth, derivative_floor)))

    return math.fsum(logarithmic_growth) / len(logarithmic_growth)


def _normalize_vector(
    values: Sequence[float],
    *,
    expected_dimension: int,
    value_name: str,
) -> Vector:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_name}は数値列である必要があります") from error
    if len(normalized) != expected_dimension:
        raise ValueError(
            f"{value_name}の次元は{expected_dimension}である必要があります"
        )
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{value_name}はすべて有限である必要があります")
    return normalized


def _normalize_tangent(
    tangent: Vector,
    *,
    state_dimension: int,
) -> Vector:
    normalized = _normalize_vector(
        tangent,
        expected_dimension=state_dimension,
        value_name="接ベクトル",
    )
    magnitude = math.sqrt(math.fsum(value * value for value in normalized))
    if magnitude == 0.0:
        raise ValueError("接ベクトルは非ゼロである必要があります")
    return tuple(value / magnitude for value in normalized)
