"""離散時間の入力駆動系を複数初期状態から再現可能に実行する。"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

Vector = tuple[float, ...]
Trajectory = tuple[Vector, ...]


class DiscreteDrivenSystem(Protocol):
    """離散時間・入力駆動系が満たす最小インターフェース。"""

    state_dimension: int
    input_dimension: int

    def step(self, state: Vector, input_value: Vector) -> Vector:
        """一時刻だけ状態を更新する。"""


@dataclass(frozen=True, slots=True)
class DiscreteSimulationResult:
    """共通入力と初期状態を含む不変なシミュレーション結果。"""

    inputs: tuple[Vector, ...]
    trajectories: tuple[Trajectory, ...]


def simulate_discrete_replicas(
    *,
    system: DiscreteDrivenSystem,
    initial_states: Sequence[Sequence[float]],
    inputs: Sequence[Sequence[float]],
) -> DiscreteSimulationResult:
    """同一入力列をすべての初期状態へ与え、初期時刻を含む軌道を返す。"""

    if system.state_dimension < 1:
        raise ValueError("状態次元は1以上必要です")
    if system.input_dimension < 1:
        raise ValueError("入力次元は1以上必要です")
    if not initial_states:
        raise ValueError("初期状態は1つ以上必要です")

    normalized_initial_states = tuple(
        _normalize_vector(
            state,
            expected_dimension=system.state_dimension,
            dimension_name="状態次元",
            value_name="初期状態",
        )
        for state in initial_states
    )
    normalized_inputs = tuple(
        _normalize_vector(
            input_value,
            expected_dimension=system.input_dimension,
            dimension_name="入力次元",
            value_name="入力",
        )
        for input_value in inputs
    )

    trajectories = tuple(
        _simulate_single_trajectory(
            system=system,
            initial_state=initial_state,
            inputs=normalized_inputs,
        )
        for initial_state in normalized_initial_states
    )
    return DiscreteSimulationResult(
        inputs=normalized_inputs,
        trajectories=trajectories,
    )


def _simulate_single_trajectory(
    *,
    system: DiscreteDrivenSystem,
    initial_state: Vector,
    inputs: tuple[Vector, ...],
) -> Trajectory:
    states = [initial_state]
    current_state = initial_state
    for input_value in inputs:
        raw_next_state = system.step(current_state, input_value)
        current_state = _normalize_vector(
            raw_next_state,
            expected_dimension=system.state_dimension,
            dimension_name="力学系が出力した状態次元",
            value_name="力学系が出力した状態",
        )
        states.append(current_state)
    return tuple(states)


def _normalize_vector(
    values: Sequence[float],
    *,
    expected_dimension: int,
    dimension_name: str,
    value_name: str,
) -> Vector:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_name}は数値列である必要があります") from error

    if len(normalized) != expected_dimension:
        raise ValueError(
            f"{dimension_name}は{expected_dimension}である必要があります"
        )
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{value_name}はすべて有限である必要があります")
    return normalized
