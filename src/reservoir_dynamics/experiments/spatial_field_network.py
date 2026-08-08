"""EXP-2026-015で共有する非対称core–reserve networkとpaired信号。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from reservoir_dynamics.systems.tanh_rnn import Matrix, Vector
from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
)


@dataclass(frozen=True, slots=True)
class SpatialCoreReserveNetwork:
    """異なる次元と非対称bridgeを持つcore–reserve parameter。"""

    core_recurrent_gain: float
    reserve_recurrent_weights: Matrix
    core_to_reserve_weights: Matrix
    reserve_to_core_weights: Matrix
    reserve_input_weights: Vector

    @property
    def core_dimension(self) -> int:
        return len(self.reserve_to_core_weights)

    @property
    def reserve_dimension(self) -> int:
        return len(self.reserve_recurrent_weights)

    @property
    def state_dimension(self) -> int:
        return self.core_dimension + self.reserve_dimension

    @property
    def full_recurrent_weights(self) -> Matrix:
        core_rows = tuple(
            tuple(
                self.core_recurrent_gain if row == column else 0.0
                for column in range(self.core_dimension)
            )
            + self.reserve_to_core_weights[row]
            for row in range(self.core_dimension)
        )
        reserve_rows = tuple(
            self.core_to_reserve_weights[row]
            + self.reserve_recurrent_weights[row]
            for row in range(self.reserve_dimension)
        )
        return core_rows + reserve_rows

    @property
    def magnitude_fingerprint(self) -> tuple[str, ...]:
        """符号対角共役で保存される絶対値をbit-exactに記録する。"""

        return tuple(
            abs(value).hex()
            for row in self.full_recurrent_weights
            for value in row
        ) + tuple(abs(value).hex() for value in self.reserve_input_weights)


@dataclass(frozen=True, slots=True)
class SpatialTrialSignals:
    """policy間で共有する入力と有界外乱。"""

    novel_inputs: Vector
    core_disturbances: tuple[Vector, ...]
    reserve_disturbances: tuple[Vector, ...]


def build_spatial_core_reserve_network(
    *,
    trial_seed: int,
    feedback_gain: float,
    core_dimension: int = 3,
    reserve_dimension: int = 5,
    core_recurrent_gain: float = 1.5,
    reserve_recurrent_gain: float = 0.85,
    core_to_reserve_gain: float = 0.12,
) -> SpatialCoreReserveNetwork:
    """連続絶対値を持つ非対称3+5 networkを決定論的に生成する。"""

    _validate_positive_integer(core_dimension, "core_dimension")
    _validate_positive_integer(reserve_dimension, "reserve_dimension")
    if core_dimension == reserve_dimension:
        raise ValueError("coreとreserveは異なるmodule sizeにしてください")
    bistable_tanh_certificate(core_recurrent_gain)
    _validate_positive_finite(feedback_gain, "feedback_gain")
    _validate_open_unit_interval(
        reserve_recurrent_gain,
        "reserve_recurrent_gain",
    )
    _validate_positive_finite(
        core_to_reserve_gain,
        "core_to_reserve_gain",
    )
    _validate_seed(trial_seed)

    random_generator = random.Random(7_000_001 + trial_seed)
    reserve_recurrent = _draw_row_normalized_matrix(
        rows=reserve_dimension,
        columns=reserve_dimension,
        maximum_row_sum=reserve_recurrent_gain,
        random_generator=random_generator,
    )
    core_to_reserve = _draw_row_normalized_matrix(
        rows=reserve_dimension,
        columns=core_dimension,
        maximum_row_sum=core_to_reserve_gain,
        random_generator=random_generator,
    )
    reserve_to_core = _draw_row_normalized_matrix(
        rows=core_dimension,
        columns=reserve_dimension,
        maximum_row_sum=feedback_gain,
        random_generator=random_generator,
    )
    reserve_input = tuple(
        random_generator.choice((-1.0, 1.0))
        * random_generator.uniform(0.45, 0.95)
        for _ in range(reserve_dimension)
    )
    return SpatialCoreReserveNetwork(
        core_recurrent_gain=core_recurrent_gain,
        reserve_recurrent_weights=reserve_recurrent,
        core_to_reserve_weights=core_to_reserve,
        reserve_to_core_weights=reserve_to_core,
        reserve_input_weights=reserve_input,
    )


def generate_spatial_trial_signals(
    *,
    trial_seed: int,
    steps: int,
    core_dimension: int,
    reserve_dimension: int,
    disturbance_bound: float,
) -> SpatialTrialSignals:
    """noise level間でも標準化乱数を共有するpaired信号を作る。"""

    _validate_seed(trial_seed)
    _validate_positive_integer(steps, "steps")
    _validate_positive_integer(core_dimension, "core_dimension")
    _validate_positive_integer(reserve_dimension, "reserve_dimension")
    if not math.isfinite(disturbance_bound) or disturbance_bound < 0.0:
        raise ValueError("disturbance_boundは有限の非負値にしてください")
    random_generator = random.Random(8_000_003 + trial_seed)
    novel_inputs: list[float] = []
    core_disturbances: list[Vector] = []
    reserve_disturbances: list[Vector] = []
    for _ in range(steps):
        novel_inputs.append(random_generator.uniform(-1.0, 1.0))
        core_disturbances.append(
            tuple(
                disturbance_bound * random_generator.uniform(-1.0, 1.0)
                for _ in range(core_dimension)
            )
        )
        reserve_disturbances.append(
            tuple(
                disturbance_bound * random_generator.uniform(-1.0, 1.0)
                for _ in range(reserve_dimension)
            )
        )
    return SpatialTrialSignals(
        novel_inputs=tuple(novel_inputs),
        core_disturbances=tuple(core_disturbances),
        reserve_disturbances=tuple(reserve_disturbances),
    )


def _draw_row_normalized_matrix(
    *,
    rows: int,
    columns: int,
    maximum_row_sum: float,
    random_generator: random.Random,
) -> Matrix:
    matrix: list[Vector] = []
    for _ in range(rows):
        raw = tuple(
            random_generator.choice((-1.0, 1.0))
            * random_generator.uniform(0.2, 1.0)
            for _ in range(columns)
        )
        target_sum = maximum_row_sum * random_generator.uniform(0.8, 1.0)
        scale = target_sum / math.fsum(abs(value) for value in raw)
        matrix.append(tuple(scale * value for value in raw))
    return tuple(matrix)


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name}は1以上の整数にしてください")


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}は有限の正値にしてください")


def _validate_open_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name}は0と1の間にしてください")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("trial_seedは整数にしてください")
