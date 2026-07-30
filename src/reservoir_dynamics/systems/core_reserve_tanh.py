"""tanh RNNを機能coreと可塑reserveのblockとして扱う。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.systems.tanh_rnn import (
    Matrix,
    TanhRnnReservoir,
    Vector,
)


@dataclass(frozen=True, slots=True)
class CoreReserveTanhRnn:
    """状態先頭をcore、末尾をreserveとして分割した不変wrapper。"""

    system: TanhRnnReservoir
    core_dimension: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.core_dimension, int)
            or isinstance(self.core_dimension, bool)
            or self.core_dimension < 1
            or self.core_dimension >= self.system.state_dimension
        ):
            raise ValueError(
                "core_dimensionは1以上かつ状態次元未満にしてください"
            )

    @property
    def state_dimension(self) -> int:
        return self.system.state_dimension

    @property
    def input_dimension(self) -> int:
        return self.system.input_dimension

    @property
    def reserve_dimension(self) -> int:
        return self.state_dimension - self.core_dimension

    @property
    def core_recurrent_infinity_norm(self) -> float:
        """core-to-core blockの誘導無限大normを返す。"""

        return max(
            math.fsum(abs(value) for value in row[: self.core_dimension])
            for row in self.system.recurrent_weights[: self.core_dimension]
        )

    @property
    def reserve_feedback_infinity_norm(self) -> float:
        """reserve-to-core feedback blockの誘導無限大normを返す。"""

        return max(
            math.fsum(abs(value) for value in row[self.core_dimension :])
            for row in self.system.recurrent_weights[: self.core_dimension]
        )

    def step(self, state: Vector, input_value: Vector) -> Vector:
        return self.system.step(state, input_value)

    def split_state(self, state: Vector) -> tuple[Vector, Vector]:
        """状態をcoreとreserveへ分け、誤った次元を早期に拒否する。"""

        if len(state) != self.state_dimension:
            raise ValueError("状態の次元は状態次元と一致させてください")
        normalized = tuple(float(value) for value in state)
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError("状態はすべて有限である必要があります")
        return (
            normalized[: self.core_dimension],
            normalized[self.core_dimension :],
        )

    def with_reserve_parameters(
        self,
        *,
        core_to_reserve_weights: Matrix,
        reserve_recurrent_weights: Matrix,
        reserve_input_weights: Matrix,
        reserve_bias: Vector,
    ) -> CoreReserveTanhRnn:
        """core更新式を変えず、reserveが受け取るparameterだけを置換する。"""

        self._validate_reserve_parameter_shapes(
            core_to_reserve_weights=core_to_reserve_weights,
            reserve_recurrent_weights=reserve_recurrent_weights,
            reserve_input_weights=reserve_input_weights,
            reserve_bias=reserve_bias,
        )
        reserve_rows = tuple(
            tuple(core_row) + tuple(reserve_row)
            for core_row, reserve_row in zip(
                core_to_reserve_weights,
                reserve_recurrent_weights,
                strict=True,
            )
        )
        adapted_system = TanhRnnReservoir(
            recurrent_weights=(
                self.system.recurrent_weights[: self.core_dimension]
                + reserve_rows
            ),
            input_weights=(
                self.system.input_weights[: self.core_dimension]
                + tuple(tuple(row) for row in reserve_input_weights)
            ),
            bias=(
                self.system.bias[: self.core_dimension]
                + tuple(reserve_bias)
            ),
        )
        return CoreReserveTanhRnn(
            system=adapted_system,
            core_dimension=self.core_dimension,
        )

    def with_core_parameters(
        self,
        *,
        core_recurrent_weights: Matrix,
        reserve_to_core_weights: Matrix,
        core_input_weights: Matrix,
        core_bias: Vector,
    ) -> CoreReserveTanhRnn:
        """比較介入のためcoreが受け取るparameterだけを置換する。"""

        self._validate_core_parameter_shapes(
            core_recurrent_weights=core_recurrent_weights,
            reserve_to_core_weights=reserve_to_core_weights,
            core_input_weights=core_input_weights,
            core_bias=core_bias,
        )
        core_rows = tuple(
            tuple(core_row) + tuple(reserve_row)
            for core_row, reserve_row in zip(
                core_recurrent_weights,
                reserve_to_core_weights,
                strict=True,
            )
        )
        adapted_system = TanhRnnReservoir(
            recurrent_weights=(
                core_rows
                + self.system.recurrent_weights[self.core_dimension :]
            ),
            input_weights=(
                tuple(tuple(row) for row in core_input_weights)
                + self.system.input_weights[self.core_dimension :]
            ),
            bias=(
                tuple(core_bias)
                + self.system.bias[self.core_dimension :]
            ),
        )
        return CoreReserveTanhRnn(
            system=adapted_system,
            core_dimension=self.core_dimension,
        )

    def _validate_reserve_parameter_shapes(
        self,
        *,
        core_to_reserve_weights: Matrix,
        reserve_recurrent_weights: Matrix,
        reserve_input_weights: Matrix,
        reserve_bias: Vector,
    ) -> None:
        if not _has_shape(
            core_to_reserve_weights,
            self.reserve_dimension,
            self.core_dimension,
        ):
            raise ValueError(
                "core_to_reserve_weightsのshapeが分割と一致しません"
            )
        if not _has_shape(
            reserve_recurrent_weights,
            self.reserve_dimension,
            self.reserve_dimension,
        ):
            raise ValueError(
                "reserve_recurrent_weightsのshapeが分割と一致しません"
            )
        if not _has_shape(
            reserve_input_weights,
            self.reserve_dimension,
            self.input_dimension,
        ):
            raise ValueError(
                "reserve_input_weightsのshapeが入力次元と一致しません"
            )
        if len(reserve_bias) != self.reserve_dimension:
            raise ValueError("reserve_biasの次元がreserveと一致しません")

    def _validate_core_parameter_shapes(
        self,
        *,
        core_recurrent_weights: Matrix,
        reserve_to_core_weights: Matrix,
        core_input_weights: Matrix,
        core_bias: Vector,
    ) -> None:
        if not _has_shape(
            core_recurrent_weights,
            self.core_dimension,
            self.core_dimension,
        ):
            raise ValueError(
                "core_recurrent_weightsのshapeが分割と一致しません"
            )
        if not _has_shape(
            reserve_to_core_weights,
            self.core_dimension,
            self.reserve_dimension,
        ):
            raise ValueError(
                "reserve_to_core_weightsのshapeが分割と一致しません"
            )
        if not _has_shape(
            core_input_weights,
            self.core_dimension,
            self.input_dimension,
        ):
            raise ValueError(
                "core_input_weightsのshapeが入力次元と一致しません"
            )
        if len(core_bias) != self.core_dimension:
            raise ValueError("core_biasの次元がcoreと一致しません")


def _has_shape(matrix: Matrix, rows: int, columns: int) -> bool:
    return len(matrix) == rows and all(len(row) == columns for row in matrix)
