"""多次元tanh RNNを入力駆動リザバーとして表現する。"""

import math
from dataclasses import dataclass

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True, slots=True)
class TanhRnnReservoir:
    """`x[t+1] = tanh(W*x[t] + B*u[t] + bias)` を表す。"""

    recurrent_weights: Matrix
    input_weights: Matrix
    bias: Vector

    def __post_init__(self) -> None:
        state_dimension = len(self.recurrent_weights)
        if state_dimension < 1:
            raise ValueError("recurrent_weightsは1行以上必要です")
        if any(
            len(weight_row) != state_dimension
            for weight_row in self.recurrent_weights
        ):
            raise ValueError("recurrent_weightsは正方行列である必要があります")
        if len(self.input_weights) != state_dimension:
            raise ValueError("input_weightsの行数は状態次元と一致させてください")
        if len(self.bias) != state_dimension:
            raise ValueError("biasの要素数は状態次元と一致させてください")

        input_dimension = len(self.input_weights[0])
        if input_dimension < 1:
            raise ValueError("入力次元は1以上必要です")
        if any(
            len(weight_row) != input_dimension
            for weight_row in self.input_weights
        ):
            raise ValueError("input_weightsの列数を一致させてください")
        if any(
            not math.isfinite(value)
            for matrix in (self.recurrent_weights, self.input_weights)
            for row in matrix
            for value in row
        ) or any(not math.isfinite(value) for value in self.bias):
            raise ValueError("重みとbiasはすべて有限である必要があります")

    @property
    def state_dimension(self) -> int:
        return len(self.recurrent_weights)

    @property
    def input_dimension(self) -> int:
        return len(self.input_weights[0])

    def step(self, state: Vector, input_value: Vector) -> Vector:
        """指定入力の下で一時刻だけ状態を更新する。"""

        activations = self._activations(state, input_value)
        return tuple(math.tanh(activation) for activation in activations)

    def jacobian_vector_product(
        self,
        state: Vector,
        input_value: Vector,
        tangent: Vector,
    ) -> Vector:
        """状態Jacobianを接ベクトルへ作用させる。"""

        if len(tangent) != self.state_dimension:
            raise ValueError("接ベクトルの次元は状態次元と一致させてください")
        if any(not math.isfinite(value) for value in tangent):
            raise ValueError("接ベクトルはすべて有限である必要があります")

        next_state = tuple(
            math.tanh(activation)
            for activation in self._activations(state, input_value)
        )
        recurrent_projection = tuple(
            math.fsum(
                weight * tangent_value
                for weight, tangent_value in zip(
                    weight_row,
                    tangent,
                    strict=True,
                )
            )
            for weight_row in self.recurrent_weights
        )
        return tuple(
            (1.0 - state_value * state_value) * projected_value
            for state_value, projected_value in zip(
                next_state,
                recurrent_projection,
                strict=True,
            )
        )

    def _activations(self, state: Vector, input_value: Vector) -> Vector:
        if len(state) != self.state_dimension:
            raise ValueError("状態の次元は状態次元と一致させてください")
        if len(input_value) != self.input_dimension:
            raise ValueError("入力の次元は入力次元と一致させてください")
        if any(
            not math.isfinite(value)
            for vector in (state, input_value)
            for value in vector
        ):
            raise ValueError("状態と入力はすべて有限である必要があります")

        return tuple(
            math.fsum(
                (
                    math.fsum(
                        weight * state_value
                        for weight, state_value in zip(
                            recurrent_row,
                            state,
                            strict=True,
                        )
                    ),
                    math.fsum(
                        weight * input_component
                        for weight, input_component in zip(
                            input_row,
                            input_value,
                            strict=True,
                        )
                    ),
                    bias_value,
                )
            )
            for recurrent_row, input_row, bias_value in zip(
                self.recurrent_weights,
                self.input_weights,
                self.bias,
                strict=True,
            )
        )
