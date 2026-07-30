"""大域的収縮条件を解析できる最小の入力駆動リザバー。"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScalarTanhReservoir:
    """`x[t+1] = tanh(a*x[t] + b*u[t] + c)` を表す。"""

    recurrent_gain: float
    input_gain: float
    bias: float = 0.0
    state_dimension: int = 1
    input_dimension: int = 1

    def __post_init__(self) -> None:
        parameter_values = (
            self.recurrent_gain,
            self.input_gain,
            self.bias,
        )
        if any(not math.isfinite(value) for value in parameter_values):
            raise ValueError("すべてのパラメータは有限である必要があります")
        if self.state_dimension != 1 or self.input_dimension != 1:
            raise ValueError("ScalarTanhReservoirの状態と入力は1次元です")

    @property
    def global_state_lipschitz_bound(self) -> float:
        """状態写像の大域的Lipschitz上界を返す。"""

        return abs(self.recurrent_gain)

    @property
    def is_globally_contractive(self) -> bool:
        """十分条件 `|recurrent_gain| < 1` を満たすか返す。"""

        return self.global_state_lipschitz_bound < 1.0

    def step(
        self,
        state: tuple[float, ...],
        input_value: tuple[float, ...],
    ) -> tuple[float, ...]:
        """指定入力の下で一時刻だけ更新する。"""

        activation = self._activation(state, input_value)
        return (math.tanh(activation),)

    def state_jacobian_magnitude(
        self,
        state: tuple[float, ...],
        input_value: tuple[float, ...],
    ) -> float:
        """現在状態から次状態への局所微分絶対値を返す。"""

        activation = self._activation(state, input_value)
        next_state = math.tanh(activation)
        return abs(self.recurrent_gain) * (1.0 - next_state * next_state)

    def _activation(
        self,
        state: tuple[float, ...],
        input_value: tuple[float, ...],
    ) -> float:
        if len(state) != 1 or len(input_value) != 1:
            raise ValueError("状態と入力は1次元である必要があります")
        if not math.isfinite(state[0]) or not math.isfinite(input_value[0]):
            raise ValueError("状態と入力は有限である必要があります")

        return (
            self.recurrent_gain * state[0]
            + self.input_gain * input_value[0]
            + self.bias
        )
