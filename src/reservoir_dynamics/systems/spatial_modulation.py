"""局所sourceを拡散させる有界な離散空間変調場。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.systems.tanh_rnn import Matrix, Vector


@dataclass(frozen=True, slots=True)
class DiffusiveModulationField:
    """凸結合更新により`[0, 1]^n`を保つ抑制場。"""

    diffusion_kernel: Matrix
    diffusion_rate: float
    source_rate: float
    minimum_gate: float = 0.0

    def __post_init__(self) -> None:
        normalized_kernel = _normalize_diffusion_kernel(
            self.diffusion_kernel,
        )
        _validate_unit_interval(self.diffusion_rate, "diffusion_rate")
        _validate_unit_interval(self.source_rate, "source_rate")
        _validate_unit_interval(self.minimum_gate, "minimum_gate")
        if self.diffusion_rate + self.source_rate > 1.0 + 1e-12:
            raise ValueError(
                "diffusion_rateとsource_rateの合計は1以下にしてください"
            )
        object.__setattr__(self, "diffusion_kernel", normalized_kernel)

    @property
    def dimension(self) -> int:
        return len(self.diffusion_kernel)

    def step(self, *, state: Vector, source: Vector) -> Vector:
        """場を一step進め、数値誤差を含めhypercube外を拒否する。"""

        normalized_state = _normalize_hypercube_vector(
            state,
            expected_dimension=self.dimension,
            value_name="state",
        )
        normalized_source = _normalize_hypercube_vector(
            source,
            expected_dimension=self.dimension,
            value_name="source",
        )
        carry_rate = 1.0 - self.diffusion_rate - self.source_rate
        diffused = tuple(
            math.fsum(
                weight * value
                for weight, value in zip(
                    row,
                    normalized_state,
                    strict=True,
                )
            )
            for row in self.diffusion_kernel
        )
        return tuple(
            min(
                1.0,
                max(
                    0.0,
                    carry_rate * current
                    + self.diffusion_rate * spatial_average
                    + self.source_rate * local_source,
                ),
            )
            for current, spatial_average, local_source in zip(
                normalized_state,
                diffused,
                normalized_source,
                strict=True,
            )
        )

    def gates(self, state: Vector) -> Vector:
        """抑制場を`[minimum_gate, 1]`の乗法gateへ写す。"""

        normalized_state = _normalize_hypercube_vector(
            state,
            expected_dimension=self.dimension,
            value_name="state",
        )
        gate_range = 1.0 - self.minimum_gate
        return tuple(1.0 - gate_range * value for value in normalized_state)


def chain_diffusion_kernel(node_count: int) -> Matrix:
    """端点で質量を失わない最近傍chain kernelを返す。"""

    if (
        not isinstance(node_count, int)
        or isinstance(node_count, bool)
        or node_count < 1
    ):
        raise ValueError("node_countは1以上の整数にしてください")
    if node_count == 1:
        return ((1.0,),)
    rows: list[Vector] = []
    for node_index in range(node_count):
        row = [0.0] * node_count
        row[node_index] = 0.5
        if node_index == 0:
            row[1] = 0.5
        elif node_index == node_count - 1:
            row[node_count - 2] = 0.5
        else:
            row[node_index - 1] = 0.25
            row[node_index + 1] = 0.25
        rows.append(tuple(row))
    return tuple(rows)


def _normalize_diffusion_kernel(kernel: Matrix) -> Matrix:
    if not kernel:
        raise ValueError("diffusion_kernelは空にできません")
    dimension = len(kernel)
    normalized_rows: list[Vector] = []
    for row in kernel:
        if len(row) != dimension:
            raise ValueError("diffusion_kernelは正方行列にしてください")
        normalized = tuple(float(value) for value in row)
        if any(not math.isfinite(value) or value < 0.0 for value in normalized):
            raise ValueError(
                "diffusion_kernelは有限の非負値にしてください"
            )
        if not math.isclose(
            math.fsum(normalized),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "diffusion_kernelはrow-stochasticにしてください"
            )
        normalized_rows.append(normalized)
    return tuple(normalized_rows)


def _normalize_hypercube_vector(
    values: Vector,
    *,
    expected_dimension: int,
    value_name: str,
) -> Vector:
    if len(values) != expected_dimension:
        raise ValueError(
            f"{value_name}の次元はfield dimensionと一致させてください"
        )
    normalized = tuple(float(value) for value in values)
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in normalized
    ):
        raise ValueError(f"{value_name}は[0, 1]内にしてください")
    return normalized


def _validate_unit_interval(value: float, value_name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{value_name}は[0, 1]内の有限値にしてください")
