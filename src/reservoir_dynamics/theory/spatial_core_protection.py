"""空間gateによるcore保護と介入energy比較の純粋関数。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.systems.tanh_rnn import Matrix, Vector
from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
)


@dataclass(frozen=True, slots=True)
class BistableCoordinateProtection:
    """座標別双安定marginに対するforcing bound。"""

    recurrent_gains: Vector
    critical_forcings: Vector
    applied_forcing_bounds: Vector
    margins: Vector
    certified_coordinates: tuple[bool, ...]

    @property
    def all_certified(self) -> bool:
        return all(self.certified_coordinates)

    @property
    def minimum_margin(self) -> float:
        return min(self.margins)


def row_gated_matrix(matrix: Matrix, gates: Vector) -> Matrix:
    """各rowへ一つの局所gateを掛けた新しい行列を返す。"""

    normalized = _normalize_rectangular_matrix(matrix, "matrix")
    normalized_gates = _normalize_unit_interval_vector(gates, "gates")
    if len(normalized) != len(normalized_gates):
        raise ValueError("gatesの次元はmatrixのrow数と一致させてください")
    return tuple(
        tuple(gate * value for value in row)
        for row, gate in zip(normalized, normalized_gates, strict=True)
    )


def matrix_frobenius_distance_squared(
    first: Matrix,
    second: Matrix,
) -> float:
    """同shape行列間のFrobenius距離二乗を返す。"""

    normalized_first = _normalize_rectangular_matrix(first, "first")
    normalized_second = _normalize_rectangular_matrix(second, "second")
    if (
        len(normalized_first) != len(normalized_second)
        or len(normalized_first[0]) != len(normalized_second[0])
    ):
        raise ValueError("二つの行列shapeを一致させてください")
    return math.fsum(
        (first_value - second_value) ** 2
        for first_row, second_row in zip(
            normalized_first,
            normalized_second,
            strict=True,
        )
        for first_value, second_value in zip(
            first_row,
            second_row,
            strict=True,
        )
    )


def energy_matched_global_weights(
    matrix: Matrix,
    *,
    target_energy: float,
) -> Matrix:
    """指定Frobenius摂動energyと一致する一様縮小を返す。"""

    normalized = _normalize_rectangular_matrix(matrix, "matrix")
    if not math.isfinite(target_energy) or target_energy < 0.0:
        raise ValueError("target_energyは有限の非負値にしてください")
    matrix_energy = math.fsum(
        value * value for row in normalized for value in row
    )
    if matrix_energy == 0.0:
        if target_energy > 1e-15:
            raise ValueError("zero matrixでは正のtarget_energyに一致できません")
        return normalized
    if target_energy > matrix_energy + 1e-12:
        raise ValueError(
            "target_energyはmatrix全体をzeroにするenergy以下にしてください"
        )
    attenuation = min(1.0, math.sqrt(target_energy / matrix_energy))
    scale = 1.0 - attenuation
    return tuple(tuple(scale * value for value in row) for row in normalized)


def bistable_coordinate_protection(
    *,
    recurrent_gains: Vector,
    feedback_loads: Vector,
    disturbance_bounds: Vector,
) -> BistableCoordinateProtection:
    """`abs(feedback)+noise_bound`を座標別臨界外力と比較する。"""

    normalized_gains = _normalize_finite_vector(
        recurrent_gains,
        "recurrent_gains",
    )
    normalized_loads = _normalize_finite_vector(
        feedback_loads,
        "feedback_loads",
    )
    normalized_disturbances = _normalize_non_negative_vector(
        disturbance_bounds,
        "disturbance_bounds",
    )
    if not (
        len(normalized_gains)
        == len(normalized_loads)
        == len(normalized_disturbances)
    ):
        raise ValueError("三つの入力次元を一致させてください")
    critical_forcings = tuple(
        bistable_tanh_certificate(gain).critical_forcing
        for gain in normalized_gains
    )
    applied_bounds = tuple(
        abs(load) + disturbance
        for load, disturbance in zip(
            normalized_loads,
            normalized_disturbances,
            strict=True,
        )
    )
    margins = tuple(
        critical - applied
        for critical, applied in zip(
            critical_forcings,
            applied_bounds,
            strict=True,
        )
    )
    return BistableCoordinateProtection(
        recurrent_gains=normalized_gains,
        critical_forcings=critical_forcings,
        applied_forcing_bounds=applied_bounds,
        margins=margins,
        certified_coordinates=tuple(margin >= 0.0 for margin in margins),
    )


def time_varying_core_deviation_bound(
    *,
    core_lipschitz: float,
    forcing_loads: Vector,
    initial_core_distance: float = 0.0,
) -> Vector:
    """`D[t+1] <= Lc D[t] + load[t]`の時変上界を返す。"""

    if (
        not math.isfinite(core_lipschitz)
        or core_lipschitz < 0.0
        or core_lipschitz >= 1.0
    ):
        raise ValueError("core_lipschitzは0以上1未満にしてください")
    normalized_loads = _normalize_non_negative_vector(
        forcing_loads,
        "forcing_loads",
    )
    if not math.isfinite(initial_core_distance) or initial_core_distance < 0.0:
        raise ValueError(
            "initial_core_distanceは有限の非負値にしてください"
        )
    bounds = [float(initial_core_distance)]
    for load in normalized_loads:
        bounds.append(core_lipschitz * bounds[-1] + load)
    return tuple(bounds)


def _normalize_rectangular_matrix(matrix: Matrix, name: str) -> Matrix:
    if not matrix or not matrix[0]:
        raise ValueError(f"{name}は空にできません")
    column_count = len(matrix[0])
    normalized = tuple(tuple(float(value) for value in row) for row in matrix)
    if any(len(row) != column_count for row in normalized):
        raise ValueError(f"{name}は長方形行列にしてください")
    if any(not math.isfinite(value) for row in normalized for value in row):
        raise ValueError(f"{name}は有限値だけにしてください")
    return normalized


def _normalize_unit_interval_vector(values: Vector, name: str) -> Vector:
    normalized = _normalize_finite_vector(values, name)
    if any(value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError(f"{name}は[0, 1]内にしてください")
    return normalized


def _normalize_non_negative_vector(values: Vector, name: str) -> Vector:
    normalized = _normalize_finite_vector(values, name)
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"{name}は非負値にしてください")
    return normalized


def _normalize_finite_vector(values: Vector, name: str) -> Vector:
    if not values:
        raise ValueError(f"{name}は空にできません")
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name}は有限値にしてください")
    return normalized
