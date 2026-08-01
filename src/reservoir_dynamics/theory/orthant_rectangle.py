"""座標別下側境界を持つorthant rectangleのロバスト正不変性。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from reservoir_dynamics.theory.orthant_box import Matrix, Signs


@dataclass(frozen=True, slots=True)
class OrthantRectangleCertificate:
    """固定された座標別境界に対する一様外乱margin。"""

    attractor_signs: Signs
    lower_boundaries: tuple[float, ...]
    row_margins: tuple[float, ...]
    raw_uniform_disturbance_margin: float
    maximum_uniform_disturbance: float
    limiting_coordinates: tuple[int, ...]
    is_certified: bool


def orthant_rectangle_certificate(
    *,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    lower_boundaries: tuple[float, ...],
) -> OrthantRectangleCertificate:
    """指定rectangleを変えず、その外乱marginを厳密な行下界で評価する。"""

    dimension = _validate_square_matrix(
        recurrent_weights,
        name="recurrent_weights",
    )
    _validate_signs(attractor_signs, dimension)
    if len(lower_boundaries) != dimension or any(
        not math.isfinite(boundary) or not 0.0 < boundary < 1.0
        for boundary in lower_boundaries
    ):
        raise ValueError(
            "lower_boundariesは状態次元と同じ長さの有限な0から1の値にしてください"
        )

    transformed_weights = tuple(
        tuple(
            attractor_signs[row_index]
            * weight
            * attractor_signs[column_index]
            for column_index, weight in enumerate(row)
        )
        for row_index, row in enumerate(recurrent_weights)
    )
    row_margins = tuple(
        math.fsum(
            max(weight, 0.0) * lower_boundaries[column_index]
            + min(weight, 0.0)
            for column_index, weight in enumerate(row)
        )
        - math.atanh(lower_boundaries[row_index])
        for row_index, row in enumerate(transformed_weights)
    )
    raw_margin = min(row_margins)
    limiting_coordinates = tuple(
        index
        for index, margin in enumerate(row_margins)
        if math.isclose(margin, raw_margin, abs_tol=1e-10)
    )
    return OrthantRectangleCertificate(
        attractor_signs=attractor_signs,
        lower_boundaries=lower_boundaries,
        row_margins=row_margins,
        raw_uniform_disturbance_margin=raw_margin,
        maximum_uniform_disturbance=max(raw_margin, 0.0),
        limiting_coordinates=limiting_coordinates,
        is_certified=raw_margin > 0.0,
    )


def matrix_infinity_norm_difference(
    first: Matrix,
    second: Matrix,
) -> float:
    """二行列の差の誘導infinity normを返す。"""

    first_dimension = _validate_square_matrix(first, name="first")
    second_dimension = _validate_square_matrix(second, name="second")
    if first_dimension != second_dimension:
        raise ValueError("firstとsecondの次元を一致させてください")
    return max(
        math.fsum(
            abs(first_value - second_value)
            for first_value, second_value in zip(
                first_row,
                second_row,
                strict=True,
            )
        )
        for first_row, second_row in zip(first, second, strict=True)
    )


def _validate_square_matrix(matrix: Matrix, *, name: str) -> int:
    if not matrix:
        raise ValueError(f"{name}は空にできません")
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError(f"{name}は正方行列にしてください")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name}は有限値にしてください")
    return dimension


def _validate_signs(attractor_signs: Signs, dimension: int) -> None:
    if len(attractor_signs) != dimension or any(
        not isinstance(sign, int)
        or isinstance(sign, bool)
        or sign not in (-1, 1)
        for sign in attractor_signs
    ):
        raise ValueError(
            "attractor_signsは状態次元と同じ長さの-1または1にしてください"
        )
