"""非対角tanh RNNの符号orthantに対するロバストbox証明。"""

from __future__ import annotations

import math
from dataclasses import dataclass

Matrix = tuple[tuple[float, ...], ...]
Signs = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RobustOrthantBoxCertificate:
    """共通内側境界を持つorthant boxの一様外力余裕。"""

    attractor_signs: Signs
    invariant_boundary: float
    raw_uniform_disturbance_margin: float
    maximum_uniform_disturbance: float
    limiting_coordinates: tuple[int, ...]
    transformed_positive_row_sums: tuple[float, ...]
    transformed_negative_row_sums: tuple[float, ...]
    is_certified: bool


def robust_orthant_box_certificate(
    *,
    recurrent_weights: Matrix,
    attractor_signs: Signs,
    search_iterations: int = 240,
) -> RobustOrthantBoxCertificate:
    """`x'=tanh(Wx+eta)` の符号付き共通境界boxを認証する。"""

    dimension = _validate_recurrent_weights(recurrent_weights)
    _validate_attractor_signs(attractor_signs, dimension)
    if (
        not isinstance(search_iterations, int)
        or isinstance(search_iterations, bool)
        or search_iterations < 1
    ):
        raise ValueError("search_iterationsは1以上の整数にしてください")

    transformed_weights = tuple(
        tuple(
            attractor_signs[row_index]
            * weight
            * attractor_signs[column_index]
            for column_index, weight in enumerate(row)
        )
        for row_index, row in enumerate(recurrent_weights)
    )
    positive_row_sums = tuple(
        math.fsum(max(weight, 0.0) for weight in row)
        for row in transformed_weights
    )
    negative_row_sums = tuple(
        math.fsum(min(weight, 0.0) for weight in row)
        for row in transformed_weights
    )

    def row_margins(boundary: float) -> tuple[float, ...]:
        inverse_boundary = math.atanh(boundary)
        return tuple(
            positive_sum * boundary + negative_sum - inverse_boundary
            for positive_sum, negative_sum in zip(
                positive_row_sums,
                negative_row_sums,
                strict=True,
            )
        )

    def objective(boundary: float) -> float:
        return min(row_margins(boundary))

    # 下側包絡は直線の最小値から共通のatanhを引いた形なので、
    # 極値候補は各滑らかな枝の停留点と枝同士の交点に限られる。
    candidates = [1e-12]
    candidates.extend(
        math.sqrt(1.0 - 1.0 / positive_sum)
        for positive_sum in positive_row_sums
        if positive_sum > 1.0
    )
    for first_index, first_positive_sum in enumerate(positive_row_sums):
        for second_index in range(first_index + 1, dimension):
            slope_difference = (
                first_positive_sum - positive_row_sums[second_index]
            )
            if math.isclose(slope_difference, 0.0, abs_tol=1e-15):
                continue
            intersection = (
                negative_row_sums[second_index]
                - negative_row_sums[first_index]
            ) / slope_difference
            if 0.0 < intersection < 1.0:
                candidates.append(intersection)
    # 有限gridも含め、丸め誤差で交点候補を落とした場合に備える。
    candidates.extend(
        index / (search_iterations + 1)
        for index in range(1, search_iterations + 1)
    )
    invariant_boundary = max(candidates, key=objective)
    margins = row_margins(invariant_boundary)
    raw_margin = min(margins)
    limiting_coordinates = tuple(
        index
        for index, margin in enumerate(margins)
        if math.isclose(margin, raw_margin, abs_tol=1e-10)
    )
    is_certified = raw_margin > 0.0
    return RobustOrthantBoxCertificate(
        attractor_signs=attractor_signs,
        invariant_boundary=invariant_boundary,
        raw_uniform_disturbance_margin=raw_margin,
        maximum_uniform_disturbance=max(raw_margin, 0.0),
        limiting_coordinates=limiting_coordinates,
        transformed_positive_row_sums=positive_row_sums,
        transformed_negative_row_sums=negative_row_sums,
        is_certified=is_certified,
    )


def _validate_recurrent_weights(recurrent_weights: Matrix) -> int:
    if not recurrent_weights:
        raise ValueError("recurrent_weightsは空でない正方行列にしてください")
    dimension = len(recurrent_weights)
    if any(len(row) != dimension for row in recurrent_weights):
        raise ValueError("recurrent_weightsは正方行列にしてください")
    if any(
        not math.isfinite(weight)
        for row in recurrent_weights
        for weight in row
    ):
        raise ValueError("recurrent_weightsは有限値だけにしてください")
    return dimension


def _validate_attractor_signs(
    attractor_signs: Signs,
    dimension: int,
) -> None:
    if (
        len(attractor_signs) != dimension
        or any(
            not isinstance(sign, int)
            or isinstance(sign, bool)
            or sign not in (-1, 1)
            for sign in attractor_signs
        )
    ):
        raise ValueError(
            "attractor_signsは状態次元と同じ長さの-1または1にしてください"
        )
