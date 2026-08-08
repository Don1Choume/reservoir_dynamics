"""一つの固定方向を振幅だけ変えて再利用できる重み摂動。"""

from __future__ import annotations

import math
import random

Matrix = tuple[tuple[float, ...], ...]


def sample_entrywise_bounded_perturbation(
    recurrent_weights: Matrix,
    *,
    maximum_absolute_change: float,
    random_seed: int,
) -> Matrix:
    """対角を保ち、各非対角entryを指定した絶対上限内で摂動する。"""

    dimension = _validate_square_matrix(recurrent_weights)
    if (
        isinstance(maximum_absolute_change, bool)
        or not math.isfinite(float(maximum_absolute_change))
        or maximum_absolute_change < 0.0
    ):
        raise ValueError("maximum_absolute_changeは有限の非負値にしてください")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seedは整数にしてください")
    amplitude = float(maximum_absolute_change)
    if amplitude == 0.0:
        return tuple(tuple(row) for row in recurrent_weights)
    random_generator = random.Random(random_seed)
    return tuple(
        tuple(
            recurrent_weights[row][column]
            if row == column
            else recurrent_weights[row][column]
            + amplitude * random_generator.uniform(-1.0, 1.0)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _validate_square_matrix(matrix: Matrix) -> int:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("recurrent_weightsは空でない正方行列にしてください")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("recurrent_weightsは有限値にしてください")
    return len(matrix)
