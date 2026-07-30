"""RNN構造と局所安定性の比較baseline。"""

from __future__ import annotations

import math

from reservoir_dynamics.theory.orthant_box import Matrix, Signs


def off_diagonal_infinity_norm(recurrent_weights: Matrix) -> float:
    """対角自己結合を除いた最大絶対行和を返す。"""

    _validate_square_matrix(recurrent_weights)
    return max(
        math.fsum(
            abs(weight)
            for column_index, weight in enumerate(row)
            if column_index != row_index
        )
        for row_index, row in enumerate(recurrent_weights)
    )


def matrix_nonnormality_commutator_norm(
    recurrent_weights: Matrix,
) -> float:
    """W^T WとW W^Tの交換子のFrobenius normを返す。"""

    _validate_square_matrix(recurrent_weights)
    dimension = len(recurrent_weights)
    squared_difference = math.fsum(
        (
            math.fsum(
                recurrent_weights[index][row]
                * recurrent_weights[index][column]
                for index in range(dimension)
            )
            - math.fsum(
                recurrent_weights[row][index]
                * recurrent_weights[column][index]
                for index in range(dimension)
            )
        )
        ** 2
        for row in range(dimension)
        for column in range(dimension)
    )
    return math.sqrt(squared_difference)


def local_jacobian_infinity_norm(
    *,
    recurrent_weights: Matrix,
    state: tuple[float, ...],
) -> float:
    """tanh RNN Jacobianの最大絶対行和を固定点局所baselineとする。"""

    _validate_square_matrix(recurrent_weights)
    if len(state) != len(recurrent_weights):
        raise ValueError("stateとrecurrent_weightsの次元が一致しません")
    if any(not math.isfinite(value) for value in state):
        raise ValueError("stateは有限値にしてください")
    return max(
        (1.0 - state_value**2)
        * math.fsum(abs(weight) for weight in row)
        for state_value, row in zip(
            state,
            recurrent_weights,
            strict=True,
        )
    )


def signed_minimum_coordinate(
    *,
    state: tuple[float, ...],
    attractor_signs: Signs,
) -> float:
    """対象orthant境界から最も近い符号付き座標を返す。"""

    if len(state) != len(attractor_signs):
        raise ValueError("stateとattractor_signsの次元が一致しません")
    if not state:
        raise ValueError("stateは空にできません")
    if any(not math.isfinite(value) for value in state):
        raise ValueError("stateは有限値にしてください")
    if any(sign not in (-1, 1) for sign in attractor_signs):
        raise ValueError("attractor_signsは-1または1にしてください")
    return min(
        sign * value
        for sign, value in zip(attractor_signs, state, strict=True)
    )


def _validate_square_matrix(recurrent_weights: Matrix) -> None:
    if not recurrent_weights:
        raise ValueError("recurrent_weightsは空にできません")
    dimension = len(recurrent_weights)
    if any(len(row) != dimension for row in recurrent_weights):
        raise ValueError("recurrent_weightsは正方行列にしてください")
    if any(
        not math.isfinite(weight)
        for row in recurrent_weights
        for weight in row
    ):
        raise ValueError("recurrent_weightsは有限値にしてください")
