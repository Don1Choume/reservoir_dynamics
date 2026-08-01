"""符号座標変換で同値なRNN構造の監査。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from reservoir_dynamics.theory.orthant_box import Matrix


@dataclass(frozen=True)
class SignedCoordinateConjugacyAudit:
    """符号対角共役で割ったnetwork集合の要約。"""

    raw_network_count: int
    effective_class_count: int
    representative_indices: tuple[int, ...]
    class_index_by_network: tuple[int, ...]
    class_sizes: tuple[int, ...]

    @property
    def effective_fraction(self) -> float:
        """投入network数に対する異なる共役class数の割合を返す。"""

        return self.effective_class_count / self.raw_network_count


def signed_coordinate_conjugacy_witness(
    first: Matrix,
    second: Matrix,
    *,
    tolerance: float = 1e-12,
) -> tuple[int, ...] | None:
    """``second = D first D``を満たす符号対角行列Dの対角を返す。

    tanhの奇関数性により、この関係は座標符号を変えた力学系の共役を
    与える。ただしtaskまで同値とみなすには、入力、外乱、readout、評価集合も
    同じ座標変換で閉じていることを別途確認する必要がある。
    """

    _validate_tolerance(tolerance)
    first_dimension = _validate_square_matrix(first, name="first")
    second_dimension = _validate_square_matrix(second, name="second")
    if first_dimension != second_dimension:
        raise ValueError("firstとsecondは同じ次元にしてください")

    constraints: list[list[tuple[int, int]]] = [
        [] for _ in range(first_dimension)
    ]
    for row_index in range(first_dimension):
        for column_index in range(first_dimension):
            relation = _entry_sign_relation(
                first[row_index][column_index],
                second[row_index][column_index],
                tolerance=tolerance,
            )
            if relation is None:
                return None
            if relation == 0:
                continue
            if row_index == column_index:
                if relation != 1:
                    return None
                continue
            constraints[row_index].append((column_index, relation))
            constraints[column_index].append((row_index, relation))

    witness = _solve_sign_constraints(constraints)
    if witness is None:
        return None
    if not _witness_reconstructs_second(
        first,
        second,
        witness,
        tolerance=tolerance,
    ):
        return None
    return witness


def audit_signed_coordinate_conjugacy(
    recurrent_weights: tuple[Matrix, ...],
    *,
    tolerance: float = 1e-12,
) -> SignedCoordinateConjugacyAudit:
    """投入networkを符号対角共役classへ決定論的に分割する。"""

    _validate_tolerance(tolerance)
    if not recurrent_weights:
        raise ValueError("recurrent_weightsは空にできません")

    expected_dimension = _validate_square_matrix(
        recurrent_weights[0],
        name="recurrent_weights[0]",
    )
    for network_index, matrix in enumerate(recurrent_weights[1:], start=1):
        actual_dimension = _validate_square_matrix(
            matrix,
            name=f"recurrent_weights[{network_index}]",
        )
        if actual_dimension != expected_dimension:
            raise ValueError("recurrent_weightsはすべて同じ次元にしてください")

    representative_indices: list[int] = []
    class_index_by_network: list[int] = []
    class_sizes: list[int] = []
    for network_index, matrix in enumerate(recurrent_weights):
        matching_class_index = _matching_representative_class(
            matrix=matrix,
            recurrent_weights=recurrent_weights,
            representative_indices=representative_indices,
            tolerance=tolerance,
        )
        if matching_class_index is None:
            matching_class_index = len(representative_indices)
            representative_indices.append(network_index)
            class_sizes.append(0)
        class_index_by_network.append(matching_class_index)
        class_sizes[matching_class_index] += 1

    return SignedCoordinateConjugacyAudit(
        raw_network_count=len(recurrent_weights),
        effective_class_count=len(representative_indices),
        representative_indices=tuple(representative_indices),
        class_index_by_network=tuple(class_index_by_network),
        class_sizes=tuple(class_sizes),
    )


def weakly_connected_components(
    recurrent_weights: Matrix,
    *,
    tolerance: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    """非零の有向結合を無向化したconnected componentを返す。"""

    _validate_tolerance(tolerance)
    dimension = _validate_square_matrix(
        recurrent_weights,
        name="recurrent_weights",
    )
    neighbors = tuple(
        tuple(
            column_index
            for column_index in range(dimension)
            if column_index != row_index
            and (
                abs(recurrent_weights[row_index][column_index]) > tolerance
                or abs(recurrent_weights[column_index][row_index]) > tolerance
            )
        )
        for row_index in range(dimension)
    )
    observed: set[int] = set()
    components: list[tuple[int, ...]] = []
    for root_index in range(dimension):
        if root_index in observed:
            continue
        pending_indices = [root_index]
        component: set[int] = set()
        while pending_indices:
            current_index = pending_indices.pop()
            if current_index in component:
                continue
            component.add(current_index)
            pending_indices.extend(neighbors[current_index])
        observed.update(component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _entry_sign_relation(
    first_value: float,
    second_value: float,
    *,
    tolerance: float,
) -> int | None:
    first_is_zero = abs(first_value) <= tolerance
    second_is_zero = abs(second_value) <= tolerance
    if first_is_zero and second_is_zero:
        return 0
    if first_is_zero or second_is_zero:
        return None
    if not math.isclose(
        abs(first_value),
        abs(second_value),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        return None
    return 1 if first_value * second_value > 0.0 else -1


def _solve_sign_constraints(
    constraints: list[list[tuple[int, int]]],
) -> tuple[int, ...] | None:
    signs = [0] * len(constraints)
    for root_index in range(len(constraints)):
        if signs[root_index] != 0:
            continue
        # 各非連結成分のglobal signは任意なので、最小indexを+1に固定する。
        signs[root_index] = 1
        pending_indices = [root_index]
        while pending_indices:
            current_index = pending_indices.pop()
            for neighbor_index, relation in constraints[current_index]:
                expected_sign = signs[current_index] * relation
                if signs[neighbor_index] == 0:
                    signs[neighbor_index] = expected_sign
                    pending_indices.append(neighbor_index)
                    continue
                if signs[neighbor_index] != expected_sign:
                    return None
    return tuple(signs)


def _witness_reconstructs_second(
    first: Matrix,
    second: Matrix,
    witness: tuple[int, ...],
    *,
    tolerance: float,
) -> bool:
    return all(
        math.isclose(
            second[row_index][column_index],
            witness[row_index]
            * first[row_index][column_index]
            * witness[column_index],
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
        for row_index in range(len(first))
        for column_index in range(len(first))
    )


def _matching_representative_class(
    *,
    matrix: Matrix,
    recurrent_weights: tuple[Matrix, ...],
    representative_indices: list[int],
    tolerance: float,
) -> int | None:
    for class_index, representative_index in enumerate(
        representative_indices
    ):
        if signed_coordinate_conjugacy_witness(
            recurrent_weights[representative_index],
            matrix,
            tolerance=tolerance,
        ) is not None:
            return class_index
    return None


def _validate_tolerance(tolerance: float) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")


def _validate_square_matrix(matrix: Matrix, *, name: str) -> int:
    if not matrix:
        raise ValueError(f"{name}は空にできません")
    dimension = len(matrix)
    if any(len(row) != dimension for row in matrix):
        raise ValueError(f"{name}は正方行列にしてください")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name}は有限値にしてください")
    return dimension
