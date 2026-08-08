"""多成分の方向別負荷と直積を作らないcertificate集約。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.metrics.module_partition import (
    Partition,
    normalize_partition,
)
from reservoir_dynamics.theory.orthant_box import Matrix


@dataclass(frozen=True, slots=True)
class MultiComponentMarginSummary:
    """局所margin分布から因子化した方向別・global保証。"""

    total_inbound_loads: tuple[float, ...]
    global_load: float
    directional_certified_fraction: float
    global_certified_fraction: float
    mean_directional_slack: float
    minimum_directional_slack: float


@dataclass(frozen=True, slots=True)
class CertificateEnumerationComplexity:
    """局所列挙とmonolithic列挙の状態数比較。"""

    module_sizes: tuple[int, ...]
    local_orthant_count: int
    monolithic_orthant_count: int
    local_to_monolithic_ratio: float


def component_inbound_load_matrix(
    recurrent_weights: Matrix,
    partition: Partition,
) -> Matrix:
    """source module別の最大target行絶対値和を保持する。"""

    dimension = _validate_square_matrix(recurrent_weights)
    normalized = normalize_partition(partition, dimension=dimension)
    return tuple(
        tuple(
            0.0
            if target_index == source_index
            else max(
                math.fsum(
                    abs(recurrent_weights[row][column])
                    for column in source_component
                )
                for row in target_component
            )
            for source_index, source_component in enumerate(normalized)
        )
        for target_index, target_component in enumerate(normalized)
    )


def summarize_multicomponent_margins(
    *,
    component_margins: tuple[tuple[float, ...], ...],
    disturbance_bound: float,
    inbound_load_matrix: Matrix,
    tolerance: float = 1e-12,
) -> MultiComponentMarginSummary:
    """独立な局所orthant分布をsurvival積で厳密集約する。"""

    normalized_disturbance = _validate_nonnegative_finite(
        disturbance_bound,
        "disturbance_bound",
    )
    if not component_margins or any(not values for values in component_margins):
        raise ValueError("component_marginsはmoduleごとに1件以上必要です")
    normalized_margins = tuple(
        tuple(float(value) for value in values) for values in component_margins
    )
    if any(
        not math.isfinite(value)
        for values in normalized_margins
        for value in values
    ):
        raise ValueError("component marginは有限値にしてください")
    module_count = len(normalized_margins)
    loads = _validate_load_matrix(inbound_load_matrix, module_count)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")
    total_loads = tuple(math.fsum(row) for row in loads)
    global_load = max(total_loads)
    directional_slacks = tuple(
        tuple(
            margin - normalized_disturbance - total_load
            for margin in margins
        )
        for margins, total_load in zip(
            normalized_margins,
            total_loads,
            strict=True,
        )
    )
    global_slacks = tuple(
        tuple(
            margin - normalized_disturbance - global_load
            for margin in margins
        )
        for margins in normalized_margins
    )
    return MultiComponentMarginSummary(
        total_inbound_loads=total_loads,
        global_load=global_load,
        directional_certified_fraction=_factorized_nonnegative_fraction(
            directional_slacks,
            tolerance=tolerance,
        ),
        global_certified_fraction=_factorized_nonnegative_fraction(
            global_slacks,
            tolerance=tolerance,
        ),
        mean_directional_slack=_mean_minimum_without_product(
            directional_slacks
        ),
        minimum_directional_slack=min(
            value for values in directional_slacks for value in values
        ),
    )


def certificate_enumeration_complexity(
    module_sizes: tuple[int, ...],
) -> CertificateEnumerationComplexity:
    """局所profile再利用によるorthant列挙数の削減を返す。"""

    if len(module_sizes) < 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in module_sizes
    ):
        raise ValueError("module_sizesは正整数二つ以上にしてください")
    local_count = sum(2**size for size in module_sizes)
    monolithic_count = 2 ** sum(module_sizes)
    return CertificateEnumerationComplexity(
        module_sizes=module_sizes,
        local_orthant_count=local_count,
        monolithic_orthant_count=monolithic_count,
        local_to_monolithic_ratio=local_count / monolithic_count,
    )


def _factorized_nonnegative_fraction(
    slack_distributions: tuple[tuple[float, ...], ...],
    *,
    tolerance: float,
) -> float:
    return math.prod(
        sum(value >= -tolerance for value in values) / len(values)
        for values in slack_distributions
    )


def _mean_minimum_without_product(
    slack_distributions: tuple[tuple[float, ...], ...],
) -> float:
    support = tuple(
        sorted({value for values in slack_distributions for value in values})
    )
    return math.fsum(
        value
        * (
            _minimum_survival_probability(slack_distributions, value, strict=False)
            - _minimum_survival_probability(slack_distributions, value, strict=True)
        )
        for value in support
    )


def _minimum_survival_probability(
    distributions: tuple[tuple[float, ...], ...],
    threshold: float,
    *,
    strict: bool,
) -> float:
    comparator = (
        (lambda value: value > threshold)
        if strict
        else (lambda value: value >= threshold)
    )
    return math.prod(
        sum(comparator(value) for value in values) / len(values)
        for values in distributions
    )


def _validate_load_matrix(matrix: Matrix, module_count: int) -> Matrix:
    if len(matrix) != module_count or any(
        len(row) != module_count for row in matrix
    ):
        raise ValueError("inbound load matrixの次元をmodule数と一致させてください")
    normalized = tuple(tuple(float(value) for value in row) for row in matrix)
    if any(
        not math.isfinite(value) or value < 0.0
        for row in normalized
        for value in row
    ) or any(abs(normalized[index][index]) > 1e-15 for index in range(module_count)):
        raise ValueError("inbound loadは有限非負で対角を0にしてください")
    return normalized


def _validate_square_matrix(matrix: Matrix) -> int:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("recurrent_weightsは空でない正方行列にしてください")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("recurrent_weightsは有限値にしてください")
    return len(matrix)


def _validate_nonnegative_finite(value: float, name: str) -> float:
    normalized = float(value)
    if isinstance(value, bool) or not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name}は有限の非負値にしてください")
    return normalized
