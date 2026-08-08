"""重みaffinityの明示的gapからmodule分割を推定する。"""

from __future__ import annotations

import math
from dataclasses import dataclass

Matrix = tuple[tuple[float, ...], ...]
Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class AffinityGapPartition:
    """最大affinity gapと、そのthresholdで得た連結成分。"""

    components: Partition
    threshold: float
    selected_gap: float
    relative_gap: float
    gap_is_unique: bool


@dataclass(frozen=True, slots=True)
class PartitionSeparation:
    """指定partitionのinter/intra affinity分離証明。"""

    minimum_within_affinity: float
    maximum_between_affinity: float
    gap: float
    separated: bool


@dataclass(frozen=True, slots=True)
class AffinityGapRobustness:
    """最大gap partitionが不変なentrywise摂動の十分半径。"""

    partition: AffinityGapPartition
    selected_gap: float
    runner_up_gap: float
    gap_dominance: float
    edge_classification_radius: float
    gap_selection_radius: float
    certified_entrywise_radius: float
    relative_certified_radius: float
    guaranteed: bool


def infer_affinity_gap_partition(
    recurrent_weights: Matrix,
    *,
    tolerance: float = 1e-12,
) -> AffinityGapPartition:
    """pair affinityの一意な最大gapをthresholdとして連結成分を返す。"""

    dimension = _validate_square_matrix(recurrent_weights)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")
    pair_affinities = _pair_affinities(recurrent_weights, dimension)
    unique_values = tuple(sorted(set(pair_affinities)))
    if len(unique_values) < 2:
        raise ValueError("partition推定に使えるaffinity gapがありません")
    gaps = tuple(
        (upper - lower, lower, upper)
        for lower, upper in zip(unique_values, unique_values[1:])
    )
    maximum_gap = max(value[0] for value in gaps)
    if maximum_gap <= tolerance:
        raise ValueError("partition推定に使える正のaffinity gapがありません")
    maximum_candidates = tuple(
        value for value in gaps if abs(value[0] - maximum_gap) <= tolerance
    )
    selected_gap, lower, upper = maximum_candidates[0]
    threshold = (lower + upper) / 2.0
    components = _threshold_components(
        recurrent_weights,
        threshold=threshold,
    )
    if len(components) < 2:
        raise ValueError("affinity gapから複数moduleを推定できません")
    return AffinityGapPartition(
        components=components,
        threshold=threshold,
        selected_gap=selected_gap,
        relative_gap=selected_gap / upper if upper > 0.0 else 0.0,
        gap_is_unique=len(maximum_candidates) == 1,
    )


def certify_affinity_gap_partition(
    recurrent_weights: Matrix,
    *,
    tolerance: float = 1e-12,
) -> AffinityGapRobustness:
    """最大gapの選択とedge分類を同時に保つ厳密な十分半径を返す。"""

    dimension = _validate_square_matrix(recurrent_weights)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")
    partition = infer_affinity_gap_partition(
        recurrent_weights,
        tolerance=tolerance,
    )
    sorted_affinities = tuple(
        sorted(_pair_affinities(recurrent_weights, dimension))
    )
    gaps = tuple(
        upper - lower
        for lower, upper in zip(sorted_affinities, sorted_affinities[1:])
    )
    selected_gap = max(gaps)
    selected_indices = tuple(
        index
        for index, gap in enumerate(gaps)
        if abs(gap - selected_gap) <= tolerance
    )
    if len(selected_indices) == 1:
        selected_index = selected_indices[0]
        runner_up_gap = max(
            (gap for index, gap in enumerate(gaps) if index != selected_index),
            default=0.0,
        )
    else:
        runner_up_gap = selected_gap
    gap_dominance = max(0.0, selected_gap - runner_up_gap)
    edge_radius = selected_gap / 2.0
    selection_radius = gap_dominance / 4.0
    certified_radius = min(edge_radius, selection_radius)
    largest_affinity = max(sorted_affinities)
    unique_selection = (
        len(selected_indices) == 1 and partition.gap_is_unique
    )
    guaranteed = unique_selection and certified_radius > tolerance
    if not guaranteed:
        certified_radius = 0.0
    return AffinityGapRobustness(
        partition=partition,
        selected_gap=selected_gap,
        runner_up_gap=runner_up_gap,
        gap_dominance=gap_dominance if unique_selection else 0.0,
        edge_classification_radius=edge_radius,
        gap_selection_radius=selection_radius if unique_selection else 0.0,
        certified_entrywise_radius=certified_radius,
        relative_certified_radius=(
            certified_radius / largest_affinity
            if largest_affinity > 0.0
            else 0.0
        ),
        guaranteed=guaranteed,
    )


def partition_separation(
    recurrent_weights: Matrix,
    partition: Partition,
    *,
    tolerance: float = 1e-12,
) -> PartitionSeparation:
    """module内最小affinityとmodule間最大affinityのgapを計算する。"""

    dimension = _validate_square_matrix(recurrent_weights)
    normalized = normalize_partition(partition, dimension=dimension)
    if any(len(component) < 2 for component in normalized):
        raise ValueError("partitionの各componentは2 node以上にしてください")
    module_by_node = {
        node: module_index
        for module_index, component in enumerate(normalized)
        for node in component
    }
    within = tuple(
        _pair_affinity(recurrent_weights, first, second)
        for first in range(dimension)
        for second in range(first + 1, dimension)
        if module_by_node[first] == module_by_node[second]
    )
    between = tuple(
        _pair_affinity(recurrent_weights, first, second)
        for first in range(dimension)
        for second in range(first + 1, dimension)
        if module_by_node[first] != module_by_node[second]
    )
    if not within or not between:
        raise ValueError("partitionにはmodule内・module間pairが必要です")
    minimum_within = min(within)
    maximum_between = max(between)
    gap = minimum_within - maximum_between
    return PartitionSeparation(
        minimum_within_affinity=minimum_within,
        maximum_between_affinity=maximum_between,
        gap=gap,
        separated=gap > tolerance,
    )


def partitions_equivalent(first: Partition, second: Partition) -> bool:
    """module labelとcomponent内順序を除いて二partitionを比較する。"""

    return frozenset(frozenset(component) for component in first) == frozenset(
        frozenset(component) for component in second
    )


def partition_pair_disagreement(first: Partition, second: Partition) -> float:
    """node pairの共所属判定が異なる割合をmodule label非依存で返す。"""

    first_nodes = _validate_partition_node_set(first)
    second_nodes = _validate_partition_node_set(second)
    if first_nodes != second_nodes:
        raise ValueError("二partitionのnode集合を一致させてください")
    ordered_nodes = tuple(sorted(first_nodes))
    if len(ordered_nodes) < 2:
        raise ValueError("partitionは2個以上のnodeを含めてください")
    first_membership = _component_membership(first)
    second_membership = _component_membership(second)
    disagreements = sum(
        (first_membership[first_node] == first_membership[second_node])
        != (second_membership[first_node] == second_membership[second_node])
        for first_index, first_node in enumerate(ordered_nodes)
        for second_node in ordered_nodes[first_index + 1 :]
    )
    pair_count = len(ordered_nodes) * (len(ordered_nodes) - 1) // 2
    return disagreements / pair_count


def maximum_pair_affinity_change(first: Matrix, second: Matrix) -> float:
    """二重み行列間で生じたpair affinity変化の最大値を返す。"""

    first_dimension = _validate_square_matrix(first)
    second_dimension = _validate_square_matrix(second)
    if first_dimension != second_dimension:
        raise ValueError("二重み行列の次元を一致させてください")
    return max(
        abs(
            _pair_affinity(first, first_node, second_node)
            - _pair_affinity(second, first_node, second_node)
        )
        for first_node in range(first_dimension)
        for second_node in range(first_node + 1, first_dimension)
    )


def normalize_partition(partition: Partition, *, dimension: int) -> Partition:
    """全nodeを一度ずつ覆うpartitionを安定順序へ正規化する。"""

    if not partition or any(not component for component in partition):
        raise ValueError("partitionは空でないcomponentを含めてください")
    flattened = tuple(node for component in partition for node in component)
    if any(
        not isinstance(node, int) or isinstance(node, bool)
        for node in flattened
    ) or sorted(flattened) != list(range(dimension)):
        raise ValueError("partitionは全nodeを重複なく一度ずつ覆ってください")
    return tuple(
        sorted(
            (tuple(sorted(component)) for component in partition),
            key=lambda component: component[0],
        )
    )


def _threshold_components(
    recurrent_weights: Matrix,
    *,
    threshold: float,
) -> Partition:
    dimension = len(recurrent_weights)
    neighbors = tuple(
        tuple(
            other
            for other in range(dimension)
            if other != node
            and _pair_affinity(recurrent_weights, node, other) > threshold
        )
        for node in range(dimension)
    )
    remaining = set(range(dimension))
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        frontier = [start]
        reached: set[int] = set()
        while frontier:
            node = frontier.pop()
            if node in reached:
                continue
            reached.add(node)
            frontier.extend(
                neighbor for neighbor in neighbors[node] if neighbor not in reached
            )
        remaining.difference_update(reached)
        components.append(tuple(sorted(reached)))
    return tuple(sorted(components, key=lambda component: component[0]))


def _pair_affinity(matrix: Matrix, first: int, second: int) -> float:
    return max(abs(matrix[first][second]), abs(matrix[second][first]))


def _pair_affinities(matrix: Matrix, dimension: int) -> tuple[float, ...]:
    return tuple(
        _pair_affinity(matrix, first, second)
        for first in range(dimension)
        for second in range(first + 1, dimension)
    )


def _validate_partition_node_set(partition: Partition) -> frozenset[int]:
    if not partition or any(not component for component in partition):
        raise ValueError("partitionは空でないcomponentを含めてください")
    flattened = tuple(node for component in partition for node in component)
    if any(
        not isinstance(node, int) or isinstance(node, bool)
        for node in flattened
    ) or len(set(flattened)) != len(flattened):
        raise ValueError("partitionのnodeは重複しない整数にしてください")
    return frozenset(flattened)


def _component_membership(partition: Partition) -> dict[int, int]:
    return {
        node: component_index
        for component_index, component in enumerate(partition)
        for node in component
    }


def _validate_square_matrix(matrix: Matrix) -> int:
    if len(matrix) < 2 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("recurrent_weightsは2次元以上の正方行列にしてください")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("recurrent_weightsは有限値にしてください")
    return len(matrix)
