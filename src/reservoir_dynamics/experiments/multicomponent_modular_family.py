"""task前に分割回復を監査できる三module tanh RNN family。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from reservoir_dynamics.metrics.module_partition import (
    AffinityGapPartition,
    Partition,
    PartitionSeparation,
    infer_affinity_gap_partition,
    partition_separation,
    partitions_equivalent,
)
from reservoir_dynamics.theory.multicomponent_coupling import (
    component_inbound_load_matrix,
)
from reservoir_dynamics.theory.orthant_box import Matrix


@dataclass(frozen=True, slots=True)
class MultiComponentModularNetwork:
    """座標permutation後の重みと生成時partitionを分離して保持する。"""

    trial_seed: int
    module_sizes: tuple[int, ...]
    internal_gain: float
    maximum_total_bridge_strength: float
    recurrent_weights: Matrix
    true_partition: Partition
    module_weights: tuple[Matrix, ...]
    inbound_load_matrix: Matrix
    total_inbound_loads: tuple[float, ...]
    coordinate_permutation: tuple[int, ...]
    partition_separation: PartitionSeparation
    inferred_partition: AffinityGapPartition
    internal_blocks_asymmetric: bool
    all_module_pairs_bidirectional: bool
    bridges_nontransposed: bool
    magnitude_fingerprint: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultiComponentStructureGate:
    """task値を生成せずfamilyとpartition推定だけを監査する。"""

    trial_seeds: tuple[int, ...]
    module_sizes: tuple[int, ...]
    internal_gains: tuple[float, ...]
    maximum_total_bridge_strengths: tuple[float, ...]
    group_class_counts: tuple[tuple[str, int], ...]
    fingerprints_unique: bool
    partition_recovery_exact: bool
    affinity_separation_valid: bool
    selected_gap_matches_separation: bool
    unique_maximum_gap: bool
    internal_blocks_asymmetric: bool
    positive_bridges_valid: bool
    bridge_loads_valid: bool
    task_values_generated: bool
    passed: bool


def build_multicomponent_modular_network(
    *,
    trial_seed: int,
    module_sizes: tuple[int, ...],
    internal_gain: float,
    maximum_total_bridge_strength: float,
    diagonal_gain: float = 1.5,
) -> MultiComponentModularNetwork:
    """一意なaffinity gapと非対称bridgeを持つ座標scramble系を生成する。"""

    _validate_configuration(
        trial_seed=trial_seed,
        module_sizes=module_sizes,
        internal_gain=internal_gain,
        maximum_total_bridge_strength=maximum_total_bridge_strength,
        diagonal_gain=diagonal_gain,
    )
    random_generator = random.Random(
        trial_seed * 1_000_003
        + sum(
            (index + 1) * size * 10_007
            for index, size in enumerate(module_sizes)
        )
        + round(internal_gain * 1_000_000)
    )
    dimension = sum(module_sizes)
    mutable_weights = [
        [0.0 for _ in range(dimension)] for _ in range(dimension)
    ]
    for index in range(dimension):
        mutable_weights[index][index] = diagonal_gain
    canonical_partition = _contiguous_partition(module_sizes)
    for component in canonical_partition:
        _fill_internal_component(
            mutable_weights,
            component=component,
            internal_gain=internal_gain,
            random_generator=random_generator,
        )
    _fill_all_bridges(
        mutable_weights,
        partition=canonical_partition,
        maximum_total_bridge_strength=maximum_total_bridge_strength,
        random_generator=random_generator,
    )
    canonical_weights = _freeze_matrix(mutable_weights)
    coordinate_permutation = list(range(dimension))
    random_generator.shuffle(coordinate_permutation)
    observed_weights = _permute_matrix(
        canonical_weights,
        observed_order=tuple(coordinate_permutation),
    )
    canonical_to_observed = {
        canonical_index: observed_index
        for observed_index, canonical_index in enumerate(coordinate_permutation)
    }
    true_partition = tuple(
        tuple(
            sorted(canonical_to_observed[index] for index in component)
        )
        for component in canonical_partition
    )
    module_weights = tuple(
        _submatrix_by_indices(observed_weights, component)
        for component in true_partition
    )
    load_matrix = component_inbound_load_matrix(
        observed_weights,
        true_partition,
    )
    separation = partition_separation(observed_weights, true_partition)
    inferred = infer_affinity_gap_partition(observed_weights)
    return MultiComponentModularNetwork(
        trial_seed=trial_seed,
        module_sizes=module_sizes,
        internal_gain=internal_gain,
        maximum_total_bridge_strength=maximum_total_bridge_strength,
        recurrent_weights=observed_weights,
        true_partition=true_partition,
        module_weights=module_weights,
        inbound_load_matrix=load_matrix,
        total_inbound_loads=tuple(math.fsum(row) for row in load_matrix),
        coordinate_permutation=tuple(coordinate_permutation),
        partition_separation=separation,
        inferred_partition=inferred,
        internal_blocks_asymmetric=all(
            _is_asymmetric(matrix) for matrix in module_weights
        ),
        all_module_pairs_bidirectional=_all_module_pairs_bidirectional(
            observed_weights,
            true_partition,
        ),
        bridges_nontransposed=_all_bridges_nontransposed(
            observed_weights,
            true_partition,
        ),
        magnitude_fingerprint=tuple(
            abs(value).hex() for row in observed_weights for value in row
        ),
    )


def audit_multicomponent_structure(
    *,
    trial_seeds: tuple[int, ...],
    module_sizes: tuple[int, ...],
    internal_gains: tuple[float, ...],
    maximum_total_bridge_strengths: tuple[float, ...],
    diagonal_gain: float = 1.5,
    tolerance: float = 1e-12,
) -> MultiComponentStructureGate:
    """taskを呼ばず、全登録群の構造独立性と分割回復を監査する。"""

    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    if not internal_gains or not maximum_total_bridge_strengths:
        raise ValueError("gainとbridge strengthは1件以上必要です")
    group_counts: list[tuple[str, int]] = []
    fingerprints_unique = True
    partition_exact = True
    separation_valid = True
    gap_matches = True
    gap_unique = True
    internal_valid = True
    positive_bridges_valid = True
    bridge_loads_valid = True
    for internal_gain in internal_gains:
        for bridge_strength in maximum_total_bridge_strengths:
            networks = tuple(
                build_multicomponent_modular_network(
                    trial_seed=seed,
                    module_sizes=module_sizes,
                    internal_gain=internal_gain,
                    maximum_total_bridge_strength=bridge_strength,
                    diagonal_gain=diagonal_gain,
                )
                for seed in trial_seeds
            )
            class_count = len(
                {network.magnitude_fingerprint for network in networks}
            )
            group_counts.append(
                (
                    f"{'+'.join(str(size) for size in module_sizes)}:"
                    f"{internal_gain.hex()}:{bridge_strength.hex()}",
                    class_count,
                )
            )
            fingerprints_unique = (
                fingerprints_unique and class_count == len(trial_seeds)
            )
            partition_exact = partition_exact and all(
                partitions_equivalent(
                    network.inferred_partition.components,
                    network.true_partition,
                )
                for network in networks
            )
            separation_valid = separation_valid and all(
                network.partition_separation.separated for network in networks
            )
            gap_matches = gap_matches and all(
                math.isclose(
                    network.inferred_partition.selected_gap,
                    network.partition_separation.gap,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for network in networks
            )
            gap_unique = gap_unique and all(
                network.inferred_partition.gap_is_unique for network in networks
            )
            internal_valid = internal_valid and all(
                network.internal_blocks_asymmetric for network in networks
            )
            if bridge_strength > 0.0:
                positive_bridges_valid = positive_bridges_valid and all(
                    network.all_module_pairs_bidirectional
                    and network.bridges_nontransposed
                    for network in networks
                )
            bridge_loads_valid = bridge_loads_valid and all(
                math.isclose(
                    max(network.total_inbound_loads),
                    bridge_strength,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for network in networks
            )
    passed = all(
        (
            fingerprints_unique,
            partition_exact,
            separation_valid,
            gap_matches,
            gap_unique,
            internal_valid,
            positive_bridges_valid,
            bridge_loads_valid,
        )
    )
    return MultiComponentStructureGate(
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_total_bridge_strengths=maximum_total_bridge_strengths,
        group_class_counts=tuple(group_counts),
        fingerprints_unique=fingerprints_unique,
        partition_recovery_exact=partition_exact,
        affinity_separation_valid=separation_valid,
        selected_gap_matches_separation=gap_matches,
        unique_maximum_gap=gap_unique,
        internal_blocks_asymmetric=internal_valid,
        positive_bridges_valid=positive_bridges_valid,
        bridge_loads_valid=bridge_loads_valid,
        task_values_generated=False,
        passed=passed,
    )


def _fill_internal_component(
    matrix: list[list[float]],
    *,
    component: tuple[int, ...],
    internal_gain: float,
    random_generator: random.Random,
) -> None:
    if len(component) == 2:
        first, second = component
        matrix[first][second] = _signed_magnitude(
            internal_gain * (0.90 + 0.05 * random_generator.random()),
            random_generator,
        )
        matrix[second][first] = _signed_magnitude(
            internal_gain * (0.72 + 0.05 * random_generator.random()),
            random_generator,
        )
        return
    size = len(component)
    for local_index, row in enumerate(component):
        next_node = component[(local_index + 1) % size]
        previous_node = component[(local_index - 1) % size]
        matrix[row][next_node] = _signed_magnitude(
            internal_gain * (0.82 + 0.04 * random_generator.random()),
            random_generator,
        )
        matrix[row][previous_node] = _signed_magnitude(
            internal_gain * (0.18 + 0.04 * random_generator.random()),
            random_generator,
        )


def _fill_all_bridges(
    matrix: list[list[float]],
    *,
    partition: Partition,
    maximum_total_bridge_strength: float,
    random_generator: random.Random,
) -> None:
    all_nodes = frozenset(node for component in partition for node in component)
    for component in partition:
        source_nodes = tuple(sorted(all_nodes.difference(component)))
        for local_row, row in enumerate(component):
            row_strength = maximum_total_bridge_strength * (
                1.0
                if local_row == 0
                else 0.72 + 0.18 * random_generator.random()
            )
            raw_magnitudes = tuple(
                0.9 + 0.2 * random_generator.random() for _ in source_nodes
            )
            magnitude_sum = math.fsum(raw_magnitudes)
            for column, raw_magnitude in zip(
                source_nodes,
                raw_magnitudes,
                strict=True,
            ):
                matrix[row][column] = _signed_magnitude(
                    row_strength * raw_magnitude / magnitude_sum,
                    random_generator,
                )


def _signed_magnitude(
    magnitude: float,
    random_generator: random.Random,
) -> float:
    return magnitude if random_generator.random() < 0.5 else -magnitude


def _contiguous_partition(module_sizes: tuple[int, ...]) -> Partition:
    components: list[tuple[int, ...]] = []
    start = 0
    for size in module_sizes:
        components.append(tuple(range(start, start + size)))
        start += size
    return tuple(components)


def _permute_matrix(
    matrix: Matrix,
    *,
    observed_order: tuple[int, ...],
) -> Matrix:
    return tuple(
        tuple(matrix[row][column] for column in observed_order)
        for row in observed_order
    )


def _submatrix_by_indices(matrix: Matrix, indices: tuple[int, ...]) -> Matrix:
    return tuple(
        tuple(matrix[row][column] for column in indices) for row in indices
    )


def _freeze_matrix(matrix: list[list[float]]) -> Matrix:
    return tuple(tuple(row) for row in matrix)


def _is_asymmetric(matrix: Matrix, tolerance: float = 1e-15) -> bool:
    return any(
        abs(matrix[row][column] - matrix[column][row]) > tolerance
        for row in range(len(matrix))
        for column in range(row)
    )


def _all_module_pairs_bidirectional(
    matrix: Matrix,
    partition: Partition,
) -> bool:
    return all(
        any(matrix[row][column] != 0.0 for row in first for column in second)
        and any(matrix[row][column] != 0.0 for row in second for column in first)
        for first_index, first in enumerate(partition)
        for second in partition[first_index + 1 :]
    )


def _all_bridges_nontransposed(matrix: Matrix, partition: Partition) -> bool:
    return all(
        any(
            matrix[row][column] != matrix[column][row]
            for row in first
            for column in second
        )
        for first_index, first in enumerate(partition)
        for second in partition[first_index + 1 :]
    )


def _validate_configuration(
    *,
    trial_seed: int,
    module_sizes: tuple[int, ...],
    internal_gain: float,
    maximum_total_bridge_strength: float,
    diagonal_gain: float,
) -> None:
    if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
        raise ValueError("trial_seedは整数にしてください")
    if len(module_sizes) < 3 or any(
        not isinstance(size, int)
        or isinstance(size, bool)
        or size not in (2, 3)
        for size in module_sizes
    ) or sum(module_sizes) > 10:
        raise ValueError(
            "module_sizesは合計10以下の2または3を三つ以上指定してください"
        )
    if not math.isfinite(internal_gain) or internal_gain <= 0.0:
        raise ValueError("internal_gainは有限の正値にしてください")
    if (
        not math.isfinite(maximum_total_bridge_strength)
        or maximum_total_bridge_strength < 0.0
        or maximum_total_bridge_strength > internal_gain * 1.6
    ):
        raise ValueError(
            "maximum_total_bridge_strengthは分離条件内の有限非負値にしてください"
        )
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
