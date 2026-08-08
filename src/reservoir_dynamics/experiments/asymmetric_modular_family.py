"""異なるmodule sizeと非対称bridgeを持つ多重安定RNN family。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from reservoir_dynamics.theory.component_coupling import (
    directional_bridge_norms,
)
from reservoir_dynamics.theory.orthant_box import Matrix

ModuleSizePair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class AsymmetricModularNetwork:
    """component profileに必要な分割情報を保持する不変network。"""

    trial_seed: int
    module_sizes: ModuleSizePair
    internal_gain: float
    maximum_bridge_strength: float
    recurrent_weights: Matrix
    base_recurrent_weights: Matrix
    module_weights: tuple[Matrix, Matrix]
    inbound_bridge_norms: tuple[float, float]
    internal_blocks_asymmetric: bool
    bridges_nontransposed: bool
    magnitude_fingerprint: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AsymmetricModularStructureGate:
    """task値を見る前にfamilyの独立性と非対称性を監査する。"""

    trial_seeds: tuple[int, ...]
    module_size_pairs: tuple[ModuleSizePair, ...]
    internal_gains: tuple[float, ...]
    maximum_bridge_strengths: tuple[float, ...]
    group_class_counts: tuple[tuple[str, int], ...]
    fingerprints_unique: bool
    internal_blocks_asymmetric: bool
    positive_bridges_valid: bool
    bridge_norms_valid: bool
    passed: bool


def build_asymmetric_modular_network(
    *,
    trial_seed: int,
    module_sizes: ModuleSizePair,
    internal_gain: float,
    maximum_bridge_strength: float,
    diagonal_gain: float = 1.5,
) -> AsymmetricModularNetwork:
    """行和を制御した内部blockと方向別normが異なるbridgeを生成する。"""

    _validate_configuration(
        trial_seed=trial_seed,
        module_sizes=module_sizes,
        internal_gain=internal_gain,
        maximum_bridge_strength=maximum_bridge_strength,
        diagonal_gain=diagonal_gain,
    )
    first_size, second_size = module_sizes
    dimension = first_size + second_size
    random_generator = random.Random(
        trial_seed * 1_000_003
        + first_size * 10_007
        + second_size * 100_003
        + round(internal_gain * 1_000_000)
    )
    mutable_weights = [
        [0.0 for _ in range(dimension)] for _ in range(dimension)
    ]
    for index in range(dimension):
        mutable_weights[index][index] = diagonal_gain
    _fill_internal_block(
        mutable_weights,
        start=0,
        size=first_size,
        maximum_row_sum=internal_gain,
        random_generator=random_generator,
    )
    _fill_internal_block(
        mutable_weights,
        start=first_size,
        size=second_size,
        maximum_row_sum=internal_gain,
        random_generator=random_generator,
    )
    base_weights = _freeze_matrix(mutable_weights)

    ratio = 0.35 + 0.5 * random_generator.random()
    if trial_seed % 2 == 0:
        inbound_targets = (
            maximum_bridge_strength * ratio,
            maximum_bridge_strength,
        )
    else:
        inbound_targets = (
            maximum_bridge_strength,
            maximum_bridge_strength * ratio,
        )
    _fill_bridge_rows(
        mutable_weights,
        target_start=0,
        target_size=first_size,
        source_start=first_size,
        source_size=second_size,
        maximum_row_sum=inbound_targets[0],
        random_generator=random_generator,
    )
    _fill_bridge_rows(
        mutable_weights,
        target_start=first_size,
        target_size=second_size,
        source_start=0,
        source_size=first_size,
        maximum_row_sum=inbound_targets[1],
        random_generator=random_generator,
    )
    recurrent_weights = _freeze_matrix(mutable_weights)
    module_weights = (
        _submatrix(recurrent_weights, 0, first_size),
        _submatrix(recurrent_weights, first_size, second_size),
    )
    inbound_norms = directional_bridge_norms(
        recurrent_weights,
        split_index=first_size,
    )
    return AsymmetricModularNetwork(
        trial_seed=trial_seed,
        module_sizes=module_sizes,
        internal_gain=internal_gain,
        maximum_bridge_strength=maximum_bridge_strength,
        recurrent_weights=recurrent_weights,
        base_recurrent_weights=base_weights,
        module_weights=module_weights,
        inbound_bridge_norms=inbound_norms,
        internal_blocks_asymmetric=all(
            _is_asymmetric(matrix) for matrix in module_weights
        ),
        bridges_nontransposed=_bridges_are_nontransposed(
            recurrent_weights,
            first_size,
        ),
        magnitude_fingerprint=tuple(
            abs(value).hex() for row in recurrent_weights for value in row
        ),
    )


def audit_asymmetric_modular_structure(
    *,
    trial_seeds: tuple[int, ...],
    module_size_pairs: tuple[ModuleSizePair, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    diagonal_gain: float = 1.5,
    tolerance: float = 1e-12,
) -> AsymmetricModularStructureGate:
    """family・gain・strengthごとの絶対値classとbridge条件を監査する。"""

    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    group_counts: list[tuple[str, int]] = []
    fingerprints_unique = True
    internal_valid = True
    positive_bridges_valid = True
    bridge_norms_valid = True
    for module_sizes in module_size_pairs:
        for internal_gain in internal_gains:
            for bridge_strength in maximum_bridge_strengths:
                networks = tuple(
                    build_asymmetric_modular_network(
                        trial_seed=seed,
                        module_sizes=module_sizes,
                        internal_gain=internal_gain,
                        maximum_bridge_strength=bridge_strength,
                        diagonal_gain=diagonal_gain,
                    )
                    for seed in trial_seeds
                )
                fingerprints = tuple(
                    network.magnitude_fingerprint for network in networks
                )
                class_count = len(set(fingerprints))
                group_counts.append(
                    (
                        f"{module_sizes[0]}+{module_sizes[1]}:"
                        f"{internal_gain.hex()}:{bridge_strength.hex()}",
                        class_count,
                    )
                )
                fingerprints_unique = (
                    fingerprints_unique and class_count == len(trial_seeds)
                )
                internal_valid = internal_valid and all(
                    network.internal_blocks_asymmetric for network in networks
                )
                if bridge_strength > 0.0:
                    positive_bridges_valid = positive_bridges_valid and all(
                        min(network.inbound_bridge_norms) > 0.0
                        and network.bridges_nontransposed
                        for network in networks
                    )
                bridge_norms_valid = bridge_norms_valid and all(
                    math.isclose(
                        max(network.inbound_bridge_norms),
                        bridge_strength,
                        rel_tol=tolerance,
                        abs_tol=tolerance,
                    )
                    for network in networks
                )
    passed = (
        fingerprints_unique
        and internal_valid
        and positive_bridges_valid
        and bridge_norms_valid
    )
    return AsymmetricModularStructureGate(
        trial_seeds=trial_seeds,
        module_size_pairs=module_size_pairs,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        group_class_counts=tuple(group_counts),
        fingerprints_unique=fingerprints_unique,
        internal_blocks_asymmetric=internal_valid,
        positive_bridges_valid=positive_bridges_valid,
        bridge_norms_valid=bridge_norms_valid,
        passed=passed,
    )


def _fill_internal_block(
    matrix: list[list[float]],
    *,
    start: int,
    size: int,
    maximum_row_sum: float,
    random_generator: random.Random,
) -> None:
    for local_row in range(size):
        raw_values = tuple(
            random_generator.uniform(-1.0, 1.0)
            for _ in range(size - 1)
        )
        target = maximum_row_sum * (0.65 + 0.35 * random_generator.random())
        scaled = _scale_values(raw_values, target)
        value_index = 0
        for local_column in range(size):
            if local_column == local_row:
                continue
            matrix[start + local_row][start + local_column] = scaled[value_index]
            value_index += 1


def _fill_bridge_rows(
    matrix: list[list[float]],
    *,
    target_start: int,
    target_size: int,
    source_start: int,
    source_size: int,
    maximum_row_sum: float,
    random_generator: random.Random,
) -> None:
    for target_row in range(target_size):
        raw_values = tuple(
            random_generator.uniform(-1.0, 1.0) for _ in range(source_size)
        )
        row_factor = 1.0 if target_row == 0 else 0.45 + 0.45 * random_generator.random()
        scaled = _scale_values(raw_values, maximum_row_sum * row_factor)
        for source_column, value in enumerate(scaled):
            matrix[target_start + target_row][source_start + source_column] = value


def _scale_values(values: tuple[float, ...], target_absolute_sum: float) -> tuple[float, ...]:
    absolute_sum = math.fsum(abs(value) for value in values)
    if target_absolute_sum == 0.0:
        return (0.0,) * len(values)
    if absolute_sum == 0.0:
        raise RuntimeError("乱数行の絶対値和が0です")
    return tuple(value * target_absolute_sum / absolute_sum for value in values)


def _submatrix(matrix: Matrix, start: int, size: int) -> Matrix:
    return tuple(
        tuple(matrix[row][column] for column in range(start, start + size))
        for row in range(start, start + size)
    )


def _freeze_matrix(matrix: list[list[float]]) -> Matrix:
    return tuple(tuple(row) for row in matrix)


def _is_asymmetric(matrix: Matrix, tolerance: float = 1e-15) -> bool:
    return any(
        abs(matrix[row][column] - matrix[column][row]) > tolerance
        for row in range(len(matrix))
        for column in range(row)
    )


def _bridges_are_nontransposed(matrix: Matrix, split_index: int) -> bool:
    return any(
        matrix[row][column] != matrix[column][row]
        for row in range(split_index)
        for column in range(split_index, len(matrix))
    )


def _validate_configuration(
    *,
    trial_seed: int,
    module_sizes: ModuleSizePair,
    internal_gain: float,
    maximum_bridge_strength: float,
    diagonal_gain: float,
) -> None:
    if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
        raise ValueError("trial_seedは整数にしてください")
    if len(module_sizes) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 2
        for size in module_sizes
    ):
        raise ValueError("module_sizesは2以上の整数二つにしてください")
    if sum(module_sizes) > 10:
        raise ValueError("全dimensionは10以下にしてください")
    for value, name in (
        (internal_gain, "internal_gain"),
        (maximum_bridge_strength, "maximum_bridge_strength"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}は有限の非負値にしてください")
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")

