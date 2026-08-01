"""外的妥当性検証に使う再帰重みfamilyの決定論的生成。"""

from __future__ import annotations

import math
import random
from typing import Literal

from reservoir_dynamics.experiments.orthant_margin_sweep import (
    _signed_symmetric_weights,
)
from reservoir_dynamics.theory.orthant_box import Matrix

RecurrentWeightFamily = Literal[
    "dense_symmetric",
    "sparse_symmetric",
    "modular_paired",
    "modular_heterogeneous",
    "asymmetric_dense",
    "feedforward_nonnormal",
]

RECURRENT_WEIGHT_FAMILIES: tuple[RecurrentWeightFamily, ...] = (
    "dense_symmetric",
    "sparse_symmetric",
    "modular_paired",
    "modular_heterogeneous",
    "asymmetric_dense",
    "feedforward_nonnormal",
)


def build_recurrent_weights(
    *,
    network_family: RecurrentWeightFamily,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    trial_seed: int,
) -> Matrix:
    """同一seed規約で構造だけが異なるtanh RNN重みを生成する。"""

    _validate_arguments(
        network_family=network_family,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gain=coupling_gain,
        trial_seed=trial_seed,
    )
    if network_family == "dense_symmetric":
        # 既存実験との数値的連続性を保つため、元のgeneratorを再利用する。
        return _signed_symmetric_weights(
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gain=coupling_gain,
            trial_seed=trial_seed,
        )

    random_generator = random.Random(trial_seed)
    if network_family == "sparse_symmetric":
        return _sparse_symmetric_weights(
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gain=coupling_gain,
            random_generator=random_generator,
        )
    if network_family == "modular_paired":
        return _modular_paired_weights(
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gain=coupling_gain,
            random_generator=random_generator,
        )
    if network_family == "modular_heterogeneous":
        return _modular_heterogeneous_weights(
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gain=coupling_gain,
            random_generator=random_generator,
        )
    if network_family == "asymmetric_dense":
        return _asymmetric_dense_weights(
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            coupling_gain=coupling_gain,
            random_generator=random_generator,
        )
    return _feedforward_nonnormal_weights(
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gain=coupling_gain,
        random_generator=random_generator,
    )


def _sparse_symmetric_weights(
    *,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    random_generator: random.Random,
) -> Matrix:
    # ringは全nodeの次数を揃え、疎性と孤立nodeの影響を分離する。
    undirected_edges = tuple(
        sorted(
            {
                (min(index, (index + 1) % dimension),
                 max(index, (index + 1) % dimension))
                for index in range(dimension)
            }
        )
    )
    edge_signs = {
        edge: random_generator.choice((-1.0, 1.0))
        for edge in undirected_edges
    }
    return tuple(
        tuple(
            _symmetric_entry(
                row_index=row_index,
                column_index=column_index,
                diagonal_gain=diagonal_gain,
                coupling_gain=coupling_gain,
                edge_signs=edge_signs,
            )
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )


def _modular_paired_weights(
    *,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    random_generator: random.Random,
) -> Matrix:
    # 独立な2-node moduleは局所結合とmodule間分離を同時に固定できる。
    paired_edges = tuple(
        (index, index + 1)
        for index in range(0, dimension - 1, 2)
    )
    edge_signs = {
        edge: random_generator.choice((-1.0, 1.0))
        for edge in paired_edges
    }
    return tuple(
        tuple(
            _symmetric_entry(
                row_index=row_index,
                column_index=column_index,
                diagonal_gain=diagonal_gain,
                coupling_gain=coupling_gain,
                edge_signs=edge_signs,
            )
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )


def _modular_heterogeneous_weights(
    *,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    random_generator: random.Random,
) -> Matrix:
    # 符号共役では消えないmodule差を作るため、各pairの絶対結合も独立化する。
    paired_edges = tuple(
        (index, index + 1)
        for index in range(0, dimension - 1, 2)
    )
    relative_edge_weights: dict[tuple[int, int], float] = {}
    for edge in paired_edges:
        magnitude_scale = 0.75 + 0.5 * random_generator.random()
        edge_sign = random_generator.choice((-1.0, 1.0))
        relative_edge_weights[edge] = magnitude_scale * edge_sign
    return tuple(
        tuple(
            _symmetric_entry(
                row_index=row_index,
                column_index=column_index,
                diagonal_gain=diagonal_gain,
                coupling_gain=coupling_gain,
                edge_signs=relative_edge_weights,
            )
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )
def _symmetric_entry(
    *,
    row_index: int,
    column_index: int,
    diagonal_gain: float,
    coupling_gain: float,
    edge_signs: dict[tuple[int, int], float],
) -> float:
    if row_index == column_index:
        return diagonal_gain
    edge = (
        min(row_index, column_index),
        max(row_index, column_index),
    )
    return coupling_gain * edge_signs.get(edge, 0.0)


def _asymmetric_dense_weights(
    *,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    random_generator: random.Random,
) -> Matrix:
    return tuple(
        tuple(
            diagonal_gain
            if row_index == column_index
            else coupling_gain * random_generator.choice((-1.0, 1.0))
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )


def _feedforward_nonnormal_weights(
    *,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    random_generator: random.Random,
) -> Matrix:
    # 三角構造は固有値を対角に固定したまま非正規な過渡結合を導入できる。
    return tuple(
        tuple(
            diagonal_gain
            if row_index == column_index
            else (
                coupling_gain * random_generator.choice((-1.0, 1.0))
                if column_index > row_index
                else 0.0
            )
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )


def _validate_arguments(
    *,
    network_family: str,
    dimension: int,
    diagonal_gain: float,
    coupling_gain: float,
    trial_seed: int,
) -> None:
    if network_family not in RECURRENT_WEIGHT_FAMILIES:
        raise ValueError(
            f"network_familyは{RECURRENT_WEIGHT_FAMILIES}から選んでください"
        )
    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 2
        or dimension > 8
    ):
        raise ValueError("dimensionは2以上8以下の整数にしてください")
    if not math.isfinite(diagonal_gain) or diagonal_gain <= 1.0:
        raise ValueError("diagonal_gainは1より大きい有限値にしてください")
    if not math.isfinite(coupling_gain) or coupling_gain < 0.0:
        raise ValueError("coupling_gainは有限の非負値にしてください")
    if not isinstance(trial_seed, int) or isinstance(trial_seed, bool):
        raise ValueError("trial_seedは整数にしてください")
