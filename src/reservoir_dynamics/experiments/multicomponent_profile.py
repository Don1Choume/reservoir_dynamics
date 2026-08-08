"""未知partitionから多成分certificateと固定predictor特徴を抽出する。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.aligned_sign_memory import (
    AlignedDisturbanceEvaluation,
    AlignedSignMemoryProfile,
    evaluate_aligned_sign_memory_network,
)
from reservoir_dynamics.experiments.component_profile import (
    COMPONENT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
)
from reservoir_dynamics.experiments.multicomponent_modular_family import (
    MultiComponentModularNetwork,
)
from reservoir_dynamics.metrics.module_partition import (
    Partition,
    normalize_partition,
)
from reservoir_dynamics.metrics.network_diagnostics import (
    matrix_nonnormality_commutator_norm,
    off_diagonal_infinity_norm,
)
from reservoir_dynamics.theory.multicomponent_coupling import (
    certificate_enumeration_complexity,
    component_inbound_load_matrix,
    summarize_multicomponent_margins,
)
from reservoir_dynamics.theory.orthant_box import Matrix
from reservoir_dynamics.theory.orthant_rectangle import (
    orthant_rectangle_certificate,
)


@dataclass(frozen=True, slots=True)
class MultiComponentProfilePoint:
    """既存固定modelへ渡せる対称集約済み三module profile。"""

    trial_seed: int
    module_sizes: tuple[int, ...]
    internal_gain: float
    maximum_bridge_strength: float
    disturbance_bound: float
    observed_task_retention: float
    isolated_task_product: float
    task_product_residual: float
    raw_attractor_fraction: float
    coupled_common_certified_fraction: float
    coupled_mean_uniform_margin: float
    full_off_diagonal_infinity_norm: float
    dimension_normalized_nonnormality: float
    maximum_directional_bridge_norm: float
    isolated_component_certified_fraction_product: float
    directional_certified_fraction: float
    transported_certified_fraction: float
    global_shifted_certified_fraction: float
    mean_directional_slack: float
    minimum_directional_slack: float
    directional_load_imbalance: float
    module_size_imbalance: float
    factorized_directional_certified_fraction: float
    enumerated_directional_certified_fraction: float
    local_orthant_count: int
    global_orthant_count: int

    @property
    def dimension(self) -> int:
        return sum(self.module_sizes)

    @property
    def global_feature_row(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in GLOBAL_FEATURE_NAMES)

    @property
    def component_feature_row(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in COMPONENT_FEATURE_NAMES)

    @property
    def product_feature_row(self) -> tuple[float, ...]:
        return (self.isolated_task_product,)


def evaluate_multicomponent_profile(
    *,
    network: MultiComponentModularNetwork,
    partition: Partition,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[MultiComponentProfilePoint, ...]:
    """観測座標を保ったまま局所profileと結合後targetを評価する。"""

    return evaluate_multicomponent_partitions(
        network=network,
        recurrent_weights=network.recurrent_weights,
        partitions=(partition,),
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )[0]


def evaluate_multicomponent_partitions(
    *,
    network: MultiComponentModularNetwork,
    partitions: tuple[Partition, ...],
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    recurrent_weights: Matrix | None = None,
) -> tuple[tuple[MultiComponentProfilePoint, ...], ...]:
    """同じ結合後trajectoryを共有し、複数partitionの局所表現を比較する。"""

    if not partitions:
        raise ValueError("partitionsは1件以上必要です")
    effective_weights = (
        network.recurrent_weights
        if recurrent_weights is None
        else recurrent_weights
    )
    dimension = len(effective_weights)
    if dimension != len(network.recurrent_weights) or any(
        len(row) != dimension for row in effective_weights
    ):
        raise ValueError("recurrent_weightsはnetworkと同次元の正方行列にしてください")
    if any(not math.isfinite(value) for row in effective_weights for value in row):
        raise ValueError("recurrent_weightsは有限値にしてください")
    normalized_partitions = tuple(
        normalize_partition(partition, dimension=dimension)
        for partition in partitions
    )

    profile_arguments = {
        "disturbance_bounds": disturbance_bounds,
        "task_steps": task_steps,
        "autonomous_steps": autonomous_steps,
        "convergence_tolerance": convergence_tolerance,
    }
    coupled_profile = evaluate_aligned_sign_memory_network(
        recurrent_weights=effective_weights,
        coordinate_indices=tuple(range(dimension)),
        **profile_arguments,
    )
    profiles_by_partition: dict[
        Partition,
        tuple[MultiComponentProfilePoint, ...],
    ] = {}
    for normalized_partition in normalized_partitions:
        if normalized_partition in profiles_by_partition:
            continue
        component_profiles = tuple(
            evaluate_aligned_sign_memory_network(
                recurrent_weights=_submatrix_by_indices(
                    effective_weights,
                    component,
                ),
                coordinate_indices=component,
                **profile_arguments,
            )
            for component in normalized_partition
        )
        load_matrix = component_inbound_load_matrix(
            effective_weights,
            normalized_partition,
        )
        profiles_by_partition[normalized_partition] = tuple(
            _build_point(
                network=network,
                recurrent_weights=effective_weights,
                partition=normalized_partition,
                coupled_profile=coupled_profile,
                component_profiles=component_profiles,
                inbound_load_matrix=load_matrix,
                disturbance_index=index,
            )
            for index in range(len(disturbance_bounds))
        )
    return tuple(
        profiles_by_partition[partition]
        for partition in normalized_partitions
    )


def _build_point(
    *,
    network: MultiComponentModularNetwork,
    recurrent_weights: Matrix,
    partition: Partition,
    coupled_profile: AlignedSignMemoryProfile,
    component_profiles: tuple[AlignedSignMemoryProfile, ...],
    inbound_load_matrix: Matrix,
    disturbance_index: int,
) -> MultiComponentProfilePoint:
    coupled_evaluation = coupled_profile.disturbance_evaluations[disturbance_index]
    component_evaluations = tuple(
        profile.disturbance_evaluations[disturbance_index]
        for profile in component_profiles
    )
    isolated_product = _aligned_task_product(component_evaluations)
    margin_distributions = tuple(
        tuple(
            orthant.maximum_uniform_disturbance for orthant in profile.orthants
        )
        for profile in component_profiles
    )
    directional_summary = summarize_multicomponent_margins(
        component_margins=margin_distributions,
        disturbance_bound=coupled_evaluation.disturbance_bound,
        inbound_load_matrix=inbound_load_matrix,
    )
    transported, enumerated_directional, global_shifted = _certificate_fractions(
        recurrent_weights=recurrent_weights,
        partition=partition,
        coupled_profile=coupled_profile,
        component_profiles=component_profiles,
        disturbance_bound=coupled_evaluation.disturbance_bound,
        total_inbound_loads=directional_summary.total_inbound_loads,
    )
    module_sizes = tuple(sorted(len(component) for component in partition))
    complexity = certificate_enumeration_complexity(module_sizes)
    total_load_sum = math.fsum(directional_summary.total_inbound_loads)
    point = MultiComponentProfilePoint(
        trial_seed=network.trial_seed,
        module_sizes=module_sizes,
        internal_gain=network.internal_gain,
        maximum_bridge_strength=network.maximum_total_bridge_strength,
        disturbance_bound=coupled_evaluation.disturbance_bound,
        observed_task_retention=coupled_evaluation.mean_task_retention,
        isolated_task_product=isolated_product,
        task_product_residual=(
            coupled_evaluation.mean_task_retention - isolated_product
        ),
        raw_attractor_fraction=(
            coupled_profile.raw_attractor_count / (2 ** sum(module_sizes))
        ),
        coupled_common_certified_fraction=(
            coupled_evaluation.certified_robust_fraction
        ),
        coupled_mean_uniform_margin=(
            coupled_profile.mean_uniform_disturbance_margin
        ),
        full_off_diagonal_infinity_norm=off_diagonal_infinity_norm(
            recurrent_weights
        ),
        dimension_normalized_nonnormality=(
            matrix_nonnormality_commutator_norm(recurrent_weights)
            / sum(module_sizes)
        ),
        maximum_directional_bridge_norm=directional_summary.global_load,
        isolated_component_certified_fraction_product=math.prod(
            evaluation.certified_robust_fraction
            for evaluation in component_evaluations
        ),
        directional_certified_fraction=(
            directional_summary.directional_certified_fraction
        ),
        transported_certified_fraction=transported,
        global_shifted_certified_fraction=global_shifted,
        mean_directional_slack=directional_summary.mean_directional_slack,
        minimum_directional_slack=(
            directional_summary.minimum_directional_slack
        ),
        directional_load_imbalance=(
            (
                max(directional_summary.total_inbound_loads)
                - min(directional_summary.total_inbound_loads)
            )
            / total_load_sum
            if total_load_sum > 0.0
            else 0.0
        ),
        module_size_imbalance=(
            (max(module_sizes) - min(module_sizes)) / sum(module_sizes)
        ),
        factorized_directional_certified_fraction=(
            directional_summary.directional_certified_fraction
        ),
        enumerated_directional_certified_fraction=enumerated_directional,
        local_orthant_count=complexity.local_orthant_count,
        global_orthant_count=complexity.monolithic_orthant_count,
    )
    if any(
        not math.isfinite(value)
        for value in point.component_feature_row
        + (point.observed_task_retention,)
    ):
        raise RuntimeError("multicomponent profileに非有限値があります")
    return point


def _aligned_task_product(
    evaluations: tuple[AlignedDisturbanceEvaluation, ...],
) -> float:
    retention_by_component = tuple(
        {
            value.code: value.task_retention
            for value in evaluation.direction_evaluations
        }
        for evaluation in evaluations
    )
    code_sets = tuple(set(values) for values in retention_by_component)
    if not code_sets or any(values != code_sets[0] for values in code_sets[1:]):
        raise RuntimeError("component間でdirection codeが一致しません")
    return math.fsum(
        math.prod(values[code] for values in retention_by_component)
        for code in retention_by_component[0]
    ) / len(retention_by_component[0])


def _certificate_fractions(
    *,
    recurrent_weights: Matrix,
    partition: Partition,
    coupled_profile: AlignedSignMemoryProfile,
    component_profiles: tuple[AlignedSignMemoryProfile, ...],
    disturbance_bound: float,
    total_inbound_loads: tuple[float, ...],
) -> tuple[float, float, float]:
    coupled_by_signs = {
        orthant.attractor_signs: orthant for orthant in coupled_profile.orthants
    }
    transported_count = 0
    directional_count = 0
    global_count = 0
    global_load = max(total_inbound_loads)
    combination_count = 0
    for component_orthants in itertools.product(
        *(profile.orthants for profile in component_profiles)
    ):
        combination_count += 1
        signs = [0 for _ in recurrent_weights]
        boundaries = [0.0 for _ in recurrent_weights]
        for component, orthant in zip(
            partition,
            component_orthants,
            strict=True,
        ):
            for local_index, node in enumerate(component):
                signs[node] = orthant.attractor_signs[local_index]
                boundaries[node] = orthant.invariant_boundary
        expected_signs = tuple(signs)
        expected_boundaries = tuple(boundaries)
        coupled_orthant = coupled_by_signs[expected_signs]
        inside = coupled_orthant.fixed_point_retained and all(
            sign * value >= boundary - 1e-12
            for sign, value, boundary in zip(
                expected_signs,
                coupled_orthant.fixed_point,
                expected_boundaries,
                strict=True,
            )
        )
        if not inside:
            continue
        rectangle = orthant_rectangle_certificate(
            recurrent_weights=recurrent_weights,
            attractor_signs=expected_signs,
            lower_boundaries=expected_boundaries,
        )
        transported_count += (
            rectangle.maximum_uniform_disturbance
            >= disturbance_bound - 1e-12
        )
        margins = tuple(
            orthant.maximum_uniform_disturbance
            for orthant in component_orthants
        )
        directional_count += all(
            margin >= disturbance_bound + load - 1e-12
            for margin, load in zip(margins, total_inbound_loads, strict=True)
        )
        global_count += all(
            margin >= disturbance_bound + global_load - 1e-12
            for margin in margins
        )
    if combination_count != len(coupled_profile.orthants):
        raise RuntimeError("component直積とfull orthant数が一致しません")
    return (
        transported_count / combination_count,
        directional_count / combination_count,
        global_count / combination_count,
    )


def _submatrix_by_indices(matrix: Matrix, indices: tuple[int, ...]) -> Matrix:
    return tuple(
        tuple(matrix[row][column] for column in indices) for row in indices
    )
