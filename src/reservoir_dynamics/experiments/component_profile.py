"""非対称module系からglobal・component予測特徴を抽出する。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.aligned_sign_memory import (
    AlignedDisturbanceEvaluation,
    AlignedSignMemoryProfile,
    evaluate_aligned_sign_memory_network,
)
from reservoir_dynamics.experiments.asymmetric_modular_family import (
    AsymmetricModularNetwork,
)
from reservoir_dynamics.metrics.network_diagnostics import (
    matrix_nonnormality_commutator_norm,
    off_diagonal_infinity_norm,
)
from reservoir_dynamics.theory.component_coupling import (
    summarize_directional_margins,
)
from reservoir_dynamics.theory.orthant_rectangle import (
    orthant_rectangle_certificate,
)

GLOBAL_FEATURE_NAMES = (
    "dimension",
    "raw_attractor_fraction",
    "coupled_common_certified_fraction",
    "coupled_mean_uniform_margin",
    "full_off_diagonal_infinity_norm",
    "dimension_normalized_nonnormality",
    "maximum_directional_bridge_norm",
)
COMPONENT_ONLY_FEATURE_NAMES = (
    "isolated_task_product",
    "isolated_component_certified_fraction_product",
    "directional_certified_fraction",
    "transported_certified_fraction",
    "mean_directional_slack",
    "minimum_directional_slack",
    "directional_load_imbalance",
    "module_size_imbalance",
)
COMPONENT_FEATURE_NAMES = GLOBAL_FEATURE_NAMES + COMPONENT_ONLY_FEATURE_NAMES
PRODUCT_FEATURE_NAMES = ("isolated_task_product",)


@dataclass(frozen=True, slots=True)
class ComponentProfilePoint:
    """一network・一外乱におけるnested predictorの入力とtarget。"""

    trial_seed: int
    module_sizes: tuple[int, int]
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


def evaluate_component_profile(
    *,
    network: AsymmetricModularNetwork,
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[ComponentProfilePoint, ...]:
    """結合後targetと、targetを含まないnested特徴を同じgridで評価する。"""

    first_size, _ = network.module_sizes
    profile_arguments = {
        "disturbance_bounds": disturbance_bounds,
        "task_steps": task_steps,
        "autonomous_steps": autonomous_steps,
        "convergence_tolerance": convergence_tolerance,
    }
    coupled_profile = evaluate_aligned_sign_memory_network(
        recurrent_weights=network.recurrent_weights,
        coordinate_offset=0,
        **profile_arguments,
    )
    component_profiles = (
        evaluate_aligned_sign_memory_network(
            recurrent_weights=network.module_weights[0],
            coordinate_offset=0,
            **profile_arguments,
        ),
        evaluate_aligned_sign_memory_network(
            recurrent_weights=network.module_weights[1],
            coordinate_offset=first_size,
            **profile_arguments,
        ),
    )
    return tuple(
        _build_point(
            network=network,
            coupled_profile=coupled_profile,
            component_profiles=component_profiles,
            disturbance_index=index,
        )
        for index in range(len(disturbance_bounds))
    )


def _build_point(
    *,
    network: AsymmetricModularNetwork,
    coupled_profile: AlignedSignMemoryProfile,
    component_profiles: tuple[AlignedSignMemoryProfile, AlignedSignMemoryProfile],
    disturbance_index: int,
) -> ComponentProfilePoint:
    coupled_evaluation = coupled_profile.disturbance_evaluations[disturbance_index]
    component_evaluations = (
        component_profiles[0].disturbance_evaluations[disturbance_index],
        component_profiles[1].disturbance_evaluations[disturbance_index],
    )
    isolated_product = _aligned_task_product(component_evaluations)
    margin_pairs = tuple(
        (
            first.maximum_uniform_disturbance,
            second.maximum_uniform_disturbance,
        )
        for first, second in itertools.product(
            component_profiles[0].orthants,
            component_profiles[1].orthants,
        )
    )
    directional_summary = summarize_directional_margins(
        component_margin_pairs=margin_pairs,
        disturbance_bound=coupled_evaluation.disturbance_bound,
        inbound_loads=network.inbound_bridge_norms,
    )
    transported, directional, global_shifted = _certificate_fractions(
        network=network,
        coupled_profile=coupled_profile,
        component_profiles=component_profiles,
        disturbance_bound=coupled_evaluation.disturbance_bound,
    )
    dimension = sum(network.module_sizes)
    load_sum = math.fsum(network.inbound_bridge_norms)
    point = ComponentProfilePoint(
        trial_seed=network.trial_seed,
        module_sizes=network.module_sizes,
        internal_gain=network.internal_gain,
        maximum_bridge_strength=network.maximum_bridge_strength,
        disturbance_bound=coupled_evaluation.disturbance_bound,
        observed_task_retention=coupled_evaluation.mean_task_retention,
        isolated_task_product=isolated_product,
        task_product_residual=(
            coupled_evaluation.mean_task_retention - isolated_product
        ),
        raw_attractor_fraction=(
            coupled_profile.raw_attractor_count / (2**dimension)
        ),
        coupled_common_certified_fraction=(
            coupled_evaluation.certified_robust_fraction
        ),
        coupled_mean_uniform_margin=(
            coupled_profile.mean_uniform_disturbance_margin
        ),
        full_off_diagonal_infinity_norm=off_diagonal_infinity_norm(
            network.recurrent_weights
        ),
        dimension_normalized_nonnormality=(
            matrix_nonnormality_commutator_norm(network.recurrent_weights)
            / dimension
        ),
        maximum_directional_bridge_norm=max(network.inbound_bridge_norms),
        isolated_component_certified_fraction_product=math.prod(
            evaluation.certified_robust_fraction
            for evaluation in component_evaluations
        ),
        directional_certified_fraction=directional,
        transported_certified_fraction=transported,
        global_shifted_certified_fraction=global_shifted,
        mean_directional_slack=directional_summary.mean_directional_slack,
        minimum_directional_slack=(
            directional_summary.minimum_directional_slack
        ),
        directional_load_imbalance=(
            abs(
                network.inbound_bridge_norms[0]
                - network.inbound_bridge_norms[1]
            )
            / load_sum
            if load_sum > 0.0
            else 0.0
        ),
        module_size_imbalance=(
            abs(network.module_sizes[0] - network.module_sizes[1])
            / dimension
        ),
    )
    if any(
        not math.isfinite(value)
        for value in point.component_feature_row
        + (point.observed_task_retention,)
    ):
        raise RuntimeError("component profileに非有限値があります")
    return point


def _aligned_task_product(
    evaluations: tuple[AlignedDisturbanceEvaluation, AlignedDisturbanceEvaluation],
) -> float:
    first_by_code = {
        value.code: value.task_retention
        for value in evaluations[0].direction_evaluations
    }
    second_by_code = {
        value.code: value.task_retention
        for value in evaluations[1].direction_evaluations
    }
    if first_by_code.keys() != second_by_code.keys():
        raise RuntimeError("component間でdirection codeが一致しません")
    return math.fsum(
        first_by_code[code] * second_by_code[code]
        for code in first_by_code
    ) / len(first_by_code)


def _certificate_fractions(
    *,
    network: AsymmetricModularNetwork,
    coupled_profile: AlignedSignMemoryProfile,
    component_profiles: tuple[AlignedSignMemoryProfile, AlignedSignMemoryProfile],
    disturbance_bound: float,
) -> tuple[float, float, float]:
    first_size, second_size = network.module_sizes
    component_pairs = tuple(
        itertools.product(
            component_profiles[0].orthants,
            component_profiles[1].orthants,
        )
    )
    if len(component_pairs) != len(coupled_profile.orthants):
        raise RuntimeError("component直積とfull orthant数が一致しません")
    transported_count = 0
    directional_count = 0
    global_count = 0
    global_load = max(network.inbound_bridge_norms)
    for coupled_orthant, component_pair in zip(
        coupled_profile.orthants,
        component_pairs,
        strict=True,
    ):
        expected_signs = (
            component_pair[0].attractor_signs
            + component_pair[1].attractor_signs
        )
        boundaries = (
            (component_pair[0].invariant_boundary,) * first_size
            + (component_pair[1].invariant_boundary,) * second_size
        )
        if coupled_orthant.attractor_signs != expected_signs:
            raise RuntimeError("orthant列挙順がcomponent直積と一致しません")
        inside = coupled_orthant.fixed_point_retained and all(
            sign * value >= boundary - 1e-12
            for sign, value, boundary in zip(
                expected_signs,
                coupled_orthant.fixed_point,
                boundaries,
                strict=True,
            )
        )
        if not inside:
            continue
        rectangle = orthant_rectangle_certificate(
            recurrent_weights=network.recurrent_weights,
            attractor_signs=expected_signs,
            lower_boundaries=boundaries,
        )
        transported_count += (
            rectangle.maximum_uniform_disturbance
            >= disturbance_bound - 1e-12
        )
        margins = tuple(
            value.maximum_uniform_disturbance for value in component_pair
        )
        directional_count += all(
            margin >= disturbance_bound + load - 1e-12
            for margin, load in zip(
                margins,
                network.inbound_bridge_norms,
                strict=True,
            )
        )
        global_count += all(
            margin >= disturbance_bound + global_load - 1e-12
            for margin in margins
        )
    orthant_count = len(coupled_profile.orthants)
    return (
        transported_count / orthant_count,
        directional_count / orthant_count,
        global_count / orthant_count,
    )
