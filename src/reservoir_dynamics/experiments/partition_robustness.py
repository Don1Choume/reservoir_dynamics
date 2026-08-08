"""EXP-2026-018のtask-free分割摂動保証phase。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.multicomponent_modular_family import (
    MultiComponentModularNetwork,
    build_multicomponent_modular_network,
)
from reservoir_dynamics.metrics.module_partition import (
    Partition,
    certify_affinity_gap_partition,
    infer_affinity_gap_partition,
    maximum_pair_affinity_change,
    partition_pair_disagreement,
    partitions_equivalent,
)
from reservoir_dynamics.simulation.weight_perturbation import (
    sample_entrywise_bounded_perturbation,
)

EXPERIMENT_ID = "EXP-2026-018"
PREREGISTERED_DEVELOPMENT_SEEDS = tuple(range(2401, 2431))
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(2501, 2531))
PREREGISTERED_MODULE_SIZES = (2, 2, 3)
PREREGISTERED_INTERNAL_GAINS = (0.025, 0.05)
PREREGISTERED_BRIDGE_STRENGTHS = (0.01, 0.02, 0.04)
PREREGISTERED_PERTURBATION_DIRECTION_COUNT = 8
PREREGISTERED_RELATIVE_AMPLITUDES = (0.0, 0.25, 0.5, 0.9, 1.1, 2.0, 4.0)


@dataclass(frozen=True, slots=True)
class PartitionPerturbationPoint:
    """一つのbase network、摂動方向、相対振幅に対する構造観測。"""

    trial_seed: int
    internal_gain: float
    maximum_bridge_strength: float
    perturbation_direction_index: int
    perturbation_seed: int
    relative_amplitude: float
    absolute_amplitude: float
    certified_entrywise_radius: float
    relative_certified_radius: float
    maximum_affinity_change: float
    inference_succeeded: bool
    partition_recovered: bool
    pair_disagreement: float | None
    inferred_component_count: int | None
    inferred_module_sizes: tuple[int, ...] | None
    inferred_partition: Partition | None


@dataclass(frozen=True, slots=True)
class PartitionRobustnessDecisions:
    """事前登録した構造phaseの八判定。"""

    base_recovery: bool
    positive_radius: bool
    affinity_lipschitz: bool
    subradius_exactness: bool
    subradius_pair_distance: bool
    strict_boundary: bool
    task_free: bool
    seed_independence: bool
    all_passed: bool


@dataclass(frozen=True, slots=True)
class PartitionRobustnessResult:
    """task値を含まない分割摂動gridと判定結果。"""

    experiment_id: str
    phase: str
    base_network_count: int
    group_class_counts: tuple[tuple[str, int], ...]
    points: tuple[PartitionPerturbationPoint, ...]
    decisions: PartitionRobustnessDecisions
    task_values_generated: bool


def run_partition_robustness_development(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_DEVELOPMENT_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    perturbation_direction_count: int = PREREGISTERED_PERTURBATION_DIRECTION_COUNT,
    relative_amplitudes: tuple[float, ...] = PREREGISTERED_RELATIVE_AMPLITUDES,
    tolerance: float = 1e-12,
) -> PartitionRobustnessResult:
    """開発seedで構造phaseの実装と識別性を確認する。"""

    return _run_partition_robustness(
        phase="development",
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        perturbation_direction_count=perturbation_direction_count,
        relative_amplitudes=relative_amplitudes,
        tolerance=tolerance,
    )


def run_partition_robustness_confirmation(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    perturbation_direction_count: int = PREREGISTERED_PERTURBATION_DIRECTION_COUNT,
    relative_amplitudes: tuple[float, ...] = PREREGISTERED_RELATIVE_AMPLITUDES,
    tolerance: float = 1e-12,
) -> PartitionRobustnessResult:
    """developmentと分離したseedで固定構造phaseを評価する。"""

    if set(trial_seeds).intersection(PREREGISTERED_DEVELOPMENT_SEEDS):
        raise ValueError("confirmation seedはdevelopment seedと分離してください")
    return _run_partition_robustness(
        phase="confirmation",
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        perturbation_direction_count=perturbation_direction_count,
        relative_amplitudes=relative_amplitudes,
        tolerance=tolerance,
    )


def _run_partition_robustness(
    *,
    phase: str,
    trial_seeds: tuple[int, ...],
    module_sizes: tuple[int, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    perturbation_direction_count: int,
    relative_amplitudes: tuple[float, ...],
    tolerance: float,
) -> PartitionRobustnessResult:
    amplitudes = _validate_grid(
        trial_seeds=trial_seeds,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        perturbation_direction_count=perturbation_direction_count,
        relative_amplitudes=relative_amplitudes,
        tolerance=tolerance,
    )
    networks = tuple(
        build_multicomponent_modular_network(
            trial_seed=trial_seed,
            module_sizes=module_sizes,
            internal_gain=internal_gain,
            maximum_total_bridge_strength=bridge_strength,
        )
        for trial_seed in trial_seeds
        for internal_gain in internal_gains
        for bridge_strength in maximum_bridge_strengths
    )
    certificates = tuple(
        certify_affinity_gap_partition(
            network.recurrent_weights,
            tolerance=tolerance,
        )
        for network in networks
    )
    points = tuple(
        point
        for network, certificate in zip(networks, certificates, strict=True)
        for direction_index in range(perturbation_direction_count)
        for point in _evaluate_direction(
            network=network,
            base_partition=certificate.partition.components,
            certified_radius=certificate.certified_entrywise_radius,
            relative_certified_radius=certificate.relative_certified_radius,
            direction_index=direction_index,
            relative_amplitudes=amplitudes,
            tolerance=tolerance,
        )
    )
    group_counts = _group_class_counts(
        networks=networks,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
    )
    base_recovery = all(
        partitions_equivalent(
            certificate.partition.components,
            network.true_partition,
        )
        for network, certificate in zip(networks, certificates, strict=True)
    )
    positive_radius = all(
        certificate.guaranteed
        and math.isfinite(certificate.certified_entrywise_radius)
        and certificate.certified_entrywise_radius > tolerance
        for certificate in certificates
    )
    subradius = tuple(
        point for point in points if point.relative_amplitude < 1.0
    )
    decision_values = (
        base_recovery,
        positive_radius,
        all(
            point.maximum_affinity_change
            <= point.absolute_amplitude + tolerance
            for point in points
        ),
        bool(subradius) and all(point.partition_recovered for point in subradius),
        bool(subradius)
        and all(point.pair_disagreement == 0.0 for point in subradius),
        all(abs(value - 1.0) > tolerance for value in amplitudes),
        True,
        all(count == len(trial_seeds) for _, count in group_counts),
    )
    decisions = PartitionRobustnessDecisions(
        base_recovery=decision_values[0],
        positive_radius=decision_values[1],
        affinity_lipschitz=decision_values[2],
        subradius_exactness=decision_values[3],
        subradius_pair_distance=decision_values[4],
        strict_boundary=decision_values[5],
        task_free=decision_values[6],
        seed_independence=decision_values[7],
        all_passed=all(decision_values),
    )
    return PartitionRobustnessResult(
        experiment_id=EXPERIMENT_ID,
        phase=phase,
        base_network_count=len(networks),
        group_class_counts=group_counts,
        points=points,
        decisions=decisions,
        task_values_generated=False,
    )


def _evaluate_direction(
    *,
    network: MultiComponentModularNetwork,
    base_partition: Partition,
    certified_radius: float,
    relative_certified_radius: float,
    direction_index: int,
    relative_amplitudes: tuple[float, ...],
    tolerance: float,
) -> tuple[PartitionPerturbationPoint, ...]:
    perturbation_seed = partition_perturbation_seed(
        trial_seed=network.trial_seed,
        internal_gain=network.internal_gain,
        bridge_strength=network.maximum_total_bridge_strength,
        direction_index=direction_index,
    )
    points: list[PartitionPerturbationPoint] = []
    for relative_amplitude in relative_amplitudes:
        absolute_amplitude = relative_amplitude * certified_radius
        perturbed = sample_entrywise_bounded_perturbation(
            network.recurrent_weights,
            maximum_absolute_change=absolute_amplitude,
            random_seed=perturbation_seed,
        )
        try:
            inferred = infer_affinity_gap_partition(
                perturbed,
                tolerance=tolerance,
            ).components
        except ValueError:
            inferred = None
        recovered = inferred is not None and partitions_equivalent(
            inferred,
            base_partition,
        )
        points.append(
            PartitionPerturbationPoint(
                trial_seed=network.trial_seed,
                internal_gain=network.internal_gain,
                maximum_bridge_strength=network.maximum_total_bridge_strength,
                perturbation_direction_index=direction_index,
                perturbation_seed=perturbation_seed,
                relative_amplitude=relative_amplitude,
                absolute_amplitude=absolute_amplitude,
                certified_entrywise_radius=certified_radius,
                relative_certified_radius=relative_certified_radius,
                maximum_affinity_change=maximum_pair_affinity_change(
                    network.recurrent_weights,
                    perturbed,
                ),
                inference_succeeded=inferred is not None,
                partition_recovered=recovered,
                pair_disagreement=(
                    partition_pair_disagreement(base_partition, inferred)
                    if inferred is not None
                    else None
                ),
                inferred_component_count=(len(inferred) if inferred is not None else None),
                inferred_module_sizes=(
                    tuple(sorted(len(component) for component in inferred))
                    if inferred is not None
                    else None
                ),
                inferred_partition=inferred,
            )
        )
    return tuple(points)


def partition_perturbation_seed(
    *,
    trial_seed: int,
    internal_gain: float,
    bridge_strength: float,
    direction_index: int,
) -> int:
    """base networkと方向indexから再利用可能な決定論的seedを作る。"""

    return (
        trial_seed * 1_000_003
        + round(internal_gain * 1_000_000) * 1_009
        + round(bridge_strength * 1_000_000) * 9_176
        + direction_index
    )


def _group_class_counts(
    *,
    networks: tuple[MultiComponentModularNetwork, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            f"{internal_gain.hex()}:{bridge_strength.hex()}",
            len(
                {
                    network.magnitude_fingerprint
                    for network in networks
                    if network.internal_gain == internal_gain
                    and network.maximum_total_bridge_strength == bridge_strength
                }
            ),
        )
        for internal_gain in internal_gains
        for bridge_strength in maximum_bridge_strengths
    )


def _validate_grid(
    *,
    trial_seeds: tuple[int, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    perturbation_direction_count: int,
    relative_amplitudes: tuple[float, ...],
    tolerance: float,
) -> tuple[float, ...]:
    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    if not internal_gains or not maximum_bridge_strengths:
        raise ValueError("gainとbridge strengthは1件以上必要です")
    if (
        not isinstance(perturbation_direction_count, int)
        or isinstance(perturbation_direction_count, bool)
        or perturbation_direction_count < 1
    ):
        raise ValueError("perturbation_direction_countは正整数にしてください")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")
    amplitudes = tuple(float(value) for value in relative_amplitudes)
    if not amplitudes or any(
        isinstance(value, bool) or not math.isfinite(value) or value < 0.0
        for value in relative_amplitudes
    ) or len(set(amplitudes)) != len(amplitudes):
        raise ValueError("relative amplitudeは重複しない有限非負値にしてください")
    if (
        not any(value < 1.0 for value in amplitudes)
        or not any(value > 1.0 for value in amplitudes)
    ):
        raise ValueError("保証半径の内側と外側を両方指定してください")
    return amplitudes
