"""EXP-2026-019の分割誤差から機能解析誤差への接続実験。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.component_predictor import (
    NamedComponentPredictor,
    fixed_predictor_feature_shift_bound,
    predictor_payload_sha256,
    predictors_to_payload,
)
from reservoir_dynamics.experiments.multicomponent_modular_family import (
    MultiComponentModularNetwork,
    build_multicomponent_modular_network,
)
from reservoir_dynamics.experiments.multicomponent_profile import (
    MultiComponentProfilePoint,
    evaluate_multicomponent_partitions,
)
from reservoir_dynamics.experiments.partition_robustness import (
    partition_perturbation_seed,
)
from reservoir_dynamics.metrics.module_partition import (
    Partition,
    certify_affinity_gap_partition,
    infer_affinity_gap_partition,
    partition_pair_disagreement,
    partitions_equivalent,
)
from reservoir_dynamics.simulation.weight_perturbation import (
    sample_entrywise_bounded_perturbation,
)
from reservoir_dynamics.theory.orthant_box import Matrix

EXPERIMENT_ID = "EXP-2026-019"
PREREGISTERED_DEVELOPMENT_SEEDS = tuple(range(2601, 2631))
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(2701, 2731))
PREREGISTERED_MODULE_SIZES = (2, 2, 3)
PREREGISTERED_INTERNAL_GAINS = (0.025, 0.05)
PREREGISTERED_BRIDGE_STRENGTHS = (0.01, 0.02, 0.04)
PREREGISTERED_RELATIVE_AMPLITUDES = (0.9, 2.0, 4.0)
PREREGISTERED_DISTURBANCE_BOUNDS = (0.12, 0.20)
PREREGISTERED_MODEL_SHA256 = (
    "db0b50a648fb085ca687922a531fab5482af2a134bd01eefc3efe3dd85675a01"
)
PREREGISTERED_PERTURBATION_DIRECTION_INDEX = 0


@dataclass(frozen=True, slots=True)
class PartitionFunctionalPerturbation:
    """task生成前に固定する一つの摂動後partition観測。"""

    trial_seed: int
    internal_gain: float
    maximum_bridge_strength: float
    relative_amplitude: float
    absolute_amplitude: float
    certified_entrywise_radius: float
    perturbation_seed: int
    inference_succeeded: bool
    partition_recovered: bool
    pair_disagreement: float | None
    inferred_partition: Partition | None


@dataclass(frozen=True, slots=True)
class FixedPredictionImpact:
    """一固定modelに対するpartition変更の予測影響。"""

    model_name: str
    oracle_prediction: float
    inferred_prediction: float
    absolute_prediction_shift: float
    feature_shift_bound: float
    oracle_absolute_error: float
    inferred_absolute_error: float
    absolute_error_penalty: float


@dataclass(frozen=True, slots=True)
class PartitionFunctionalImpactPoint:
    """同じ摂動後networkを二partitionで解析した一外乱点。"""

    trial_seed: int
    internal_gain: float
    maximum_bridge_strength: float
    relative_amplitude: float
    disturbance_bound: float
    pair_disagreement: float
    observed_task_retention: float
    shared_target_difference: float
    oracle_module_sizes: tuple[int, ...]
    inferred_module_sizes: tuple[int, ...]
    oracle_component_features: tuple[float, ...]
    inferred_component_features: tuple[float, ...]
    maximum_component_feature_difference: float
    component_feature_rmse: float
    oracle_directional_certificate: float
    inferred_directional_certificate: float
    oracle_transported_certificate: float
    inferred_transported_certificate: float
    oracle_global_certificate: float
    inferred_global_certificate: float
    inferred_factorized_certificate: float
    inferred_enumerated_certificate: float
    oracle_local_orthant_count: int
    inferred_local_orthant_count: int
    local_orthant_count_ratio: float
    prediction_impacts: tuple[FixedPredictionImpact, ...]

    @property
    def component_prediction_shift(self) -> float:
        return self.prediction_impacts[0].absolute_prediction_shift

    @property
    def component_prediction_shift_bound(self) -> float:
        return self.prediction_impacts[0].feature_shift_bound


@dataclass(frozen=True, slots=True)
class PartitionFunctionalDecisions:
    """development前に固定した九判定。"""

    fixed_model: bool
    subradius_recovery: bool
    subradius_profile_identity: bool
    shared_target_identity: bool
    prediction_shift_bound: bool
    factorized_exactness: bool
    certificate_chain: bool
    seed_independence: bool
    task_free_inference: bool
    all_passed: bool


@dataclass(frozen=True, slots=True)
class PartitionFunctionalImpactResult:
    """構造観測とpartition条件付き機能比較を分離した結果。"""

    experiment_id: str
    phase: str
    model_sha256: str
    base_network_count: int
    group_class_counts: tuple[tuple[str, int], ...]
    perturbations: tuple[PartitionFunctionalPerturbation, ...]
    points: tuple[PartitionFunctionalImpactPoint, ...]
    decisions: PartitionFunctionalDecisions
    inference_completed_before_task: bool


@dataclass(frozen=True, slots=True)
class _PreparedPerturbation:
    network: MultiComponentModularNetwork
    weights: Matrix
    observation: PartitionFunctionalPerturbation


def run_partition_functional_development(
    *,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...] = PREREGISTERED_DEVELOPMENT_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    relative_amplitudes: tuple[float, ...] = PREREGISTERED_RELATIVE_AMPLITUDES,
    disturbance_bounds: tuple[float, ...] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 80,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    tolerance: float = 1e-12,
) -> PartitionFunctionalImpactResult:
    """development seedで機能影響の識別性を測る。"""

    return _run_partition_functional_impact(
        phase="development",
        fitted_models=fitted_models,
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        relative_amplitudes=relative_amplitudes,
        disturbance_bounds=disturbance_bounds,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        tolerance=tolerance,
    )


def run_partition_functional_confirmation(
    *,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
    relative_amplitudes: tuple[float, ...] = PREREGISTERED_RELATIVE_AMPLITUDES,
    disturbance_bounds: tuple[float, ...] = PREREGISTERED_DISTURBANCE_BOUNDS,
    diagonal_gain: float = 1.5,
    task_steps: int = 80,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    tolerance: float = 1e-12,
) -> PartitionFunctionalImpactResult:
    """developmentと分離したseedへ固定手順を適用する。"""

    if set(trial_seeds).intersection(PREREGISTERED_DEVELOPMENT_SEEDS):
        raise ValueError("confirmation seedはdevelopment seedと分離してください")
    return _run_partition_functional_impact(
        phase="confirmation",
        fitted_models=fitted_models,
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        relative_amplitudes=relative_amplitudes,
        disturbance_bounds=disturbance_bounds,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        tolerance=tolerance,
    )


def _run_partition_functional_impact(
    *,
    phase: str,
    fitted_models: tuple[NamedComponentPredictor, ...],
    trial_seeds: tuple[int, ...],
    module_sizes: tuple[int, ...],
    internal_gains: tuple[float, ...],
    maximum_bridge_strengths: tuple[float, ...],
    relative_amplitudes: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
    tolerance: float,
) -> PartitionFunctionalImpactResult:
    amplitudes = _validate_grid(
        trial_seeds=trial_seeds,
        internal_gains=internal_gains,
        maximum_bridge_strengths=maximum_bridge_strengths,
        relative_amplitudes=relative_amplitudes,
        disturbance_bounds=disturbance_bounds,
        tolerance=tolerance,
    )
    model_sha256 = predictor_payload_sha256(
        predictors_to_payload(fitted_models)
    )
    if model_sha256 != PREREGISTERED_MODEL_SHA256:
        raise ValueError("固定model hashが事前登録値と一致しません")
    networks = tuple(
        build_multicomponent_modular_network(
            trial_seed=trial_seed,
            module_sizes=module_sizes,
            internal_gain=internal_gain,
            maximum_total_bridge_strength=bridge_strength,
            diagonal_gain=diagonal_gain,
        )
        for trial_seed in trial_seeds
        for internal_gain in internal_gains
        for bridge_strength in maximum_bridge_strengths
    )
    # 全partition推定を先に完了し、task値による解析条件の選択を防ぐ。
    prepared = tuple(
        _prepare_perturbation(network, relative_amplitude, tolerance)
        for network in networks
        for relative_amplitude in amplitudes
    )
    points = tuple(
        point
        for value in prepared
        if value.observation.inferred_partition is not None
        for point in _evaluate_prepared(
            prepared=value,
            fitted_models=fitted_models,
            disturbance_bounds=disturbance_bounds,
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
    )
    group_counts = _group_class_counts(
        networks,
        internal_gains,
        maximum_bridge_strengths,
    )
    observations = tuple(value.observation for value in prepared)
    decisions = _decisions(
        model_sha256=model_sha256,
        observations=observations,
        points=points,
        group_counts=group_counts,
        expected_group_count=len(trial_seeds),
        tolerance=tolerance,
    )
    return PartitionFunctionalImpactResult(
        experiment_id=EXPERIMENT_ID,
        phase=phase,
        model_sha256=model_sha256,
        base_network_count=len(networks),
        group_class_counts=group_counts,
        perturbations=observations,
        points=points,
        decisions=decisions,
        inference_completed_before_task=True,
    )


def _prepare_perturbation(
    network: MultiComponentModularNetwork,
    relative_amplitude: float,
    tolerance: float,
) -> _PreparedPerturbation:
    robustness = certify_affinity_gap_partition(
        network.recurrent_weights,
        tolerance=tolerance,
    )
    absolute_amplitude = (
        relative_amplitude * robustness.certified_entrywise_radius
    )
    perturbation_seed = partition_perturbation_seed(
        trial_seed=network.trial_seed,
        internal_gain=network.internal_gain,
        bridge_strength=network.maximum_total_bridge_strength,
        direction_index=PREREGISTERED_PERTURBATION_DIRECTION_INDEX,
    )
    weights = sample_entrywise_bounded_perturbation(
        network.recurrent_weights,
        maximum_absolute_change=absolute_amplitude,
        random_seed=perturbation_seed,
    )
    try:
        inferred = infer_affinity_gap_partition(
            weights,
            tolerance=tolerance,
        ).components
    except ValueError:
        inferred = None
    return _PreparedPerturbation(
        network=network,
        weights=weights,
        observation=PartitionFunctionalPerturbation(
            trial_seed=network.trial_seed,
            internal_gain=network.internal_gain,
            maximum_bridge_strength=network.maximum_total_bridge_strength,
            relative_amplitude=relative_amplitude,
            absolute_amplitude=absolute_amplitude,
            certified_entrywise_radius=robustness.certified_entrywise_radius,
            perturbation_seed=perturbation_seed,
            inference_succeeded=inferred is not None,
            partition_recovered=(
                inferred is not None
                and partitions_equivalent(inferred, network.true_partition)
            ),
            pair_disagreement=(
                partition_pair_disagreement(network.true_partition, inferred)
                if inferred is not None
                else None
            ),
            inferred_partition=inferred,
        ),
    )


def _evaluate_prepared(
    *,
    prepared: _PreparedPerturbation,
    fitted_models: tuple[NamedComponentPredictor, ...],
    disturbance_bounds: tuple[float, ...],
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[PartitionFunctionalImpactPoint, ...]:
    inferred = prepared.observation.inferred_partition
    if inferred is None:
        return ()
    oracle_profiles, inferred_profiles = evaluate_multicomponent_partitions(
        network=prepared.network,
        recurrent_weights=prepared.weights,
        partitions=(prepared.network.true_partition, inferred),
        disturbance_bounds=disturbance_bounds,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    return tuple(
        _functional_point(
            prepared=prepared,
            oracle=oracle,
            inferred=inferred_point,
            fitted_models=fitted_models,
        )
        for oracle, inferred_point in zip(
            oracle_profiles,
            inferred_profiles,
            strict=True,
        )
    )


def _functional_point(
    *,
    prepared: _PreparedPerturbation,
    oracle: MultiComponentProfilePoint,
    inferred: MultiComponentProfilePoint,
    fitted_models: tuple[NamedComponentPredictor, ...],
) -> PartitionFunctionalImpactPoint:
    feature_differences = tuple(
        abs(first - second)
        for first, second in zip(
            oracle.component_feature_row,
            inferred.component_feature_row,
            strict=True,
        )
    )
    prediction_impacts = tuple(
        _prediction_impact(model, oracle, inferred)
        for model in fitted_models
    )
    return PartitionFunctionalImpactPoint(
        trial_seed=prepared.network.trial_seed,
        internal_gain=prepared.network.internal_gain,
        maximum_bridge_strength=prepared.network.maximum_total_bridge_strength,
        relative_amplitude=prepared.observation.relative_amplitude,
        disturbance_bound=oracle.disturbance_bound,
        pair_disagreement=prepared.observation.pair_disagreement or 0.0,
        observed_task_retention=oracle.observed_task_retention,
        shared_target_difference=abs(
            oracle.observed_task_retention - inferred.observed_task_retention
        ),
        oracle_module_sizes=oracle.module_sizes,
        inferred_module_sizes=inferred.module_sizes,
        oracle_component_features=oracle.component_feature_row,
        inferred_component_features=inferred.component_feature_row,
        maximum_component_feature_difference=max(feature_differences),
        component_feature_rmse=math.sqrt(
            math.fsum(value * value for value in feature_differences)
            / len(feature_differences)
        ),
        oracle_directional_certificate=oracle.directional_certified_fraction,
        inferred_directional_certificate=inferred.directional_certified_fraction,
        oracle_transported_certificate=oracle.transported_certified_fraction,
        inferred_transported_certificate=inferred.transported_certified_fraction,
        oracle_global_certificate=oracle.global_shifted_certified_fraction,
        inferred_global_certificate=inferred.global_shifted_certified_fraction,
        inferred_factorized_certificate=(
            inferred.factorized_directional_certified_fraction
        ),
        inferred_enumerated_certificate=(
            inferred.enumerated_directional_certified_fraction
        ),
        oracle_local_orthant_count=oracle.local_orthant_count,
        inferred_local_orthant_count=inferred.local_orthant_count,
        local_orthant_count_ratio=(
            inferred.local_orthant_count / oracle.local_orthant_count
        ),
        prediction_impacts=prediction_impacts,
    )


def _prediction_impact(
    model: NamedComponentPredictor,
    oracle: MultiComponentProfilePoint,
    inferred: MultiComponentProfilePoint,
) -> FixedPredictionImpact:
    oracle_prediction = model.predict(oracle)
    inferred_prediction = model.predict(inferred)
    oracle_error = abs(oracle.observed_task_retention - oracle_prediction)
    inferred_error = abs(inferred.observed_task_retention - inferred_prediction)
    return FixedPredictionImpact(
        model_name=model.name,
        oracle_prediction=oracle_prediction,
        inferred_prediction=inferred_prediction,
        absolute_prediction_shift=abs(oracle_prediction - inferred_prediction),
        feature_shift_bound=fixed_predictor_feature_shift_bound(
            model,
            oracle,
            inferred,
        ),
        oracle_absolute_error=oracle_error,
        inferred_absolute_error=inferred_error,
        absolute_error_penalty=inferred_error - oracle_error,
    )


def _decisions(
    *,
    model_sha256: str,
    observations: tuple[PartitionFunctionalPerturbation, ...],
    points: tuple[PartitionFunctionalImpactPoint, ...],
    group_counts: tuple[tuple[str, int], ...],
    expected_group_count: int,
    tolerance: float,
) -> PartitionFunctionalDecisions:
    subradius_observations = tuple(
        value for value in observations if value.relative_amplitude < 1.0
    )
    subradius_points = tuple(
        point for point in points if point.relative_amplitude < 1.0
    )
    values = (
        model_sha256 == PREREGISTERED_MODEL_SHA256,
        bool(subradius_observations)
        and all(value.partition_recovered for value in subradius_observations),
        bool(subradius_points)
        and all(
            point.maximum_component_feature_difference <= tolerance
            and all(
                impact.absolute_prediction_shift <= tolerance
                for impact in point.prediction_impacts
            )
            for point in subradius_points
        ),
        all(point.shared_target_difference <= tolerance for point in points),
        all(
            impact.absolute_prediction_shift
            <= impact.feature_shift_bound + tolerance
            for point in points
            for impact in point.prediction_impacts
        ),
        all(
            abs(
                point.inferred_factorized_certificate
                - point.inferred_enumerated_certificate
            ) <= tolerance
            for point in points
        ),
        all(
            point.observed_task_retention + tolerance
            >= point.inferred_transported_certificate
            and point.inferred_transported_certificate + tolerance
            >= point.inferred_directional_certificate
            and point.inferred_directional_certificate + tolerance
            >= point.inferred_global_certificate
            for point in points
        ),
        all(count == expected_group_count for _, count in group_counts),
        True,
    )
    return PartitionFunctionalDecisions(
        fixed_model=values[0],
        subradius_recovery=values[1],
        subradius_profile_identity=values[2],
        shared_target_identity=values[3],
        prediction_shift_bound=values[4],
        factorized_exactness=values[5],
        certificate_chain=values[6],
        seed_independence=values[7],
        task_free_inference=values[8],
        all_passed=all(values),
    )


def _group_class_counts(
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
    relative_amplitudes: tuple[float, ...],
    disturbance_bounds: tuple[float, ...],
    tolerance: float,
) -> tuple[float, ...]:
    if len(trial_seeds) < 2 or len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("trial_seedsは重複しない2個以上にしてください")
    if not internal_gains or not maximum_bridge_strengths:
        raise ValueError("gainとbridge strengthは1件以上必要です")
    amplitudes = tuple(float(value) for value in relative_amplitudes)
    if (
        not amplitudes
        or len(set(amplitudes)) != len(amplitudes)
        or any(not math.isfinite(value) or value < 0.0 for value in amplitudes)
        or not any(value < 1.0 for value in amplitudes)
        or not any(value > 1.0 for value in amplitudes)
    ):
        raise ValueError("relative amplitudeは保証半径の内外を含めてください")
    if not disturbance_bounds or any(
        not math.isfinite(value) or value < 0.0
        for value in disturbance_bounds
    ):
        raise ValueError("disturbance boundは有限非負値にしてください")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("toleranceは有限の非負値にしてください")
    return amplitudes
