"""空間局所gateを持つ双安定core–reserve系の一条件評価。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from reservoir_dynamics.experiments.spatial_field_network import (
    SpatialCoreReserveNetwork,
    SpatialTrialSignals,
    build_spatial_core_reserve_network,
    generate_spatial_trial_signals,
)
from reservoir_dynamics.metrics.linear_memory import linear_memory_curve
from reservoir_dynamics.systems.spatial_modulation import (
    DiffusiveModulationField,
    chain_diffusion_kernel,
)
from reservoir_dynamics.systems.tanh_rnn import Matrix, Vector
from reservoir_dynamics.theory.bistable_margin import (
    bistable_tanh_certificate,
    positive_bistable_fixed_point,
)
from reservoir_dynamics.theory.spatial_core_protection import (
    bistable_coordinate_protection,
    energy_matched_global_weights,
    matrix_frobenius_distance_squared,
    row_gated_matrix,
)


@dataclass(frozen=True, slots=True)
class SpatialPolicyTrace:
    """一つのcore orthantに対するpolicy軌道。"""

    core_states: tuple[Vector, ...]
    reserve_states: tuple[Vector, ...]
    field_states: tuple[Vector, ...]
    intervention_energies: Vector
    certified_challenges: int
    certificate_challenges: int
    certificate_violations: int
    safe_box_retained: bool
    sign_retained: bool
    raw_feedback_loads: Vector
    gated_feedback_loads: Vector


@dataclass(frozen=True, slots=True)
class SpatialFieldConditionEvaluation:
    """一seed・feedback・noise条件の三policy比較。"""

    trial_seed: int
    core_dimension: int
    reserve_dimension: int
    feedback_gain: float
    disturbance_bound: float
    local_safe_box_retention: float
    ungated_safe_box_retention: float
    global_safe_box_retention: float
    local_sign_retention: float
    ungated_sign_retention: float
    global_sign_retention: float
    local_reserve_capacity: float
    ungated_reserve_capacity: float
    global_reserve_capacity: float
    certified_challenge_fraction: float
    certificate_violation_count: int
    maximum_energy_match_error: float
    minimum_field_value: float
    maximum_field_value: float
    mean_field_spatial_variance: float
    mean_raw_feedback_load: float
    mean_gated_feedback_load: float
    maximum_gated_load_ratio: float


def evaluate_spatial_field_condition(
    *,
    trial_seed: int,
    feedback_gain: float,
    disturbance_bound: float,
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
    diffusion_rate: float,
    source_rate: float,
    minimum_gate: float,
) -> SpatialFieldConditionEvaluation:
    """全core orthantでlocal、ungated、energy-matched globalを比較する。"""

    network = build_spatial_core_reserve_network(
        trial_seed=trial_seed,
        feedback_gain=feedback_gain,
    )
    critical_forcing = bistable_tanh_certificate(
        network.core_recurrent_gain,
    ).critical_forcing
    if disturbance_bound >= critical_forcing:
        raise ValueError(
            "disturbance_boundはcoreのcritical forcing未満にしてください"
        )
    steps = max(washout, max_delay) + training_steps + testing_steps
    signals = generate_spatial_trial_signals(
        trial_seed=trial_seed,
        steps=steps,
        core_dimension=network.core_dimension,
        reserve_dimension=network.reserve_dimension,
        disturbance_bound=disturbance_bound,
    )
    field = DiffusiveModulationField(
        diffusion_kernel=chain_diffusion_kernel(network.state_dimension),
        diffusion_rate=diffusion_rate,
        source_rate=source_rate,
        minimum_gate=minimum_gate,
    )
    positive_fixed_point = positive_bistable_fixed_point(
        network.core_recurrent_gain,
    )
    sign_patterns = tuple(
        tuple(float(sign) for sign in signs)
        for signs in itertools.product(
            (-1, 1),
            repeat=network.core_dimension,
        )
    )

    local_traces: list[SpatialPolicyTrace] = []
    ungated_traces: list[SpatialPolicyTrace] = []
    global_traces: list[SpatialPolicyTrace] = []
    maximum_energy_error = 0.0
    for signs in sign_patterns:
        initial_core = tuple(sign * positive_fixed_point for sign in signs)
        local_trace = _simulate_local_policy(
            network=network,
            field=field,
            signals=signals,
            signs=signs,
            initial_core=initial_core,
            disturbance_bound=disturbance_bound,
        )
        ungated_trace = _simulate_matrix_policy(
            network=network,
            signals=signals,
            signs=signs,
            initial_core=initial_core,
            recurrent_schedule=(network.full_recurrent_weights,) * steps,
        )
        global_schedule = tuple(
            energy_matched_global_weights(
                network.full_recurrent_weights,
                target_energy=energy,
            )
            for energy in local_trace.intervention_energies
        )
        global_trace = _simulate_matrix_policy(
            network=network,
            signals=signals,
            signs=signs,
            initial_core=initial_core,
            recurrent_schedule=global_schedule,
        )
        maximum_energy_error = max(
            maximum_energy_error,
            *(
                abs(
                    matrix_frobenius_distance_squared(
                        network.full_recurrent_weights,
                        weights,
                    )
                    - target
                )
                for weights, target in zip(
                    global_schedule,
                    local_trace.intervention_energies,
                    strict=True,
                )
            ),
        )
        local_traces.append(local_trace)
        ungated_traces.append(ungated_trace)
        global_traces.append(global_trace)

    reference_index = sign_patterns.index((1.0,) * network.core_dimension)
    local_reference = local_traces[reference_index]
    ungated_reference = ungated_traces[reference_index]
    global_reference = global_traces[reference_index]
    local_fields = tuple(
        value
        for trace in local_traces
        for state in trace.field_states
        for value in state
    )
    field_variances = tuple(
        _population_variance(state)
        for trace in local_traces
        for state in trace.field_states
    )
    certified_count = sum(
        trace.certified_challenges for trace in local_traces
    )
    certificate_count = sum(
        trace.certificate_challenges for trace in local_traces
    )
    return SpatialFieldConditionEvaluation(
        trial_seed=trial_seed,
        core_dimension=network.core_dimension,
        reserve_dimension=network.reserve_dimension,
        feedback_gain=feedback_gain,
        disturbance_bound=disturbance_bound,
        local_safe_box_retention=_retention_fraction(
            local_traces,
            safe_box=True,
        ),
        ungated_safe_box_retention=_retention_fraction(
            ungated_traces,
            safe_box=True,
        ),
        global_safe_box_retention=_retention_fraction(
            global_traces,
            safe_box=True,
        ),
        local_sign_retention=_retention_fraction(
            local_traces,
            safe_box=False,
        ),
        ungated_sign_retention=_retention_fraction(
            ungated_traces,
            safe_box=False,
        ),
        global_sign_retention=_retention_fraction(
            global_traces,
            safe_box=False,
        ),
        local_reserve_capacity=_reserve_capacity(
            trace=local_reference,
            signals=signals,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            max_delay=max_delay,
            ridge=ridge,
        ),
        ungated_reserve_capacity=_reserve_capacity(
            trace=ungated_reference,
            signals=signals,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            max_delay=max_delay,
            ridge=ridge,
        ),
        global_reserve_capacity=_reserve_capacity(
            trace=global_reference,
            signals=signals,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            max_delay=max_delay,
            ridge=ridge,
        ),
        certified_challenge_fraction=(
            certified_count / certificate_count
            if certificate_count > 0
            else 0.0
        ),
        certificate_violation_count=sum(
            trace.certificate_violations for trace in local_traces
        ),
        maximum_energy_match_error=maximum_energy_error,
        minimum_field_value=min(local_fields),
        maximum_field_value=max(local_fields),
        mean_field_spatial_variance=(
            math.fsum(field_variances) / len(field_variances)
        ),
        mean_raw_feedback_load=_mean_trace_load(
            local_traces,
            gated=False,
        ),
        mean_gated_feedback_load=_mean_trace_load(
            local_traces,
            gated=True,
        ),
        maximum_gated_load_ratio=max(
            (
                gated / raw if raw > 0.0 else 0.0
                for trace in local_traces
                for raw, gated in zip(
                    trace.raw_feedback_loads,
                    trace.gated_feedback_loads,
                    strict=True,
                )
            ),
            default=0.0,
        ),
    )


def _simulate_local_policy(
    *,
    network: SpatialCoreReserveNetwork,
    field: DiffusiveModulationField,
    signals: SpatialTrialSignals,
    signs: Vector,
    initial_core: Vector,
    disturbance_bound: float,
) -> SpatialPolicyTrace:
    core_state = initial_core
    reserve_state = (0.0,) * network.reserve_dimension
    field_state = (0.0,) * network.state_dimension
    core_states = [core_state]
    reserve_states = [reserve_state]
    field_states = [field_state]
    intervention_energies: list[float] = []
    raw_loads: list[float] = []
    gated_loads: list[float] = []
    certificate_challenges = 0
    certified_challenges = 0
    certificate_violations = 0
    boundary = bistable_tanh_certificate(
        network.core_recurrent_gain,
    ).invariant_boundary
    safe_box_retained = True
    sign_retained = True
    full_weights = network.full_recurrent_weights

    for novel_input, core_noise, reserve_noise in zip(
        signals.novel_inputs,
        signals.core_disturbances,
        signals.reserve_disturbances,
        strict=True,
    ):
        raw_feedback = _matrix_vector_product(
            network.reserve_to_core_weights,
            reserve_state,
        )
        risk_scale = max(
            1e-12,
            bistable_tanh_certificate(
                network.core_recurrent_gain,
            ).critical_forcing
            - disturbance_bound,
        )
        source = tuple(
            min(1.0, abs(load) / risk_scale) for load in raw_feedback
        ) + (0.0,) * network.reserve_dimension
        gates = field.gates(field_state)[: network.core_dimension]
        gated_feedback_weights = row_gated_matrix(
            network.reserve_to_core_weights,
            gates,
        )
        gated_feedback = _matrix_vector_product(
            gated_feedback_weights,
            reserve_state,
        )
        local_weights = _replace_feedback_block(
            full_weights=full_weights,
            core_dimension=network.core_dimension,
            gated_feedback=gated_feedback_weights,
        )
        intervention_energies.append(
            matrix_frobenius_distance_squared(full_weights, local_weights)
        )
        raw_loads.append(max(abs(value) for value in raw_feedback))
        gated_loads.append(max(abs(value) for value in gated_feedback))
        certificate = bistable_coordinate_protection(
            recurrent_gains=(network.core_recurrent_gain,)
            * network.core_dimension,
            feedback_loads=gated_feedback,
            disturbance_bounds=(disturbance_bound,)
            * network.core_dimension,
        )
        next_core, next_reserve = _step_network(
            recurrent_weights=local_weights,
            core_dimension=network.core_dimension,
            reserve_input_weights=network.reserve_input_weights,
            core_state=core_state,
            reserve_state=reserve_state,
            novel_input=novel_input,
            core_disturbance=core_noise,
            reserve_disturbance=reserve_noise,
        )
        for coordinate, certified in enumerate(
            certificate.certified_coordinates
        ):
            current_safe = signs[coordinate] * core_state[coordinate] >= boundary
            if current_safe:
                certificate_challenges += 1
                if certified:
                    certified_challenges += 1
                    if signs[coordinate] * next_core[coordinate] < boundary - 1e-12:
                        certificate_violations += 1
        safe_box_retained = safe_box_retained and _state_in_safe_box(
            next_core,
            signs,
            boundary,
        )
        sign_retained = sign_retained and all(
            sign * value > 0.0
            for sign, value in zip(signs, next_core, strict=True)
        )
        core_state = next_core
        reserve_state = next_reserve
        field_state = field.step(state=field_state, source=source)
        core_states.append(core_state)
        reserve_states.append(reserve_state)
        field_states.append(field_state)
    return SpatialPolicyTrace(
        core_states=tuple(core_states),
        reserve_states=tuple(reserve_states),
        field_states=tuple(field_states),
        intervention_energies=tuple(intervention_energies),
        certified_challenges=certified_challenges,
        certificate_challenges=certificate_challenges,
        certificate_violations=certificate_violations,
        safe_box_retained=safe_box_retained,
        sign_retained=sign_retained,
        raw_feedback_loads=tuple(raw_loads),
        gated_feedback_loads=tuple(gated_loads),
    )


def _simulate_matrix_policy(
    *,
    network: SpatialCoreReserveNetwork,
    signals: SpatialTrialSignals,
    signs: Vector,
    initial_core: Vector,
    recurrent_schedule: tuple[Matrix, ...],
) -> SpatialPolicyTrace:
    core_state = initial_core
    reserve_state = (0.0,) * network.reserve_dimension
    core_states = [core_state]
    reserve_states = [reserve_state]
    boundary = bistable_tanh_certificate(
        network.core_recurrent_gain,
    ).invariant_boundary
    safe_box_retained = True
    sign_retained = True
    for weights, novel_input, core_noise, reserve_noise in zip(
        recurrent_schedule,
        signals.novel_inputs,
        signals.core_disturbances,
        signals.reserve_disturbances,
        strict=True,
    ):
        core_state, reserve_state = _step_network(
            recurrent_weights=weights,
            core_dimension=network.core_dimension,
            reserve_input_weights=network.reserve_input_weights,
            core_state=core_state,
            reserve_state=reserve_state,
            novel_input=novel_input,
            core_disturbance=core_noise,
            reserve_disturbance=reserve_noise,
        )
        core_states.append(core_state)
        reserve_states.append(reserve_state)
        safe_box_retained = safe_box_retained and _state_in_safe_box(
            core_state,
            signs,
            boundary,
        )
        sign_retained = sign_retained and all(
            sign * value > 0.0
            for sign, value in zip(signs, core_state, strict=True)
        )
    return SpatialPolicyTrace(
        core_states=tuple(core_states),
        reserve_states=tuple(reserve_states),
        field_states=(),
        intervention_energies=(),
        certified_challenges=0,
        certificate_challenges=0,
        certificate_violations=0,
        safe_box_retained=safe_box_retained,
        sign_retained=sign_retained,
        raw_feedback_loads=(),
        gated_feedback_loads=(),
    )


def _step_network(
    *,
    recurrent_weights: Matrix,
    core_dimension: int,
    reserve_input_weights: Vector,
    core_state: Vector,
    reserve_state: Vector,
    novel_input: float,
    core_disturbance: Vector,
    reserve_disturbance: Vector,
) -> tuple[Vector, Vector]:
    state = core_state + reserve_state
    recurrent = _matrix_vector_product(recurrent_weights, state)
    next_core = tuple(
        math.tanh(value + disturbance)
        for value, disturbance in zip(
            recurrent[:core_dimension],
            core_disturbance,
            strict=True,
        )
    )
    next_reserve = tuple(
        math.tanh(
            value + input_weight * novel_input + disturbance
        )
        for value, input_weight, disturbance in zip(
            recurrent[core_dimension:],
            reserve_input_weights,
            reserve_disturbance,
            strict=True,
        )
    )
    return next_core, next_reserve


def _reserve_capacity(
    *,
    trace: SpatialPolicyTrace,
    signals: SpatialTrialSignals,
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
) -> float:
    return linear_memory_curve(
        states=trace.reserve_states,
        inputs=signals.novel_inputs,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    ).total_capacity


def _replace_feedback_block(
    *,
    full_weights: Matrix,
    core_dimension: int,
    gated_feedback: Matrix,
) -> Matrix:
    return tuple(
        full_weights[row][:core_dimension] + gated_feedback[row]
        if row < core_dimension
        else full_weights[row]
        for row in range(len(full_weights))
    )


def _matrix_vector_product(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        math.fsum(
            value * state_value
            for value, state_value in zip(row, vector, strict=True)
        )
        for row in matrix
    )


def _state_in_safe_box(
    state: Vector,
    signs: Vector,
    boundary: float,
) -> bool:
    return all(
        sign * value >= boundary - 1e-12
        for sign, value in zip(signs, state, strict=True)
    )


def _retention_fraction(
    traces: list[SpatialPolicyTrace],
    *,
    safe_box: bool,
) -> float:
    retained = sum(
        trace.safe_box_retained if safe_box else trace.sign_retained
        for trace in traces
    )
    return retained / len(traces)


def _mean_trace_load(
    traces: list[SpatialPolicyTrace],
    *,
    gated: bool,
) -> float:
    values = tuple(
        value
        for trace in traces
        for value in (
            trace.gated_feedback_loads
            if gated
            else trace.raw_feedback_loads
        )
    )
    return math.fsum(values) / len(values)


def _population_variance(values: Vector) -> float:
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)

