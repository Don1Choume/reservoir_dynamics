"""core–reserve保護実験の一seed評価。"""

from __future__ import annotations

import random
from dataclasses import dataclass

from reservoir_dynamics.metrics.linear_memory import (
    linear_memory_curve,
    shared_readout_memory_capacity,
)
from reservoir_dynamics.systems.core_reserve_tanh import CoreReserveTanhRnn
from reservoir_dynamics.systems.tanh_rnn import Matrix, TanhRnnReservoir, Vector
from reservoir_dynamics.theory.core_protection import (
    core_deviation_bound_curve,
)


@dataclass(frozen=True, slots=True)
class AdaptationChoice:
    """calibration入力で選択したreserve parameter。"""

    recurrent_gain: float
    input_gain: float
    calibration_novel_capacity: float


@dataclass(frozen=True, slots=True)
class ProtectedUpdatePoint:
    """reserve-only更新後のcore保持と新規容量。"""

    trial_seed: int
    feedback_gain: float
    selected_recurrent_gain: float
    selected_input_gain: float
    pre_novel_capacity: float
    post_novel_capacity: float
    core_reference_capacity: float
    core_worst_capacity: float
    core_retention: float
    max_core_deviation: float
    max_core_deviation_bound: float
    bound_satisfied: bool


@dataclass(frozen=True, slots=True)
class EntangledUpdatePoint:
    """同数の可塑parameter枠をcore側へ置いた干渉対照。"""

    trial_seed: int
    selected_recurrent_gain: float
    selected_input_gain: float
    post_novel_capacity: float
    core_reference_capacity: float
    core_worst_capacity: float
    core_retention: float
    max_core_deviation: float


def select_adaptation(
    *,
    core_dimension: int,
    core_direction: Vector,
    reserve_direction: Vector,
    core_recurrent_gain: float,
    core_input_gain: float,
    candidates: tuple[tuple[float, float], ...],
    inputs: tuple[Vector, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
) -> AdaptationChoice:
    """calibration新規容量を最大化するreserve gain対を選ぶ。"""

    base_system = _build_base_system(
        dimension=core_dimension,
        core_direction=core_direction,
        core_recurrent_gain=core_recurrent_gain,
        core_input_gain=core_input_gain,
        feedback_gain=0.0,
    )
    novel_inputs = tuple(input_value[1] for input_value in inputs)
    best_choice: AdaptationChoice | None = None
    for recurrent_gain, input_gain in candidates:
        candidate_system = _adapt_reserve(
            system=base_system,
            reserve_direction=reserve_direction,
            recurrent_gain=recurrent_gain,
            input_gain=input_gain,
        )
        trajectory = _simulate(candidate_system, inputs)
        reserve_states = _state_block(
            system=candidate_system,
            trajectory=trajectory,
            block="reserve",
        )
        capacity = linear_memory_curve(
            states=reserve_states,
            inputs=novel_inputs,
            max_delay=max_delay,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            ridge=ridge,
        ).total_capacity
        if (
            best_choice is None
            or capacity > best_choice.calibration_novel_capacity
        ):
            best_choice = AdaptationChoice(
                recurrent_gain=recurrent_gain,
                input_gain=input_gain,
                calibration_novel_capacity=capacity,
            )
    if best_choice is None:
        raise RuntimeError("adaptation候補の選択に失敗しました")
    return best_choice


def select_entangled_adaptation(
    *,
    core_dimension: int,
    core_direction: Vector,
    reserve_direction: Vector,
    core_recurrent_gain: float,
    core_input_gain: float,
    candidates: tuple[tuple[float, float], ...],
    inputs: tuple[Vector, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
) -> AdaptationChoice:
    """同じ探索予算でcore更新対照の新規容量を最大化する。"""

    base_system = _build_base_system(
        dimension=core_dimension,
        core_direction=core_direction,
        core_recurrent_gain=core_recurrent_gain,
        core_input_gain=core_input_gain,
        feedback_gain=0.0,
    )
    novel_inputs = tuple(input_value[1] for input_value in inputs)
    best_choice: AdaptationChoice | None = None
    for recurrent_gain, input_gain in candidates:
        choice = AdaptationChoice(
            recurrent_gain=recurrent_gain,
            input_gain=input_gain,
            calibration_novel_capacity=0.0,
        )
        candidate_system = _adapt_core(
            system=base_system,
            core_direction=core_direction,
            reserve_direction=reserve_direction,
            core_input_gain=core_input_gain,
            adaptation=choice,
        )
        trajectory = _simulate(candidate_system, inputs)
        core_states = _state_block(
            system=candidate_system,
            trajectory=trajectory,
            block="core",
        )
        capacity = _memory_capacity(
            states=core_states,
            inputs=novel_inputs,
            max_delay=max_delay,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            ridge=ridge,
        )
        if (
            best_choice is None
            or capacity > best_choice.calibration_novel_capacity
        ):
            best_choice = AdaptationChoice(
                recurrent_gain=recurrent_gain,
                input_gain=input_gain,
                calibration_novel_capacity=capacity,
            )
    if best_choice is None:
        raise RuntimeError("entangled adaptation候補の選択に失敗しました")
    return best_choice


def evaluate_protected_update(
    *,
    trial_seed: int,
    dimension: int,
    core_direction: Vector,
    reserve_direction: Vector,
    core_recurrent_gain: float,
    core_input_gain: float,
    feedback_gain: float,
    adaptation: AdaptationChoice,
    inputs: tuple[Vector, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
) -> ProtectedUpdatePoint:
    """reserve-only更新によるcore保持と新規容量を評価する。"""

    pre_system = _build_base_system(
        dimension=dimension,
        core_direction=core_direction,
        core_recurrent_gain=core_recurrent_gain,
        core_input_gain=core_input_gain,
        feedback_gain=feedback_gain,
    )
    post_system = _adapt_reserve(
        system=pre_system,
        reserve_direction=reserve_direction,
        recurrent_gain=adaptation.recurrent_gain,
        input_gain=adaptation.input_gain,
    )
    pre_trajectory = _simulate(pre_system, inputs)
    post_trajectory = _simulate(post_system, inputs)
    core_inputs = tuple(input_value[0] for input_value in inputs)
    novel_inputs = tuple(input_value[1] for input_value in inputs)
    shared_core = shared_readout_memory_capacity(
        trajectories=(pre_trajectory, post_trajectory),
        inputs=core_inputs,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    pre_reserve = _state_block(
        system=pre_system,
        trajectory=pre_trajectory,
        block="reserve",
    )
    post_reserve = _state_block(
        system=post_system,
        trajectory=post_trajectory,
        block="reserve",
    )
    observed_deviations = _core_deviation_curve(
        system=pre_system,
        first=pre_trajectory,
        second=post_trajectory,
    )
    theoretical_bounds = core_deviation_bound_curve(
        core_lipschitz=pre_system.core_recurrent_infinity_norm,
        feedback_lipschitz=pre_system.reserve_feedback_infinity_norm,
        reserve_difference_bound=1.0,
        steps=len(inputs),
    )
    bound_satisfied = all(
        observed <= bound + 1e-12
        for observed, bound in zip(
            observed_deviations,
            theoretical_bounds,
            strict=True,
        )
    )
    return ProtectedUpdatePoint(
        trial_seed=trial_seed,
        feedback_gain=feedback_gain,
        selected_recurrent_gain=adaptation.recurrent_gain,
        selected_input_gain=adaptation.input_gain,
        pre_novel_capacity=_memory_capacity(
            states=pre_reserve,
            inputs=novel_inputs,
            max_delay=max_delay,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            ridge=ridge,
        ),
        post_novel_capacity=_memory_capacity(
            states=post_reserve,
            inputs=novel_inputs,
            max_delay=max_delay,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            ridge=ridge,
        ),
        core_reference_capacity=shared_core.reference_total_capacity,
        core_worst_capacity=shared_core.worst_total_capacity,
        core_retention=shared_core.worst_to_reference_ratio,
        max_core_deviation=max(observed_deviations),
        max_core_deviation_bound=max(theoretical_bounds),
        bound_satisfied=bound_satisfied,
    )


def evaluate_entangled_update(
    *,
    trial_seed: int,
    dimension: int,
    core_direction: Vector,
    reserve_direction: Vector,
    core_recurrent_gain: float,
    core_input_gain: float,
    adaptation: AdaptationChoice,
    inputs: tuple[Vector, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    ridge: float,
) -> EntangledUpdatePoint:
    """同じgain対をcoreへ書き込む干渉対照を評価する。"""

    pre_system = _build_base_system(
        dimension=dimension,
        core_direction=core_direction,
        core_recurrent_gain=core_recurrent_gain,
        core_input_gain=core_input_gain,
        feedback_gain=0.0,
    )
    entangled_system = _adapt_core(
        system=pre_system,
        core_direction=core_direction,
        reserve_direction=reserve_direction,
        core_input_gain=core_input_gain,
        adaptation=adaptation,
    )
    pre_trajectory = _simulate(pre_system, inputs)
    post_trajectory = _simulate(entangled_system, inputs)
    core_inputs = tuple(input_value[0] for input_value in inputs)
    novel_inputs = tuple(input_value[1] for input_value in inputs)
    shared_core = shared_readout_memory_capacity(
        trajectories=(pre_trajectory, post_trajectory),
        inputs=core_inputs,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    post_core = _state_block(
        system=entangled_system,
        trajectory=post_trajectory,
        block="core",
    )
    return EntangledUpdatePoint(
        trial_seed=trial_seed,
        selected_recurrent_gain=adaptation.recurrent_gain,
        selected_input_gain=adaptation.input_gain,
        post_novel_capacity=_memory_capacity(
            states=post_core,
            inputs=novel_inputs,
            max_delay=max_delay,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            ridge=ridge,
        ),
        core_reference_capacity=shared_core.reference_total_capacity,
        core_worst_capacity=shared_core.worst_total_capacity,
        core_retention=shared_core.worst_to_reference_ratio,
        max_core_deviation=max(
            _core_deviation_curve(
                system=pre_system,
                first=pre_trajectory,
                second=post_trajectory,
            )
        ),
    )


def seeded_directions(
    *,
    dimension: int,
    trial_seed: int,
) -> tuple[Vector, Vector]:
    """seedごとのcore入力方向とreserve入力方向を作る。"""

    random_generator = random.Random(trial_seed)

    def draw_direction() -> Vector:
        return tuple(
            random_generator.choice((-1.0, 1.0))
            * random_generator.uniform(0.5, 1.0)
            for _ in range(dimension)
        )

    return draw_direction(), draw_direction()


def generate_trial_inputs(
    *,
    trial_seed: int,
    steps: int,
) -> tuple[Vector, ...]:
    """独立なcore入力とnovel入力を生成する。"""

    random_generator = random.Random(trial_seed)
    return tuple(
        (
            random_generator.uniform(-1.0, 1.0),
            random_generator.uniform(-1.0, 1.0),
        )
        for _ in range(steps)
    )


def _build_base_system(
    *,
    dimension: int,
    core_direction: Vector,
    core_recurrent_gain: float,
    core_input_gain: float,
    feedback_gain: float,
) -> CoreReserveTanhRnn:
    core_recurrent = _diagonal_profile(dimension, core_recurrent_gain)
    feedback = _diagonal_profile(dimension, feedback_gain)
    recurrent_weights = tuple(
        core_row + feedback_row
        for core_row, feedback_row in zip(
            core_recurrent,
            feedback,
            strict=True,
        )
    ) + _zero_matrix(dimension, dimension * 2)
    input_weights = tuple(
        (direction * core_input_gain, 0.0)
        for direction in core_direction
    ) + _zero_matrix(dimension, 2)
    return CoreReserveTanhRnn(
        system=TanhRnnReservoir(
            recurrent_weights=recurrent_weights,
            input_weights=input_weights,
            bias=(0.0,) * (dimension * 2),
        ),
        core_dimension=dimension,
    )


def _adapt_reserve(
    *,
    system: CoreReserveTanhRnn,
    reserve_direction: Vector,
    recurrent_gain: float,
    input_gain: float,
) -> CoreReserveTanhRnn:
    dimension = system.reserve_dimension
    return system.with_reserve_parameters(
        core_to_reserve_weights=_zero_matrix(
            dimension,
            system.core_dimension,
        ),
        reserve_recurrent_weights=_diagonal_profile(
            dimension,
            recurrent_gain,
        ),
        reserve_input_weights=tuple(
            (0.0, direction * input_gain)
            for direction in reserve_direction
        ),
        reserve_bias=(0.0,) * dimension,
    )


def _adapt_core(
    *,
    system: CoreReserveTanhRnn,
    core_direction: Vector,
    reserve_direction: Vector,
    core_input_gain: float,
    adaptation: AdaptationChoice,
) -> CoreReserveTanhRnn:
    dimension = system.core_dimension
    return system.with_core_parameters(
        core_recurrent_weights=_diagonal_profile(
            dimension,
            adaptation.recurrent_gain,
        ),
        reserve_to_core_weights=_zero_matrix(
            dimension,
            system.reserve_dimension,
        ),
        core_input_weights=tuple(
            (
                core_direction[index] * core_input_gain,
                reserve_direction[index] * adaptation.input_gain,
            )
            for index in range(dimension)
        ),
        core_bias=(0.0,) * dimension,
    )


def _simulate(
    system: CoreReserveTanhRnn,
    inputs: tuple[Vector, ...],
) -> tuple[Vector, ...]:
    state = (0.0,) * system.state_dimension
    trajectory = [state]
    for input_value in inputs:
        state = system.step(state, input_value)
        trajectory.append(state)
    return tuple(trajectory)


def _state_block(
    *,
    system: CoreReserveTanhRnn,
    trajectory: tuple[Vector, ...],
    block: str,
) -> tuple[Vector, ...]:
    split_states = tuple(system.split_state(state) for state in trajectory)
    if block == "core":
        return tuple(core for core, _ in split_states)
    if block == "reserve":
        return tuple(reserve for _, reserve in split_states)
    raise ValueError("blockはcoreまたはreserveにしてください")


def _memory_capacity(
    *,
    states: tuple[Vector, ...],
    inputs: tuple[float, ...],
    max_delay: int,
    washout: int,
    training_steps: int,
    testing_steps: int,
    ridge: float,
) -> float:
    return linear_memory_curve(
        states=states,
        inputs=inputs,
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    ).total_capacity


def _core_deviation_curve(
    *,
    system: CoreReserveTanhRnn,
    first: tuple[Vector, ...],
    second: tuple[Vector, ...],
) -> tuple[float, ...]:
    return tuple(
        max(
            abs(first_value - second_value)
            for first_value, second_value in zip(
                system.split_state(first_state)[0],
                system.split_state(second_state)[0],
                strict=True,
            )
        )
        for first_state, second_state in zip(first, second, strict=True)
    )


def _diagonal_profile(dimension: int, gain: float) -> Matrix:
    return tuple(
        tuple(
            gain * (row_index + 1) / dimension
            if row_index == column_index
            else 0.0
            for column_index in range(dimension)
        )
        for row_index in range(dimension)
    )


def _zero_matrix(rows: int, columns: int) -> Matrix:
    return tuple((0.0,) * columns for _ in range(rows))
