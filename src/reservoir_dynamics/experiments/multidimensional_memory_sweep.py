"""多次元tanh RNNの安定性、同期、線形記憶を同時に掃引する。"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass

from reservoir_dynamics.metrics.linear_memory import linear_memory_curve
from reservoir_dynamics.metrics.replica import pairwise_replica_distance_curve
from reservoir_dynamics.metrics.top_conditional_lyapunov import (
    top_conditional_lyapunov_exponent,
)
from reservoir_dynamics.simulation.discrete import simulate_discrete_replicas
from reservoir_dynamics.systems.tanh_rnn import TanhRnnReservoir

EXPERIMENT_ID = "EXP-2026-003"


@dataclass(frozen=True, slots=True)
class MultidimensionalMemoryPoint:
    """一つのgainとseedに対する三種類の評価結果。"""

    recurrent_gain: float
    input_gain: float
    trial_seed: int
    top_conditional_lyapunov_exponent: float
    tail_replica_rms_distance: float
    replica_synchronized: bool
    memory_curve: tuple[float, ...]
    linear_memory_capacity: float


@dataclass(frozen=True, slots=True)
class MultidimensionalMemorySweepResult:
    """多次元RNNのparameter sweep全体。"""

    experiment_id: str
    state_dimension: int
    recurrent_gains: tuple[float, ...]
    input_gains: tuple[float, ...]
    trial_seeds: tuple[int, ...]
    washout: int
    training_steps: int
    testing_steps: int
    max_delay: int
    tail_window: int
    synchronization_tolerance: float
    ridge: float
    points: tuple[MultidimensionalMemoryPoint, ...]


def run_multidimensional_memory_sweep(
    *,
    state_dimension: int = 8,
    recurrent_gains: tuple[float, ...] = (0.6, 0.9, 1.2, 1.5),
    input_gains: tuple[float, ...] = (0.1, 0.5, 1.5),
    trial_seeds: tuple[int, ...] = (11, 29, 47),
    washout: int = 200,
    training_steps: int = 800,
    testing_steps: int = 400,
    max_delay: int = 12,
    tail_window: int = 100,
    synchronization_tolerance: float = 1e-7,
    ridge: float = 1e-8,
) -> MultidimensionalMemorySweepResult:
    """直交再帰核のgain、入力gain、seedを掃引する。"""

    _validate_configuration(
        state_dimension=state_dimension,
        recurrent_gains=recurrent_gains,
        input_gains=input_gains,
        trial_seeds=trial_seeds,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        max_delay=max_delay,
        tail_window=tail_window,
        synchronization_tolerance=synchronization_tolerance,
        ridge=ridge,
    )
    first_sample_time = max(washout, max_delay)
    steps = first_sample_time + training_steps + testing_steps
    points = tuple(
        _evaluate_condition(
            state_dimension=state_dimension,
            recurrent_gain=recurrent_gain,
            input_gain=input_gain,
            trial_seed=trial_seed,
            steps=steps,
            washout=washout,
            training_steps=training_steps,
            testing_steps=testing_steps,
            max_delay=max_delay,
            tail_window=tail_window,
            synchronization_tolerance=synchronization_tolerance,
            ridge=ridge,
        )
        for trial_seed in trial_seeds
        for recurrent_gain in recurrent_gains
        for input_gain in input_gains
    )
    return MultidimensionalMemorySweepResult(
        experiment_id=EXPERIMENT_ID,
        state_dimension=state_dimension,
        recurrent_gains=recurrent_gains,
        input_gains=input_gains,
        trial_seeds=trial_seeds,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        max_delay=max_delay,
        tail_window=tail_window,
        synchronization_tolerance=synchronization_tolerance,
        ridge=ridge,
        points=points,
    )


def _evaluate_condition(
    *,
    state_dimension: int,
    recurrent_gain: float,
    input_gain: float,
    trial_seed: int,
    steps: int,
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    tail_window: int,
    synchronization_tolerance: float,
    ridge: float,
) -> MultidimensionalMemoryPoint:
    system = create_seeded_orthogonal_reservoir(
        state_dimension=state_dimension,
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        random_seed=trial_seed,
    )
    inputs = generate_trial_inputs(random_seed=trial_seed, steps=steps)
    simulation = simulate_discrete_replicas(
        system=system,
        initial_states=standard_replica_initial_states(state_dimension),
        inputs=inputs,
    )
    reference_trajectory = simulation.trajectories[0]
    conditional_exponent = top_conditional_lyapunov_exponent(
        system=system,
        trajectory=reference_trajectory,
        inputs=inputs,
        washout=washout,
    )
    replica_distances = pairwise_replica_distance_curve(
        simulation.trajectories
    )
    tail_distances = replica_distances[-tail_window:]
    tail_rms_distance = math.sqrt(
        math.fsum(distance * distance for distance in tail_distances)
        / len(tail_distances)
    )
    memory_result = linear_memory_curve(
        states=reference_trajectory,
        inputs=tuple(input_value[0] for input_value in inputs),
        max_delay=max_delay,
        washout=washout,
        training_steps=training_steps,
        testing_steps=testing_steps,
        ridge=ridge,
    )
    return MultidimensionalMemoryPoint(
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        trial_seed=trial_seed,
        top_conditional_lyapunov_exponent=conditional_exponent,
        tail_replica_rms_distance=tail_rms_distance,
        replica_synchronized=tail_rms_distance <= synchronization_tolerance,
        memory_curve=memory_result.capacity_by_delay,
        linear_memory_capacity=memory_result.total_capacity,
    )


def create_seeded_orthogonal_reservoir(
    *,
    state_dimension: int,
    recurrent_gain: float,
    input_gain: float,
    random_seed: int,
) -> TanhRnnReservoir:
    random_generator = random.Random(random_seed)
    orthogonal_rows = _orthogonal_rows(
        state_dimension=state_dimension,
        random_generator=random_generator,
    )
    recurrent_weights = tuple(
        tuple(recurrent_gain * value for value in row)
        for row in orthogonal_rows
    )
    raw_input_direction = tuple(
        random_generator.gauss(0.0, 1.0) for _ in range(state_dimension)
    )
    input_norm = math.sqrt(
        math.fsum(value * value for value in raw_input_direction)
    )
    input_weights = tuple(
        (input_gain * value / input_norm,) for value in raw_input_direction
    )
    return TanhRnnReservoir(
        recurrent_weights=recurrent_weights,
        input_weights=input_weights,
        bias=(0.0,) * state_dimension,
    )


def _orthogonal_rows(
    *,
    state_dimension: int,
    random_generator: random.Random,
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for _ in range(state_dimension):
        candidate = [
            random_generator.gauss(0.0, 1.0)
            for _ in range(state_dimension)
        ]
        for existing_row in rows:
            projection = math.fsum(
                value * basis_value
                for value, basis_value in zip(
                    candidate,
                    existing_row,
                    strict=True,
                )
            )
            candidate = [
                value - projection * basis_value
                for value, basis_value in zip(
                    candidate,
                    existing_row,
                    strict=True,
                )
            ]
        candidate_norm = math.sqrt(
            math.fsum(value * value for value in candidate)
        )
        if candidate_norm <= 1e-12:
            raise RuntimeError("直交再帰核の生成に失敗しました")
        rows.append(tuple(value / candidate_norm for value in candidate))
    return tuple(rows)


def generate_trial_inputs(
    *,
    random_seed: int,
    steps: int,
) -> tuple[tuple[float, ...], ...]:
    random_generator = random.Random(random_seed + 1_000_003)
    return tuple((random_generator.uniform(-1.0, 1.0),) for _ in range(steps))


def standard_replica_initial_states(
    state_dimension: int,
) -> tuple[tuple[float, ...], ...]:
    return (
        (0.0,) * state_dimension,
        (-1.0,) * state_dimension,
        (1.0,) * state_dimension,
        tuple(
            1.0 if index % 2 == 0 else -1.0
            for index in range(state_dimension)
        ),
    )


def _validate_configuration(
    *,
    state_dimension: int,
    recurrent_gains: tuple[float, ...],
    input_gains: tuple[float, ...],
    trial_seeds: tuple[int, ...],
    washout: int,
    training_steps: int,
    testing_steps: int,
    max_delay: int,
    tail_window: int,
    synchronization_tolerance: float,
    ridge: float,
) -> None:
    _validate_positive_integer(state_dimension, "state_dimension")
    if not recurrent_gains:
        raise ValueError("recurrent_gainsは1要素以上必要です")
    if not input_gains:
        raise ValueError("input_gainsは1要素以上必要です")
    if not trial_seeds:
        raise ValueError("trial_seedsは1要素以上必要です")
    if any(
        not math.isfinite(gain) or gain < 0.0 for gain in recurrent_gains
    ):
        raise ValueError("recurrent_gainsは有限の非負値にしてください")
    if any(not math.isfinite(gain) or gain < 0.0 for gain in input_gains):
        raise ValueError("input_gainsは有限の非負値にしてください")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool)
        for seed in trial_seeds
    ):
        raise ValueError("trial_seedsは整数にしてください")
    _validate_non_negative_integer(washout, "washout")
    _validate_positive_integer(training_steps, "training_steps")
    _validate_positive_integer(testing_steps, "testing_steps")
    _validate_positive_integer(max_delay, "max_delay")
    _validate_positive_integer(tail_window, "tail_window")
    steps = max(washout, max_delay) + training_steps + testing_steps
    if tail_window > steps + 1:
        raise ValueError("tail_windowは軌道長以下にしてください")
    if (
        not math.isfinite(synchronization_tolerance)
        or synchronization_tolerance <= 0.0
    ):
        raise ValueError(
            "synchronization_toleranceは有限の正数にしてください"
        )
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridgeは有限の非負値にしてください")


def _validate_positive_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{value_name}は1以上の整数にしてください")


def _validate_non_negative_integer(value: int, value_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{value_name}は0以上の整数にしてください")


def main() -> None:
    """既定specの実験結果をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_multidimensional_memory_sweep()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
