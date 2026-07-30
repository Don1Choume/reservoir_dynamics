"""局所条件付き安定性と大域的replica同期を分離する基準実験。"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass

from reservoir_dynamics.metrics.conditional_lyapunov import (
    finite_time_conditional_lyapunov_exponent,
)
from reservoir_dynamics.metrics.replica import (
    pairwise_replica_distance_curve,
)
from reservoir_dynamics.simulation.discrete import (
    simulate_discrete_replicas,
)
from reservoir_dynamics.systems.scalar_tanh import ScalarTanhReservoir

EXPERIMENT_ID = "EXP-2026-002"
_RECURRENT_GAINS = (0.6, 1.0, 1.2, 1.5)
_INPUT_GAINS = (0.0, 0.5, 2.0, 4.0)
_INITIAL_STATES = ((-2.0,), (-1.0,), (0.0,), (1.0,), (2.0,))


@dataclass(frozen=True, slots=True)
class StabilitySweepPoint:
    """一組のgainに対する局所安定性とreplica同期の測定値。"""

    recurrent_gain: float
    input_gain: float
    global_contraction_guaranteed: bool
    conditional_lyapunov_exponent: float
    tail_replica_rms_distance: float
    replica_synchronized: bool


@dataclass(frozen=True, slots=True)
class InputConditionedStabilitySweepResult:
    """入力強度とrecurrent gainを掃引した結果。"""

    experiment_id: str
    random_seed: int
    steps: int
    washout: int
    tail_window: int
    derivative_floor: float
    synchronization_tolerance: float
    points: tuple[StabilitySweepPoint, ...]
    local_stability_without_global_sync_observed: bool
    sync_beyond_global_contraction_observed: bool


def run_input_conditioned_stability_sweep(
    *,
    random_seed: int = 20_260_730,
    steps: int = 2_500,
    washout: int = 500,
    tail_window: int = 100,
    derivative_floor: float = 1e-300,
    synchronization_tolerance: float = 1e-8,
) -> InputConditionedStabilitySweepResult:
    """条件付き指数と複製同期を同じ入力実現で比較する。"""

    if steps <= washout:
        raise ValueError("stepsはwashoutより大きい必要があります")
    if tail_window < 1 or tail_window > steps + 1:
        raise ValueError("tail_windowは軌道長以下の正数である必要があります")
    if (
        not math.isfinite(synchronization_tolerance)
        or synchronization_tolerance <= 0.0
    ):
        raise ValueError(
            "synchronization_toleranceは有限の正数である必要があります"
        )

    input_sequence = _generate_input_sequence(
        random_seed=random_seed,
        steps=steps,
    )
    points = tuple(
        _evaluate_sweep_point(
            recurrent_gain=recurrent_gain,
            input_gain=input_gain,
            inputs=input_sequence,
            washout=washout,
            tail_window=tail_window,
            derivative_floor=derivative_floor,
            synchronization_tolerance=synchronization_tolerance,
        )
        for recurrent_gain in _RECURRENT_GAINS
        for input_gain in _INPUT_GAINS
    )

    return InputConditionedStabilitySweepResult(
        experiment_id=EXPERIMENT_ID,
        random_seed=random_seed,
        steps=steps,
        washout=washout,
        tail_window=tail_window,
        derivative_floor=derivative_floor,
        synchronization_tolerance=synchronization_tolerance,
        points=points,
        local_stability_without_global_sync_observed=any(
            point.conditional_lyapunov_exponent < 0.0
            and not point.replica_synchronized
            for point in points
        ),
        sync_beyond_global_contraction_observed=any(
            not point.global_contraction_guaranteed
            and point.conditional_lyapunov_exponent < 0.0
            and point.replica_synchronized
            for point in points
        ),
    )


def _generate_input_sequence(
    *,
    random_seed: int,
    steps: int,
) -> tuple[tuple[float, ...], ...]:
    random_generator = random.Random(random_seed)
    return tuple((random_generator.uniform(-1.0, 1.0),) for _ in range(steps))


def _evaluate_sweep_point(
    *,
    recurrent_gain: float,
    input_gain: float,
    inputs: tuple[tuple[float, ...], ...],
    washout: int,
    tail_window: int,
    derivative_floor: float,
    synchronization_tolerance: float,
) -> StabilitySweepPoint:
    system = ScalarTanhReservoir(
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
    )
    simulation = simulate_discrete_replicas(
        system=system,
        initial_states=_INITIAL_STATES,
        inputs=inputs,
    )
    reference_trajectory = simulation.trajectories[0]
    local_derivative_magnitudes = tuple(
        system.state_jacobian_magnitude(
            reference_trajectory[time_index],
            input_value,
        )
        for time_index, input_value in enumerate(simulation.inputs)
    )
    conditional_exponent = finite_time_conditional_lyapunov_exponent(
        local_derivative_magnitudes,
        washout=washout,
        derivative_floor=derivative_floor,
    )
    replica_distances = pairwise_replica_distance_curve(
        simulation.trajectories
    )
    tail_distances = replica_distances[-tail_window:]
    tail_rms_distance = math.sqrt(
        math.fsum(distance * distance for distance in tail_distances)
        / len(tail_distances)
    )

    return StabilitySweepPoint(
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        global_contraction_guaranteed=system.is_globally_contractive,
        conditional_lyapunov_exponent=conditional_exponent,
        tail_replica_rms_distance=tail_rms_distance,
        replica_synchronized=tail_rms_distance <= synchronization_tolerance,
    )


def main() -> None:
    """実験結果をJSONとして標準出力へ出す。"""

    result = run_input_conditioned_stability_sweep()
    print(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
