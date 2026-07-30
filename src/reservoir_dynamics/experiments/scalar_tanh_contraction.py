"""スカラーtanhリザバーの大域的収縮上界を数値検証する。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from reservoir_dynamics.metrics.replica import (
    pairwise_replica_distance_curve,
)
from reservoir_dynamics.simulation.discrete import (
    simulate_discrete_replicas,
)
from reservoir_dynamics.systems.scalar_tanh import ScalarTanhReservoir
from reservoir_dynamics.theory.contraction import (
    iterated_lipschitz_distance_bound,
)

EXPERIMENT_ID = "EXP-2026-001"
_DEFAULT_INPUTS = (
    (0.75,),
    (-0.25,),
    (0.5,),
    (-1.0,),
) * 5


@dataclass(frozen=True, slots=True)
class ScalarTanhContractionResult:
    """収縮上界のground-truth照合結果。"""

    experiment_id: str
    recurrent_gain: float
    input_gain: float
    bias: float
    initial_states: tuple[tuple[float, ...], ...]
    inputs: tuple[tuple[float, ...], ...]
    observed_replica_distances: tuple[float, ...]
    theoretical_distance_bounds: tuple[float, ...]
    maximum_bound_violation: float
    tolerance: float
    passed: bool


def run_scalar_tanh_contraction_experiment(
    *,
    tolerance: float = 1e-12,
) -> ScalarTanhContractionResult:
    """解析的Lipschitz上界と複製軌道距離を照合する。"""

    system = ScalarTanhReservoir(
        recurrent_gain=0.6,
        input_gain=0.8,
        bias=-0.05,
    )
    initial_states = ((-2.0,), (2.0,))
    simulation = simulate_discrete_replicas(
        system=system,
        initial_states=initial_states,
        inputs=_DEFAULT_INPUTS,
    )
    observed_distances = pairwise_replica_distance_curve(
        simulation.trajectories
    )
    theoretical_bounds = iterated_lipschitz_distance_bound(
        initial_distance=observed_distances[0],
        lipschitz_constant=system.global_state_lipschitz_bound,
        steps=len(simulation.inputs),
    )
    maximum_bound_violation = max(
        observed_distance - theoretical_bound
        for observed_distance, theoretical_bound in zip(
            observed_distances,
            theoretical_bounds,
            strict=True,
        )
    )

    return ScalarTanhContractionResult(
        experiment_id=EXPERIMENT_ID,
        recurrent_gain=system.recurrent_gain,
        input_gain=system.input_gain,
        bias=system.bias,
        initial_states=initial_states,
        inputs=simulation.inputs,
        observed_replica_distances=observed_distances,
        theoretical_distance_bounds=theoretical_bounds,
        maximum_bound_violation=maximum_bound_violation,
        tolerance=tolerance,
        passed=maximum_bound_violation <= tolerance,
    )


def main() -> None:
    """実験結果を機械可読JSONとして標準出力へ出す。"""

    result = run_scalar_tanh_contraction_experiment()
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
