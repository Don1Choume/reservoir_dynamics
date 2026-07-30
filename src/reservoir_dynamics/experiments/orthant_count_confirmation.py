"""EXP-007で選んだcount-matched条件を独立seedで確認する。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from reservoir_dynamics.experiments.orthant_margin_sweep import (
    OrthantMarginSweepResult,
    run_orthant_margin_sweep,
)

EXPERIMENT_ID = "EXP-2026-008"
DEFAULT_TRIAL_SEEDS = tuple(range(501, 531))
DISCOVERY_SEEDS = frozenset(range(401, 431))


def run_orthant_count_confirmation_study(
    *,
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    safe_trials: int = 16,
    simulation_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 20_260_734,
) -> OrthantMarginSweepResult:
    """発見用seedを拒否し、0.04対0.07の確認specを実行する。"""

    if DISCOVERY_SEEDS.intersection(trial_seeds):
        raise ValueError("EXP-007の発見用seedは確認実験に再利用できません")
    discovery_result = run_orthant_margin_sweep(
        trial_seeds=trial_seeds,
        dimension=4,
        diagonal_gain=1.5,
        coupling_gains=(0.04, 0.07),
        safe_disturbance_ratio=0.9,
        witness_disturbance_ratio=1.1,
        safe_trials=safe_trials,
        simulation_steps=simulation_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return replace(discovery_result, experiment_id=EXPERIMENT_ID)


def main() -> None:
    """既定30 seed確認実験をJSONとして出力する。"""

    print(
        json.dumps(
            asdict(run_orthant_count_confirmation_study()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
