"""候補選択から隔離したmodular familyへの外挿確認。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.cross_family_robust_task_confirmation import (
    DEFAULT_CONFIRMATION_SEEDS,
    DEFAULT_DISCOVERY_SEEDS,
    DEFAULT_FAMILY_SPECIFICATIONS,
    FamilyTaskSpecification,
)
from reservoir_dynamics.experiments.external_family_robust_task import (
    EXPERIMENT_ID,
    ExternalFamilyConfirmationResult,
    evaluate_external_family_confirmation,
)
from reservoir_dynamics.experiments.family_holdout_confirmation import (
    PREREGISTERED_BASELINE_CANDIDATES,
    PREREGISTERED_SELECTED_CANDIDATE,
)
from reservoir_dynamics.experiments.robust_repertoire_task import (
    RobustRepertoireTaskPoint,
    run_robust_repertoire_task_diagnostics,
)

PREREGISTERED_TRAINING_SEEDS = (
    DEFAULT_DISCOVERY_SEEDS + DEFAULT_CONFIRMATION_SEEDS
)
PREREGISTERED_CONFIRMATION_SEEDS = tuple(range(1301, 1331))
PREREGISTERED_EXTERNAL_FAMILY_SPECIFICATION = FamilyTaskSpecification(
    network_family="modular_paired",
    coupling_gains=(0.04, 0.05, 0.06, 0.07),
    disturbance_bound=0.16,
)


@dataclass(frozen=True, slots=True)
class IsolatedModularFamilyConfirmationResult:
    """学習用既知4族と未観測modular族を分離した実験結果。"""

    experiment_id: str
    phase: str
    training_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    training_family_specifications: tuple[
        FamilyTaskSpecification, ...
    ]
    external_family_specification: FamilyTaskSpecification
    dimension: int
    task_steps: int
    training_points: tuple[RobustRepertoireTaskPoint, ...]
    confirmation_points: tuple[RobustRepertoireTaskPoint, ...]
    evaluation: ExternalFamilyConfirmationResult


def run_isolated_modular_family_confirmation(
    *,
    training_seeds: tuple[int, ...] = PREREGISTERED_TRAINING_SEEDS,
    confirmation_seeds: tuple[
        int, ...
    ] = PREREGISTERED_CONFIRMATION_SEEDS,
    training_family_specifications: tuple[
        FamilyTaskSpecification, ...
    ] = DEFAULT_FAMILY_SPECIFICATIONS,
    external_family_specification: FamilyTaskSpecification = (
        PREREGISTERED_EXTERNAL_FAMILY_SPECIFICATION
    ),
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    task_steps: int = 100,
    autonomous_steps: int = 500,
    convergence_tolerance: float = 1e-9,
    association_threshold: float = 0.75,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
) -> IsolatedModularFamilyConfirmationResult:
    """固定済み予測器を未観測modular familyへ適用する。"""

    _validate_configuration(
        training_seeds=training_seeds,
        confirmation_seeds=confirmation_seeds,
        training_family_specifications=(
            training_family_specifications
        ),
        external_family_specification=(
            external_family_specification
        ),
        bootstrap_resamples=bootstrap_resamples,
    )
    training_points = tuple(
        point
        for specification in training_family_specifications
        for point in _run_specification(
            trial_seeds=training_seeds,
            specification=specification,
            dimension=dimension,
            diagonal_gain=diagonal_gain,
            task_steps=task_steps,
            autonomous_steps=autonomous_steps,
            convergence_tolerance=convergence_tolerance,
        )
    )
    confirmation_points = _run_specification(
        trial_seeds=confirmation_seeds,
        specification=external_family_specification,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
    )
    evaluation = evaluate_external_family_confirmation(
        training_points=training_points,
        confirmation_points=confirmation_points,
        selected_candidate=PREREGISTERED_SELECTED_CANDIDATE,
        baseline_candidates=PREREGISTERED_BASELINE_CANDIDATES,
        expected_raw_attractor_count=2**dimension,
        association_threshold=association_threshold,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
    )
    return IsolatedModularFamilyConfirmationResult(
        experiment_id=EXPERIMENT_ID,
        phase="confirmation",
        training_seeds=training_seeds,
        confirmation_seeds=confirmation_seeds,
        training_family_specifications=(
            training_family_specifications
        ),
        external_family_specification=(
            external_family_specification
        ),
        dimension=dimension,
        task_steps=task_steps,
        training_points=training_points,
        confirmation_points=confirmation_points,
        evaluation=evaluation,
    )


def _run_specification(
    *,
    trial_seeds: tuple[int, ...],
    specification: FamilyTaskSpecification,
    dimension: int,
    diagonal_gain: float,
    task_steps: int,
    autonomous_steps: int,
    convergence_tolerance: float,
) -> tuple[RobustRepertoireTaskPoint, ...]:
    # 内部の単変量診断分割は本実験のfit境界に使わないため末尾2seedだけ残す。
    diagnostics = run_robust_repertoire_task_diagnostics(
        trial_seeds=trial_seeds,
        network_family=specification.network_family,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        coupling_gains=specification.coupling_gains,
        disturbance_bounds=(specification.disturbance_bound,),
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        convergence_tolerance=convergence_tolerance,
        training_seed_count=len(trial_seeds) - 2,
    )
    return diagnostics.points


def _validate_configuration(
    *,
    training_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
    training_family_specifications: tuple[
        FamilyTaskSpecification, ...
    ],
    external_family_specification: FamilyTaskSpecification,
    bootstrap_resamples: int,
) -> None:
    if len(training_seeds) < 4 or len(confirmation_seeds) < 4:
        raise ValueError("trainingとconfirmationは各4 seed以上必要です")
    if set(training_seeds).intersection(confirmation_seeds):
        raise ValueError("trainingとconfirmationのseedは重複禁止です")
    training_families = tuple(
        specification.network_family
        for specification in training_family_specifications
    )
    if (
        len(training_families) < 2
        or len(set(training_families)) != len(training_families)
    ):
        raise ValueError("training familyは重複しない2種類以上必要です")
    if external_family_specification.network_family in training_families:
        raise ValueError("external familyをtrainingへ含めないでください")
    if external_family_specification.network_family != "modular_paired":
        raise ValueError("external familyはmodular_pairedに固定します")
    if (
        not isinstance(bootstrap_resamples, int)
        or isinstance(bootstrap_resamples, bool)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamplesは1以上の整数にしてください")


def _summary_payload(
    result: IsolatedModularFamilyConfirmationResult,
) -> dict[str, object]:
    """再計算可能性を保ちながら巨大なpoint列を除いた要約を作る。"""

    evaluation = result.evaluation
    selected = evaluation.selected_evaluation
    challenges_per_point = 2 ** (2 * result.dimension)
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "training_seeds": result.training_seeds,
        "confirmation_seeds": result.confirmation_seeds,
        "training_family_specifications": tuple(
            asdict(specification)
            for specification in result.training_family_specifications
        ),
        "external_family_specification": asdict(
            result.external_family_specification
        ),
        "dimension": result.dimension,
        "task_steps": result.task_steps,
        "training_point_count": len(result.training_points),
        "confirmation_point_count": len(result.confirmation_points),
        "confirmation_challenge_count": (
            len(result.confirmation_points) * challenges_per_point
        ),
        "guarantee_violation_count": (
            evaluation.guarantee_violation_count
        ),
        "selected_candidate": asdict(selected.candidate),
        "selected_test_mae": selected.test_mae,
        "selected_test_spearman": selected.test_spearman,
        "baselines": tuple(
            {
                "name": baseline.candidate.name,
                "test_mae": baseline.test_mae,
                "test_spearman": baseline.test_spearman,
            }
            for baseline in evaluation.baseline_evaluations
        ),
        "baseline_minus_selected_intervals": tuple(
            asdict(interval)
            for interval in (
                evaluation.baseline_minus_selected_intervals
            )
        ),
        "raw_attractor_count_values": tuple(
            sorted(
                {
                    point.raw_attractor_count
                    for point in result.confirmation_points
                }
            )
        ),
        "task_retention_range": (
            min(
                point.task_retention
                for point in result.confirmation_points
            ),
            max(
                point.task_retention
                for point in result.confirmation_points
            ),
        ),
        "decisions": asdict(evaluation.decisions),
    }


def main() -> None:
    """事前登録した外部modular family確認をJSONで出力する。"""

    print(
        json.dumps(
            _summary_payload(
                run_isolated_modular_family_confirmation()
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
