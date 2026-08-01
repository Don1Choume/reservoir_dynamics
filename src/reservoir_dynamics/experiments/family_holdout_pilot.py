"""既観測のEXP-010データだけでEXP-011候補を選択する。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from reservoir_dynamics.experiments.cross_family_robust_task_confirmation import (
    DEFAULT_CONFIRMATION_SEEDS,
    DEFAULT_DISCOVERY_SEEDS,
    DEFAULT_FAMILY_SPECIFICATIONS,
    FamilyTaskSpecification,
    run_cross_family_robust_task_confirmation,
)
from reservoir_dynamics.experiments.family_holdout_robust_task import (
    DEFAULT_PILOT_CANDIDATES,
    EXPERIMENT_ID,
    FamilyHoldoutCandidate,
    FamilyHoldoutSelection,
    select_family_holdout_candidate,
)


@dataclass(frozen=True, slots=True)
class FamilyHoldoutFoldSummary:
    """pilot成果物へ残す未知family別の最小統計。"""

    held_out_family: str
    training_point_count: int
    testing_point_count: int
    test_mae: float
    test_spearman: float


@dataclass(frozen=True, slots=True)
class FamilyHoldoutCandidateSummary:
    """candidate選択を再検証するための全fold要約。"""

    name: str
    feature_names: tuple[str, ...]
    penalty: float
    pooled_test_mae: float
    pooled_test_spearman: float
    folds: tuple[FamilyHoldoutFoldSummary, ...]


@dataclass(frozen=True, slots=True)
class FamilyHoldoutPilotResult:
    """未使用confirmation seedへ触れる前の探索結果。"""

    experiment_id: str
    phase: str
    source_experiment_id: str
    training_seeds: tuple[int, ...]
    testing_seeds: tuple[int, ...]
    source_point_count: int
    candidates: tuple[FamilyHoldoutCandidateSummary, ...]
    selected_candidate: FamilyHoldoutCandidate


def run_family_holdout_pilot(
    *,
    training_seeds: tuple[int, ...] = DEFAULT_DISCOVERY_SEEDS,
    testing_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
    family_specifications: tuple[
        FamilyTaskSpecification, ...
    ] = DEFAULT_FAMILY_SPECIFICATIONS,
    candidates: tuple[
        FamilyHoldoutCandidate, ...
    ] = DEFAULT_PILOT_CANDIDATES,
    dimension: int = 4,
    diagonal_gain: float = 1.5,
    task_steps: int = 100,
    autonomous_steps: int = 500,
) -> FamilyHoldoutPilotResult:
    """EXP-010の観測済みseedでleave-one-family-out候補を選ぶ。"""

    source_result = run_cross_family_robust_task_confirmation(
        discovery_seeds=training_seeds,
        confirmation_seeds=testing_seeds,
        family_specifications=family_specifications,
        dimension=dimension,
        diagonal_gain=diagonal_gain,
        task_steps=task_steps,
        autonomous_steps=autonomous_steps,
        bootstrap_resamples=20,
    )
    selection = select_family_holdout_candidate(
        points=source_result.points,
        training_seeds=training_seeds,
        testing_seeds=testing_seeds,
        candidates=candidates,
    )
    return _summarize_selection(
        selection=selection,
        source_experiment_id=source_result.experiment_id,
        source_point_count=len(source_result.points),
    )


def _summarize_selection(
    *,
    selection: FamilyHoldoutSelection,
    source_experiment_id: str,
    source_point_count: int,
) -> FamilyHoldoutPilotResult:
    first_evaluation = selection.evaluations[0]
    return FamilyHoldoutPilotResult(
        experiment_id=EXPERIMENT_ID,
        phase="pilot",
        source_experiment_id=source_experiment_id,
        training_seeds=first_evaluation.training_seeds,
        testing_seeds=first_evaluation.testing_seeds,
        source_point_count=source_point_count,
        candidates=tuple(
            FamilyHoldoutCandidateSummary(
                name=evaluation.candidate.name,
                feature_names=evaluation.candidate.feature_names,
                penalty=evaluation.candidate.penalty,
                pooled_test_mae=evaluation.pooled_test_mae,
                pooled_test_spearman=evaluation.pooled_test_spearman,
                folds=tuple(
                    FamilyHoldoutFoldSummary(
                        held_out_family=fold.held_out_family,
                        training_point_count=fold.training_point_count,
                        testing_point_count=fold.testing_point_count,
                        test_mae=fold.test_mae,
                        test_spearman=fold.test_spearman,
                    )
                    for fold in evaluation.folds
                ),
            )
            for evaluation in selection.evaluations
        ),
        selected_candidate=selection.selected_candidate,
    )


def main() -> None:
    """既定pilotを実行し、機械可読な要約を標準出力へ書く。"""

    print(
        json.dumps(
            asdict(run_family_holdout_pilot()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
