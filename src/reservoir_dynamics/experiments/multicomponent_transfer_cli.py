"""EXP-2026-017のtask前gateと固定model confirmationを保存するCLI。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from reservoir_dynamics.experiments.component_predictor import (
    ComponentPredictorEvaluation,
    NamedComponentPredictor,
    predictor_payload_sha256,
    predictors_from_payload,
    predictors_to_payload,
)
from reservoir_dynamics.experiments.multicomponent_profile import (
    MultiComponentProfilePoint,
)
from reservoir_dynamics.experiments.multicomponent_modular_family import (
    audit_multicomponent_structure,
)
from reservoir_dynamics.experiments.multicomponent_transfer import (
    EXPERIMENT_ID,
    PREREGISTERED_BRIDGE_STRENGTHS,
    PREREGISTERED_CONFIRMATION_SEEDS,
    PREREGISTERED_DEVELOPMENT_SEEDS,
    PREREGISTERED_INTERNAL_GAINS,
    PREREGISTERED_MODULE_SIZES,
    MultiComponentTransferResult,
    run_multicomponent_confirmation,
    run_multicomponent_development,
)


def load_frozen_models(
    pilot_artifact: Path,
    *,
    expected_sha256: str,
) -> tuple[tuple[NamedComponentPredictor, ...], str]:
    """EXP-2026-016 artifactを再fitせずhash照合して復元する。"""

    if len(expected_sha256) != 64:
        raise ValueError("expected_sha256は64文字にしてください")
    try:
        payload = json.loads(pilot_artifact.read_text(encoding="utf-8"))
        model_payload = payload["fitted_models"]
        artifact_hash = str(payload["model_sha256"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("pilot artifactを読み込めません") from error
    models = predictors_from_payload(model_payload)
    computed_hash = predictor_payload_sha256(predictors_to_payload(models))
    if artifact_hash != computed_hash or computed_hash != expected_sha256:
        raise ValueError("pilot artifactのmodel hashが一致しません")
    return models, computed_hash


def structure_gate_payload(
    *,
    trial_seeds: tuple[int, ...] = PREREGISTERED_CONFIRMATION_SEEDS,
    module_sizes: tuple[int, ...] = PREREGISTERED_MODULE_SIZES,
    internal_gains: tuple[float, ...] = PREREGISTERED_INTERNAL_GAINS,
    maximum_bridge_strengths: tuple[float, ...] = PREREGISTERED_BRIDGE_STRENGTHS,
) -> dict[str, object]:
    """task関数を呼ばずに登録familyの分割回復だけを直列化する。"""

    gate = audit_multicomponent_structure(
        trial_seeds=trial_seeds,
        module_sizes=module_sizes,
        internal_gains=internal_gains,
        maximum_total_bridge_strengths=maximum_bridge_strengths,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": "confirmation",
        "task_values_generated": False,
        "structure_gate": asdict(gate),
    }


def confirmation_summary_payload(
    result: MultiComponentTransferResult,
    *,
    fixed_model_sha256: str,
) -> dict[str, object]:
    """予測、certificate、bootstrap区間を完全なJSON証拠へ変換する。"""

    if len(fixed_model_sha256) != 64:
        raise ValueError("fixed_model_sha256は64文字にしてください")
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "fixed_model_sha256": fixed_model_sha256,
        "structure_gate": asdict(result.structure_gate),
        "theory_decisions": asdict(result.theory_decisions),
        "decisions": asdict(result.decisions),
        "point_count": len(result.points),
        "challenge_count": sum(
            4 * point.global_orthant_count for point in result.points
        ),
        "maximum_absolute_task_product_residual": max(
            abs(point.task_product_residual) for point in result.points
        ),
        "maximum_factorized_enumerated_certificate_difference": max(
            abs(
                point.factorized_directional_certified_fraction
                - point.enumerated_directional_certified_fraction
            )
            for point in result.points
        ),
        "minimum_task_minus_transported_certificate": min(
            point.observed_task_retention
            - point.transported_certified_fraction
            for point in result.points
        ),
        "enumeration_complexity": {
            "local_orthant_count": result.points[0].local_orthant_count,
            "monolithic_orthant_count": result.points[0].global_orthant_count,
            "local_to_monolithic_ratio": (
                result.points[0].local_orthant_count
                / result.points[0].global_orthant_count
            ),
        },
        "evaluations": [
            _evaluation_payload(value) for value in result.evaluations
        ],
        "error_intervals": [asdict(value) for value in result.error_intervals],
        "group_summaries": _group_summaries(result.points),
        "points": [asdict(point) for point in result.points],
    }


def _evaluation_payload(
    value: ComponentPredictorEvaluation,
) -> dict[str, object]:
    payload = asdict(value)
    predictions = payload.pop("predictions")
    return {**payload, "predictions": predictions}


def _group_summaries(
    points: tuple[MultiComponentProfilePoint, ...],
) -> list[dict[str, object]]:
    strengths = sorted({point.maximum_bridge_strength for point in points})
    return [_group_summary(points, strength) for strength in strengths]


def _group_summary(
    points: tuple[MultiComponentProfilePoint, ...],
    bridge_strength: float,
) -> dict[str, object]:
    selected = tuple(
        point
        for point in points
        if point.maximum_bridge_strength == bridge_strength
    )
    return {
        "maximum_bridge_strength": bridge_strength,
        "point_count": len(selected),
        "mean_observed_task_retention": _mean(
            tuple(point.observed_task_retention for point in selected)
        ),
        "mean_absolute_task_product_residual": _mean(
            tuple(abs(point.task_product_residual) for point in selected)
        ),
        "mean_transported_certified_fraction": _mean(
            tuple(point.transported_certified_fraction for point in selected)
        ),
        "mean_directional_certified_fraction": _mean(
            tuple(point.directional_certified_fraction for point in selected)
        ),
        "mean_global_shifted_certified_fraction": _mean(
            tuple(point.global_shifted_certified_fraction for point in selected)
        ),
    }


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _write_payload(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("development", "confirmation"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot-artifact", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--structure-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.structure_only:
        seeds = (
            PREREGISTERED_DEVELOPMENT_SEEDS
            if arguments.phase == "development"
            else PREREGISTERED_CONFIRMATION_SEEDS
        )
        _write_payload(
            arguments.output,
            structure_gate_payload(trial_seeds=seeds),
        )
        return
    if arguments.pilot_artifact is None or arguments.expected_model_sha256 is None:
        parser.error("task実行にはpilot artifactとexpected model hashが必要です")
    models, model_sha256 = load_frozen_models(
        arguments.pilot_artifact,
        expected_sha256=arguments.expected_model_sha256,
    )
    result = (
        run_multicomponent_development(fitted_models=models)
        if arguments.phase == "development"
        else run_multicomponent_confirmation(fitted_models=models)
    )
    _write_payload(
        arguments.output,
        confirmation_summary_payload(
            result,
            fixed_model_sha256=model_sha256,
        ),
    )


if __name__ == "__main__":
    main()
