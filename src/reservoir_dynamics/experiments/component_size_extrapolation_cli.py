"""EXP-2026-016を一度だけ実行し、固定modelを含むartifactを保存するCLI。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from reservoir_dynamics.experiments.asymmetric_modular_family import (
    audit_asymmetric_modular_structure,
)
from reservoir_dynamics.experiments.component_predictor import (
    predictor_payload_sha256,
    predictors_from_payload,
    predictors_to_payload,
)
from reservoir_dynamics.experiments.component_size_experiment import (
    PREREGISTERED_BRIDGE_STRENGTHS,
    PREREGISTERED_CONFIRMATION_MODULE_SIZES,
    PREREGISTERED_CONFIRMATION_SEEDS,
    PREREGISTERED_INTERNAL_GAINS,
    PREREGISTERED_PILOT_MODULE_SIZES,
    PREREGISTERED_PILOT_SEEDS,
    ComponentSizeConfirmationResult,
    ComponentSizePilotResult,
    run_component_size_confirmation,
    run_component_size_pilot,
)


def pilot_summary_payload(result: ComponentSizePilotResult) -> dict[str, object]:
    """pilot結果とconfirmationへ渡す係数をJSON互換にする。"""

    model_payload = predictors_to_payload(result.fitted_models)
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "structure_gate": asdict(result.structure_gate),
        "decisions": asdict(result.decisions),
        "point_count": len(result.points),
        "challenge_count": _challenge_count(result.points),
        "maximum_absolute_task_product_residual": max(
            abs(point.task_product_residual) for point in result.points
        ),
        "minimum_task_minus_transported_certificate": min(
            point.observed_task_retention
            - point.transported_certified_fraction
            for point in result.points
        ),
        "cross_validated_evaluations": [
            _evaluation_payload(value)
            for value in result.cross_validated_evaluations
        ],
        "fitted_models": model_payload,
        "model_sha256": predictor_payload_sha256(model_payload),
        "group_summaries": _group_summaries(result.points),
        "points": [asdict(point) for point in result.points],
    }


def confirmation_summary_payload(
    result: ComponentSizeConfirmationResult,
    *,
    pilot_model_sha256: str,
) -> dict[str, object]:
    """未知3+5 familyの予測・paired区間・保証を保存する。"""

    if len(pilot_model_sha256) != 64:
        raise ValueError("pilot_model_sha256は64文字にしてください")
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "pilot_model_sha256": pilot_model_sha256,
        "structure_gate": asdict(result.structure_gate),
        "decisions": asdict(result.decisions),
        "point_count": len(result.points),
        "challenge_count": _challenge_count(result.points),
        "maximum_absolute_task_product_residual": max(
            abs(point.task_product_residual) for point in result.points
        ),
        "minimum_task_minus_transported_certificate": min(
            point.observed_task_retention
            - point.transported_certified_fraction
            for point in result.points
        ),
        "evaluations": [
            _evaluation_payload(value) for value in result.evaluations
        ],
        "error_intervals": [
            asdict(value) for value in result.error_intervals
        ],
        "group_summaries": _group_summaries(result.points),
        "points": [asdict(point) for point in result.points],
    }


def structure_gate_payload(phase: str) -> dict[str, object]:
    """task値を生成せず、登録familyの絶対値classを監査する。"""

    if phase == "pilot":
        seeds = PREREGISTERED_PILOT_SEEDS
        module_sizes = PREREGISTERED_PILOT_MODULE_SIZES
    elif phase == "confirmation":
        seeds = PREREGISTERED_CONFIRMATION_SEEDS
        module_sizes = (PREREGISTERED_CONFIRMATION_MODULE_SIZES,)
    else:
        raise ValueError("phaseはpilotまたはconfirmationにしてください")
    gate = audit_asymmetric_modular_structure(
        trial_seeds=seeds,
        module_size_pairs=module_sizes,
        internal_gains=PREREGISTERED_INTERNAL_GAINS,
        maximum_bridge_strengths=PREREGISTERED_BRIDGE_STRENGTHS,
    )
    return {
        "experiment_id": "EXP-2026-016",
        "phase": phase,
        "task_values_generated": False,
        "structure_gate": asdict(gate),
    }


def _evaluation_payload(value: object) -> dict[str, object]:
    payload = asdict(value)
    predictions = payload.pop("predictions")
    return {**payload, "predictions": predictions}


def _group_summaries(points: tuple[object, ...]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                point.module_sizes,
                point.maximum_bridge_strength,
            )
            for point in points
        }
    )
    return [
        _group_summary(points, module_sizes, bridge_strength)
        for module_sizes, bridge_strength in keys
    ]


def _group_summary(
    points: tuple[object, ...],
    module_sizes: tuple[int, int],
    bridge_strength: float,
) -> dict[str, object]:
    selected = tuple(
        point
        for point in points
        if point.module_sizes == module_sizes
        and point.maximum_bridge_strength == bridge_strength
    )
    return {
        "module_sizes": list(module_sizes),
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


def _challenge_count(points: tuple[object, ...]) -> int:
    return sum(4 * (2 ** sum(point.module_sizes)) for point in points)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _write_payload(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("pilot", "confirmation"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot-artifact", type=Path)
    parser.add_argument("--structure-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.structure_only:
        _write_payload(arguments.output, structure_gate_payload(arguments.phase))
        return
    if arguments.phase == "pilot":
        _write_payload(
            arguments.output,
            pilot_summary_payload(run_component_size_pilot()),
        )
        return
    if arguments.pilot_artifact is None:
        parser.error("confirmationには--pilot-artifactが必要です")
    pilot_payload = json.loads(arguments.pilot_artifact.read_text(encoding="utf-8"))
    models = predictors_from_payload(pilot_payload["fitted_models"])
    expected_hash = predictor_payload_sha256(predictors_to_payload(models))
    if pilot_payload.get("model_sha256") != expected_hash:
        raise ValueError("pilot artifactのmodel hashが一致しません")
    result = run_component_size_confirmation(fitted_models=models)
    _write_payload(
        arguments.output,
        confirmation_summary_payload(
            result,
            pilot_model_sha256=expected_hash,
        ),
    )


if __name__ == "__main__":
    main()
