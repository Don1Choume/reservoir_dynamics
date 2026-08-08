"""EXP-2026-019を固定modelで実行し、再現可能なJSONへ保存する。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from reservoir_dynamics.experiments.component_predictor import (
    predictors_from_payload,
)
from reservoir_dynamics.experiments.partition_functional_impact import (
    PartitionFunctionalImpactPoint,
    PartitionFunctionalImpactResult,
    run_partition_functional_confirmation,
    run_partition_functional_development,
)


def result_payload(result: PartitionFunctionalImpactResult) -> dict[str, object]:
    """abstentionと条件付き機能誤差を混同しないJSON payloadを作る。"""

    amplitudes = tuple(
        sorted({value.relative_amplitude for value in result.perturbations})
    )
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "model_sha256": result.model_sha256,
        "base_network_count": result.base_network_count,
        "perturbation_count": len(result.perturbations),
        "functional_point_count": len(result.points),
        "inference_completed_before_task": result.inference_completed_before_task,
        "group_class_counts": [
            {"group": group, "class_count": count}
            for group, count in result.group_class_counts
        ],
        "decisions": asdict(result.decisions),
        "amplitude_summaries": [
            _amplitude_summary(result, amplitude) for amplitude in amplitudes
        ],
        "perturbations": [asdict(value) for value in result.perturbations],
        "points": [asdict(value) for value in result.points],
    }


def _amplitude_summary(
    result: PartitionFunctionalImpactResult,
    relative_amplitude: float,
) -> dict[str, object]:
    observations = tuple(
        value
        for value in result.perturbations
        if value.relative_amplitude == relative_amplitude
    )
    points = tuple(
        point
        for point in result.points
        if point.relative_amplitude == relative_amplitude
    )
    inferred = tuple(value for value in observations if value.inference_succeeded)
    component_impacts = tuple(point.prediction_impacts[0] for point in points)
    mispartition_points = tuple(point for point in points if point.pair_disagreement > 0.0)
    return {
        "relative_amplitude": relative_amplitude,
        "perturbation_count": len(observations),
        "inference_success_rate": sum(
            value.inference_succeeded for value in observations
        ) / len(observations),
        "partition_recovery_rate": sum(
            value.partition_recovered for value in observations
        ) / len(observations),
        "conditional_recovery_rate": (
            sum(value.partition_recovered for value in inferred) / len(inferred)
            if inferred
            else None
        ),
        "mean_pair_disagreement": _mean_or_none(
            tuple(
                value.pair_disagreement
                for value in inferred
                if value.pair_disagreement is not None
            )
        ),
        "functional_point_count": len(points),
        "mispartition_functional_point_count": len(mispartition_points),
        "mean_component_prediction_shift": _mean_or_none(
            tuple(value.absolute_prediction_shift for value in component_impacts)
        ),
        "mean_component_prediction_bound": _mean_or_none(
            tuple(value.feature_shift_bound for value in component_impacts)
        ),
        "mean_component_error_penalty": _mean_or_none(
            tuple(value.absolute_error_penalty for value in component_impacts)
        ),
        "mean_feature_rmse": _mean_or_none(
            tuple(point.component_feature_rmse for point in points)
        ),
        "mean_directional_certificate_shift": _mean_or_none(
            tuple(
                point.inferred_directional_certificate
                - point.oracle_directional_certificate
                for point in points
            )
        ),
        "mean_transported_certificate_shift": _mean_or_none(
            tuple(
                point.inferred_transported_certificate
                - point.oracle_transported_certificate
                for point in points
            )
        ),
        "mean_local_orthant_count_ratio": _mean_or_none(
            tuple(point.local_orthant_count_ratio for point in points)
        ),
    }


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return math.fsum(values) / len(values)


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXP-2026-019のpartition条件付き機能影響を評価します",
    )
    parser.add_argument(
        "--phase",
        choices=("development", "confirmation"),
        required=True,
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=Path(
            "docs/research/artifacts/EXP-2026-016-pilot-summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """固定phaseを一度実行し、完全payloadを標準出力とfileへ保存する。"""

    parsed = _parse_arguments(arguments)
    model_artifact = json.loads(
        parsed.model_artifact.read_text(encoding="utf-8")
    )
    models = predictors_from_payload(model_artifact["fitted_models"])
    runner = (
        run_partition_functional_development
        if parsed.phase == "development"
        else run_partition_functional_confirmation
    )
    payload = result_payload(runner(fitted_models=models))
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
