"""EXP-2026-018のtask-free構造artifactを保存するCLI。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from reservoir_dynamics.experiments.partition_robustness import (
    PartitionPerturbationPoint,
    PartitionRobustnessResult,
    run_partition_robustness_confirmation,
    run_partition_robustness_development,
)


def structure_result_payload(
    result: PartitionRobustnessResult,
) -> dict[str, object]:
    """全pointと判断に必要な集約を欠落なくJSON互換へ変換する。"""

    amplitudes = tuple(
        sorted({point.relative_amplitude for point in result.points})
    )
    radii = tuple(
        {
            (
                point.trial_seed,
                point.internal_gain,
                point.maximum_bridge_strength,
            ): point.certified_entrywise_radius
            for point in result.points
        }.values()
    )
    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "task_values_generated": result.task_values_generated,
        "base_network_count": result.base_network_count,
        "point_count": len(result.points),
        "decisions": asdict(result.decisions),
        "group_class_counts": [
            {"group": group, "class_count": count}
            for group, count in result.group_class_counts
        ],
        "certified_radius_summary": {
            "minimum": min(radii),
            "mean": _mean(radii),
            "maximum": max(radii),
        },
        "amplitude_summaries": [
            _amplitude_summary(result.points, amplitude)
            for amplitude in amplitudes
        ],
        "points": [asdict(point) for point in result.points],
    }


def _amplitude_summary(
    points: tuple[PartitionPerturbationPoint, ...],
    relative_amplitude: float,
) -> dict[str, object]:
    selected = tuple(
        point
        for point in points
        if point.relative_amplitude == relative_amplitude
    )
    disagreements = tuple(
        point.pair_disagreement
        for point in selected
        if point.pair_disagreement is not None
    )
    normalized_affinity_changes = tuple(
        (
            point.maximum_affinity_change / point.absolute_amplitude
            if point.absolute_amplitude > 0.0
            else 0.0
        )
        for point in selected
    )
    return {
        "relative_amplitude": relative_amplitude,
        "point_count": len(selected),
        "inference_success_rate": (
            sum(point.inference_succeeded for point in selected) / len(selected)
        ),
        "partition_recovery_rate": (
            sum(point.partition_recovered for point in selected) / len(selected)
        ),
        "mean_pair_disagreement": (
            _mean(disagreements) if disagreements else None
        ),
        "maximum_pair_disagreement": (
            max(disagreements) if disagreements else None
        ),
        "maximum_affinity_change_to_bound_ratio": max(
            normalized_affinity_changes
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
    arguments = parser.parse_args()
    result = (
        run_partition_robustness_development()
        if arguments.phase == "development"
        else run_partition_robustness_confirmation()
    )
    _write_payload(arguments.output, structure_result_payload(result))


if __name__ == "__main__":
    main()
