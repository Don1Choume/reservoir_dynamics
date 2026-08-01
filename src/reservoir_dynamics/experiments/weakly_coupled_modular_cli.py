"""EXP-2026-014のpilotまたはconfirmation集約をJSON出力する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math

from reservoir_dynamics.experiments.weakly_coupled_modular_task import (
    WeakCouplingFactorizationPoint,
    WeakCouplingFactorizationResult,
    run_weak_coupling_factorization,
)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def _strength_summary(
    points: tuple[WeakCouplingFactorizationPoint, ...],
    strength: float,
) -> dict[str, object]:
    selected = tuple(
        point
        for point in points
        if point.cross_coupling_strength == strength
    )
    absolute_residuals = tuple(
        point.absolute_task_product_residual for point in selected
    )
    return {
        "cross_coupling_strength": strength,
        "point_count": len(selected),
        "mean_signed_task_product_residual": _mean(
            tuple(point.task_product_residual for point in selected)
        ),
        "mean_absolute_task_product_residual": _mean(absolute_residuals),
        "maximum_absolute_task_product_residual": max(absolute_residuals),
        "nonzero_residual_fraction": (
            sum(value > 1e-12 for value in absolute_residuals)
            / len(absolute_residuals)
        ),
        "coupled_task_retention_range": (
            min(point.coupled_task_retention for point in selected),
            max(point.coupled_task_retention for point in selected),
        ),
        "coupled_raw_attractor_count_values": tuple(
            sorted({point.coupled_raw_attractor_count for point in selected})
        ),
        "mean_transported_rectangle_certified_fraction": _mean(
            tuple(
                point.transported_rectangle_certified_fraction
                for point in selected
            )
        ),
        "mean_norm_shifted_certified_fraction": _mean(
            tuple(point.norm_shifted_certified_fraction for point in selected)
        ),
    }


def summary_payload(
    result: WeakCouplingFactorizationResult,
) -> dict[str, object]:
    """artifactへ保存する集約要約を構成する。"""

    return {
        "experiment_id": result.experiment_id,
        "phase": result.phase,
        "dimension": result.dimension,
        "diagonal_gain": result.diagonal_gain,
        "task_steps": result.task_steps,
        "autonomous_steps": result.autonomous_steps,
        "convergence_tolerance": result.convergence_tolerance,
        "structure_gate": asdict(result.structure_gate),
        "point_count": len(result.points),
        "challenge_count": len(result.points) * 256,
        "decisions": asdict(result.decisions),
        "maximum_absolute_task_product_residual": (
            result.maximum_absolute_task_product_residual
        ),
        "minimum_task_minus_transported_certificate": (
            result.minimum_task_minus_transported_certificate
        ),
        "minimum_task_minus_norm_shifted_certificate": (
            result.minimum_task_minus_norm_shifted_certificate
        ),
        "minimum_transported_minus_norm_shifted_certificate": (
            result.minimum_transported_minus_norm_shifted_certificate
        ),
        "strength_summaries": tuple(
            _strength_summary(result.points, strength)
            for strength in result.structure_gate.cross_coupling_strengths
        ),
    }


def main() -> None:
    """事前登録したphaseを選び、標準出力へ要約を返す。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("pilot", "confirmation"),
        default="pilot",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            summary_payload(
                run_weak_coupling_factorization(phase=arguments.phase)
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
