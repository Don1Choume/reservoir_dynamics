"""globalとcomponent-aware nested ridgeの固定比較。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from reservoir_dynamics.experiments.component_profile import (
    COMPONENT_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    PRODUCT_FEATURE_NAMES,
    ComponentProfilePoint,
)
from reservoir_dynamics.experiments.robust_repertoire_task import _spearman
from reservoir_dynamics.metrics.standardized_ridge import (
    StandardizedRidgeModel,
    fit_standardized_ridge,
)

PREREGISTERED_RIDGE_PENALTY = 0.001


@dataclass(frozen=True, slots=True)
class NamedComponentPredictor:
    """特徴名と標準化統計を固定した一つの予測器。"""

    name: str
    feature_names: tuple[str, ...]
    model: StandardizedRidgeModel

    def predict(self, point: ComponentProfilePoint) -> float:
        return self.model.predict(
            _feature_row(point, self.name),
            clip_to_unit_interval=True,
        )


@dataclass(frozen=True, slots=True)
class ComponentPrediction:
    """seed集約bootstrapに使える一条件の予測誤差。"""

    trial_seed: int
    module_sizes: tuple[int, ...]
    internal_gain: float
    maximum_bridge_strength: float
    disturbance_bound: float
    observed_task_retention: float
    predicted_task_retention: float
    absolute_error: float


@dataclass(frozen=True, slots=True)
class ComponentPredictorEvaluation:
    """一modelのout-of-sample予測要約。"""

    name: str
    feature_names: tuple[str, ...]
    mae: float
    spearman: float
    predictions: tuple[ComponentPrediction, ...]


def fixed_predictor_feature_shift_bound(
    model: NamedComponentPredictor,
    first: ComponentProfilePoint,
    second: ComponentProfilePoint,
) -> float:
    """固定standardized ridgeの特徴変化に対する予測差上界を返す。"""

    first_row = _feature_row(first, model.name)
    second_row = _feature_row(second, model.name)
    if len(first_row) != len(second_row) or len(first_row) != len(
        model.model.coefficients
    ):
        raise ValueError("比較する特徴量次元を固定modelと一致させてください")
    if any(
        not math.isfinite(value)
        for value in first_row + second_row
    ):
        raise ValueError("比較する特徴量は有限値にしてください")
    return math.fsum(
        abs(coefficient) * abs(first_value - second_value) / scale
        for coefficient, scale, first_value, second_value in zip(
            model.model.coefficients,
            model.model.feature_scales,
            first_row,
            second_row,
            strict=True,
        )
    )


def fit_preregistered_component_models(
    points: tuple[ComponentProfilePoint, ...],
) -> tuple[NamedComponentPredictor, ...]:
    """事前固定した三modelを同じtraining pointへfitする。"""

    _validate_points(points, require_multiple_seeds=False)
    return tuple(
        _fit_named_model(points, name)
        for name in (
            "component_aware",
            "global_profile",
            "product_only",
        )
    )


def leave_one_seed_out_evaluations(
    points: tuple[ComponentProfilePoint, ...],
) -> tuple[ComponentPredictorEvaluation, ...]:
    """pilot seedを一つずつ完全に除外し、model選択なしで評価する。"""

    trial_seeds = _validate_points(points, require_multiple_seeds=True)
    predictions_by_name: dict[str, list[ComponentPrediction]] = {
        "component_aware": [],
        "global_profile": [],
        "product_only": [],
    }
    for held_out_seed in trial_seeds:
        training_points = tuple(
            point for point in points if point.trial_seed != held_out_seed
        )
        testing_points = tuple(
            point for point in points if point.trial_seed == held_out_seed
        )
        for model in fit_preregistered_component_models(training_points):
            predictions_by_name[model.name].extend(
                _predict_points(model, testing_points)
            )
    return tuple(
        _summarize_predictions(name, tuple(predictions_by_name[name]))
        for name in (
            "component_aware",
            "global_profile",
            "product_only",
        )
    )


def evaluate_fixed_component_models(
    *,
    models: tuple[NamedComponentPredictor, ...],
    points: tuple[ComponentProfilePoint, ...],
) -> tuple[ComponentPredictorEvaluation, ...]:
    """再fitせず、固定modelを未知familyへ適用する。"""

    _validate_points(points, require_multiple_seeds=False)
    expected_names = (
        "component_aware",
        "global_profile",
        "product_only",
    )
    if tuple(model.name for model in models) != expected_names:
        raise ValueError("modelsは事前固定順の三modelにしてください")
    return tuple(
        _summarize_predictions(
            model.name,
            _predict_points(model, points),
        )
        for model in models
    )


def predictors_to_payload(
    models: tuple[NamedComponentPredictor, ...],
) -> tuple[dict[str, object], ...]:
    """固定modelをJSON互換な不変順序のpayloadへ変換する。"""

    expected_names = (
        "component_aware",
        "global_profile",
        "product_only",
    )
    if tuple(model.name for model in models) != expected_names:
        raise ValueError("modelsは事前固定順の三modelにしてください")
    return tuple(
        {
            "name": model.name,
            "feature_names": list(model.feature_names),
            "intercept": model.model.intercept,
            "coefficients": list(model.model.coefficients),
            "feature_means": list(model.model.feature_means),
            "feature_scales": list(model.model.feature_scales),
            "penalty": model.model.penalty,
        }
        for model in models
    )


def predictors_from_payload(
    payload: Sequence[Mapping[str, object]],
) -> tuple[NamedComponentPredictor, ...]:
    """pilot artifactの係数を再fitせず厳密に復元する。"""

    models = tuple(_predictor_from_mapping(value) for value in payload)
    expected_names = (
        "component_aware",
        "global_profile",
        "product_only",
    )
    if tuple(model.name for model in models) != expected_names:
        raise ValueError("pilot model名または順序が事前登録と一致しません")
    return models


def predictor_payload_sha256(
    payload: Sequence[Mapping[str, object]],
) -> str:
    """model payloadを順序・float表現込みで再現可能なhashへ固定する。"""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _fit_named_model(
    points: tuple[ComponentProfilePoint, ...],
    name: str,
) -> NamedComponentPredictor:
    feature_names = _feature_names(name)
    model = fit_standardized_ridge(
        tuple(_feature_row(point, name) for point in points),
        tuple(point.observed_task_retention for point in points),
        penalty=PREREGISTERED_RIDGE_PENALTY,
    )
    return NamedComponentPredictor(
        name=name,
        feature_names=feature_names,
        model=model,
    )


def _predictor_from_mapping(
    payload: Mapping[str, object],
) -> NamedComponentPredictor:
    try:
        name = str(payload["name"])
        feature_names = tuple(str(value) for value in payload["feature_names"])
        coefficients = tuple(float(value) for value in payload["coefficients"])
        feature_means = tuple(float(value) for value in payload["feature_means"])
        feature_scales = tuple(float(value) for value in payload["feature_scales"])
        intercept = float(payload["intercept"])
        penalty = float(payload["penalty"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pilot model payloadの形式が不正です") from error
    expected_features = _feature_names(name)
    if feature_names != expected_features:
        raise ValueError("pilot feature名が事前登録と一致しません")
    feature_count = len(feature_names)
    if not (
        len(coefficients)
        == len(feature_means)
        == len(feature_scales)
        == feature_count
    ):
        raise ValueError("pilot modelの特徴量次元が一致しません")
    numeric_values = (
        intercept,
        penalty,
        *coefficients,
        *feature_means,
        *feature_scales,
    )
    if (
        any(not math.isfinite(value) for value in numeric_values)
        or penalty != PREREGISTERED_RIDGE_PENALTY
        or any(scale <= 0.0 for scale in feature_scales)
    ):
        raise ValueError("pilot modelの数値またはpenaltyが不正です")
    return NamedComponentPredictor(
        name=name,
        feature_names=feature_names,
        model=StandardizedRidgeModel(
            intercept=intercept,
            coefficients=coefficients,
            feature_means=feature_means,
            feature_scales=feature_scales,
            penalty=penalty,
        ),
    )


def _predict_points(
    model: NamedComponentPredictor,
    points: tuple[ComponentProfilePoint, ...],
) -> tuple[ComponentPrediction, ...]:
    return tuple(
        _predict_point(model, point) for point in points
    )


def _predict_point(
    model: NamedComponentPredictor,
    point: ComponentProfilePoint,
) -> ComponentPrediction:
    prediction = model.predict(point)
    return ComponentPrediction(
        trial_seed=point.trial_seed,
        module_sizes=point.module_sizes,
        internal_gain=point.internal_gain,
        maximum_bridge_strength=point.maximum_bridge_strength,
        disturbance_bound=point.disturbance_bound,
        observed_task_retention=point.observed_task_retention,
        predicted_task_retention=prediction,
        absolute_error=abs(point.observed_task_retention - prediction),
    )


def _summarize_predictions(
    name: str,
    predictions: tuple[ComponentPrediction, ...],
) -> ComponentPredictorEvaluation:
    if not predictions:
        raise ValueError("predictionsは1件以上必要です")
    return ComponentPredictorEvaluation(
        name=name,
        feature_names=_feature_names(name),
        mae=math.fsum(
            prediction.absolute_error for prediction in predictions
        )
        / len(predictions),
        spearman=_spearman(
            tuple(
                prediction.predicted_task_retention
                for prediction in predictions
            ),
            tuple(
                prediction.observed_task_retention
                for prediction in predictions
            ),
        ),
        predictions=predictions,
    )


def _feature_names(name: str) -> tuple[str, ...]:
    mapping = {
        "component_aware": COMPONENT_FEATURE_NAMES,
        "global_profile": GLOBAL_FEATURE_NAMES,
        "product_only": PRODUCT_FEATURE_NAMES,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"未対応model名です: {name}") from error


def _feature_row(
    point: ComponentProfilePoint,
    name: str,
) -> tuple[float, ...]:
    row_getters: dict[str, Callable[[ComponentProfilePoint], tuple[float, ...]]] = {
        "component_aware": lambda value: value.component_feature_row,
        "global_profile": lambda value: value.global_feature_row,
        "product_only": lambda value: value.product_feature_row,
    }
    try:
        return row_getters[name](point)
    except KeyError as error:
        raise ValueError(f"未対応model名です: {name}") from error


def _validate_points(
    points: tuple[ComponentProfilePoint, ...],
    *,
    require_multiple_seeds: bool,
) -> tuple[int, ...]:
    if not points:
        raise ValueError("pointsは1件以上必要です")
    trial_seeds = tuple(sorted({point.trial_seed for point in points}))
    if require_multiple_seeds and len(trial_seeds) < 3:
        raise ValueError("leave-one-seed-outには3 seed以上必要です")
    if any(
        not math.isfinite(value)
        for point in points
        for value in point.component_feature_row
        + (point.observed_task_retention,)
    ):
        raise ValueError("pointの特徴とtargetは有限値にしてください")
    return trial_seeds
