import math
import unittest

from reservoir_dynamics.metrics.standardized_ridge import (
    StandardizedRidgeModel,
    fit_standardized_ridge,
)


class StandardizedRidgeTest(unittest.TestCase):
    def test_recovers_exact_affine_plane_without_penalty(self) -> None:
        feature_rows = (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 1.0),
            (1.0, 2.0),
        )
        targets = tuple(
            2.0 + 3.0 * first - 4.0 * second
            for first, second in feature_rows
        )

        model = fit_standardized_ridge(
            feature_rows,
            targets,
            penalty=0.0,
        )

        self.assertIsInstance(model, StandardizedRidgeModel)
        for feature_row, target in zip(
            feature_rows,
            targets,
            strict=True,
        ):
            self.assertAlmostEqual(
                model.predict(feature_row),
                target,
                places=10,
            )

    def test_constant_feature_uses_unit_scale_under_ridge(self) -> None:
        model = fit_standardized_ridge(
            (
                (5.0, 0.0),
                (5.0, 1.0),
                (5.0, 2.0),
                (5.0, 3.0),
            ),
            (0.0, 1.0, 2.0, 3.0),
            penalty=0.1,
        )

        self.assertEqual(model.feature_scales[0], 1.0)
        self.assertEqual(model.coefficients[0], 0.0)
        self.assertTrue(math.isfinite(model.predict((5.0, 1.5))))

    def test_prediction_can_be_clipped_to_probability_range(self) -> None:
        model = fit_standardized_ridge(
            ((0.0,), (1.0,)),
            (-1.0, 2.0),
            penalty=0.0,
        )

        self.assertAlmostEqual(model.predict((0.0,)), -1.0)
        self.assertAlmostEqual(
            model.predict((0.0,), clip_to_unit_interval=True),
            0.0,
        )
        self.assertAlmostEqual(
            model.predict((1.0,), clip_to_unit_interval=True),
            1.0,
        )

    def test_rejects_invalid_training_and_prediction_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "1行以上"):
            fit_standardized_ridge((), ())

        with self.assertRaisesRegex(ValueError, "列数"):
            fit_standardized_ridge(
                ((1.0, 2.0), (3.0,)),
                (1.0, 2.0),
            )

        with self.assertRaisesRegex(ValueError, "target"):
            fit_standardized_ridge(
                ((1.0,), (2.0,)),
                (1.0,),
            )

        with self.assertRaisesRegex(ValueError, "有限"):
            fit_standardized_ridge(
                ((1.0,), (float("nan"),)),
                (1.0, 2.0),
            )

        with self.assertRaisesRegex(ValueError, "penalty"):
            fit_standardized_ridge(
                ((1.0,), (2.0,)),
                (1.0, 2.0),
                penalty=-1.0,
            )

        model = fit_standardized_ridge(
            ((0.0,), (1.0,)),
            (0.0, 1.0),
            penalty=0.0,
        )
        with self.assertRaisesRegex(ValueError, "特徴量数"):
            model.predict((1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "有限"):
            model.predict((float("inf"),))


if __name__ == "__main__":
    unittest.main()
