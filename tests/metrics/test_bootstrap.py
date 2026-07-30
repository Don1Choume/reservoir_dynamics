import unittest

from reservoir_dynamics.metrics.bootstrap import bootstrap_mean_interval


class BootstrapMeanIntervalTest(unittest.TestCase):
    def test_constant_sample_has_degenerate_interval(self) -> None:
        interval = bootstrap_mean_interval(
            (3.5, 3.5, 3.5),
            resamples=200,
            random_seed=19,
        )

        self.assertEqual(interval.estimate, 3.5)
        self.assertEqual(interval.lower, 3.5)
        self.assertEqual(interval.upper, 3.5)

    def test_is_deterministic_and_contains_sample_mean(self) -> None:
        first_interval = bootstrap_mean_interval(
            (1.0, 2.0, 3.0, 4.0),
            resamples=500,
            random_seed=31,
        )
        second_interval = bootstrap_mean_interval(
            (1.0, 2.0, 3.0, 4.0),
            resamples=500,
            random_seed=31,
        )

        self.assertEqual(first_interval, second_interval)
        self.assertEqual(first_interval.estimate, 2.5)
        self.assertLess(first_interval.lower, 2.5)
        self.assertGreater(first_interval.upper, 2.5)

    def test_rejects_invalid_sample_and_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "2要素"):
            bootstrap_mean_interval((1.0,))

        with self.assertRaisesRegex(ValueError, "confidence_level"):
            bootstrap_mean_interval(
                (1.0, 2.0),
                confidence_level=1.0,
            )

        with self.assertRaisesRegex(ValueError, "resamples"):
            bootstrap_mean_interval(
                (1.0, 2.0),
                resamples=0,
            )


if __name__ == "__main__":
    unittest.main()
