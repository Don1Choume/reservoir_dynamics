import math
import unittest

from reservoir_dynamics.metrics.replica import pairwise_replica_distance_curve


class PairwiseReplicaDistanceCurveTest(unittest.TestCase):
    def test_identical_replicas_have_zero_distance(self) -> None:
        trajectories = (
            ((1.0, 2.0), (3.0, 4.0)),
            ((1.0, 2.0), (3.0, 4.0)),
        )

        self.assertEqual(pairwise_replica_distance_curve(trajectories), (0.0, 0.0))

    def test_returns_coordinate_normalized_root_mean_square_distance(self) -> None:
        trajectories = (
            ((0.0, 0.0), (1.0, 1.0)),
            ((2.0, 0.0), (3.0, 3.0)),
        )

        distances = pairwise_replica_distance_curve(trajectories)

        self.assertAlmostEqual(distances[0], math.sqrt(2.0))
        self.assertAlmostEqual(distances[1], 2.0)

    def test_averages_over_all_replica_pairs(self) -> None:
        trajectories = (
            ((0.0,),),
            ((1.0,),),
            ((3.0,),),
        )

        expected = math.sqrt((1.0 + 9.0 + 4.0) / 3.0)
        self.assertAlmostEqual(pairwise_replica_distance_curve(trajectories)[0], expected)

    def test_rejects_single_replica(self) -> None:
        with self.assertRaisesRegex(ValueError, "2つ以上"):
            pairwise_replica_distance_curve((((0.0,),),))

    def test_rejects_inconsistent_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "時系列長"):
            pairwise_replica_distance_curve(
                (
                    ((0.0,), (1.0,)),
                    ((0.0,),),
                )
            )

    def test_rejects_inconsistent_state_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "状態次元"):
            pairwise_replica_distance_curve(
                (
                    ((0.0, 1.0),),
                    ((0.0,),),
                )
            )

    def test_rejects_non_finite_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限"):
            pairwise_replica_distance_curve(
                (
                    ((0.0,),),
                    ((math.inf,),),
                )
            )


if __name__ == "__main__":
    unittest.main()
