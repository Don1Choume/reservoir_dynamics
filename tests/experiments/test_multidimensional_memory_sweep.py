import math
import unittest

from reservoir_dynamics.experiments.multidimensional_memory_sweep import (
    EXPERIMENT_ID,
    run_multidimensional_memory_sweep,
)


class MultidimensionalMemorySweepTest(unittest.TestCase):
    def test_measures_stability_sync_and_memory_for_each_condition(self) -> None:
        result = run_multidimensional_memory_sweep(
            state_dimension=4,
            recurrent_gains=(0.6, 1.2),
            input_gains=(0.2,),
            trial_seeds=(17,),
            washout=30,
            training_steps=80,
            testing_steps=40,
            max_delay=4,
            tail_window=20,
        )

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(result.points), 2)
        self.assertTrue(
            all(len(point.memory_curve) == 4 for point in result.points)
        )
        self.assertTrue(
            all(math.isfinite(point.top_conditional_lyapunov_exponent)
                for point in result.points)
        )
        self.assertTrue(
            all(0.0 <= point.linear_memory_capacity <= 4.000001
                for point in result.points)
        )
        self.assertTrue(
            all(point.tail_replica_rms_distance >= 0.0
                for point in result.points)
        )

    def test_small_sweep_is_deterministic(self) -> None:
        experiment_arguments = {
            "state_dimension": 3,
            "recurrent_gains": (0.7,),
            "input_gains": (0.3,),
            "trial_seeds": (23,),
            "washout": 20,
            "training_steps": 60,
            "testing_steps": 30,
            "max_delay": 3,
            "tail_window": 10,
        }

        first_result = run_multidimensional_memory_sweep(
            **experiment_arguments,
        )
        second_result = run_multidimensional_memory_sweep(
            **experiment_arguments,
        )

        self.assertEqual(first_result, second_result)

    def test_rejects_empty_sweep_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "recurrent_gains"):
            run_multidimensional_memory_sweep(recurrent_gains=())

if __name__ == "__main__":
    unittest.main()
