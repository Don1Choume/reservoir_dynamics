import json
import math
import subprocess
import sys
import unittest

from reservoir_dynamics.experiments.input_conditioned_stability_sweep import (
    EXPERIMENT_ID,
    run_input_conditioned_stability_sweep,
)


class InputConditionedStabilitySweepTest(unittest.TestCase):
    def test_distinguishes_global_contraction_local_stability_and_sync(self) -> None:
        result = run_input_conditioned_stability_sweep()
        points = {
            (point.recurrent_gain, point.input_gain): point
            for point in result.points
        }

        globally_contractive = points[(0.6, 0.0)]
        locally_stable_multistable = points[(1.2, 0.0)]
        input_synchronized = points[(1.2, 4.0)]

        self.assertTrue(globally_contractive.global_contraction_guaranteed)
        self.assertTrue(globally_contractive.replica_synchronized)
        self.assertLess(
            locally_stable_multistable.conditional_lyapunov_exponent,
            0.0,
        )
        self.assertFalse(locally_stable_multistable.replica_synchronized)
        self.assertFalse(input_synchronized.global_contraction_guaranteed)
        self.assertLess(input_synchronized.conditional_lyapunov_exponent, 0.0)
        self.assertTrue(input_synchronized.replica_synchronized)
        self.assertTrue(result.local_stability_without_global_sync_observed)
        self.assertTrue(result.sync_beyond_global_contraction_observed)

    def test_is_deterministic(self) -> None:
        first_result = run_input_conditioned_stability_sweep()
        second_result = run_input_conditioned_stability_sweep()

        self.assertEqual(first_result, second_result)

    def test_rejects_invalid_evaluation_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps"):
            run_input_conditioned_stability_sweep(
                steps=500,
                washout=500,
            )

        with self.assertRaisesRegex(ValueError, "tail_window"):
            run_input_conditioned_stability_sweep(tail_window=0)

    def test_rejects_non_positive_synchronization_tolerance(self) -> None:
        with self.assertRaisesRegex(ValueError, "synchronization_tolerance"):
            run_input_conditioned_stability_sweep(
                synchronization_tolerance=0.0,
            )

        with self.assertRaisesRegex(ValueError, "synchronization_tolerance"):
            run_input_conditioned_stability_sweep(
                synchronization_tolerance=math.nan,
            )

    def test_module_entrypoint_outputs_machine_readable_summary(self) -> None:
        completed_process = subprocess.run(
            [
                sys.executable,
                "-m",
                (
                    "reservoir_dynamics.experiments."
                    "input_conditioned_stability_sweep"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        output = json.loads(completed_process.stdout)
        self.assertEqual(output["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(completed_process.stderr, "")


if __name__ == "__main__":
    unittest.main()
