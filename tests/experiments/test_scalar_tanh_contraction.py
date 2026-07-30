import json
import subprocess
import sys
import unittest

from reservoir_dynamics.experiments.scalar_tanh_contraction import (
    EXPERIMENT_ID,
    run_scalar_tanh_contraction_experiment,
)


class ScalarTanhContractionExperimentTest(unittest.TestCase):
    def test_observed_replica_distance_respects_global_bound(self) -> None:
        result = run_scalar_tanh_contraction_experiment()

        self.assertEqual(result.experiment_id, EXPERIMENT_ID)
        self.assertTrue(result.passed)
        self.assertLessEqual(result.maximum_bound_violation, result.tolerance)
        self.assertEqual(
            len(result.observed_replica_distances),
            len(result.theoretical_distance_bounds),
        )
        self.assertLess(
            result.observed_replica_distances[-1],
            result.observed_replica_distances[0],
        )

    def test_module_entrypoint_outputs_json_without_runtime_warning(self) -> None:
        completed_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "reservoir_dynamics.experiments.scalar_tanh_contraction",
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
