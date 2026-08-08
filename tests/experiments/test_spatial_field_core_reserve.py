import unittest

from reservoir_dynamics.experiments.spatial_field_core_reserve import (
    EXPERIMENT_ID,
    audit_spatial_field_structures,
    run_spatial_field_study,
)


class SpatialFieldCoreReserveStudyTest(unittest.TestCase):
    def test_small_study_is_deterministic_and_satisfies_theory_gates(self) -> None:
        arguments = {
            "trial_seeds": (31, 37),
            "feedback_gains": (0.08,),
            "disturbance_bounds": (0.0, 0.02),
            "washout": 10,
            "training_steps": 30,
            "testing_steps": 20,
            "max_delay": 3,
            "bootstrap_resamples": 50,
            "minimum_local_safe_box_retention": 1.0,
            "minimum_max_feedback_core_advantage": -1.0,
            "minimum_core_advantage_lower": -1.0,
            "minimum_reserve_advantage_lower": -1.0,
        }

        first = run_spatial_field_study(**arguments)
        second = run_spatial_field_study(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(first.experiment_id, EXPERIMENT_ID)
        self.assertEqual(len(first.points), 4)
        self.assertTrue(first.decisions.field_hypercube_invariant)
        self.assertTrue(first.decisions.intervention_energy_matched)
        self.assertTrue(first.decisions.certificate_lower_bound_valid)
        self.assertTrue(first.decisions.structures_effectively_distinct)
        self.assertTrue(first.decisions.local_core_fully_protected)
        self.assertTrue(first.decisions.maximum_feedback_advantage)
        self.assertTrue(first.decisions.local_beats_global_core)
        self.assertTrue(first.decisions.local_beats_global_reserve)
        self.assertTrue(all(point.core_dimension == 3 for point in first.points))
        self.assertTrue(all(point.reserve_dimension == 5 for point in first.points))

    def test_structure_gate_has_unique_magnitudes_and_asymmetric_blocks(self) -> None:
        gate = audit_spatial_field_structures(
            trial_seeds=(41, 43, 47),
            feedback_gains=(0.08, 0.12),
        )

        self.assertEqual(gate.raw_network_count, 6)
        self.assertEqual(gate.effective_magnitude_class_count, 6)
        self.assertTrue(gate.all_reserve_blocks_asymmetric)
        self.assertTrue(gate.all_bridges_bidirectional_nonzero)
        self.assertTrue(gate.unequal_module_sizes)

    def test_rejects_invalid_seed_and_disturbance_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial_seeds"):
            run_spatial_field_study(trial_seeds=(1,))
        with self.assertRaisesRegex(ValueError, "disturbance_bounds"):
            run_spatial_field_study(
                trial_seeds=(1, 2),
                disturbance_bounds=(0.0, 0.3),
            )


if __name__ == "__main__":
    unittest.main()
