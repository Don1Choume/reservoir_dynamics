import unittest

from reservoir_dynamics.experiments.multicomponent_modular_family import (
    audit_multicomponent_structure,
    build_multicomponent_modular_network,
)
from reservoir_dynamics.metrics.module_partition import (
    infer_affinity_gap_partition,
    partitions_equivalent,
)


class MultiComponentModularFamilyTest(unittest.TestCase):
    def test_generator_hides_but_preserves_exact_three_component_partition(self) -> None:
        network = build_multicomponent_modular_network(
            trial_seed=2201,
            module_sizes=(2, 2, 3),
            internal_gain=0.025,
            maximum_total_bridge_strength=0.04,
        )

        inferred = infer_affinity_gap_partition(network.recurrent_weights)

        self.assertEqual(len(network.recurrent_weights), 7)
        self.assertTrue(partitions_equivalent(inferred.components, network.true_partition))
        self.assertTrue(network.internal_blocks_asymmetric)
        self.assertTrue(network.all_module_pairs_bidirectional)
        self.assertTrue(network.bridges_nontransposed)
        self.assertTrue(network.partition_separation.separated)

    def test_structure_gate_is_task_free_and_counts_unique_networks(self) -> None:
        gate = audit_multicomponent_structure(
            trial_seeds=(2201, 2202, 2203),
            module_sizes=(2, 2, 3),
            internal_gains=(0.025,),
            maximum_total_bridge_strengths=(0.0, 0.02),
        )

        self.assertTrue(gate.passed)
        self.assertTrue(gate.partition_recovery_exact)
        self.assertTrue(gate.affinity_separation_valid)
        self.assertTrue(gate.fingerprints_unique)
        self.assertFalse(gate.task_values_generated)
        self.assertEqual(
            tuple(count for _, count in gate.group_class_counts),
            (3, 3),
        )

    def test_rejects_invalid_module_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "module_sizes"):
            build_multicomponent_modular_network(
                trial_seed=2201,
                module_sizes=(2, 5),
                internal_gain=0.025,
                maximum_total_bridge_strength=0.02,
            )


if __name__ == "__main__":
    unittest.main()
