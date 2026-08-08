import unittest

from reservoir_dynamics.experiments.multicomponent_modular_family import (
    build_multicomponent_modular_network,
)
from reservoir_dynamics.experiments.multicomponent_profile import (
    evaluate_multicomponent_partitions,
    evaluate_multicomponent_profile,
)
from reservoir_dynamics.metrics.module_partition import (
    infer_affinity_gap_partition,
)


class MultiComponentProfileTest(unittest.TestCase):
    def test_multiple_partitions_share_target_and_honor_weight_override(self) -> None:
        network = build_multicomponent_modular_network(
            trial_seed=2203,
            module_sizes=(2, 2, 3),
            internal_gain=0.025,
            maximum_total_bridge_strength=0.02,
        )
        mutable_weights = [list(row) for row in network.recurrent_weights]
        mutable_weights[0][1] += 0.005
        perturbed_weights = tuple(tuple(row) for row in mutable_weights)
        alternative_partition = ((0, 1, 2), (3, 4), (5, 6))

        profiles = evaluate_multicomponent_partitions(
            network=network,
            recurrent_weights=perturbed_weights,
            partitions=(network.true_partition, alternative_partition),
            disturbance_bounds=(0.12,),
            task_steps=10,
            autonomous_steps=500,
            convergence_tolerance=1e-9,
        )

        self.assertEqual(len(profiles), 2)
        oracle_point = profiles[0][0]
        alternative_point = profiles[1][0]
        self.assertEqual(
            oracle_point.observed_task_retention,
            alternative_point.observed_task_retention,
        )
        self.assertEqual(
            oracle_point.full_off_diagonal_infinity_norm,
            alternative_point.full_off_diagonal_infinity_norm,
        )
        self.assertNotEqual(
            oracle_point.component_feature_row,
            alternative_point.component_feature_row,
        )

    def test_inferred_and_oracle_profiles_match_after_coordinate_permutation(self) -> None:
        network = build_multicomponent_modular_network(
            trial_seed=2201,
            module_sizes=(2, 2, 3),
            internal_gain=0.025,
            maximum_total_bridge_strength=0.02,
        )
        inferred = infer_affinity_gap_partition(network.recurrent_weights)
        arguments = {
            "disturbance_bounds": (0.12,),
            "task_steps": 10,
            "autonomous_steps": 500,
            "convergence_tolerance": 1e-9,
        }

        inferred_point = evaluate_multicomponent_profile(
            network=network,
            partition=inferred.components,
            **arguments,
        )[0]
        oracle_point = evaluate_multicomponent_profile(
            network=network,
            partition=network.true_partition,
            **arguments,
        )[0]

        self.assertEqual(inferred_point.component_feature_row, oracle_point.component_feature_row)
        self.assertAlmostEqual(
            inferred_point.observed_task_retention,
            oracle_point.observed_task_retention,
        )
        self.assertAlmostEqual(
            inferred_point.factorized_directional_certified_fraction,
            inferred_point.enumerated_directional_certified_fraction,
        )

    def test_zero_coupling_factorizes_and_multicomponent_chain_holds(self) -> None:
        arguments = {
            "disturbance_bounds": (0.12,),
            "task_steps": 10,
            "autonomous_steps": 500,
            "convergence_tolerance": 1e-9,
        }
        zero_network = build_multicomponent_modular_network(
            trial_seed=2202,
            module_sizes=(2, 2, 3),
            internal_gain=0.025,
            maximum_total_bridge_strength=0.0,
        )
        coupled_network = build_multicomponent_modular_network(
            trial_seed=2202,
            module_sizes=(2, 2, 3),
            internal_gain=0.025,
            maximum_total_bridge_strength=0.02,
        )

        zero_point = evaluate_multicomponent_profile(
            network=zero_network,
            partition=infer_affinity_gap_partition(
                zero_network.recurrent_weights
            ).components,
            **arguments,
        )[0]
        coupled_point = evaluate_multicomponent_profile(
            network=coupled_network,
            partition=infer_affinity_gap_partition(
                coupled_network.recurrent_weights
            ).components,
            **arguments,
        )[0]

        self.assertAlmostEqual(
            zero_point.observed_task_retention,
            zero_point.isolated_task_product,
        )
        self.assertGreaterEqual(
            coupled_point.observed_task_retention + 1e-12,
            coupled_point.transported_certified_fraction,
        )
        self.assertGreaterEqual(
            coupled_point.transported_certified_fraction + 1e-12,
            coupled_point.directional_certified_fraction,
        )
        self.assertGreaterEqual(
            coupled_point.directional_certified_fraction + 1e-12,
            coupled_point.global_shifted_certified_fraction,
        )
        self.assertEqual(coupled_point.local_orthant_count, 16)
        self.assertEqual(coupled_point.global_orthant_count, 128)


if __name__ == "__main__":
    unittest.main()
