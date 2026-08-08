import unittest

from reservoir_dynamics.metrics.module_partition import (
    certify_affinity_gap_partition,
    infer_affinity_gap_partition,
    maximum_pair_affinity_change,
    partition_pair_disagreement,
    partition_separation,
    partitions_equivalent,
)
from reservoir_dynamics.simulation.weight_perturbation import (
    sample_entrywise_bounded_perturbation,
)


class ModulePartitionTest(unittest.TestCase):
    def test_recovers_noncontiguous_components_from_unique_affinity_gap(self) -> None:
        weights = _separated_weights()

        result = infer_affinity_gap_partition(weights)

        expected = ((0, 3), (1, 4), (2, 5))
        self.assertTrue(partitions_equivalent(result.components, expected))
        self.assertTrue(result.gap_is_unique)
        self.assertGreater(result.selected_gap, 0.1)
        separation = partition_separation(weights, expected)
        self.assertTrue(separation.separated)
        self.assertGreater(
            separation.minimum_within_affinity,
            separation.maximum_between_affinity,
        )

    def test_rejects_matrix_without_an_affinity_gap(self) -> None:
        weights = (
            (1.5, 0.1, 0.1),
            (0.1, 1.5, 0.1),
            (0.1, 0.1, 1.5),
        )

        with self.assertRaisesRegex(ValueError, "gap"):
            infer_affinity_gap_partition(weights)

    def test_rejects_invalid_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "partition"):
            partition_separation(
                _separated_weights(),
                ((0, 1), (1, 2, 3, 4, 5)),
            )

    def test_certified_radius_preserves_partition_under_subradius_noise(self) -> None:
        weights = _separated_weights()
        certificate = certify_affinity_gap_partition(weights)

        perturbed = sample_entrywise_bounded_perturbation(
            weights,
            maximum_absolute_change=0.9 * certificate.certified_entrywise_radius,
            random_seed=18,
        )
        inferred = infer_affinity_gap_partition(perturbed)

        self.assertTrue(certificate.guaranteed)
        self.assertGreater(certificate.selected_gap, certificate.runner_up_gap)
        self.assertGreater(certificate.gap_dominance, 0.0)
        self.assertGreater(certificate.certified_entrywise_radius, 0.0)
        self.assertTrue(
            partitions_equivalent(
                certificate.partition.components,
                inferred.components,
            )
        )

    def test_nonunique_maximum_gap_has_zero_certified_radius(self) -> None:
        weights = _nonunique_gap_weights()

        certificate = certify_affinity_gap_partition(weights)

        self.assertFalse(certificate.partition.gap_is_unique)
        self.assertFalse(certificate.guaranteed)
        self.assertAlmostEqual(certificate.selected_gap, certificate.runner_up_gap)
        self.assertEqual(certificate.gap_dominance, 0.0)
        self.assertEqual(certificate.certified_entrywise_radius, 0.0)

    def test_pair_disagreement_is_label_free_and_normalized(self) -> None:
        first = ((0, 1), (2, 3))
        relabeled = ((3, 2), (1, 0))
        crossed = ((0, 2), (1, 3))

        self.assertEqual(partition_pair_disagreement(first, relabeled), 0.0)
        self.assertAlmostEqual(
            partition_pair_disagreement(first, crossed),
            4.0 / 6.0,
        )

    def test_pair_disagreement_rejects_different_node_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "node"):
            partition_pair_disagreement(
                ((0, 1), (2, 3)),
                ((0, 1), (2, 4)),
            )

    def test_pair_affinity_change_obeys_entrywise_bound(self) -> None:
        weights = _separated_weights()
        perturbed = sample_entrywise_bounded_perturbation(
            weights,
            maximum_absolute_change=0.03,
            random_seed=1804,
        )

        maximum_change = maximum_pair_affinity_change(weights, perturbed)

        self.assertLessEqual(maximum_change, 0.03 + 1e-15)


def _separated_weights() -> tuple[tuple[float, ...], ...]:
    dimension = 6
    modules = ((0, 3), (1, 4), (2, 5))
    matrix = [
        [1.5 if row == column else 0.006 for column in range(dimension)]
        for row in range(dimension)
    ]
    for module_index, module in enumerate(modules):
        first, second = module
        matrix[first][second] = 0.18 + 0.01 * module_index
        matrix[second][first] = -0.17 - 0.01 * module_index
    return tuple(tuple(row) for row in matrix)


def _nonunique_gap_weights() -> tuple[tuple[float, ...], ...]:
    affinities = {
        (0, 1): 0.32,
        (0, 2): 0.33,
        (1, 2): 0.53,
        (0, 3): 0.10,
        (1, 3): 0.11,
        (2, 3): 0.12,
    }
    return tuple(
        tuple(
            1.5
            if row == column
            else affinities[tuple(sorted((row, column)))]
            for column in range(4)
        )
        for row in range(4)
    )


if __name__ == "__main__":
    unittest.main()
