import unittest

from reservoir_dynamics.metrics.module_partition import (
    infer_affinity_gap_partition,
    partition_separation,
    partitions_equivalent,
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


if __name__ == "__main__":
    unittest.main()
