import unittest

from reservoir_dynamics.metrics.structural_equivalence import (
    audit_signed_coordinate_conjugacy,
    signed_coordinate_conjugacy_witness,
    weakly_connected_components,
)


class StructuralEquivalenceTest(unittest.TestCase):
    def test_finds_a_signed_coordinate_conjugacy_witness(self) -> None:
        first = (
            (1.5, 0.2, -0.3),
            (0.4, 1.6, 0.5),
            (0.7, -0.8, 1.7),
        )
        expected_witness = (1, -1, 1)
        second = _conjugate(first, expected_witness)

        actual_witness = signed_coordinate_conjugacy_witness(
            first,
            second,
        )

        self.assertEqual(actual_witness, expected_witness)

    def test_rejects_inconsistent_cycle_and_magnitude_change(self) -> None:
        triangle = (
            (1.5, 0.2, 0.2),
            (0.2, 1.5, 0.2),
            (0.2, 0.2, 1.5),
        )
        inconsistent_cycle = (
            (1.5, 0.2, -0.2),
            (0.2, 1.5, 0.2),
            (-0.2, 0.2, 1.5),
        )
        changed_magnitude = (
            (1.5, 0.2, 0.2),
            (0.2, 1.5, 0.3),
            (0.2, 0.3, 1.5),
        )

        self.assertIsNone(
            signed_coordinate_conjugacy_witness(
                triangle,
                inconsistent_cycle,
            )
        )
        self.assertIsNone(
            signed_coordinate_conjugacy_witness(
                triangle,
                changed_magnitude,
            )
        )

    def test_audit_counts_effective_structural_classes(self) -> None:
        representative = (
            (1.5, 0.1, 0.0, 0.0),
            (0.1, 1.5, 0.0, 0.0),
            (0.0, 0.0, 1.5, 0.1),
            (0.0, 0.0, 0.1, 1.5),
        )
        matrices = (
            representative,
            _conjugate(representative, (1, -1, 1, 1)),
            _conjugate(representative, (1, 1, 1, -1)),
            (
                (1.5, 0.1, 0.0, 0.0),
                (0.1, 1.5, 0.0, 0.0),
                (0.0, 0.0, 1.5, 0.2),
                (0.0, 0.0, 0.2, 1.5),
            ),
        )

        audit = audit_signed_coordinate_conjugacy(matrices)

        self.assertEqual(audit.raw_network_count, 4)
        self.assertEqual(audit.effective_class_count, 2)
        self.assertEqual(audit.representative_indices, (0, 3))
        self.assertEqual(audit.class_index_by_network, (0, 0, 0, 1))
        self.assertEqual(audit.class_sizes, (3, 1))
        self.assertAlmostEqual(audit.effective_fraction, 0.5)

    def test_validates_matrix_shape_values_and_tolerance(self) -> None:
        valid = ((1.5, 0.1), (0.1, 1.5))

        with self.assertRaisesRegex(ValueError, "同じ次元"):
            signed_coordinate_conjugacy_witness(
                valid,
                ((1.5,),),
            )
        with self.assertRaisesRegex(ValueError, "正方行列"):
            signed_coordinate_conjugacy_witness(
                valid,
                ((1.5, 0.1),),
            )
        with self.assertRaisesRegex(ValueError, "有限値"):
            signed_coordinate_conjugacy_witness(
                valid,
                ((1.5, float("nan")), (0.1, 1.5)),
            )
        with self.assertRaisesRegex(ValueError, "tolerance"):
            signed_coordinate_conjugacy_witness(
                valid,
                valid,
                tolerance=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "空"):
            audit_signed_coordinate_conjugacy(())

    def test_weak_components_ignore_edge_direction(self) -> None:
        matrix = (
            (1.5, 0.2, 0.0, 0.0),
            (0.0, 1.5, 0.0, 0.0),
            (0.0, 0.0, 1.5, 0.0),
            (0.0, 0.0, -0.3, 1.5),
        )

        self.assertEqual(
            weakly_connected_components(matrix),
            ((0, 1), (2, 3)),
        )


def _conjugate(
    matrix: tuple[tuple[float, ...], ...],
    signs: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            signs[row] * matrix[row][column] * signs[column]
            for column in range(len(matrix))
        )
        for row in range(len(matrix))
    )


if __name__ == "__main__":
    unittest.main()
