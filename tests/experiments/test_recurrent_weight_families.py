import unittest

from reservoir_dynamics.experiments.orthant_margin_sweep import (
    _signed_symmetric_weights,
)
from reservoir_dynamics.experiments.recurrent_weight_families import (
    RECURRENT_WEIGHT_FAMILIES,
    build_recurrent_weights,
)


class RecurrentWeightFamiliesTest(unittest.TestCase):
    def test_dense_symmetric_preserves_existing_seed_protocol(self) -> None:
        arguments = {
            "dimension": 4,
            "diagonal_gain": 1.5,
            "coupling_gain": 0.07,
            "trial_seed": 401,
        }

        expected = _signed_symmetric_weights(**arguments)
        actual = build_recurrent_weights(
            network_family="dense_symmetric",
            **arguments,
        )

        self.assertEqual(actual, expected)

    def test_family_topologies_are_structurally_distinct(self) -> None:
        matrices = {
            family: build_recurrent_weights(
                network_family=family,
                dimension=4,
                diagonal_gain=1.5,
                coupling_gain=0.07,
                trial_seed=409,
            )
            for family in RECURRENT_WEIGHT_FAMILIES
        }

        for matrix in matrices.values():
            self.assertEqual(
                tuple(matrix[index][index] for index in range(4)),
                (1.5, 1.5, 1.5, 1.5),
            )

        sparse = matrices["sparse_symmetric"]
        self.assertEqual(
            sum(
                sparse[row][column] != 0.0
                for row in range(4)
                for column in range(4)
                if row != column
            ),
            8,
        )
        self.assertTrue(_is_symmetric(sparse))

        modular = matrices["modular_paired"]
        self.assertEqual(
            sum(
                modular[row][column] != 0.0
                for row in range(4)
                for column in range(4)
                if row != column
            ),
            4,
        )
        self.assertTrue(_is_symmetric(modular))
        self.assertTrue(
            all(
                modular[row][column] == 0.0
                for row, column in (
                    (0, 2),
                    (0, 3),
                    (1, 2),
                    (1, 3),
                    (2, 0),
                    (3, 0),
                    (2, 1),
                    (3, 1),
                )
            )
        )

        asymmetric = matrices["asymmetric_dense"]
        self.assertFalse(_is_symmetric(asymmetric))
        self.assertTrue(
            all(
                asymmetric[row][column] != 0.0
                for row in range(4)
                for column in range(4)
                if row != column
            )
        )

        feedforward = matrices["feedforward_nonnormal"]
        self.assertTrue(
            all(
                feedforward[row][column] == 0.0
                for row in range(4)
                for column in range(row)
            )
        )
        self.assertTrue(
            any(
                feedforward[row][column] != 0.0
                for row in range(4)
                for column in range(row + 1, 4)
            )
        )
        self.assertFalse(_commutes_with_transpose(feedforward))

    def test_generation_is_deterministic_and_validates_inputs(self) -> None:
        arguments = {
            "network_family": "asymmetric_dense",
            "dimension": 4,
            "diagonal_gain": 1.5,
            "coupling_gain": 0.06,
            "trial_seed": 419,
        }
        self.assertEqual(
            build_recurrent_weights(**arguments),
            build_recurrent_weights(**arguments),
        )

        with self.assertRaisesRegex(ValueError, "network_family"):
            build_recurrent_weights(
                network_family="unknown",
                dimension=4,
                diagonal_gain=1.5,
                coupling_gain=0.06,
                trial_seed=419,
            )

        with self.assertRaisesRegex(ValueError, "dimension"):
            build_recurrent_weights(
                network_family="dense_symmetric",
                dimension=1,
                diagonal_gain=1.5,
                coupling_gain=0.06,
                trial_seed=419,
            )

    def test_heterogeneous_modules_vary_magnitude_within_fixed_range(
        self,
    ) -> None:
        matrix = build_recurrent_weights(
            network_family="modular_heterogeneous",
            dimension=4,
            diagonal_gain=1.5,
            coupling_gain=0.07,
            trial_seed=1401,
        )

        first_magnitude = abs(matrix[0][1])
        second_magnitude = abs(matrix[2][3])
        self.assertGreaterEqual(first_magnitude, 0.07 * 0.75)
        self.assertLess(first_magnitude, 0.07 * 1.25)
        self.assertGreaterEqual(second_magnitude, 0.07 * 0.75)
        self.assertLess(second_magnitude, 0.07 * 1.25)
        self.assertNotEqual(first_magnitude, second_magnitude)
        self.assertEqual(matrix[0][1], matrix[1][0])
        self.assertEqual(matrix[2][3], matrix[3][2])
        self.assertTrue(
            all(
                matrix[row][column] == 0.0
                for row, column in (
                    (0, 2),
                    (0, 3),
                    (1, 2),
                    (1, 3),
                    (2, 0),
                    (3, 0),
                    (2, 1),
                    (3, 1),
                )
            )
        )

def _is_symmetric(matrix: tuple[tuple[float, ...], ...]) -> bool:
    return all(
        matrix[row][column] == matrix[column][row]
        for row in range(len(matrix))
        for column in range(len(matrix))
    )


def _commutes_with_transpose(
    matrix: tuple[tuple[float, ...], ...],
) -> bool:
    size = len(matrix)
    matrix_times_transpose = tuple(
        tuple(
            sum(matrix[row][index] * matrix[column][index]
                for index in range(size))
            for column in range(size)
        )
        for row in range(size)
    )
    transpose_times_matrix = tuple(
        tuple(
            sum(matrix[index][row] * matrix[index][column]
                for index in range(size))
            for column in range(size)
        )
        for row in range(size)
    )
    return matrix_times_transpose == transpose_times_matrix


if __name__ == "__main__":
    unittest.main()
