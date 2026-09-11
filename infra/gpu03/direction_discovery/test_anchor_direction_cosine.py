import unittest

from .anchor_direction_cosine import signed_cosine


class CosineTests(unittest.TestCase):
    def test_identical_opposite_orthogonal_and_zero(self):
        self.assertAlmostEqual(signed_cosine([1, 2], [2, 4])['cosine'], 1)
        self.assertAlmostEqual(signed_cosine([1, 2], [-2, -4])['cosine'], -1)
        self.assertAlmostEqual(signed_cosine([1, 0], [0, 1])['angle_degrees'], 90)
        self.assertIsNone(signed_cosine([0, 0], [1, 0])['cosine'])

    def test_invalid_dimensions_or_nonfinite_rejected(self):
        for a, b in (([1], [1,2]), ([[1]], [[1]]), ([float('nan')], [1]), ([1], [float('inf')])):
            with self.assertRaises(ValueError):
                signed_cosine(a, b)


if __name__ == '__main__':
    unittest.main()
