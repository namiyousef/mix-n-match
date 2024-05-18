import unittest

import numpy as np

from mix_n_match.utils import PolarsDuration, find_contiguous_segments


class TestUtils(unittest.TestCase):
    def test_PolarsDuration(self):
        duration = "1ns"
        assert PolarsDuration(duration)._decompose_duration(duration) == [
            (1, "ns")
        ]

        duration = "3d12h4m25s"
        assert PolarsDuration(duration)._decompose_duration(duration) == [
            (3, "d"),
            (12, "h"),
            (4, "m"),
            (25, "s"),
        ]

    def test_find_contiguous_segments(self):
        # -- test single contiguous segment
        array = np.array([0])
        indices = find_contiguous_segments(array)
        expected_indices = [[0, 1]]

        assert indices == expected_indices

        # -- test all different segments
        array = np.arange(0, 3)
        indices = find_contiguous_segments(array)
        expected_indices = [[0, 1], [1, 2], [2, 3]]

        assert indices == expected_indices

        # -- test repeated contiguous segment
        array = np.array([0, 0, 1, 1, 0, 0])
        indices = find_contiguous_segments(array)
        expected_indices = [[0, 2], [2, 4], [4, 6]]

        assert indices == expected_indices


if __name__ == "__main__":
    unittest.main()
