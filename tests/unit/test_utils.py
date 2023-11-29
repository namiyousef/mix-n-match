import unittest

from mix_n_match.utils import PolarsDuration


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


if __name__ == "__main__":
    unittest.main()
