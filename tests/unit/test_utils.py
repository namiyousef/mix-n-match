import unittest

import numpy as np
import polars as pl

from mix_n_match.utils import (
    PolarsDuration,
    detect_timeseries_frequency,
    find_contiguous_segments,
)


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
        expected_indices = [[0, 0]]

        assert indices == expected_indices

        # -- test all different segments
        array = np.arange(0, 3)
        indices = find_contiguous_segments(array)
        expected_indices = [[0, 0], [1, 1], [2, 2]]

        assert indices == expected_indices

        # -- test repeated contiguous segment
        array = np.array([0, 0, 1, 1, 0, 0])
        indices = find_contiguous_segments(array)
        expected_indices = [[0, 1], [2, 3], [4, 5]]

        assert indices == expected_indices

        # -- test filtered items on boundary
        array = np.array(
            [0, 0, 1, 1, 0, 0],
        )
        indices = find_contiguous_segments(array, array == 0)
        expected_indices = [[0, 1], [4, 5]]

        assert indices == expected_indices

        # -- test filtered items in middle
        array = np.array(
            [0, 0, 1, 1, 0, 0],
        )
        indices = find_contiguous_segments(array, array == 1)
        expected_indices = [[2, 3]]

        assert indices == expected_indices

        # -- test all filtered
        array = np.array([0, 0, 1, 1, 0, 0])
        indices = find_contiguous_segments(array, array >= 0)
        expected_indices = [[0, 1], [2, 3], [4, 5]]

        assert indices == expected_indices

        # -- test empty after filter
        array = np.array(
            [0, 0, 1, 1, 0, 0],
        )
        indices = find_contiguous_segments(array, array < 0)
        expected_indices = []

        assert indices == expected_indices

        # -- test min length
        array = np.array(
            [0, 0, 1, 1, 0, 0],
        )
        indices = find_contiguous_segments(array, min_length=2)
        expected_indices = [[0, 1], [2, 3], [4, 5]]

        assert indices == expected_indices

        array = np.array(
            [0, 0, 1, 1, 0, 0, 0],
        )
        indices = find_contiguous_segments(array, min_length=3)
        expected_indices = [[4, 6]]

        assert indices == expected_indices

    def test_detect_timeseries_frequency(self):
        # -- simple case
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01 00:45:00",
                    "2023-01-01 00:30:00",
                    "2023-01-01 00:15:00",
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
        frequency = detect_timeseries_frequency(df, time_column="date")

        assert frequency == 15 * 60  # 15 mins

        # -- case with missing data (gaps)
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01 01:15:00",
                    "2023-01-01 00:45:00",
                    "2023-01-01 00:30:00",
                    "2023-01-01 00:15:00",
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        # 1: fail with mode `exact` (default)
        with self.assertRaises(ValueError):
            frequency = detect_timeseries_frequency(df, time_column="date")

        # 2: get most frequent with mode `mode`
        frequency = detect_timeseries_frequency(
            df, time_column="date", how="mode"
        )

        assert frequency == 15 * 60  # 15 mins

        # 3: get worst case (max) with mode `max`

        frequency = detect_timeseries_frequency(
            df, time_column="date", how="max"
        )

        assert frequency == 15 * 60 * 2  # 30 mins

        # -- case with duplicated data
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01 01:15:00",
                    "2023-01-01 01:15:00",
                    "2023-01-01 01:15:00",
                    "2023-01-01 01:15:00",
                    "2023-01-01 00:45:00",
                    "2023-01-01 00:30:00",
                    "2023-01-01 00:15:00",
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        frequency = detect_timeseries_frequency(
            df, time_column="date", how="mode"
        )

        assert frequency == 15 * 60  # 15 mins


if __name__ == "__main__":
    unittest.main()
