import unittest

import polars as pl

from mix_n_match.main import FilterDataBasedOnTime

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TestFilterDataBasedOnTime(unittest.TestCase):
    """def test_time_pattern_parsing(self):
    # -- test reject if multiple boundaries provided

    time_patterns = [">1h<3h>5h"]

    with self.assertRaises(ValueError):
        processor = FilterDataBasedOnTime(time_column="dummy", time_patterns=time_patterns)  # noqa

    # -- test patterns
    time_patterns = [">1h"]
    processor = FilterDataBasedOnTime(time_column="dummy", time_patterns=time_patterns)  # noqa
    assert processor.filtering_rules == [
        [
            [{"value": 1, "unit_method": "hour", "operator_method": 'gt'}]
        ]
    ]

    time_patterns = [">1h, <3h"]
    processor = FilterDataBasedOnTime(time_column="dummy", time_patterns=time_patterns)  # noqa
    assert processor.filtering_rules == [
        [
            [{"value": 1, "unit_method": "hour", "operator_method": 'gt'}],
            [{"value": 3, "unit_method": "hour", "operator_method": 'gt'}]

        ]
    ]

    # -- test multiple patterns




    pass"""

    def test_filter_data_based_on_time(self):
        df = pl.DataFrame(
            {
                "date": [
                    "2023-01-01 00:00:00",
                    "2023-01-01 01:00:00",
                    "2023-01-01 01:15:00",
                    "2023-01-01 02:00:00",
                    "2023-01-01 03:00:00",
                    "2023-01-01 04:00:00",
                    "2023-01-01 05:00:00",
                    "2023-01-01 06:00:00",
                    "2023-01-01 07:00:00",
                    "2023-01-01 08:00:00",
                    "2023-01-01 09:00:00",
                    "2023-01-02 00:00:00",
                    "2023-03-01 00:00:00",
                    "2024-01-01 00:00:00",
                ]
            }
        ).with_columns(pl.col("date").str.strptime(pl.Datetime))

        # -- remove any dates where hour > 1. Note: this does not include 01:15:00 since we make no filter on minutes  # noqa
        processor = FilterDataBasedOnTime(
            time_column="date", time_patterns=[">1h"]
        )

        df_transformed = processor.transform(df)

        assert df_transformed["date"].dt.strftime(DATE_FORMAT).to_list() == [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 01:15:00",
            "2023-01-02 00:00:00",
            "2023-03-01 00:00:00",
            "2024-01-01 00:00:00",
        ]

        # -- remove any dates where hour > 1
        processor = FilterDataBasedOnTime(
            time_column="date", time_patterns=[">1h*"]
        )

        df_transformed = processor.transform(df)

        assert df_transformed["date"].dt.strftime(DATE_FORMAT).to_list() == [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-02 00:00:00",
            "2023-03-01 00:00:00",
            "2024-01-01 00:00:00",
        ]

        # -- remove any times between 1-8 am
        processor = FilterDataBasedOnTime(
            time_column="date", time_patterns=[">1h*<8h"]
        )

        df_transformed = processor.transform(df)

        assert df_transformed["date"].dt.strftime(DATE_FORMAT).to_list() == [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 08:00:00",
            "2023-01-01 09:00:00",
            "2023-01-02 00:00:00",
            "2023-03-01 00:00:00",
            "2024-01-01 00:00:00",
        ]


if __name__ == "__main__":
    unittest.main()
