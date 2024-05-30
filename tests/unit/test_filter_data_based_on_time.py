import unittest

import polars as pl

from mix_n_match.main import FilterDataBasedOnTime
from mix_n_match.utils import generate_polars_condition

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TestFilterDataBasedOnTime(unittest.TestCase):
    def test_parse_time_pattern(self):
        # -- basic pattern
        duration = "1h*"
        for (
            operator
        ) in FilterDataBasedOnTime.OPERATOR_TO_POLARS_METHOD_MAPPING:
            (
                duration_strings,
                operators,
            ) = FilterDataBasedOnTime._parse_time_pattern(
                self=None, pattern=f"{operator}{duration}"
            )

            assert duration_strings == [duration]
            assert operators == [operator]

        # -- repeating pattern
        pattern = ">1h1h"
        (
            duration_strings,
            operators,
        ) = FilterDataBasedOnTime._parse_time_pattern(
            self=None, pattern=pattern
        )

        assert duration_strings == ["1h1h"]
        assert operators == [">"]

        # -- repeating pattern with condition
        pattern = ">1h>1h"
        (
            duration_strings,
            operators,
        ) = FilterDataBasedOnTime._parse_time_pattern(
            self=None, pattern=pattern
        )

        assert duration_strings == ["1h", "1h"]
        assert operators == [">", ">"]

        # -- repeating pattern with different condition
        pattern = ">1h<1h"
        (
            duration_strings,
            operators,
        ) = FilterDataBasedOnTime._parse_time_pattern(
            self=None, pattern=pattern
        )

        assert duration_strings == ["1h", "1h"]
        assert operators == [">", "<"]

    def test_create_rule_metadata_from_condition(self):
        # -- test simple
        duration_string = "1d1h"
        operator = ">"
        rule_metadata = (
            FilterDataBasedOnTime._create_rule_metadata_from_condition(
                self=None, duration_string=duration_string, operator=operator
            )
        )

        assert rule_metadata == {
            "operator": "gt",
            "decomposed_duration": [(1, "d"), (1, "h")],
            "how": "simple",
        }

        # -- test cascade
        duration_string = "1h*"
        operator = ">="
        rule_metadata = (
            FilterDataBasedOnTime._create_rule_metadata_from_condition(
                self=None, duration_string=duration_string, operator=operator
            )
        )

        assert rule_metadata == {
            "operator": "ge",
            "decomposed_duration": [(1, "h")],
            "how": "cascade",
        }

        # -- test invalid cascade operators

        with self.assertRaises(NotImplementedError):
            rule_metadata = (
                FilterDataBasedOnTime._create_rule_metadata_from_condition(
                    self=None, duration_string=duration_string, operator="!="
                )
            )

        with self.assertRaises(NotImplementedError):
            rule_metadata = (
                FilterDataBasedOnTime._create_rule_metadata_from_condition(
                    self=None, duration_string=duration_string, operator="=="
                )
            )

        # -- test invalid cascade durations

        with self.assertRaises(ValueError):
            rule_metadata = (
                FilterDataBasedOnTime._create_rule_metadata_from_condition(
                    self=None, duration_string="1q*", operator=operator
                )
            )

        with self.assertRaises(ValueError):
            rule_metadata = (
                FilterDataBasedOnTime._create_rule_metadata_from_condition(
                    self=None, duration_string="1w*", operator=operator
                )
            )

    def test_generate_simple_condition(self):
        # -- simple example
        processor = FilterDataBasedOnTime(
            time_column="date", time_patterns=[">1h"]  # dummy
        )
        expression = processor._generate_simple_condition("h", 1, "lt")

        expected_expression = pl.col("date").dt.hour() < 1

        assert str(expression) == str(expected_expression)

    def test_generate_cascade_condition(self):
        # -- simple example
        processor = FilterDataBasedOnTime(
            time_column="date", time_patterns=[">1h"]  # dummy
        )
        expression = processor._generate_cascade_condition("d", 1, "gt")

        expressions = [
            pl.col("date").dt.hour() > 0,
            pl.col("date").dt.minute() > 0,
            pl.col("date").dt.second() > 0,
            pl.col("date").dt.millisecond() > 0,
            pl.col("date").dt.microsecond() > 0,
            pl.col("date").dt.nanosecond() > 0,
        ]

        or_expression = generate_polars_condition(expressions, "or_")
        expected_expression = or_expression.and_(
            (pl.col("date").dt.day() == 1)
        ).or_((pl.col("date").dt.day() > 1))

        assert str(expression) == str(expected_expression)

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
