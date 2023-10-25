import datetime
import unittest
from copy import deepcopy

import polars as pl
from polars.testing import assert_frame_equal

from mix_n_match.main import ResampleData


class TestResample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        timestamps = [
            "2021-01-01 00:00:00",
            "2021-01-01 00:00:01",
            "2021-01-01 00:01:00",
            "2021-01-01 00:05:00",
            "2021-01-01 01:00:00",
            "2021-01-01 06:00:00",
            "2021-12-31 23:55:00",
            "2022-01-01 00:00:00",
        ]
        date_fmt = "%Y-%m-%d %H:%M:%S"
        timestamps_in_epoch_time = [
            int(datetime.datetime.strptime(timestamp, date_fmt).strftime("%s"))
            for timestamp in timestamps
        ]
        cls.dataframe = pl.DataFrame(
            {
                "timestamp_string": timestamps,
                "date": timestamps,
                "values": timestamps_in_epoch_time,
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

    def test_resample(self):
        dataframe = deepcopy(self.dataframe)

        def _filter_dataframe(dataframe, column, filter_list):
            return (
                dataframe.lazy()
                .filter(pl.col(column).is_in(filter_list))
                .collect()
            )

        # -- test 1: test resampling left closed (default), label left (default)
        processor = ResampleData("date", "1d", "sum")
        df_transformed = processor.transform(dataframe)

        df_expected = pl.DataFrame(
            {
                "date": [
                    "2021-01-01 00:00:00",
                    "2021-12-31 00:00:00",
                    "2022-01-01 00:00:00",
                ],
                "values": [
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-01-01 00:00:00",
                            "2021-01-01 00:00:01",
                            "2021-01-01 00:01:00",
                            "2021-01-01 00:05:00",
                            "2021-01-01 01:00:00",
                            "2021-01-01 06:00:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe, "timestamp_string", ["2021-12-31 23:55:00"]
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe, "timestamp_string", ["2022-01-01 00:00:00"]
                    )["values"].sum(),
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        assert_frame_equal(
            df_expected[["date", "values"]], df_transformed[["date", "values"]]
        )

        # -- test 2: test resampling right closed, label left (default)
        processor = ResampleData(
            "date", "1d", "sum", closed_boundaries="right"
        )
        df_transformed = processor.transform(dataframe)

        df_expected = pl.DataFrame(
            {
                "date": [
                    "2020-12-31 00:00:00",
                    "2021-01-01 00:00:00",
                    "2021-12-31 00:00:00",
                ],
                "values": [
                    _filter_dataframe(
                        dataframe, "timestamp_string", ["2021-01-01 00:00:00"]
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-01-01 00:00:01",
                            "2021-01-01 00:01:00",
                            "2021-01-01 00:05:00",
                            "2021-01-01 01:00:00",
                            "2021-01-01 06:00:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        ["2021-12-31 23:55:00", "2022-01-01 00:00:00"],
                    )["values"].sum(),
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        assert_frame_equal(
            df_expected[["date", "values"]], df_transformed[["date", "values"]]
        )

        # -- test 3: test resampling left closed, label right
        processor = ResampleData(
            "date",
            "1d",
            "sum",
            closed_boundaries="left",
            labelling_strategy="right",
        )
        df_transformed = processor.transform(dataframe)

        df_expected = pl.DataFrame(
            {
                "date": [
                    "2021-01-02 00:00:00",
                    "2022-01-01 00:00:00",
                    "2022-01-02 00:00:00",
                ],
                "values": [
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-01-01 00:00:00",
                            "2021-01-01 00:00:01",
                            "2021-01-01 00:01:00",
                            "2021-01-01 00:05:00",
                            "2021-01-01 01:00:00",
                            "2021-01-01 06:00:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-12-31 23:55:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe, "timestamp_string", ["2022-01-01 00:00:00"]
                    )["values"].sum(),
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        assert_frame_equal(
            df_expected[["date", "values"]], df_transformed[["date", "values"]]
        )

        # -- test 4: test resampling left right, label right
        processor = ResampleData(
            "date",
            "1d",
            "sum",
            closed_boundaries="right",
            labelling_strategy="right",
        )
        df_transformed = processor.transform(dataframe)

        df_expected = pl.DataFrame(
            {
                "date": [
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2022-01-01 00:00:00",
                ],
                "values": [
                    _filter_dataframe(
                        dataframe, "timestamp_string", ["2021-01-01 00:00:00"]
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-01-01 00:00:01",
                            "2021-01-01 00:01:00",
                            "2021-01-01 00:05:00",
                            "2021-01-01 01:00:00",
                            "2021-01-01 06:00:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        ["2021-12-31 23:55:00", "2022-01-01 00:00:00"],
                    )["values"].sum(),
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        assert_frame_equal(
            df_expected[["date", "values"]], df_transformed[["date", "values"]]
        )

        # -- test 2: test resampling manual offset
        processor = ResampleData("date", "1d", "sum", start_window_offset="6h")
        df_transformed = processor.transform(dataframe)
        df_expected = pl.DataFrame(
            {
                "date": [
                    "2020-12-31 00:00:00",
                    "2021-01-01 00:00:00",
                    "2021-12-31 00:00:00",
                ],
                "values": [
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-01-01 00:00:00",
                            "2021-01-01 00:00:01",
                            "2021-01-01 00:01:00",
                            "2021-01-01 00:05:00",
                            "2021-01-01 01:00:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2021-01-01 06:00:00",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        ["2021-12-31 23:55:00", "2022-01-01 00:00:00"],
                    )["values"].sum(),
                ],
            }
        ).with_columns(
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )

        assert_frame_equal(
            df_expected[["date", "values"]], df_transformed[["date", "values"]]
        )


if __name__ == "__main__":
    unittest.main()
