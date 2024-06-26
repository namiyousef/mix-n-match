import datetime
import unittest
from copy import deepcopy
from functools import partial

import dateutil
import polars as pl
from polars.testing import assert_frame_equal

from mix_n_match.main import ResampleData

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _prepare_dataframe(timestamps, date_fmt, timezone=None):
    timestamps_in_epoch_time = [
        int(
            dateutil.parser.parse(
                f"{timestamp} UTC"
                if date_fmt == "%Y-%m-%d %H:%M:%S"
                else timestamp
            ).strftime("%s")
        )
        for timestamp in timestamps
    ]
    dataframe = pl.DataFrame(
        {
            "timestamp_string": timestamps,
            "date": timestamps,
            "values": timestamps_in_epoch_time,
        }
    ).with_columns(pl.col("date").str.strptime(pl.Datetime, date_fmt))

    if timezone is not None:
        dataframe = dataframe.with_columns(
            [pl.col("date").dt.replace_time_zone(timezone)]
        )

    return dataframe


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

        # -- test that 24 hour resampling freq same as daily
        dataframe = _prepare_dataframe(
            [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-02-01",
                "2023-02-02",
            ],
            "%Y-%m-%d",
        )
        processor = ResampleData(
            "date",
            "1mo",
            "sum",
            start_window_offset="1d",  # e.g. start on the second day of each month
        )
        df_expected = pl.DataFrame(
            {
                "date": [
                    "2022-12-01",
                    "2023-01-01",
                    "2023-02-01",
                ],
                "values": [
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2023-01-01",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2023-01-02",
                            "2023-01-03",
                            "2023-02-01",
                        ],
                    )["values"].sum(),
                    _filter_dataframe(
                        dataframe,
                        "timestamp_string",
                        [
                            "2023-02-02",
                        ],
                    )["values"].sum(),
                ],
            }
        ).with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d"))
        df_transformed = processor.transform(dataframe)

        assert_frame_equal(
            df_expected[["date", "values"]], df_transformed[["date", "values"]]
        )

        # -- test against partial data
        dataframe = _prepare_dataframe(
            [
                "2021-03-28 00:00:00",
                "2021-03-28 01:00:00",  # Note: 02:00:00 does not exist cos
                # of daylight savings. We actually have 23 hours!
                "2021-03-28 03:00:00",
                "2021-03-28 04:00:00",
                "2021-03-28 05:00:00",
                "2021-03-28 06:00:00",
                "2021-03-28 07:00:00",
                "2021-03-28 08:00:00",
                "2021-03-28 09:00:00",
                "2021-03-28 10:00:00",
                "2021-03-28 11:00:00",
                "2021-03-28 12:00:00",
                "2021-03-28 13:00:00",
                "2021-03-28 14:00:00",
                "2021-03-28 15:00:00",
                "2021-03-28 16:00:00",
                "2021-03-28 17:00:00",
                "2021-03-28 18:00:00",
                "2021-03-28 19:00:00",
                "2021-03-28 20:00:00",
                "2021-03-28 21:00:00",
                "2021-03-28 22:00:00",
                "2021-03-28 23:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
            "Europe/Brussels",
        )

        # -- test that 23 days does not fail!
        processor = ResampleData(
            "date", "1d", "sum", partial_data_resolution_strategy="fail"
        )

        df_transformed_1 = processor.transform(dataframe)

        # -- test that offset does not affect it either!
        dataframe_2 = _prepare_dataframe(
            [
                "2021-03-28 01:00:00",  # Note: 02:00:00 does not
                # exist cos of daylight savings. We actually have 23 hours!
                "2021-03-28 03:00:00",
                "2021-03-28 04:00:00",
                "2021-03-28 05:00:00",
                "2021-03-28 06:00:00",
                "2021-03-28 07:00:00",
                "2021-03-28 08:00:00",
                "2021-03-28 09:00:00",
                "2021-03-28 10:00:00",
                "2021-03-28 11:00:00",
                "2021-03-28 12:00:00",
                "2021-03-28 13:00:00",
                "2021-03-28 14:00:00",
                "2021-03-28 15:00:00",
                "2021-03-28 16:00:00",
                "2021-03-28 17:00:00",
                "2021-03-28 18:00:00",
                "2021-03-28 19:00:00",
                "2021-03-28 20:00:00",
                "2021-03-28 21:00:00",
                "2021-03-28 22:00:00",
                "2021-03-28 23:00:00",
                "2021-03-29 00:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
            "Europe/Brussels",
        )

        # -- subtract by the diff between the two dataframes to
        # "fix" the unique timestamp values
        dataframe = dataframe_2.with_columns(
            pl.col("values") - (pl.col("values") - dataframe["values"])
        )

        processor = ResampleData(
            "date",
            "1d",
            "sum",
            partial_data_resolution_strategy="fail",
            start_window_offset="1h",
        )

        df_transformed_2 = processor.transform(dataframe)

        assert_frame_equal(df_transformed_1, df_transformed_2)

    def test_resample_with_partial_data_resolution(self):
        # -- case partial but keep row
        dataframe = _prepare_dataframe(
            [
                "2023-01-01 06:00:00",
                "2023-01-01 12:00:00",
                "2023-01-01 18:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
        )

        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="keep",
        )

        transformed = processor.transform(dataframe)

        assert transformed.shape[0] == 1  # no rows dropped

        # -- case partial but drop row
        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="drop",
        )

        transformed = processor.transform(dataframe)

        assert transformed.shape[0] == 0  # row dropped since partial

        # -- case partial but strategy fail
        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="fail",
        )

        with self.assertRaises(ValueError):
            transformed = processor.transform(dataframe)

        # -- case partial but strategy null

        _dataframe = _prepare_dataframe(
            [
                "2023-01-02 00:00:00",
                "2023-01-02 06:00:00",
                "2023-01-02 12:00:00",
                "2023-01-02 18:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
        )

        dataframe = pl.concat([dataframe, _dataframe])

        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="null",
        )

        transformed = processor.transform(dataframe)

        assert transformed["values"].to_list() == [
            None,
            _dataframe["values"].sum(),
        ]

    def test_resample_with_data_filtration(self):
        # -- test that removal of row does not make the data partial
        dataframe = _prepare_dataframe(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 06:00:00",
                "2023-01-01 12:00:00",
                "2023-01-01 18:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
        )

        def _remove_n_rows(df, rows):
            df = df.head(df.shape[0] - rows)
            return df

        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="drop",  # if partial, then empty
            # dataframe returned
            filter_data_method=partial(
                _remove_n_rows, rows=1
            ),  # remove last row
        )

        transformed = processor.transform(dataframe)

        assert (
            transformed["values"].item()
            == _remove_n_rows(dataframe, 1)["values"].sum()
        )

        # -- with duplicates

        # 1: case when duplicates not removed
        dataframe = _prepare_dataframe(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:00:00",
                "2023-01-01 06:00:00",
                "2023-01-01 12:00:00",
                "2023-01-01 18:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
        )

        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="drop",  # if partial, then empty
            # dataframe returned
            filter_data_method=partial(
                _remove_n_rows, rows=1
            ),  # remove last row
        )

        transformed = processor.transform(dataframe)
        assert (
            transformed["values"].item()
            == _remove_n_rows(dataframe, 1)["values"].sum()
        )

        # 2: case when duplicates removed
        dataframe = _prepare_dataframe(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 06:00:00",
                "2023-01-01 12:00:00",
                "2023-01-01 18:00:00",
                "2023-01-01 18:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
        )

        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            partial_data_resolution_strategy="drop",  # if partial, then empty
            # dataframe returned
            filter_data_method=partial(
                _remove_n_rows, rows=2
            ),  # remove two rows
        )

        transformed = processor.transform(dataframe)
        assert (
            transformed["values"].item()
            == _remove_n_rows(dataframe, 2)["values"].sum()
        )

        # -- with offset
        dataframe = _prepare_dataframe(
            [
                "2023-01-01 06:00:00",
                "2023-01-01 12:00:00",
                "2023-01-01 18:00:00",
                "2023-01-02 00:00:00",
            ],
            "%Y-%m-%d %H:%M:%S",
        )

        def _remove_n_rows(df, rows):
            df = df.head(df.shape[0] - rows)
            return df

        processor = ResampleData(
            time_column="date",
            resampling_frequency="1d",
            resampling_function="sum",
            start_window_offset="6h",
            partial_data_resolution_strategy="drop",  # if partial, then empty
            # dataframe returned
            filter_data_method=partial(
                _remove_n_rows, rows=1
            ),  # remove last row
        )

        transformed = processor.transform(dataframe)

        assert (
            transformed["values"].item()
            == _remove_n_rows(dataframe, 1)["values"].sum()
        )


if __name__ == "__main__":
    unittest.main()
