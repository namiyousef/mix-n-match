# -- allow for multiple resampling types? what about case of rename?
# -- I guess we won't allow rename perse, resampling function if string will
# apply to all target cols that behave, unless specified otherwise
import logging

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__file__)

SUPPORTED_BOUNDARIES = {"left", "right"}


class ResampleData(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        time_column,
        resampling_frequency,  # how to do for multiple items? Need to allow
        # support for this!
        resampling_function,
        closed_boundaries: str = "left",
        manual_offset: str | None = None,
    ):
        self.time_column = time_column
        self.resampling_frequency = resampling_frequency
        self.resampling_function = resampling_function
        self.manual_offset = manual_offset

        if closed_boundaries not in SUPPORTED_BOUNDARIES:
            msg = (
                "ResampleData only supports the following boundaries: "
                f"{sorted(SUPPORTED_BOUNDARIES)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.closed_boundaries = closed_boundaries

    def _groupby(self, X):
        X = X.sort(self.time_column)

        if self.manual_offset:
            logger.info(
                f"Detected offset... applying offset={self.manual_offset}"
            )
            X = X.with_columns(
                pl.col(self.time_column).dt.offset_by(self.manual_offset)
            )

        groupby_obj = X.groupby_dynamic(
            self.time_column,
            every=self.resampling_frequency,
            # period="6h",  # if you give a period, for each "every", end date
            # becomes start_date + period
            # offset="6h",  # offset shifts how your every windows are created.
            # It modifies the start boundary by doing
            # start_boundary = start_boundary + offset. Consequence: you can
            # lose some data because it filters your data if ourside the start
            # boundary!
            check_sorted=False,
            truncate=True,
            include_boundaries=True,
            closed=self.closed_boundaries,  # how to treat values at the
            # boundaries. Also affects the boundarties themselves.
            # If left, starts looking at [left, )
            start_by="window",
        )

        # -- start window takes a datapoint, then normalises it by "every",
        # applies the offset!
        # -- closed affects how the boundaru is created!!!
        # -- for each datapoint,

        return groupby_obj

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        # -- sorting is necessary
        groupby_obj = self._groupby(X)
        # for item in groupby_obj:
        #    print(item)
        df_agg = groupby_obj.agg(pl.col("values").sum())

        print(df_agg)

    def inverse_transform(
        self,
    ):
        pass


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02", "2023-02-01"],
            "values": [1, 2, 3],
        }
    )
    df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d"))

    df = pl.DataFrame(
        {
            "date": [
                "2021-12-31 00:00:00",
                "2021-12-31 00:00:01",
                "2021-12-31 23:50:00",
                "2022-01-01 00:00:00",
                "2022-01-01 00:05:00",
                "2022-01-01 05:00:00",
                "2022-01-01 06:00:00",
                "2022-01-01 12:00:00",
                "2022-01-01 23:59:59",
                "2022-01-02 00:00:00",
                "2022-01-02 07:00:00",
            ],
            "values": [
                400000000,
                90000000000,
                30000000,
                1,
                10,
                100,
                1000,
                10000,
                2000000,
                100000,
                10**6,
            ],
        }
    )
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    print("expects this:")
    df_expected = pl.DataFrame(
        {
            "date": [
                "2021-12-30",
                "2021-12-31",
                "2022-01-01",
                "2022-01-02",
            ],
            "values": [
                90000000000 + 400000000,
                30000000 + 1 + 10 + 100,
                1000 + 10000 + 2000000 + 100000,
                10**6,
            ],
        }
    )
    print(df_expected)
    processor = ResampleData("date", "1d", None, "left")
    processor.transform(df)

    print(df.to_pandas().to_string())
    # import pandas as pd

    # pd_frame = df.to_pandas()
    # pd_frame = pd_frame.set_index("date")
    # print(pd_frame.resample("35T").sum())
