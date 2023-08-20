# -- allow for multiple resampling types? what about case of rename?
# -- I guess we won't allow rename perse, resampling function if string will
# apply to all target cols that behave, unless specified otherwise
import logging

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__file__)

SUPPORTED_BOUNDARIES = {"left", "right"}


# TODO this is resample, need to add check that makes sure time column is
# indeed a datetime (or date?? check polars types!)
# TODO make sure all target columns are valid for aggregation!

# TODO add multi group by support

SUPPORTED_RESAMPLING_OPERATIONS = {
    "sum": "sum",
    "max": "max",
    "min": "min",
    "mean": "mean",
}
# TODO need to extend this to allow some default operations...
# will need to modify code!


# If no target cols provided, tries to apply to all!
class ResampleData(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        time_column: str,
        resampling_frequency: str,
        resampling_function: list[str] | str | list[dict],
        target_columns: list[str] | None = None,
        closed_boundaries: str = "left",
        manual_offset: str | None = None,
    ):
        self.time_column = time_column
        self.resampling_frequency = resampling_frequency
        self.manual_offset = manual_offset
        self.target_columns = target_columns

        if closed_boundaries not in SUPPORTED_BOUNDARIES:
            msg = (
                "ResampleData only supports the following boundaries: "
                f"{sorted(SUPPORTED_BOUNDARIES)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.closed_boundaries = closed_boundaries

        if isinstance(resampling_function, str):
            resampling_function = [resampling_function]

        if isinstance(resampling_function[0], dict):
            self._check_resampling_function_names_unique(resampling_function)
        else:
            resampling_function = (
                self._convert_resampling_function_to_record_format(
                    resampling_function
                )
            )

        self.resampling_function = resampling_function

    def _convert_resampling_function_to_record_format(
        self, resampling_functions: list[str]
    ):
        resampling_function_list = []
        for resampling_function in resampling_functions:
            resampling_function_list.append({"name": resampling_function})

        return resampling_function_list

    def _check_resampling_function_names_unique(self, resampling_functions):
        unique_func_names = set()
        num_functions = len(resampling_functions)
        for function_id, resampling_function in enumerate(
            resampling_functions
        ):
            func_identifier = resampling_function.get("name")
            if func_identifier is None:
                msg = (
                    f"Function {function_id+1}/{num_functions} does not have "
                    "key `name`"
                )
                logger.error(msg)
                raise ValueError(msg)

            if func_identifier in unique_func_names:
                msg = (
                    f"Function {function_id+1}/{num_functions} defines `func` "
                    f"as `{func_identifier}` but this already exists in inputs"
                )
                logger.error(msg)
                raise ValueError(msg)

            unique_func_names.add(func_identifier)

            func_callable = resampling_function.get("func")
            if func_callable is None:
                if func_identifier not in SUPPORTED_RESAMPLING_OPERATIONS:
                    msg = (
                        f"Function {function_id+1}/{num_functions} with `name`"
                        f" `{func_identifier}` is not a default supported "
                        "operation and does not have key `func`"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

    def _groupby(self, X):
        # TODO add sorting support for group by with the "by" key, e.g. sort
        # the time WITHIN a group!
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
        # validation on the target functions!
        if self.target_columns is None:
            target_columns = set(X.columns)
            target_columns.remove(self.time_column)
            target_columns = list(target_columns)

        # -- sorting is necessary
        groupby_obj = self._groupby(X)
        # for item in groupby_obj:
        #    print(item)
        agg_func_list = []
        multiple_resampling_functions = len(self.resampling_function) > 1
        for target_column in target_columns:
            for resampling_function_metadata in self.resampling_function:
                func_name = resampling_function_metadata["name"]
                print(func_name)
                # TODO add support for arguments to these, e.g.
                # "sum with truncation" if these are native!
                target_column_obj = pl.col(target_column)
                if func_name in SUPPORTED_RESAMPLING_OPERATIONS:
                    agg_func = getattr(target_column_obj, func_name)()
                    if multiple_resampling_functions:
                        agg_func = agg_func.alias(
                            f"{target_column}_{func_name}"
                        )
                else:
                    raise NotImplementedError(
                        (
                            "no support for custom functions yet... need to "
                            "learn how this is done with polars natively!"
                        )
                    )

                agg_func_list.append(agg_func)
        df_agg = groupby_obj.agg(agg_func_list)

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
    processor = ResampleData(
        "date", "1d", ["sum", "max", "min", "mean"], None, "left"
    )
    processor.transform(df)

    print(df.to_pandas().to_string())
    # import pandas as pd

    # pd_frame = df.to_pandas()
    # pd_frame = pd_frame.set_index("date")
    # print(pd_frame.resample("35T").sum())
