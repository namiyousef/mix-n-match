# -- allow for multiple resampling types? what about case of rename?
# -- I guess we won't allow rename perse, resampling function if string will
# apply to all target cols that behave, unless specified otherwise
import logging
from enum import Enum
from typing import Callable

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from mix_n_match.utils import PolarsDuration, detect_timeseries_frequency

logger = logging.getLogger(__file__)

SUPPORTED_BOUNDARIES = {"left", "right"}
SUPPORTED_LABELS = {"left", "right", "first"}


class PartialDataResolutionStrategy(str, Enum):
    """Strategy on how to deal with partial data in a resampling window Partial
    data is when a resampling window (e.g. day) does not have enough points of
    a specific frequency for it to be considered complete. E.g. in a day, if I
    have 5 minute samples, I would expect 288 points (at least, to account for
    duplicates)

    `KEEP`: ignore the presence of partial `DROP`: drop resampled
    timestamps where partial data is present `NULL`: set all values for
    timestamps with partial data to null `FAIL`: fail if any partial
    data is detected
    """

    KEEP = "keep"
    DROP = "drop"
    NULL = "null"
    FAIL = "fail"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def supported_values(cls):
        return sorted(cls._value2member_map_.keys())


# TODO this is resample, need to add check that makes sure time column is
# indeed a datetime (or date?? check polars types!)
# TODO make sure all target columns are valid for aggregation!

# TODO add multi group by support

SUPPORTED_RESAMPLING_OPERATIONS = {
    "sum": "sum",
    "max": "max",
    "min": "min",
    "mean": "mean",
    "count": "count",
    "collect": None,
}


# TODO need to extend this to allow some default operations...
# will need to modify code!


def generate_polars_condition(expressions, operator):
    final_expression = expressions.pop()
    for expression in expressions:
        final_expression = getattr(final_expression, operator)(expression)

    return final_expression


class FilterDataBasedOnTime(BaseEstimator, TransformerMixin):
    UNIT_TO_POLARS_METHOD_MAPPING = {"d": "day", "h": "hour"}

    OPERATOR_TO_POLARS_METHOD_MAPPING = {
        "=": "eq",
        "!=": "ne",
        "<=": "le",
        "<": "lt",
        ">": "gt",
        ">=": "ge",
    }

    def __init__(self, time_column, time_patterns):
        self.time_column = time_column

        self.filtering_rules = self._parse_time_patterns_into_rules(
            time_patterns
        )

    def fit(self, X, y=None):
        pass

    def _convert_rules_to_polars_expressions(self):
        rules = []  # list to store expressions for each rule
        for rule_metadata in self.filtering_rules:
            boundary_rule_expressions = []
            for boundary_rule_metadata in rule_metadata:
                unit_rule_expressions = []
                for unit_rule in boundary_rule_metadata:
                    unit_method = unit_rule["unit_method"]
                    value = unit_rule["value"]
                    operator = unit_rule["operator_method"]

                    unit_rule_expressions.append(
                        getattr(
                            getattr(
                                pl.col(self.time_column).dt, unit_method
                            )(),
                            operator,
                        )(value)
                    )

                boundary_rule_expression = generate_polars_condition(
                    unit_rule_expressions, "and_"
                )
                boundary_rule_expressions.append(boundary_rule_expression)

            rule_expression = generate_polars_condition(
                boundary_rule_expressions, "and_"
            )

            rules.append(rule_expression)

        overall_rule_expression = generate_polars_condition(
            rules, "or_"
        ).not_()

        return overall_rule_expression

    def transform(self, X):
        rule_expression = self._convert_rules_to_polars_expressions()

        X = X.filter(rule_expression)

        return X

    def _parse_time_patterns_into_rules(self, independent_time_patterns):
        rules = []  # list to store
        for pattern in independent_time_patterns:
            pattern = pattern.replace(" ", "").split(",")
            pattern = [rule_str for rule_str in pattern if rule_str]
            if len(pattern) > 2:
                raise ValueError("")  # TODO add value error

            rule_metadata = []
            for rule_str in pattern:
                operator = ""
                duration_string = ""
                operator_methods = []
                for char in rule_str:
                    if char in {">", "<", "="}:
                        operator += char
                    else:
                        duration_string += char
                        if operator:
                            operator_method = FilterDataBasedOnTime.OPERATOR_TO_POLARS_METHOD_MAPPING[  # noqa: B950
                                operator
                            ]
                            operator_methods.append(operator_method)
                        operator = ""

                polars_duration = PolarsDuration(duration=duration_string)
                decomposed_duration = polars_duration.decomposed_duration

                rule_components = []
                for (
                    value,
                    unit,
                ), operator in zip(  # noqa: B905 (strict parameter introduced in Python 3.10 but want code to be valid for 3.8)
                    decomposed_duration, operator_methods
                ):
                    unit_method = FilterDataBasedOnTime.UNIT_TO_POLARS_METHOD_MAPPING.get(  # noqa: B950
                        unit, None
                    )
                    if unit_method is None:
                        raise ValueError()  # TODO add

                    rule_component = {
                        "value": value,
                        "unit_method": unit_method,
                        "operator_method": operator,
                    }

                    rule_components.append(rule_component)

                rule_metadata.append(rule_components)

            rules.append(rule_metadata)

        return rules

    # -- <5s
    # -- >10s
    # -- =3d>6h

    # -- realistically:
    # I would like to remove: minutes from an hour
    # hours from a day
    # days from a week
    # days from a month
    # months from a year
    # potentially combinations of these, e.g. remove weekends, and also remove lunches
    # what about interaction?
    # e.g.: remove within weekends, hours of X and Y
    #


# If no target cols provided, tries to apply to all!
class ResampleData(BaseEstimator, TransformerMixin):
    """Abstraction over polars groupby_dynamic method to enable timeseries
    resampling.

    :param time_column: column to use for resampling
    :param resampling_frequency: defines frequency of bin windows
    :param resampling_function: how to resample each bin
    :param target_columns: which columns to use for resamplng. If not
        provided, resamples all columns. Defaults to `None`
    :param closed_boundaries: which boundaries are inclusive. E.g. if
        `left` on daily resample, then [2021-01-01, 2021-01-02)
    :param labelling_strategy: which boundary to use for the label. E.g.
        if `right` with closed_boundaries='left', then for [2021-01-01,
        2021-01-02) takes the value of 2021-01-01 but returns the output
        as belonging to 2021-01-02, Example use case: number of items
        sold on the day before (e.g. up to a certain date, but not
        including)
    :param start_window_offset: offset the data by a certain amount.
        This is for cases where you want to manually define a `start` of
        a specific bin. For example, offsetting by -6h on a daily
        resampling indicates that you want to start counting from 6 am
        onwards as the start of the day. Defaults to `None`
    :param partial_data_resolution_strategy: how to deal if you only
        have partial data in a resampling window. See
        `PartialDataResolutionStrategy` enum for info
    :param group_by_columns: group by columns before resampling
    :param filter_data_method: method that takes a dataframe and removes
        data from it. This is compatible with
        `partial_data_resolution_strategy`. Any extra rows removed here
        will not affect the logic of determining if a window contains
        partial data
    """

    def __init__(
        self,
        time_column: str,
        resampling_frequency: str,
        resampling_function: list[str] | str | list[dict],
        target_columns: list[str] | None = None,
        closed_boundaries: str = "left",
        labelling_strategy: str = "left",
        start_window_offset: str
        | None = None,  # For a day, offset by a few hours. For an hour,
        # offset my minutes, for a minute by seconds. For a month, by a
        # few days (although this use case doesn't make that much sense),
        # for a month, by a few weeks (this makes more sense! Though need to
        # see how weeks will be calculated, maybe should be a 7 day offset??)
        # This needs further exploration and testing... at the momnet you can
        # only guarantee that it works for daily with NO timezone!
        # TODO add parameter to return the date as offset by the
        # start_window_offset!
        truncate_window: str | None = None,
        partial_data_resolution_strategy: PartialDataResolutionStrategy = "keep",
        group_by_columns: list | None = None,
        filter_data_method: Callable[[pl.DataFrame], pl.DataFrame]
        | None = None,
    ):
        self.time_column = time_column
        self.resampling_frequency = resampling_frequency

        self._set_start_window_offset(start_window_offset)
        self.target_columns = target_columns

        self._set_closed_boundaries(closed_boundaries)

        self._set_labelling_strategy(labelling_strategy)

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
        self.truncate_window = truncate_window

        if not PartialDataResolutionStrategy.has_value(
            partial_data_resolution_strategy
        ):
            raise ValueError(
                (
                    "`partial_data_resolution_strategy` must be one of "
                    f"{PartialDataResolutionStrategy.supported_values()}"
                )
            )

        self.partial_data_resolution_strategy = (
            partial_data_resolution_strategy
        )

        self.group_by_columns = group_by_columns
        self.filter_data_method = filter_data_method

    def _set_start_window_offset(self, start_window_offset):
        if start_window_offset is not None:
            if start_window_offset.startswith("-"):
                msg = (
                    "Can only offset by a positive value. "
                    "E.g. 6h offset on a 1d resample_frequency means start "
                    "counting the day from 6 am"
                )
                logger.error(msg)
                raise ValueError(msg)
            start_window_offset = f"-{start_window_offset}"
        self.start_window_offset = start_window_offset

    def _set_closed_boundaries(self, closed_boundaries):
        if closed_boundaries not in SUPPORTED_BOUNDARIES:
            msg = (
                "ResampleData only supports the following boundaries: "
                f"{sorted(SUPPORTED_BOUNDARIES)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        self.closed_boundaries = closed_boundaries

    def _set_labelling_strategy(self, labelling_strategy):
        if labelling_strategy not in SUPPORTED_LABELS:
            msg = (
                "ResampleData only supports the following "
                f"labelling strategies: {sorted(SUPPORTED_LABELS)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        if labelling_strategy == "first":
            # TODO first internally by polars is calculaed based on index,
            # add test to ensure that indeed, we get the first
            # TIME value NOT index value
            labelling_strategy = "datapoint"

        self.labelling_strategy = labelling_strategy

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

        if self.start_window_offset:
            logger.info(
                f"Detected offset... applying offset={self.start_window_offset}"
            )
            X = X.with_columns(
                pl.col(self.time_column).dt.offset_by(self.start_window_offset)
            )

        X = X.sort(self.time_column)

        # TODO note: start_window vs. offset. With start window,
        # you can guarantee the start date because you are shifting by
        # a fixed value
        #
        # with offset, it really depends on how the start window is calculated.
        #  You cannot "fix" the value since
        # subtracting it, incases of month for example would depend on the
        # number of days in the month!
        # it doesn't even make that much sense tbh. I dont really understand it
        # One disiction it makes though is that when you use it, your
        # resampling outcome might vary when you hae
        # daylight savings etc...

        groupby_obj = X.group_by_dynamic(
            self.time_column,
            every=self.resampling_frequency,
            # TODO need to have some controls on period, e.g. disallow period
            #  exceeding size of every!
            # period=self.truncate_window,  # if you give a period, for each
            # "every", end date
            # becomes start_date + period
            # offset="6h",  # offset shifts how your every windows are created.
            # It modifies the start boundary by doing
            # start_boundary = start_boundary + offset. Consequence: you can
            # lose some data because it filters your data if ourside the start
            # boundary!
            check_sorted=False,
            include_boundaries=True,  # TODO by default we dont need this tbh,
            # but for partial check this must be set to true
            closed=self.closed_boundaries,  # how to treat values at the
            # boundaries. Also affects the boundarties themselves.
            # If left, starts looking at [left, )
            label=self.labelling_strategy,
            start_by="window",
            group_by=self.group_by_columns,
        )

        # -- start window takes a datapoint, then normalises it by "every",
        # applies the offset!
        # -- closed affects how the boundaru is created!!!
        # -- for each datapoint,

        return groupby_obj

    def _aggregate(self, X):
        # validation on the target functions!
        if self.target_columns is None:
            target_columns = set(X.columns)
            target_columns.remove(self.time_column)

            if self.group_by_columns:
                for column in self.group_by_columns:
                    target_columns.remove(column)

            target_columns = list(target_columns)
        else:
            target_columns = self.target_columns

        groupby_obj = self._groupby(X)

        agg_func_list = []
        multiple_resampling_functions = len(self.resampling_function) > 1
        for target_column in target_columns:
            for resampling_function_metadata in self.resampling_function:
                func_name = resampling_function_metadata["name"]
                # TODO add support for arguments to these, e.g.
                # "sum with truncation" if these are native!
                target_column_obj = pl.col(target_column)
                if func_name in SUPPORTED_RESAMPLING_OPERATIONS:
                    if func_name != "collect":
                        agg_func = getattr(target_column_obj, func_name)()
                    else:
                        agg_func = target_column_obj
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

        if (
            self.partial_data_resolution_strategy
            != PartialDataResolutionStrategy.KEEP
        ):
            agg_func_list.append(
                pl.col(self.time_column)
                .unique()  # unique because we don't want duplicates to affect how we
                # count rows for determining if we have a complete time window
                .count()
                .alias("_unique_timestamp_count")
            )

        df_agg = groupby_obj.agg(agg_func_list)

        return df_agg

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        if (
            self.partial_data_resolution_strategy
            != PartialDataResolutionStrategy.KEEP
        ):
            frequency = detect_timeseries_frequency(
                X, self.time_column, "mode"
            )

        # -- filter data, but keep information on number of rows prior to filtration
        if self.filter_data_method is not None:
            df_agg_non_filtered = self._groupby(X).agg(
                [
                    pl.col(self.time_column)
                    .unique()
                    .count()
                    .alias("_unique_timestamp_count_non_filtered")
                ]
            )

            X = self.filter_data_method(X)

        # -- group by and aggregate
        df_agg = self._aggregate(X)

        if (
            self.partial_data_resolution_strategy
            != PartialDataResolutionStrategy.KEEP
        ):
            df_agg = df_agg.with_columns(
                pl.col("_upper_boundary")
                .sub(pl.col("_lower_boundary"))
                .alias("_date_diff")
                .dt.total_seconds()
                .truediv(frequency)
            )

            if self.filter_data_method is not None:
                df_agg = df_agg.join(df_agg_non_filtered, on=self.time_column)
                df_agg = df_agg.with_columns(
                    pl.col("_unique_timestamp_count_non_filtered").alias(
                        "_unique_timestamp_count"
                    )
                )

            df_agg = df_agg.with_columns(
                pl.col("_unique_timestamp_count")
                .lt(pl.col("_date_diff"))
                .alias("_is_partial")
            )

            match self.partial_data_resolution_strategy:
                case PartialDataResolutionStrategy.FAIL:
                    if df_agg["_is_partial"].any():
                        _supported_values = sorted(
                            value
                            for value in (
                                PartialDataResolutionStrategy.supported_values()
                            )
                            if value != PartialDataResolutionStrategy.FAIL
                        )
                        raise ValueError(
                            (
                                "Detected partial data. If you wish to "
                                "proceed then set your "
                                "`partial_data_resolution_strategy` "
                                f"to one of {_supported_values}"
                            )
                        )
                case PartialDataResolutionStrategy.DROP:
                    df_agg = df_agg.filter(~pl.col("_is_partial"))
                case PartialDataResolutionStrategy.NULL:
                    columns_to_set_to_null = [
                        col
                        for col in df_agg.columns
                        if not col.startswith("_") and col != "date"
                    ]
                    df_agg = df_agg.with_columns(
                        [
                            pl.when(~pl.col("_is_partial"))
                            .then(pl.col(columns_to_set_to_null))
                            .keep_name()
                        ]
                    )
        return df_agg

    def inverse_transform(
        self,
    ):
        pass


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "date": [
                "2022-01-01 06:00:00",
                "2022-01-01 06:00:00",
                "2022-01-01 12:00:00",
                "2022-01-01 18:00:00",
            ],
            "values": [1, 2, 3, 4],
        }
    )
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )

    import datetime

    def method(x):
        return x.filter(
            pl.col("date") != datetime.datetime(2022, 1, 1, 6, 0, 0)
        )

    processor = ResampleData(
        time_column="date",
        resampling_frequency="1d",
        target_columns=["values"],
        partial_data_resolution_strategy="fail",
        resampling_function="sum",
        filter_data_method=method,
    )

    transformed = processor.transform(df)
    print(transformed)
    raise Exception()

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

    print(
        ResampleData(
            "date",
            "1d",
            ["sum", "max"],
            partial_data_resolution_strategy="null",
        ).transform(df)
    )
    df = pl.DataFrame(
        {
            "date": ["2021-12-31 00:00:00", "2021-12-31 00:00:01"],
            "values": [400000000, 90000000000],
        }
    )

    df = df.with_columns(
        pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    print(detect_timeseries_frequency(df, "date", "max"))
    raise Exception()
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
