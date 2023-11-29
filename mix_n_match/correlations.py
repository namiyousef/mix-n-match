import itertools
import logging
from functools import partial
from typing import Dict, Iterable, List, Optional, Union

import polars as pl
import polars.selectors as cs

from mix_n_match.utils import PolarsDuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pair_data(
    iterable: Iterable,
    method: str = "full",
    ignore_diagonal: bool = False,
) -> Iterable:
    """Function pairs items in an iterator.

    :param iterable: iterator of items to pair together
    :param method: how to pair the data.
        If `full`, pairs every item with every other item
        If `lower`, pairs each item with preceeding items (lower triangle)
        If `upper`, pairs each item with subsequent items (upper triangular)
        defaults to `full`
    :param ignore_diagonal: if `True`, ignores cases where the item is paired
        with itself. Defaults to `False`
    :yield: iterable in the form (indices, item1, item2)

    Example:
        items = iter([0, 1])
        list(pair_data(items, method="lower", ignore_diagonal=True))
        >>> [((0, 0), 1, 0)]
    """
    iterable_1, iterable_2 = itertools.tee(iterable, 2)
    for i, item1 in enumerate(iterable_1):
        iterable_2, next_iter = itertools.tee(iterable_2, 2)
        for j, item2 in enumerate(iterable_2):
            if ignore_diagonal and i == j:
                continue
            if method == "lower" and j > i:
                continue
            if method == "upper" and i > j:
                continue
            yield ((i, j), item1, item2)

        iterable_2 = next_iter


def calculate_polars_correlation(
    df1, df2, target_column, method="pearson", dof=1
):
    target_column = target_column[0]
    columns = df1.columns
    join_cols = [col for col in columns if col != target_column]
    df = df1.join(df2, on=join_cols)

    return calculate_correlation_between_columns(
        df, target_column, f"{target_column}_right", method, dof
    )


def calculate_correlation_between_columns(lazy_df, col1, col2, method, dof=1):
    return (
        lazy_df.lazy()
        .select(
            pl.corr(
                pl.col(col1),
                pl.col(col2),
                method=method,
                ddof=dof,
                # propagate_nans=True,
            )
        )
        .collect()
        .item()
    )


CORRELATION_METHODS = {
    "pearson": {
        "requires_aligned_data": True,
        "accepts_multiple_columns": False,
        "callable": partial(calculate_polars_correlation, method="pearson"),
    },
    "spearman": {
        "requires_aligned_data": True,
        "accepts_multiple_columns": False,
        "callable": partial(calculate_polars_correlation, method="spearman"),
    },
}

DEFAULT_ALIGNER = "join_aligner"


def join_aligner(df1, df2, alignment_columns, how="left"):
    df1, df2 = pl.align_frames(df1, df2, on=alignment_columns, how=how)
    return df1, df2


ALIGNERS = {"join_aligner": {"callable": join_aligner}}


class FindCorrelations:
    """Class to find correlations or distances between multiple dataframes.

    :param target_columns: target columns to use for calculations
    :param alignment_params: parameters to determine how to align
        dataframes If empty or None, then no alignment performed. If not
        empty, requires `alignment_columns` as a key. Can also take
        `alignment_method` to specify method to use, and `params` to
        override default params for `alignment_method`
    :param method: method to use for calculation correlations
    :param method_optional_params: optional params for `method` to override
        defaults
    """

    def __init__(
        self,
        target_columns: List[str] | str,
        alignment_params: dict | None = None,
        method: str = "pearson",
        method_optional_params: dict | None = None,
    ):
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        self.target_columns = target_columns
        if alignment_params:
            alignment_columns = alignment_params.get("alignment_columns", None)
            if alignment_columns is None:
                raise ValueError(
                    (
                        "Got `alignment_params` but no `alignment_columns`. "
                        "Please pass `alignment_columns`"
                    )
                )

            alignment_method = alignment_params.get("alignment_method")
            if alignment_method is None:
                logger.warning(
                    (
                        "Got `alignment_params` but no `alignment_method`. "
                        "Setting `alignment_method` to default aligner "
                        f"`{DEFAULT_ALIGNER}`"
                    )
                )
            elif ALIGNERS.get(alignment_method) is None:
                raise ValueError(
                    (
                        f"Got alignment_method `{alignment_method}` which is "
                        f"not valid. Please choose one of: {sorted(ALIGNERS)}"
                    )
                )
            alignment_params["alignment_method"] = DEFAULT_ALIGNER

        self.alignment_params = alignment_params

        method_metadata = CORRELATION_METHODS.get(method)
        if method_metadata is None:
            raise ValueError(
                (
                    f"No method `{method}`. Please choose one of: "
                    f"{sorted(CORRELATION_METHODS)}"
                )
            )

        if (
            len(self.target_columns) > 1
            and not method_metadata["accepts_multiple_columns"]
        ):
            raise ValueError(
                (
                    f"Method `{method}` only accepts a single target "
                    "column but multiple provided."
                )
            )
        self.method = method
        self.method_metadata = method_metadata
        if method_optional_params is None:
            method_optional_params = {}
        self.method_optional_params = method_optional_params

    # TODO this is a slow methed. Not exactly sure why but perhaps due to memory?
    # really scales up when size of dataframes increases
    # perhaps you can make this an iterator as well? E.g. iteratively update
    # a class dict for the mapping?
    def create_dataframe_mapping(
        self, dataframes: List[pl.LazyFrame] | Iterable[pl.LazyFrame]
    ):
        dataframes, dataframes_copy = itertools.tee(iter(dataframes), 2)
        dataframe_mapping = {
            index: f"df_{index+1}" for index, _ in enumerate(dataframes_copy)
        }

        return dataframes, dataframe_mapping

    def prepare_dataframes(self, dataframes):
        filter_columns = self.target_columns
        if self.alignment_params:
            filter_columns += self.alignment_params["alignment_columns"]

        for dataframe in dataframes:
            yield dataframe.select(filter_columns)

    def align_dataframes(self, paired_dataframes):
        aligner = ALIGNERS[self.alignment_params["alignment_method"]][
            "callable"
        ]
        alignment_columns = self.alignment_params["alignment_columns"]
        aligner_params = self.alignment_params.get("params", {})

        for indices, df1, df2 in paired_dataframes:
            df1, df2 = aligner(
                df1, df2, alignment_columns=alignment_columns, **aligner_params
            )
            yield indices, df1, df2

    def calculate_correlations(
        self,
        dataframes: List[pl.LazyFrame] | Iterable[pl.LazyFrame],
        dataframe_mapping: Dict[int, int | str] | None = None,
    ):
        if dataframe_mapping is None:
            logger.info("Creating dataframe mapping...")
            dataframes, dataframe_mapping = self.create_dataframe_mapping(
                dataframes
            )

        # -- sanity checks on data
        if (
            not self.alignment_params
            and self.method_metadata["requires_aligned_data"]
        ):
            logger.info("Checking dataframes validity...")

            # -- check that the dataframes are of the same size
            dataframes, dataframes_copy = itertools.tee(dataframes, 2)
            dataframe_shapes = {df.collect().shape for df in dataframes_copy}
            if len(dataframe_shapes) != 1:
                raise ValueError(
                    (
                        f"Method `{self.method}` requires data to be"
                        "aligned but the dataframes are of different shapes "
                        "and no alignment columns passed. "
                        "Either pass dataframes of the same shape, "
                        "or pass alignment columns"
                    )
                )

        # -- prepare dataframes
        logger.info("Preparing dataframes...")
        dataframes = self.prepare_dataframes(dataframes)

        # -- pair dataframes together
        logger.info("Pairing dataframes...")
        paired_dataframes = pair_data(dataframes)

        # -- align data
        if self.alignment_params:
            logger.info("Aligning dataframes...")
            paired_dataframes = self.align_dataframes(paired_dataframes)

        correlation_matrix = {}

        for indices, df1, df2 in paired_dataframes:
            correlation_method = self.method_metadata["callable"]
            correlation_value = correlation_method(
                df1, df2, self.target_columns, **self.method_optional_params
            )  # TODO don't likethat target columns is a required parameter
            # tbh.. need to think of better method for future
            correlation_matrix[indices] = correlation_value

        return {
            "matrix": correlation_matrix,
            "mapping": dataframe_mapping,
        }


def calculate_variogram(
    lazy_df: pl.LazyFrame,
    cardinal_direction: Union[str, List[str]],
    number_of_lags: int,
    lag_size: Union[str, int],
    delta: Union[str, int],  # TODO add support for exact variogram
    target_col: Optional[str] = None,
    normalise: bool = True,
):
    """Function to calculate variogram for a given LazyFrame. This
    implementation is based on the Matheron metric.

    :param lazy_df: input data LazyFrame
    :param cardinal_direction: column or list of columns to calculate variogram
        for
    :param number_of_lags: number of lags
    :param lag_size: distance/lag to use for determining windows
    :param delta: tolerance to use for the windows
    :param normalise: whether to normalise variogram value, defaults to True
    :return: LazyFrame containing the variogram calculation for each
        cardinal direction as well as the lags as a column `lags`
    """
    if isinstance(cardinal_direction, str):
        cardinal_direction = [cardinal_direction]

    if len(cardinal_direction) > 1:
        raise NotImplementedError(
            "There is currently no support for multiple cardinal directions"
        )

    if isinstance(target_col, str):
        target_col = [target_col]

    if target_col is None:
        target_col = [
            col for col in lazy_df.columns if col not in cardinal_direction
        ]

    if len(target_col) > 1:
        raise NotImplementedError(
            "There is currently no support for multiple target columns"
        )
    pl_duration = PolarsDuration(lag_size)
    lag_dists = [pl_duration * lags for lags in range(1, number_of_lags + 1)]

    variogram = {"lags": lag_dists}
    print(lazy_df.collect())
    target_col_variances = {
        col: lazy_df.select(pl.col(col).var()).collect().item()
        for col in target_col
    }
    for direction in cardinal_direction:
        temporal_columns = lazy_df.select(cs.temporal()).schema
        if direction not in temporal_columns:
            raise NotImplementedError(
                "There is currently only support for temporal columns"
            )

        lazy_df = lazy_df.sort(direction)
        gamma_values = []
        for lag_dist in lag_dists:
            # period = 2 * delta
            # offset = lag_distance - delta

            offset_direction = f"_{direction}"

            lazy_df_offset = lazy_df.with_columns(
                pl.col(direction)
                .dt.offset_by(f"-{delta}")
                .alias(offset_direction)
            )

            rolling_obj = lazy_df_offset.rolling(
                offset_direction, offset="0i", period=f"{lag_dist}{delta}"
            )  # TODO option for choosing closed="both"?
            # TODO add exact calculation, e.g. 0 offset

            _sum = f"_sum_{target_col}"
            _squared_sum = f"_sum_squared_{target_col}"
            _count = "_count"
            expressions = [pl.col(offset_direction).count().alias(_count)]
            for col in target_col:
                expressions.extend(
                    [
                        pl.col(col).sum().alias(_sum),
                        (pl.col(col) ** 2).sum().alias(_squared_sum),
                    ]
                )

            agg_lazy_df = rolling_obj.agg(expressions).filter(
                pl.col(_count) != 0
            )

            agg_lazy_df = agg_lazy_df.join(lazy_df_offset, on=offset_direction)

            # TODO here need to extend support to multiple columns
            metric = agg_lazy_df.select(
                (
                    (pl.col(target_col[0]) ** 2) * pl.col(_count)
                    + pl.col(_squared_sum)
                    - 2 * pl.col(target_col[0]) * pl.col(_sum)
                ).sum()
                / pl.col(_count).sum()
            )
            gamma = 0.5 * metric.collect().item()
            if normalise:
                gamma = gamma / target_col_variances[target_col[0]]
            gamma_values.append(gamma)

        variogram[direction] = gamma_values

    lazy_variogram = pl.LazyFrame(variogram)

    return lazy_variogram


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = "data/timeseries.csv"

    lazy_df = (
        pl.read_csv(path)
        .with_columns(pl.col("Date").str.strptime(pl.Datetime))
        .rename({"Date": "date"})
        .lazy()
    )

    for column in lazy_df.columns:
        if column != "date":
            plt.plot(
                lazy_df.select("date").collect(),
                lazy_df.select(column).collect(),
            )

    lazy_df = calculate_variogram(lazy_df, "date", 10, "1d", "3h", "A")
    plt.figure()
    plt.plot(lazy_df.select(pl.col("date")).collect())
    plt.show()
