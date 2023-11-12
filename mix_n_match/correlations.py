import itertools
import logging
import time
from typing import Dict, Iterable, List

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pair_data(dataframes_iter):
    dataframes_iter1, dataframes_iter2 = itertools.tee(dataframes_iter, 2)
    for i, df1 in enumerate(dataframes_iter1):
        dataframes_iter2, next_iter = itertools.tee(dataframes_iter2, 2)
        for j, df2 in enumerate(dataframes_iter2):
            yield ((i, j), df1, df2)

        dataframes_iter2 = next_iter


def calculate_pearson_correlation(lazy_df1, lazy_df2):
    lazy_df = lazy_df1.join(lazy_df2, on="date")
    return (
        lazy_df.select(pl.corr(pl.col("values_right"), pl.col("values")))
        .collect()
        .item()
    )


def calculate_correlations(
    dataframes, method="pearson", dataframe_mapping=None
):
    if dataframe_mapping is None:
        dataframes, dataframes_iter = itertools.tee(iter(dataframes), 2)
        dataframe_mapping = {
            index: f"df_{index+1}" for index, _ in enumerate(dataframes_iter)
        }

    paired_dataframes = pair_data(dataframes)
    correlation_matrix = {}
    for indices, df1, df2 in paired_dataframes:
        if method == "pearson":
            correlation_matrix[indices] = calculate_pearson_correlation(
                df1, df2
            )

    return {
        "correlation_matrix": correlation_matrix,
        "mapping": dataframe_mapping,
    }


# METHODS:
# pearson, spearman


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
        "requires_aligned_timeseries": True,
        "accepts_multiple_columns": False,
        "callable": calculate_correlation_between_columns,
    },
    "spearman": {
        "requires_aligned_timeseries": True,
        # RENAME... need to make timeseries agnostic
        "accepts_multiple_columns": False,
        "callable": calculate_correlation_between_columns,
    },
}

# Alignment methods:
# for most data, we can do joins on a column or columns
#


class FindCorrelations:
    """_summary_"""

    def __init__(
        self,
        target_columns: List[str] | str,
        alignment_columns: List[str] | None = None,
        method: str = "pearson",
    ):
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        self.target_columns = target_columns
        self.alignment_columns = alignment_columns

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
        if self.alignment_columns is not None:
            filter_columns += self.alignment_columns

        for dataframe in dataframes:
            yield dataframe.select(filter_columns)

    def align_dataframes(self, paired_dataframes):
        for indices, df1, df2 in paired_dataframes:
            df1, df2 = pl.align_frames(
                df1, df2, on=self.alignment_columns, how="left"
            )
            yield indices, df1, df2

    def calculate_correlations(
        self,
        dataframes: List[pl.LazyFrame] | Iterable[pl.LazyFrame],
        dataframe_mapping: Dict[int, int | str] | None = None,
        **kwargs,
    ):
        if dataframe_mapping is None:
            logger.info("Creating dataframe mapping...")
            s = time.time()
            dataframes, dataframe_mapping = self.create_dataframe_mapping(
                dataframes
            )
            print(time.time() - s, "for mapping")

        # -- sanity checks on data
        if (
            self.alignment_columns is None
            and self.method_metadata["requires_aligned_timeseries"]
        ):
            logger.info("Checking dataframes validity...")

            # -- check that the dataframes are of the same size
            dataframes, dataframes_copy = itertools.tee(dataframes, 2)
            dataframe_shapes = {df.collect().shape for df in dataframes_copy}
            if len(dataframe_shapes) != 1:
                raise ValueError(
                    (
                        f"Method `{self.method}` requires timeseries to be"
                        "aligned but timeseries are of different shapes "
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
        s = time.time()
        paired_dataframes = pair_data(dataframes)
        print(time.time() - s, "to pair dfs")

        # -- align timeseries
        s = time.time()

        paired_dataframes = self.align_dataframes(paired_dataframes)

        print(time.time() - s, "to align dfs")

        correlation_matrix = {}
        s = time.time()

        for indices, df1, df2 in paired_dataframes:
            # HOTFIX
            logger.info(indices)
            if self.method in {"pearson", "spearman"}:
                df = df1.join(
                    df2,
                    on=self.alignment_columns,
                )
                col1 = self.target_columns[0]
                col2 = f"{col1}_right"
                correlation = self.method_metadata["callable"](
                    df, col1=col1, col2=col2, method=self.method, **kwargs
                )
            correlation_matrix[indices] = correlation
        print(time.time() - s, "to calc dfs")

        return {
            "correlation_matrix": correlation_matrix,
            "mapping": dataframe_mapping,
        }


if __name__ == "__main__":
    df1 = pl.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02", "2023-02-01"],
            "values": [1, 2, 3],
        }
    )

    df2 = pl.DataFrame(
        {
            "date": ["2022-01-02", "2022-01-01", "2023-02-02"],
            "values": [4, 5, 6],
        }
    )

    def generate_tonnes_of_data(num_timeseries=1):
        from datetime import date

        import numpy as np

        date_range = pl.date_range(
            date(2022, 1, 1), date(2023, 1, 1), "1m", eager=True
        )  # half a million points

        multiplier = np.linspace(-1, 1, num=num_timeseries)
        for i, m in enumerate(multiplier):
            df = pl.DataFrame().with_columns(
                date=date_range,
                values=pl.lit(
                    np.sin(
                        np.linspace(-(10**6), 10**6, len(date_range)) - m
                    )
                ),
                id=i,
            )
            yield df

    s = time.time()
    dfs = generate_tonnes_of_data(20)
    print(f"took {time.time() - s} secs to gen")

    processor = FindCorrelations(["values"], alignment_columns=["date"])
    output = processor.calculate_correlations(
        dfs, dataframe_mapping={i: i for i in range(1000)}
    )

    df = df1.lazy().join(df2.lazy(), on="date", how="left")
    print(df.collect())
    print(
        calculate_correlation_between_columns(
            df, "values", "values_right", "pearson"
        )
    )
