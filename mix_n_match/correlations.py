import itertools

import polars as pl


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


if __name__ == "__main__":
    df1 = pl.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02", "2023-02-01"],
            "values": [1, 2, 3],
        }
    )

    df2 = pl.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02", "2023-02-01"],
            "values": [4, 5, 6],
        }
    )

    dfs = iter([df1.lazy(), df2.lazy()])
    print(calculate_correlations(dfs))
