import polars as pl


def detect_timeseries_frequency(
    df: pl.DataFrame, time_column: str, how: str = "exact"
) -> float:
    """Function that detects frequency of a timeseries using the diff
    operation.

    :param df: dataframe
    :param time_column: time series column
    :param how: strategy for calculating frequency. If `exact` then
        timeseries must have a single frequency (e.g. no missing data!),
        if `mode` then detects frequency as the most commonly occurring
        difference between consecutive timestamps, if `max` then detects
        frequency as the maximum occuring difference, defaults to
        "exact"
    :return: The detected frequency in seconds
    """
    # how=exact, mode, max
    SUPPORTED_METHODS = {"exact": "unique", "mode": "mode", "max": "max"}
    frequency_detector = SUPPORTED_METHODS.get(how)
    if frequency_detector is None:
        raise ValueError(
            f"Expected `how` in {sorted(SUPPORTED_METHODS)}. Got `{how}`"
        )

    diff = df.select(pl.col(time_column).diff(null_behavior="drop"))
    frequency = getattr(diff[time_column], frequency_detector)()

    if how == "exact":
        num_unique_frequencies = len(frequency)
        if num_unique_frequencies != 1:
            _remaining_methods = sorted(
                [method for method in SUPPORTED_METHODS if method != "exact"]
            )
            raise ValueError(
                (
                    f"Got {num_unique_frequencies} unique frequencies when "
                    "expected only one. If you wish to work with non-exact "
                    f"frequencies, set `how` to one of {_remaining_methods}"
                )
            )

    frequency = frequency.item().total_seconds()
    return frequency
