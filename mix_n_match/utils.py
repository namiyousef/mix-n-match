from typing import List, Tuple

import numpy as np
import polars as pl

# NANOSECOND = 1
# SECOND = NANOSECOND * 10**9
# MINUTE = SECOND * 60


# POLARS_TO_TIME_UNIT_MAPPING = {"m": MINUTE}


class PolarsDuration:
    """Class for working with polars string durations. This is created since as
    of 27.11.2023 there is no native support in polars for converting string
    durations into time components.

    :param duration: Polars string duration, e.g. "1w"
    """

    def __init__(self, duration: str) -> None:
        self.duration = duration
        self.decomposed_duration = self._decompose_duration(duration)

    def _recompose_duration(
        self, decomposed_duration: List[Tuple[int, str]]
    ) -> str:
        """Function to recompose a decomposed duration back into string form.

        :param decomposed_duration: decomposed string
        :return: recomposed duration in string format

        Example:
            decomposed = [(1, 'd'), (1, 'h')]
            PolarsDuration(None)._recompose_duration(decomposed)
            >>> "1d1h"
        """
        duration = ""
        for multipler, unit in decomposed_duration:
            duration += f"{multipler}{unit}"

        return duration

    def _decompose_duration(self, duration: str) -> List[Tuple[int, str]]:
        """Function to decompose a Polars duration string into indiviudal time
        compoennts.

        :param duration: polars duration string, e.g. "1w"
        :return: string decomposed into it's time components

        Example:
            duration = "3d12h4m25s"
            PolarsDuration(None)._decompose_duration(duration)
            >>> [
                (3, "d"),
                (12, "h"),
                (4, "m"),
                (25, "s"),
            ]
        """
        multiplier_string = ""
        time_component = ""
        components = []
        for item in duration:
            is_digit = item.isdigit()
            if is_digit:
                if time_component:
                    components.append((int(multiplier_string), time_component))
                    multiplier_string = item
                    time_component = ""
                else:
                    multiplier_string += item
            else:
                time_component += item

        components.append((int(multiplier_string), time_component))

        return components

    def __mul__(self, multiply_by: int) -> str:
        """Method to enble multiplication of a polars duration string by some
        integer.

        :param multiply_by: integer to multiply by
        :return: polars duration multiplied by value returned as a string

        Example:
            duration = "1d"
            pl_duration = PolarsDuration(duration)
            pl_duration * 5
            >>> "5d"
        """
        decomposed_duration = [
            (multiplier * multiply_by, unit)
            for multiplier, unit in self.decomposed_duration
        ]

        # TODO should it return the class here???
        # E.g. PolarsDuration(self._recompose_duration(decomposed_duration))
        return self._recompose_duration(decomposed_duration)


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


# features to add:
# only get contiguous segments of a single value


# only get contiguous segments of a specific length
def find_contiguous_segments(
    array: np.array, filter_mask=None
) -> List[List[int]]:
    """Returns a list of start, end indices to identify contiguous segments in
    an array.

    :param array: array
    """

    # NOTE: array should be 1-D
    non_matching_mask = (
        array[:-1] != array[1:]
    )  # find mask of elements where the next element is differnet to current one
    size = array.shape[0]

    index_array = np.arange(0, size).reshape(-1, 1)

    segment_end_indices = index_array[1:][non_matching_mask]

    segment_start_indices = np.concatenate(
        (index_array[:1], segment_end_indices)
    )

    segment_end_indices = np.concatenate(
        (segment_end_indices, index_array[-1:] + 1)
    )
    indices_array = np.concatenate(
        [segment_start_indices, segment_end_indices], axis=1
    )

    # Not sure if I want this, but:
    # return the actual end index
    # indices_array[:, 1] -= 1

    indices_list = indices_array.tolist()

    return indices_list
