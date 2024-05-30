# mix-n-match

A Polars-based package for efficient, easy and intuitive data processing.

## Available Methods

| Method      | Features |
| ----------- | ----------- |
| `ResampleData`      |  Extends Polars ressampling to allow dealing with partial resampling windows  |
| `FilterDataBasedOnTime`| Intuitive temporal dataframe filtration [[blog]](https://towardsdatascience.com/intuitive-temporal-dataframe-filtration-fa9d5da734b3)

## Available Utils

| Method | Features |
| ----------- | ----------- |
| `find_contiguous_segments` | Method to find start/end ids of contiguous segments within an ID, with support for selecting items based on a condition, or returning contiguous items for a specifc length [[blog]](https://namiyousef96.medium.com/finding-contiguous-segments-efficiently-b0530bca7821)|
