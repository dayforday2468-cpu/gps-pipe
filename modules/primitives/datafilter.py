from collections.abc import Iterator
from datetime import datetime

import polars as pl

from modules.primitives.timeutils import kst_to_utc_range


def filter_points(
    batches: Iterator[pl.DataFrame],
    start: datetime,
    end: datetime,
) -> Iterator[pl.DataFrame]:
    start_utc, end_utc = kst_to_utc_range(start, end)

    for df in batches:
        if df["timestamp"][0] > end_utc:
            break

        filtered = df.filter(
            (pl.col("timestamp") >= start_utc) & (pl.col("timestamp") < end_utc)
        )

        if not filtered.is_empty():
            yield filtered


def filter_intervals(
    batches: Iterator[pl.DataFrame],
    start: datetime,
    end: datetime,
) -> Iterator[pl.DataFrame]:
    start_utc, end_utc = kst_to_utc_range(start, end)

    for df in batches:
        if df["start_time"][0] > end_utc:
            break

        filtered = df.filter(
            (pl.col("start_time") < end_utc) & (pl.col("end_time") > start_utc)
        )

        if not filtered.is_empty():
            yield filtered
