from collections.abc import Iterator
from datetime import datetime

import polars as pl

from modules.primitives.decorators import measure_time
from modules.primitives.timeutils import kst_to_utc_range


@measure_time
def filter_points(
    batches: Iterator[pl.DataFrame],
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    start_utc, end_utc = kst_to_utc_range(start, end)
    filtered_batches = []

    for df in batches:
        # 시간순 정렬을 전제로 조회 범위를 지나면 읽기 중단
        if df["timestamp"][0] >= end_utc:
            break

        filtered = df.filter(
            (pl.col("timestamp") >= start_utc) & (pl.col("timestamp") < end_utc)
        )

        if not filtered.is_empty():
            filtered_batches.append(filtered)

    if not filtered_batches:
        return pl.DataFrame()

    return pl.concat(filtered_batches)


@measure_time
def filter_intervals(
    batches: Iterator[pl.DataFrame],
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    start_utc, end_utc = kst_to_utc_range(start, end)
    filtered_batches = []

    for df in batches:
        # start_time 순 정렬을 전제로 조회 범위를 지나면 읽기 중단
        if df["start_time"][0] >= end_utc:
            break

        filtered = df.filter(
            (pl.col("start_time") < end_utc) & (pl.col("end_time") > start_utc)
        )

        if not filtered.is_empty():
            filtered_batches.append(filtered)

    if not filtered_batches:
        return pl.DataFrame()

    return pl.concat(filtered_batches)
