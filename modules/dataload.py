import ijson
import polars as pl
from collections.abc import Iterator

from modules.decorators import *


def extract_raw_segments(path: str, key: str):
    with open(path, "rb") as f:
        for segment in ijson.items(f, "rawSignals.item"):
            value = segment.get(key)

            if value is not None:
                yield value


def batch_records(records, batch_size: int):
    batch = []

    for record in records:
        batch.append(record)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


@measure_generator_time
def load_raw_positions(
    path: str,
    batch_size: int = 10000,
) -> Iterator[pl.DataFrame]:

    records = extract_raw_segments(
        path,
        key="position",
    )

    for batch in batch_records(records, batch_size):
        df = (
            pl.DataFrame(batch)
            .with_columns(
                pl.col("LatLng")
                .str.split_exact(",", 1)
                .alias("coordinates"),

                pl.col("timestamp")
                .str.to_datetime(
                    format="%Y-%m-%dT%H:%M:%S%.3f%:z"
                )
                .dt.convert_time_zone("Asia/Seoul")
                .alias("timestamp"),
            )
            .with_columns(
                pl.col("coordinates")
                .struct.field("field_0")
                .str.replace("°", "")
                .str.strip_chars()
                .cast(pl.Float64)
                .alias("latitude"),

                pl.col("coordinates")
                .struct.field("field_1")
                .str.replace("°", "")
                .str.strip_chars()
                .cast(pl.Float64)
                .alias("longitude"),
            )
            .select(
                "latitude",
                "longitude",
                pl.col("accuracyMeters").alias("accuracy"),
                "timestamp",
            )
        )

        yield df
