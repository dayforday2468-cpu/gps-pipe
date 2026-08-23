import ijson
import polars as pl
from collections.abc import Iterator

from modules.primitives.config import BATCH_SIZE
from modules.primitives.decorators import measure_generator_time


def _extract_raw_segments(path: str, key: str):
    with open(path, "rb") as f:
        for segment in ijson.items(f, "rawSignals.item"):
            value = segment.get(key)

            if value is not None:
                yield value


def _extract_semantic_segments(path: str, key: str):
    with open(path, "rb") as f:
        for segment in ijson.items(
            f,
            "semanticSegments.item",
        ):
            value = segment.get(key)

            if value is not None:
                yield {
                    "startTime": segment["startTime"],
                    "endTime": segment["endTime"],
                    key: value,
                }


def _batch_records(records, batch_size: int):
    batch = []

    for record in records:
        batch.append(record)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


@measure_generator_time
def load_raw_positions_batches(
    path: str,
    batch_size: int = BATCH_SIZE,
) -> Iterator[pl.DataFrame]:
    records = _extract_raw_segments(
        path,
        key="position",
    )

    position_offset = 0

    for batch in _batch_records(records, batch_size):
        df = (
            pl.DataFrame(batch)
            .with_columns(
                pl.col("LatLng").str.split_exact(",", 1).alias("coordinates"),
                pl.col("timestamp")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%:z")
                .dt.convert_time_zone("UTC")
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
                "timestamp",
            )
            .with_row_index(
                "position_id",
                offset=position_offset,
            )
            .select(
                "position_id",
                "latitude",
                "longitude",
                "timestamp",
            )
        )

        position_offset += df.height

        yield df


@measure_generator_time
def load_timeline_paths_batches(
    path: str,
    batch_size: int = BATCH_SIZE,
) -> Iterator[pl.DataFrame]:
    records = _extract_semantic_segments(
        path,
        key="timelinePath",
    )

    position_offset = 0

    for batch in _batch_records(records, batch_size):
        df = (
            pl.DataFrame(batch)
            .explode("timelinePath")
            .with_columns(
                pl.col("timelinePath")
                .struct.field("point")
                .str.split_exact(",", 1)
                .alias("coordinates"),
                pl.col("timelinePath")
                .struct.field("time")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%:z")
                .dt.convert_time_zone("UTC")
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
                "timestamp",
            )
            .with_row_index(
                "position_id",
                offset=position_offset,
            )
            .select(
                "position_id",
                "latitude",
                "longitude",
                "timestamp",
            )
        )

        position_offset += df.height

        yield df


@measure_generator_time
def load_visits_batches(
    path: str,
    batch_size: int = BATCH_SIZE,
) -> Iterator[pl.DataFrame]:
    records = _extract_semantic_segments(
        path,
        key="visit",
    )

    for batch in _batch_records(records, batch_size):
        df = (
            pl.DataFrame(batch)
            .with_columns(
                pl.col("visit")
                .struct.field("topCandidate")
                .struct.field("placeLocation")
                .struct.field("latLng")
                .str.split_exact(",", 1)
                .alias("coordinates"),
                pl.col("startTime")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%:z")
                .dt.convert_time_zone("UTC")
                .alias("start_time"),
                pl.col("endTime")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%:z")
                .dt.convert_time_zone("UTC")
                .alias("end_time"),
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
                pl.col("visit").struct.field("probability").alias("probability"),
                pl.col("visit")
                .struct.field("topCandidate")
                .struct.field("semanticType")
                .alias("semantic_type"),
                pl.col("visit")
                .struct.field("topCandidate")
                .struct.field("probability")
                .alias("place_probability"),
            )
            .select(
                "start_time",
                "end_time",
                "latitude",
                "longitude",
                "semantic_type",
                "probability",
                "place_probability",
            )
        )

        yield df


@measure_generator_time
def load_activities_batches(
    path: str,
    batch_size: int = BATCH_SIZE,
) -> Iterator[pl.DataFrame]:
    records = _extract_semantic_segments(
        path,
        key="activity",
    )

    for batch in _batch_records(records, batch_size):
        df = (
            pl.DataFrame(batch)
            .with_columns(
                pl.col("activity")
                .struct.field("start")
                .struct.field("latLng")
                .str.split_exact(",", 1)
                .alias("start_coordinates"),
                pl.col("activity")
                .struct.field("end")
                .struct.field("latLng")
                .str.split_exact(",", 1)
                .alias("end_coordinates"),
                pl.col("startTime")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%:z")
                .dt.convert_time_zone("UTC")
                .alias("start_time"),
                pl.col("endTime")
                .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.3f%:z")
                .dt.convert_time_zone("UTC")
                .alias("end_time"),
            )
            .with_columns(
                pl.col("start_coordinates")
                .struct.field("field_0")
                .str.replace("°", "")
                .str.strip_chars()
                .cast(pl.Float64)
                .alias("start_latitude"),
                pl.col("start_coordinates")
                .struct.field("field_1")
                .str.replace("°", "")
                .str.strip_chars()
                .cast(pl.Float64)
                .alias("start_longitude"),
                pl.col("end_coordinates")
                .struct.field("field_0")
                .str.replace("°", "")
                .str.strip_chars()
                .cast(pl.Float64)
                .alias("end_latitude"),
                pl.col("end_coordinates")
                .struct.field("field_1")
                .str.replace("°", "")
                .str.strip_chars()
                .cast(pl.Float64)
                .alias("end_longitude"),
                pl.col("activity").struct.field("distanceMeters").alias("distance"),
                pl.col("activity").struct.field("probability").alias("probability"),
                pl.col("activity")
                .struct.field("topCandidate")
                .struct.field("type")
                .alias("activity_type"),
                pl.col("activity")
                .struct.field("topCandidate")
                .struct.field("probability")
                .alias("activity_probability"),
            )
            .select(
                "start_time",
                "end_time",
                "start_latitude",
                "start_longitude",
                "end_latitude",
                "end_longitude",
                "distance",
                "activity_type",
                "probability",
                "activity_probability",
            )
        )

        yield df
