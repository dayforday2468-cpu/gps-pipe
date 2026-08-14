import json

from modules.decorators import measure_time
import polars as pl


def extract_positions(raw_signals):
    for signal in raw_signals:
        position = signal.get("position")

        if position is not None:
            yield position


@measure_time
def dataload(path: str) -> pl.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pl.DataFrame(
        extract_positions(data["rawSignals"])
    )

    return (
        df
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