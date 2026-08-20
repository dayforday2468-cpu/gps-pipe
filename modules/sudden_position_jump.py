import polars as pl

from modules.haversine import haversine_expr
from modules.primitives.decorators import measure_time


def _validate_input(df: pl.DataFrame) -> None:
    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"required columns are missing: {missing_columns}")

    if not df["latitude"].is_between(-90, 90).all():
        raise ValueError("latitude must be between -90 and 90")

    if not df["longitude"].is_between(-180, 180).all():
        raise ValueError("longitude must be between -180 and 180")


@measure_time
def remove_sudden_position_jumps(
    df: pl.DataFrame,
    jump_thres: float,
    same_place_thres: float,
    max_jump_points: int,
) -> pl.DataFrame:
    _validate_input(df)

    if df.is_empty():
        return df

    segmented = (
        df.with_columns(
            haversine_expr(
                pl.col("latitude"),
                pl.col("longitude"),
                pl.col("latitude").shift(-1),
                pl.col("longitude").shift(-1),
            ).alias("_distance_to_next")
        )
        .with_columns(
            (
                pl.col("_distance_to_next")
                .shift(1)
                .fill_null(0)
                .gt(jump_thres)
                .cast(pl.Int64)
                .cum_sum()
            ).alias("segment_id")
        )
    )

    segments = (
        segmented.group_by("segment_id", maintain_order=True)
        .agg(
            pl.col("latitude").mean().alias("mean_latitude"),
            pl.col("longitude").mean().alias("mean_longitude"),
            pl.len().alias("point_count"),
        )
        .sort("segment_id")
        .with_columns(
            pl.col("mean_latitude").shift(1).alias("prev_latitude"),
            pl.col("mean_longitude").shift(1).alias("prev_longitude"),
            pl.col("mean_latitude").shift(-1).alias("next_latitude"),
            pl.col("mean_longitude").shift(-1).alias("next_longitude"),
        )
        .with_columns(
            haversine_expr(
                pl.col("prev_latitude"),
                pl.col("prev_longitude"),
                pl.col("next_latitude"),
                pl.col("next_longitude"),
            ).alias("prev_next_distance")
        )
    )

    jump_segments = (
        segments.filter(
            (pl.col("point_count") <= max_jump_points)
            & (pl.col("prev_next_distance") <= same_place_thres)
        )
        .get_column("segment_id")
        .to_list()
    )

    cleaned = segmented.filter(
        ~pl.col("segment_id").is_in(jump_segments)
    )

    return cleaned.drop(
        "_distance_to_next",
        "segment_id",
    )