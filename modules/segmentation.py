import polars as pl

from modules.haversine import haversine_expr


def segment_positions(
    df: pl.DataFrame,
    jump_thres: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    segmented = df.with_columns(
        haversine_expr(
            pl.col("latitude"),
            pl.col("longitude"),
            pl.col("latitude").shift(-1),
            pl.col("longitude").shift(-1),
        ).alias("_distance_to_next")
    ).with_columns(
        (
            pl.col("_distance_to_next")
            .shift(1)
            .fill_null(0)
            .gt(jump_thres)
            .cast(pl.Int64)
            .cum_sum()
        ).alias("segment_id")
    )

    position_segments = segmented.select(
        "position_id",
        "segment_id",
    )

    segments = (
        segmented.group_by(
            "segment_id",
            maintain_order=True,
        )
        .agg(
            pl.col("latitude").mean().alias("mean_latitude"),
            pl.col("longitude").mean().alias("mean_longitude"),
            pl.col("position_id").first().alias("head_position_id"),
            pl.col("position_id").last().alias("tail_position_id"),
            pl.len().alias("point_count"),
        )
        .sort("segment_id")
        .with_columns(
            pl.col("mean_latitude").shift(1).alias("_prev_latitude"),
            pl.col("mean_longitude").shift(1).alias("_prev_longitude"),
            pl.col("mean_latitude").shift(-1).alias("_next_latitude"),
            pl.col("mean_longitude").shift(-1).alias("_next_longitude"),
        )
        .with_columns(
            haversine_expr(
                pl.col("_prev_latitude"),
                pl.col("_prev_longitude"),
                pl.col("_next_latitude"),
                pl.col("_next_longitude"),
            ).alias("prev_next_distance")
        )
        .drop(
            "_prev_latitude",
            "_prev_longitude",
            "_next_latitude",
            "_next_longitude",
        )
    )

    return position_segments, segments
