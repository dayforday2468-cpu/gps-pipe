import polars as pl

from modules.primitives.config import MAX_JUMP_POINTS
from modules.primitives.decorators import measure_time
from modules.primitives.schema import (
    PositionSegmentSchema,
    RawPositionSchema,
    SegmentSchema,
    validate_schema_columns,
)


@measure_time
def detect_sudden_position_jumps(
    df: pl.DataFrame,
    position_segments: pl.DataFrame,
    segments: pl.DataFrame,
    same_place_thres: float,
) -> pl.DataFrame:
    validate_schema_columns(df, RawPositionSchema)
    validate_schema_columns(position_segments, PositionSegmentSchema)
    validate_schema_columns(segments, SegmentSchema)

    if df.is_empty():
        return pl.DataFrame({"position_id": [], "is_jump": []})

    jump_segments = segments.filter(
        (pl.col("point_count") <= MAX_JUMP_POINTS)
        & (pl.col("prev_next_distance") <= same_place_thres)
    ).select("segment_id")

    jump_positions = position_segments.join(
        jump_segments,
        on="segment_id",
        how="semi",
    ).select("position_id")

    position_jumps = (
        df.select("position_id")
        .join(
            jump_positions.with_columns(pl.lit(True).alias("is_jump")),
            on="position_id",
            how="left",
        )
        .with_columns(pl.col("is_jump").fill_null(False))
    )

    return position_jumps
