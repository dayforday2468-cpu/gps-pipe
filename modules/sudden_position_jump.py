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
def remove_sudden_position_jumps(
    df: pl.DataFrame,
    position_segments: pl.DataFrame,
    segments: pl.DataFrame,
    same_place_thres: float,
) -> pl.DataFrame:
    validate_schema_columns(df, RawPositionSchema)
    validate_schema_columns(position_segments, PositionSegmentSchema)
    validate_schema_columns(segments, SegmentSchema)

    if df.is_empty():
        return df

    jump_segments = segments.filter(
        (pl.col("point_count") <= MAX_JUMP_POINTS)
        & (pl.col("prev_next_distance") <= same_place_thres)
    ).select("segment_id")

    jump_positions = position_segments.join(
        jump_segments,
        on="segment_id",
        how="semi",
    )

    cleaned = df.join(
        jump_positions.select("position_id"),
        on="position_id",
        how="anti",
    )

    return cleaned
