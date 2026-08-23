import polars as pl

from modules.primitives.config import MAX_JUMP_POINTS
from modules.primitives.decorators import measure_time


def _validate_input(
    df: pl.DataFrame,
    position_segments: pl.DataFrame,
    segments: pl.DataFrame,
) -> None:
    required_position_columns = {
        "position_id",
        "latitude",
        "longitude",
        "timestamp",
    }
    missing_position_columns = required_position_columns - set(df.columns)

    if missing_position_columns:
        raise ValueError(
            f"required position columns are missing: {missing_position_columns}"
        )

    required_position_segment_columns = {
        "position_id",
        "segment_id",
    }
    missing_position_segment_columns = required_position_segment_columns - set(
        position_segments.columns
    )

    if missing_position_segment_columns:
        raise ValueError(
            "required position segment columns are missing: "
            f"{missing_position_segment_columns}"
        )

    required_segment_columns = {
        "segment_id",
        "point_count",
        "prev_next_distance",
    }
    missing_segment_columns = required_segment_columns - set(segments.columns)

    if missing_segment_columns:
        raise ValueError(
            f"required segment columns are missing: {missing_segment_columns}"
        )


@measure_time
def remove_sudden_position_jumps(
    df: pl.DataFrame,
    position_segments: pl.DataFrame,
    segments: pl.DataFrame,
    same_place_thres: float,
) -> pl.DataFrame:
    _validate_input(
        df,
        position_segments,
        segments,
    )

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
