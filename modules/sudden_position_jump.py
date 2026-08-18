import polars as pl

from modules.haversine import haversine_distance
from modules.primitives.decorators import measure_time
from modules.primitives.schema import RawPositionSchema


def _validate_input(df: pl.DataFrame) -> None:
    required_columns = {
        "latitude",
        "longitude",
        "distance_to_next",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"required columns are missing: {missing_columns}")


def _distance_between_segments(
    prev_latitude: float,
    prev_longitude: float,
    next_latitude: float,
    next_longitude: float,
) -> float:
    segment_centers = pl.DataFrame(
        {
            "latitude": [prev_latitude, next_latitude],
            "longitude": [prev_longitude, next_longitude],
        }
    )

    result = haversine_distance(segment_centers)

    return result["distance_to_next"][0]


@measure_time
def remove_sudden_position_jumps(
    df: pl.DataFrame,
    jump_thres: float,
    same_place_thres: float,
    max_jump_points: int,
) -> pl.DataFrame:
    _validate_input(df)

    if df.is_empty():
        return df.select(list(RawPositionSchema.model_fields.keys()))

    # distance_to_next가 jump_thres를 넘으면
    # 다음 point부터 새로운 segment로 분할
    segmented = df.with_columns(
        (
            pl.col("distance_to_next")
            .shift(1)
            .fill_null(0)
            .gt(jump_thres)
            .cast(pl.Int64)
            .cum_sum()
        ).alias("segment_id")
    )

    # segment별 대표 위치와 point 개수 계산
    segments = (
        segmented.group_by(
            "segment_id",
            maintain_order=True,
        )
        .agg(
            pl.col("latitude").mean().alias("mean_latitude"),
            pl.col("longitude").mean().alias("mean_longitude"),
            pl.len().alias("point_count"),
        )
        .sort("segment_id")
    )

    segment_rows = segments.to_dicts()
    jump_segments = set()

    # 첫/마지막 segment는 앞뒤 segment가 모두 없으므로 제외
    for i in range(1, len(segment_rows) - 1):
        previous = segment_rows[i - 1]
        current = segment_rows[i]
        next_segment = segment_rows[i + 1]

        # 너무 긴 segment라면 sudden jump로 보지 않음
        if current["point_count"] > max_jump_points:
            continue

        distance = _distance_between_segments(
            previous["mean_latitude"],
            previous["mean_longitude"],
            next_segment["mean_latitude"],
            next_segment["mean_longitude"],
        )

        # 앞뒤 segment가 같은 장소라면
        # 가운데 짧은 segment를 sudden jump로 판단
        if distance <= same_place_thres:
            jump_segments.add(current["segment_id"])

    cleaned = segmented.filter(~pl.col("segment_id").is_in(jump_segments))

    # 파생 컬럼 제거 후 RawPositionSchema 형태로 반환
    return cleaned.select(list(RawPositionSchema.model_fields.keys()))
