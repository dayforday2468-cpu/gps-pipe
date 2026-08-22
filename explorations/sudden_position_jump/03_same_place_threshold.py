from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl

from modules.haversine import haversine_distance, haversine_expr
from modules.primitives.config import JUMP_RATE
from modules.parameter_tuning import find_knee
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline

if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    # 1-distance graph를 이용해 jump threshold 추정
    raw_with_distance = haversine_distance(raw_filtered)

    distances = (
        raw_with_distance.get_column("distance_to_next")
        .drop_nulls()
        .sort(descending=True)
    )

    jump_thres = find_knee(distances)

    print(f"Jump Threshold: {jump_thres:.2f} m")

    # jump threshold를 기준으로 segment 분할
    segmented = raw_with_distance.with_columns(
        (
            pl.col("distance_to_next")
            .shift(1)
            .fill_null(0)
            .gt(jump_thres)
            .cast(pl.Int64)
            .cum_sum()
        ).alias("segment_id")
    )

    # segment별 평균 위치 및 point 수 계산
    segment_stats = (
        segmented.group_by("segment_id", maintain_order=True)
        .agg(
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
            pl.len().alias("point_count"),
        )
        .sort("segment_id")
        .with_columns(
            pl.col("latitude").shift(1).alias("prev_latitude"),
            pl.col("longitude").shift(1).alias("prev_longitude"),
            pl.col("latitude").shift(-1).alias("next_latitude"),
            pl.col("longitude").shift(-1).alias("next_longitude"),
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

    # jump 후보 segment의 prev-next distance 추출
    candidate_segments = segment_stats.filter(
        pl.col("prev_next_distance").is_not_null()
    ).sort("prev_next_distance", descending=True)

    print("\n=== Same Place Threshold Candidates ===")

    with pl.Config(tbl_rows=-1):
        print(
            candidate_segments.select(
                "segment_id",
                "point_count",
                pl.col("prev_next_distance").round(2),
            )
        )

    same_place_distances = candidate_segments.get_column("prev_next_distance")

    same_place_thres = same_place_distances.quantile(JUMP_RATE)

    print(f"\nSame Place Threshold: {same_place_thres:.2f} m")

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(len(same_place_distances)),
        same_place_distances.to_list(),
        marker="o",
    )

    plt.axhline(
        y=same_place_thres,
        linestyle="--",
        label=f"Same Place Threshold = {same_place_thres:.2f} m",
    )

    plt.title(f"Same Place Distance Distribution - {time_range}")
    plt.xlabel("Candidate Segments Sorted by Distance")
    plt.ylabel("Previous-Next Segment Mean Distance (m)")
    plt.grid()
    plt.legend()

    plt.show()
