from datetime import datetime

import polars as pl

from modules.haversine import haversine_distance, haversine_expr
from modules.parameter_tuning import find_knee
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer

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

    # segment별 평균 위치, point 수, head/tail 계산
    segment_stats = (
        segmented.group_by("segment_id", maintain_order=True)
        .agg(
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
            pl.len().alias("point_count"),
            pl.first("latitude").alias("head_latitude"),
            pl.first("longitude").alias("head_longitude"),
            pl.first("timestamp").alias("head_timestamp"),
            pl.last("latitude").alias("tail_latitude"),
            pl.last("longitude").alias("tail_longitude"),
            pl.last("timestamp").alias("tail_timestamp"),
        )
        .sort("segment_id")
        .with_columns(
            pl.col("latitude").shift(1).alias("prev_latitude"),
            pl.col("longitude").shift(1).alias("prev_longitude"),
            pl.col("latitude").shift(-1).alias("next_latitude"),
            pl.col("longitude").shift(-1).alias("next_longitude"),
            pl.col("head_latitude").shift(-1).alias("next_head_latitude"),
            pl.col("head_longitude").shift(-1).alias("next_head_longitude"),
            pl.col("head_timestamp").shift(-1).alias("next_head_timestamp"),
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

    print("\n=== Distance Between Previous and Next Segment Means ===")

    with pl.Config(tbl_rows=-1):
        print(
            segment_stats.filter(pl.col("prev_next_distance").is_not_null()).select(
                "segment_id",
                "point_count",
                pl.col("prev_next_distance").round(2),
            )
        )

    visualizer = GPSVisualizer(
        title=f"Sudden Position Jump Exploration - {time_range}",
        show_legend=True,
    )

    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
        "magenta",
    ]

    # segment 평균 위치
    for i, row in enumerate(segment_stats.iter_rows(named=True)):
        color = colors[i % len(colors)]

        mean_df = pl.DataFrame(
            {
                "latitude": [row["latitude"]],
                "longitude": [row["longitude"]],
            }
        )

        visualizer.add(
            mean_df,
            label="Segment mean" if i == 0 else None,
            point_size=120,
            point_color=color,
            show_line=False,
            alpha=1.0,
        )

    # segment trajectory
    segment_ids = segment_stats.get_column("segment_id").to_list()

    for i, segment_id in enumerate(segment_ids):
        color = colors[i % len(colors)]

        segment_df = segmented.filter(pl.col("segment_id") == segment_id)

        visualizer.add(
            segment_df,
            label=f"Segment {segment_id}",
            point_size=14,
            point_color=color,
            show_line=True,
            line_color=color,
            line_style="-",
            line_width=1.5,
            alpha=0.75,
        )

    # segment 사이 connector
    connector_stats = segment_stats.filter(pl.col("next_head_latitude").is_not_null())

    for i, row in enumerate(connector_stats.iter_rows(named=True)):
        color = colors[i % len(colors)]

        connector = pl.DataFrame(
            {
                "latitude": [
                    row["tail_latitude"],
                    row["next_head_latitude"],
                ],
                "longitude": [
                    row["tail_longitude"],
                    row["next_head_longitude"],
                ],
                "timestamp": [
                    row["tail_timestamp"],
                    row["next_head_timestamp"],
                ],
            }
        )

        visualizer.add(
            connector,
            label=None,
            point_size=0,
            point_color=color,
            show_line=True,
            line_color=color,
            line_style="--",
            line_width=1.2,
            alpha=0.8,
        )

    # visualizer.animate(
    #     interval=100,
    #     repeat=False,
    #     mode="time",
    # )

    visualizer.show()
