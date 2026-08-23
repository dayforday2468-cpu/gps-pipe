from datetime import datetime

import polars as pl

from modules.parameter_tuning import estimate_jump_threshold
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer
from modules.segmentation import segment_positions

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
    jump_thres = estimate_jump_threshold(raw_filtered)

    print(f"Jump Threshold: {jump_thres:.2f} m")

    # jump threshold를 기준으로 segment 분할
    position_segments, segments = segment_positions(
        raw_filtered,
        jump_thres=jump_thres,
    )

    # 시각화를 위해 원본 위치 정보와 segment 관계를 결합
    segmented = raw_filtered.join(
        position_segments,
        on="position_id",
        how="inner",
    )

    print("\n=== Distance Between Previous and Next Segment Means ===")

    with pl.Config(tbl_rows=-1):
        print(
            segments.filter(pl.col("prev_next_distance").is_not_null()).select(
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
    for i, row in enumerate(segments.iter_rows(named=True)):
        color = colors[i % len(colors)]

        mean_df = pl.DataFrame(
            {
                "latitude": [row["mean_latitude"]],
                "longitude": [row["mean_longitude"]],
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
    segment_ids = segments.get_column("segment_id").to_list()

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
    for i in range(len(segments) - 1):
        current_segment = segments.row(i, named=True)
        next_segment = segments.row(i + 1, named=True)

        connector = raw_filtered.filter(
            pl.col("position_id").is_in(
                [
                    current_segment["tail_position_id"],
                    next_segment["head_position_id"],
                ]
            )
        )

        visualizer.add(
            connector,
            label=None,
            point_size=0,
            point_color=colors[i % len(colors)],
            show_line=True,
            line_color=colors[i % len(colors)],
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
