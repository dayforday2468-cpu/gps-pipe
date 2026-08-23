from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl

from modules.parameter_tuning import (
    estimate_jump_threshold,
    estimate_same_place_threshold,
)
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
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
    _, segments = segment_positions(
        raw_filtered,
        jump_thres=jump_thres,
    )

    # jump 후보 segment의 prev-next distance 추출
    candidate_segments = segments.filter(
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

    same_place_thres = estimate_same_place_threshold(
        segments,
    )

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
