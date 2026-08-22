from datetime import datetime

import matplotlib.pyplot as plt

from modules.haversine import haversine_distance
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

    raw_with_distance = haversine_distance(raw_filtered)

    distances = (
        raw_with_distance.get_column("distance_to_next")
        .drop_nulls()
        .sort(descending=True)
    )

    jump_thres = find_knee(distances)

    print("=== Distance Summary ===")
    print(distances.describe())

    print(f"\nJump Threshold: {jump_thres:.2f} m")

    plt.figure(figsize=(10, 6))

    plt.plot(distances.to_list())

    plt.axhline(
        y=jump_thres,
        color="red",
        linestyle="--",
        label=f"Jump Threshold = {jump_thres:.2f} m",
    )

    plt.title(f"Distance distribution - {time_range}")
    plt.xlabel("Point Pairs Sorted by Distance")
    plt.ylabel("Distance to Next Point (m)")
    plt.grid()
    plt.legend()

    plt.show()
