from datetime import datetime

import matplotlib.pyplot as plt

from modules.haversine import haversine_distance
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline

if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 2, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    raw_with_distance = haversine_distance(raw_filtered)

    distances = raw_with_distance.select("distance_to_next").drop_nulls()

    print("=== Distance Summary ===")
    print(distances.describe())

    print("\n=== Largest Distances ===")
    print(
        raw_with_distance.select(
            "timestamp",
            "latitude",
            "longitude",
            "distance_to_next",
        )
        .drop_nulls()
        .sort("distance_to_next", descending=True)
        .head(20)
    )

    plt.figure(figsize=(10, 6))
    plt.hist(
        distances["distance_to_next"].to_numpy(),
        bins=50,
    )

    plt.title(f"Distribution of Distance - {time_range}")
    plt.xlabel("Distance to Next Point (m)")
    plt.ylabel("Count")

    plt.show()
