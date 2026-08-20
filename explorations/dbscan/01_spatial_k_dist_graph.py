from datetime import datetime
import math

import matplotlib.pyplot as plt

from modules.dbscan_tuning import calculate_spatial_k_distances, find_knee
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline

if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    positions = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    min_pts = math.ceil(math.log(len(positions)))
    k = min_pts

    k_distances = calculate_spatial_k_distances(
        positions,
        k=k,
    )

    eps = find_knee(k_distances)

    print(f"MinPts: {min_pts}")
    print(f"Spatial Eps: {eps:.2f} m")

    plt.plot(k_distances.to_list())
    plt.axhline(
        y=eps,
        color="red",
        linestyle="--",
        label=f"Eps = {eps:.2f} m",
    )

    plt.xlabel("Points sorted by k-distance")
    plt.ylabel(f"{k}-NN distance (m)")
    plt.title(f"Spatial {k}-distance graph - {time_range}")
    plt.grid()
    plt.legend()
    plt.show()
