from datetime import datetime
import math

import matplotlib.pyplot as plt

from modules.dbscan_tuning import calculate_k_distances
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline


if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)

    positions = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    min_pts = math.ceil(math.log(len(positions)))
    k = min_pts

    k_distances = calculate_k_distances(
        positions,
        k=k,
    )

    plt.plot(k_distances.to_list())
    plt.xlabel("Points sorted by k-distance")
    plt.ylabel(f"{k}-NN distance (m)")
    plt.title(f"{k}-distance graph")
    plt.grid()
    plt.show()