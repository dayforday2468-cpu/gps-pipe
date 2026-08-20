from datetime import datetime
import math

from modules.dbscan import st_dbscan
from modules.dbscan_tuning import (
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    find_knee,
)
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer

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

    spatial_k_distances = calculate_spatial_k_distances(
        positions,
        k=k,
    )
    temporal_k_distances = calculate_temporal_k_distances(
        positions,
        k=k,
    )

    eps_space = find_knee(spatial_k_distances)
    eps_time = find_knee(temporal_k_distances)

    clustered = st_dbscan(
        positions,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    print(f"MinPts: {min_pts}")
    print(f"Spatial Eps: {eps_space:.2f} m")
    print(f"Temporal Eps: {eps_time:.2f} s")

    visualizer = GPSVisualizer(
        title=f"ST-DBSCAN clustering - {time_range}",
    )

    noise = clustered.filter(clustered["cluster_id"] == 0)

    visualizer.add(
        noise,
        label="Noise",
        point_color="gray",
    )

    cluster_ids = (
        clustered.filter(clustered["cluster_id"] > 0)
        .get_column("cluster_id")
        .unique()
        .sort()
    )

    for cluster_id in cluster_ids:
        cluster = clustered.filter(clustered["cluster_id"] == cluster_id)

        visualizer.add(
            cluster,
            label=f"Cluster {cluster_id} ({len(cluster)} points)",
        )

    visualizer.show()
