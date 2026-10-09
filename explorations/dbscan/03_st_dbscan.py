from datetime import datetime

import math
import polars as pl

from modules.dbscan import st_dbscan
from modules.parameter_tuning import (
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    find_knee,
)
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.schema import PositionState
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
    k = min_pts - 1

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

    position_segments, clusters, movements = st_dbscan(
        positions,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    segmented = positions.join(
        position_segments,
        on="position_id",
        how="inner",
    )

    print(f"MinPts: {min_pts}")
    print(f"Spatial Eps: {eps_space:.2f} m")
    print(f"Temporal Eps: {eps_time:.2f} s")

    print("=== Clusters ===")
    print(clusters)

    print("=== Movements ===")
    print(movements)

    visualizer = GPSVisualizer(
        title=f"ST-DBSCAN segmentation - {time_range}",
    )

    removed = segmented.filter(
        pl.col("state") == PositionState.REMOVED.value
    )

    visualizer.add(
        removed,
        label=f"Removed ({len(removed)} points)",
        point_color="gray",
    )

    stay_segment_ids = (
        segmented
        .filter(pl.col("state") == PositionState.STAY.value)
        .get_column("segment_id")
        .unique()
        .sort()
    )

    for segment_id in stay_segment_ids:
        stay = segmented.filter(
            (pl.col("state") == PositionState.STAY.value)
            & (pl.col("segment_id") == segment_id)
        )

        visualizer.add(
            stay,
            label=f"Stay {segment_id} ({len(stay)} points)",
        )

    movement_segment_ids = (
        segmented
        .filter(pl.col("state") == PositionState.MOVEMENT.value)
        .get_column("segment_id")
        .unique()
        .sort()
    )

    for segment_id in movement_segment_ids:
        movement = segmented.filter(
            (pl.col("state") == PositionState.MOVEMENT.value)
            & (pl.col("segment_id") == segment_id)
        )

        visualizer.add(
            movement,
            label=f"Movement {segment_id} ({len(movement)} points)",
            point_color="black",
        )

    visualizer.animate(mode="time")