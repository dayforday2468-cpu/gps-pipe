from datetime import datetime

import math
import osmnx as ox
import polars as pl

from modules.dbscan import st_dbscan
from modules.map_matching import generate_candidate_positions
from modules.parameter_tuning import (
    calculate_road_k_distances,
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    find_knee,
)
from modules.primitives.config import ROAD_NETWORK_VIEW_MARGIN, MAX_CANDIDATES
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer
from modules.projection import project_positions
from modules.road_network import load_road_network

if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    raw_positions = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    # ST-DBSCAN을 실행한다.
    min_pts = math.ceil(math.log(len(raw_positions)))
    k = min_pts - 1

    spatial_k_distances = calculate_spatial_k_distances(
        raw_positions,
        k=k,
    )
    temporal_k_distances = calculate_temporal_k_distances(
        raw_positions,
        k=k,
    )

    eps_space = find_knee(spatial_k_distances)
    eps_time = find_knee(temporal_k_distances)

    clustered_positions = st_dbscan(
        raw_positions,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    # cluster_id == 0인 이동 point만 선택한다.
    moving_positions = clustered_positions.filter(pl.col("cluster_id") == 0)

    # 도로망과 GPS point를 동일한 평면 좌표계로 변환한다.
    road_network = load_road_network(
        moving_positions,
        margin=ROAD_NETWORK_VIEW_MARGIN,
    )

    projected_road_network = ox.project_graph(
        road_network,
    )

    projected_positions = project_positions(
        moving_positions,
        projected_road_network.graph["crs"],
    )

    edges = ox.graph_to_gdfs(
        projected_road_network,
        nodes=False,
        edges=True,
    )

    # 후보 도로 탐색을 위한 search radius를 추정한다.
    road_k = 3

    road_k_distances = calculate_road_k_distances(
        projected_positions,
        edges,
        k=road_k,
    )

    search_radius = road_k_distances.quantile(0.95)

    candidate_positions = generate_candidate_positions(
        projected_positions,
        edges,
        search_radius=search_radius,
        max_candidates=MAX_CANDIDATES,
    )

    print("=== Candidate Projection ===")
    print(f"Moving points: {projected_positions.height}")
    print(f"Candidate positions: {candidate_positions.height}")
    print(f"Search radius: {search_radius:.2f} m")
    print(candidate_positions.head())

    visualizer = GPSVisualizer(
        title=f"GPS Candidate Road Projections - {time_range}",
        show_legend=True,
    )

    visualizer.add_road_network(
        projected_road_network,
    )

    visualizer.add(
        projected_positions,
        label="GPS",
        point_size=4,
        point_color="red",
        alpha=0.7,
    )

    visualizer.add(
        candidate_positions,
        label="Candidates",
        point_size=4,
        point_color="blue",
        alpha=0.8,
    )

    visualizer.show()
