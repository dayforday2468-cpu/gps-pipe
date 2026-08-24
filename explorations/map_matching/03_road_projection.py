from datetime import datetime

import math
import osmnx as ox
import polars as pl

from modules.dbscan import st_dbscan
from modules.parameter_tuning import (
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    find_knee,
)
from modules.primitives.config import ROAD_NETWORK_VIEW_MARGIN
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer
from modules.projection import project_point_to_edge, project_positions
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
    moving_positions = clustered_positions.filter(
        pl.col("cluster_id") == 0
    )

    # 도로망과 GPS point를 동일한 평면 좌표계로 변환한다.
    road_network = load_road_network(
        moving_positions,
        margin=ROAD_NETWORK_VIEW_MARGIN,
    )

    projected_road_network = ox.project_graph(road_network)

    projected_positions = project_positions(
        moving_positions,
        projected_road_network.graph["crs"],
    )

    projected_candidates = []

    # 각 GPS point를 가장 가까운 수직 projection 가능한 도로에 투영한다.
    for position in projected_positions.iter_rows(named=True):
        best_candidate = None

        for u, v, key, edge in projected_road_network.edges(
            keys=True,
            data=True,
        ):
            projection = project_point_to_edge(
                position["x"],
                position["y"],
                edge["geometry"],
            )

            if projection is None:
                continue

            projected_x, projected_y, distance = projection

            if (
                best_candidate is None
                or distance < best_candidate["distance"]
            ):
                best_candidate = {
                    "position_id": position["position_id"],
                    "edge_u": u,
                    "edge_v": v,
                    "edge_key": key,
                    "x": projected_x,
                    "y": projected_y,
                    "distance": distance,
                }

        if best_candidate is not None:
            projected_candidates.append(best_candidate)

    candidate_positions = pl.DataFrame(projected_candidates)

    print("=== Road Projection ===")
    print(f"Moving points: {projected_positions.height}")
    print(f"Projected points: {candidate_positions.height}")
    print(candidate_positions.head())

    visualizer = GPSVisualizer(
        title=f"Project GPS Points onto Nearest Road - {time_range}",
        show_legend=True,
    )

    visualizer.add_road_network(projected_road_network)

    visualizer.add(
        projected_positions,
        label="GPS",
        point_size=4,
        point_color="red",
        alpha=0.7,
    )

    visualizer.add(
        candidate_positions,
        label="Projected",
        point_size=4,
        point_color="blue",
        alpha=0.8,
    )

    visualizer.show()