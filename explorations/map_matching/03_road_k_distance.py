from datetime import datetime

import math
import matplotlib.pyplot as plt
import osmnx as ox
import polars as pl

from modules.dbscan import st_dbscan
from modules.parameter_tuning import (
    calculate_road_k_distances,
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    estimate_jump_threshold,
    estimate_same_place_threshold,
    find_knee,
)
from modules.primitives.config import ROAD_NETWORK_VIEW_MARGIN
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.projection import project_positions
from modules.road_network import load_road_network
from modules.segmentation import segment_positions
from modules.sudden_position_jump import detect_sudden_position_jumps

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

    # Sudden Position Jump 파라미터를 추정한다.
    jump_thres = estimate_jump_threshold(raw_positions)

    # GPS 데이터를 이동 단위의 segment로 분할한다.
    position_segments, segments = segment_positions(
        raw_positions,
        jump_thres=jump_thres,
    )

    same_place_thres = estimate_same_place_threshold(
        segments,
    )

    # Sudden Position Jump를 제거하여 GPS 데이터를 정제한다.
    position_jumps = detect_sudden_position_jumps(
        raw_positions,
        position_segments,
        segments,
        same_place_thres=same_place_thres,
    )

    cleaned_positions = raw_positions.join(
        position_jumps.filter(pl.col("is_jump")),
        on="position_id",
        how="anti",
    )

    # ST-DBSCAN을 실행한다.
    min_pts = math.ceil(math.log(len(cleaned_positions)))
    k = min_pts - 1

    spatial_k_distances = calculate_spatial_k_distances(
        cleaned_positions,
        k=k,
    )

    temporal_k_distances = calculate_temporal_k_distances(
        cleaned_positions,
        k=k,
    )

    eps_space = find_knee(spatial_k_distances)
    eps_time = find_knee(temporal_k_distances)

    position_clusters, movements = st_dbscan(
        cleaned_positions,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    clustered_positions = cleaned_positions.join(
        position_clusters,
        on="position_id",
        how="inner",
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

    # 각 GPS point에서 k번째로 가까운 도로까지의 거리를 계산한다.
    road_k = 3

    road_k_distances = calculate_road_k_distances(
        projected_positions,
        edges,
        k=road_k,
    )

    # 상위 95%의 GPS point가 road_k개의 후보 도로를 확보할 수 있는
    # 거리를 search radius 후보로 사용한다.
    search_radius = road_k_distances.quantile(0.95)

    print(f"=== {road_k}-Nearest Road Distance ===")
    print(road_k_distances.describe())
    print(f"95% quantile: {search_radius:.2f} m")

    plt.hist(
        road_k_distances,
        bins=50,
    )

    plt.axvline(
        search_radius,
        linestyle="--",
        label=f"95% Quantile: {search_radius:.2f} m",
    )

    plt.xlabel(f"Distance to {road_k}-Nearest Road (m)")
    plt.ylabel("Frequency")
    plt.title(f"{road_k}-Nearest Road Distance Distribution - {time_range}")
    plt.legend()

    plt.show()