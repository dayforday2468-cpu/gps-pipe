from datetime import datetime

import math
import osmnx as ox
import polars as pl

from modules.dbscan import st_dbscan
from modules.interpolation import (
    build_corrected_positions,
    interpolate_matched_paths,
)
from modules.map_matching import (
    generate_candidate_positions,
    viterbi_map_matching,
)
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
from modules.primitives.visualization import GPSVisualizer
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

    # Sudden Position Jump 제거
    jump_thres = estimate_jump_threshold(
        raw_positions,
    )

    position_segments, segments = segment_positions(
        raw_positions,
        jump_thres=jump_thres,
    )

    same_place_thres = estimate_same_place_threshold(
        segments,
    )

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

    # ST-DBSCAN
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

    eps_space = find_knee(
        spatial_k_distances,
    )

    eps_time = find_knee(
        temporal_k_distances,
    )

    position_clusters, movements = st_dbscan(
        cleaned_positions,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    # 도로망 로드
    road_network = load_road_network(
        cleaned_positions,
        margin=ROAD_NETWORK_VIEW_MARGIN,
    )

    # Map Matching에서는 평면 좌표계 + 무방향 그래프를 사용한다.
    projected_road_network = ox.convert.to_undirected(
        ox.project_graph(
            road_network,
        )
    )

    # 전체 cleaned position을 projection한 뒤 cluster 정보를 결합한다.
    projected_all_positions = project_positions(
        cleaned_positions,
        projected_road_network.graph["crs"],
    ).join(
        position_clusters,
        on="position_id",
        how="left",
    )

    # movement point만 Map Matching에 사용한다.
    projected_moving_positions = (
        projected_all_positions.filter(pl.col("cluster_id") == 0)
        .select(
            "position_id",
            "x",
            "y",
        )
        .sort("position_id")
    )

    edges = ox.graph_to_gdfs(
        projected_road_network,
        nodes=False,
        edges=True,
    )

    # Candidate search radius 추정
    road_k_distances = calculate_road_k_distances(
        projected_moving_positions,
        edges,
        k=3,
    )

    search_radius = road_k_distances.quantile(0.95)

    candidate_positions = generate_candidate_positions(
        projected_moving_positions,
        edges,
        search_radius=search_radius,
    )

    # Viterbi Map Matching
    matched_positions = viterbi_map_matching(
        projected_road_network,
        movements,
        projected_moving_positions,
        candidate_positions,
        sigma_z=20.0,
        beta=50.0,
    ).sort("position_id")

    # Matched position 사이의 공간 및 시간 경로를 보간한다.
    matched_path_points = interpolate_matched_paths(
        projected_road_network,
        movements,
        matched_positions,
        cleaned_positions,
    )

    # Map Matching 및 경로 보간 결과를 최종 trajectory로 조립한다.
    corrected_positions = build_corrected_positions(
        cleaned_positions,
        matched_positions,
        matched_path_points,
        projected_road_network.graph["crs"],
    )

    print("=== Corrected Trajectory ===")
    print(f"Cleaned positions: {cleaned_positions.height}")
    print(f"Movements: {movements.height}")
    print(f"Matched positions: {matched_positions.height}")
    print(f"Matched path points: {matched_path_points.height}")
    print(f"Corrected positions: {corrected_positions.height}")

    # 시간 순서로 최종 trajectory를 시각화한다.
    visualizer = GPSVisualizer(
        title=f"Corrected GPS Trajectory - {time_range}",
        show_legend=True,
    )

    # corrected_positions는 위경도이므로 projection 전 도로망을 사용한다.
    visualizer.add_road_network(
        road_network,
    )

    visualizer.add(
        corrected_positions,
        label="Corrected Trajectory",
        point_size=6,
        point_color="red",
        show_line=True,
        line_color="black",
        line_width=1.2,
        alpha=0.85,
    )

    visualizer.animate(
        interval=100,
        repeat=False,
        mode="time",
    )
