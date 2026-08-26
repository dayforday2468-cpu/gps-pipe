import math
from datetime import datetime

import osmnx as ox
import polars as pl

from modules.dbscan import st_dbscan
from modules.map_matching import generate_candidate_positions
from modules.parameter_tuning import (
    calculate_road_k_distances,
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    estimate_jump_threshold,
    estimate_same_place_threshold,
    find_knee,
)
from modules.primitives.config import (
    PROCESSED_DIR,
    ROAD_NETWORK_VIEW_MARGIN,
)
from modules.primitives.datafilter import filter_points
from modules.primitives.datastore import save_dataframe
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.schema import (
    CandidatePositionSchema,
    MovementSchema,
    PositionClusterSchema,
    PositionSegmentSchema,
    ProjectedPositionSchema,
    RawPositionSchema,
    SegmentSchema,
)
from modules.projection import project_positions
from modules.road_network import load_road_network
from modules.segmentation import segment_positions
from modules.sudden_position_jump import remove_sudden_position_jumps

if __name__ == "__main__":
    batches = initialize_pipeline()

    # 하루치 데이터 필터링
    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)

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

    save_dataframe(
        position_segments.select(list(PositionSegmentSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/position_segments.csv",
        PositionSegmentSchema,
    )

    save_dataframe(
        segments.select(list(SegmentSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/segments.csv",
        SegmentSchema,
    )

    same_place_thres = estimate_same_place_threshold(
        segments,
    )

    # Sudden Position Jump를 제거하여 GPS 데이터를 정제한다.
    cleaned_data = remove_sudden_position_jumps(
        raw_positions,
        position_segments,
        segments,
        same_place_thres=same_place_thres,
    )

    save_dataframe(
        cleaned_data.select(list(RawPositionSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/cleaned_positions.csv",
        RawPositionSchema,
    )

    # ST-DBSCAN 파라미터를 추정하고 이동 및 체류 클러스터를 생성한다.
    min_pts = math.ceil(math.log(len(cleaned_data)))
    k = min_pts - 1

    spatial_k_distances = calculate_spatial_k_distances(
        cleaned_data,
        k=k,
    )

    temporal_k_distances = calculate_temporal_k_distances(
        cleaned_data,
        k=k,
    )

    eps_space = find_knee(spatial_k_distances)
    eps_time = find_knee(temporal_k_distances)

    position_clusters, movements = st_dbscan(
        cleaned_data,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    save_dataframe(
        position_clusters.select(list(PositionClusterSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/position_clusters.csv",
        PositionClusterSchema,
    )

    save_dataframe(
        movements.select(list(MovementSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/movements.csv",
        MovementSchema,
    )

    clustered_data = cleaned_data.join(
        position_clusters,
        on="position_id",
        how="inner",
    )

    # ST-DBSCAN에서 이동으로 분류된 GPS point를 선택한다.
    moving_positions = clustered_data.filter(pl.col("cluster_id") == 0)

    # Map Matching을 위한 도로망을 불러오고 평면 좌표계로 변환한다.
    road_network = load_road_network(
        moving_positions,
        margin=ROAD_NETWORK_VIEW_MARGIN,
    )

    projected_road_network = ox.project_graph(
        road_network,
    )

    # GPS 좌표를 도로망과 동일한 좌표계의 x, y 좌표로 변환한다.
    projected_positions = project_positions(
        moving_positions,
        projected_road_network.graph["crs"],
    )

    save_dataframe(
        projected_positions.select(list(ProjectedPositionSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/projected_positions.csv",
        ProjectedPositionSchema,
    )

    # 도로 edge를 GeoDataFrame으로 변환한다.
    edges = ox.graph_to_gdfs(
        projected_road_network,
        nodes=False,
        edges=True,
    )

    # Map Matching 후보 탐색을 위한 search radius를 추정한다.
    road_k = 3

    road_k_distances = calculate_road_k_distances(
        projected_positions,
        edges,
        k=road_k,
    )

    search_radius = road_k_distances.quantile(0.95)

    # Search radius 내의 도로에 projection하여 후보 위치를 생성한다.
    candidate_positions = generate_candidate_positions(
        projected_positions,
        edges,
        search_radius=search_radius,
    )

    save_dataframe(
        candidate_positions.select(list(CandidatePositionSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/candidate_positions.csv",
        CandidatePositionSchema,
    )
