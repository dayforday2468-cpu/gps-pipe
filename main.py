from datetime import datetime
import math

from modules.dbscan import st_dbscan
from modules.dbscan_tuning import (
    calculate_spatial_k_distances,
    calculate_temporal_k_distances,
    find_knee,
)
from modules.primitives.config import PROCESSED_DIR
from modules.primitives.datafilter import filter_points
from modules.primitives.datastore import save_dataframe
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.schema import ClusteredPositionSchema, RawPositionSchema
from modules.sudden_position_jump import remove_sudden_position_jumps

if __name__ == "__main__":
    batches = initialize_pipeline()

    # 하루치 데이터 필터링
    start = datetime(2026, 8, 11, 0, 0)
    end = datetime(2026, 8, 12, 0, 0)

    raw_positions = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    # Sudden Position Jump를 제거하여 GPS 데이터를 정제한다.
    cleaned_data = remove_sudden_position_jumps(
        raw_positions,
        jump_thres=300,
        same_place_thres=200,
        max_jump_points=3,
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

    clustered_data = st_dbscan(
        cleaned_data,
        eps_space=eps_space,
        eps_time=eps_time,
        min_pts=min_pts,
    )

    save_dataframe(
        clustered_data.select(list(ClusteredPositionSchema.model_fields.keys())),
        f"{PROCESSED_DIR}/clustered_positions.csv",
        ClusteredPositionSchema,
    )
