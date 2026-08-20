from datetime import datetime

from modules.primitives.config import DATA_DIR, PROCESSED_DIR
from modules.primitives.datafilter import filter_points
from modules.primitives.datastore import save_dataframe
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.schema import RawPositionSchema
from modules.sudden_position_jump import remove_sudden_position_jumps

if __name__ == "__main__":
    batches = initialize_pipeline()

    # 하루치 데이터 필터링
    start = datetime(2026, 8, 11, 0, 0)
    end = datetime(2026, 8, 12, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    raw_positions = filter_points(
        batches.raw_positions,
        start,
        end,
    )

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
