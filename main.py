from modules.dataload import *
from modules.schema import *
from modules.logger import get_logger

logger = get_logger(__name__)
data_path = "data/timeline.json"

def validate_batches(batches, schema):
    count = 0

    for df in batches:
        for row in df.iter_rows(named=True):
            schema.model_validate(row)
            count += 1

    print(f"{schema.__name__}: {count} rows validated")

if __name__=="__main__":
    # 데이터 배치 로더 생성
    raw_positions_batches = load_raw_positions_batches(data_path)
    logger.info("raw position loader initialized")

    timeline_paths_batches = load_timeline_paths_batches(data_path)
    logger.info("timeline path loader initialized")

    visits_batches = load_visits_batches(data_path)
    logger.info("visit loader initialized")

    activities_batches = load_activities_batches(data_path)
    logger.info("activity loader initialized")

    # 로드된 데이터의 스키마 검증
    validate_batches(
        raw_positions_batches,
        RawPositionSchema,
    )

    validate_batches(
        timeline_paths_batches,
        TimelinePathSchema,
    )

    validate_batches(
        visits_batches,
        VisitSchema,
    )

    validate_batches(
        activities_batches,
        ActivitySchema,
    )