from pathlib import Path

from modules.dataload import *
from modules.schema import *
from modules.datastore import *
from modules.logger import get_logger


def init_data_directory(
    data_dir: str = "data",
    preserve: str = "timeline.json",
) -> None:
    data_path = Path(data_dir)

    for path in data_path.iterdir():
        if path.name == preserve:
            continue

        if path.is_file():
            path.unlink()


logger = get_logger(__name__)
data_path = "data/timeline.json"

if __name__ == "__main__":
    # 프로젝트 초기화
    init_data_directory()
    logger.info("data directory initialized")

    # 데이터 배치 로더 생성
    raw_positions_batches = load_raw_positions_batches(data_path)
    logger.debug("raw position loader initialized")

    timeline_paths_batches = load_timeline_paths_batches(data_path)
    logger.debug("timeline path loader initialized")

    visits_batches = load_visits_batches(data_path)
    logger.debug("visit loader initialized")

    activities_batches = load_activities_batches(data_path)
    logger.debug("activity loader initialized")

    # 배치 데이터 스키마 검증 및 저장
    save_batches(
        raw_positions_batches,
        "data/raw_positions.csv",
        RawPositionSchema,
    )
    logger.debug("raw positions saved")

    save_batches(
        timeline_paths_batches,
        "data/timeline_paths.csv",
        TimelinePathSchema,
    )
    logger.debug("timeline paths saved")

    save_batches(
        visits_batches,
        "data/visits.csv",
        VisitSchema,
    )
    logger.debug("visits saved")

    save_batches(
        activities_batches,
        "data/activities.csv",
        ActivitySchema,
    )
    logger.debug("activities saved")

    logger.info("all data batches saved")

    # 배치 데이터 스키마 검증 및 로딩
    raw_positions_batches = load_csv_batches(
        "data/raw_positions.csv",
        RawPositionSchema,
    )
    logger.debug("raw positions loaded")

    timeline_paths_batches = load_csv_batches(
        "data/timeline_paths.csv",
        TimelinePathSchema,
    )
    logger.debug("timeline paths loaded")

    visits_batches = load_csv_batches(
        "data/visits.csv",
        VisitSchema,
    )
    logger.debug("visits loaded")

    activities_batches = load_csv_batches(
        "data/activities.csv",
        ActivitySchema,
    )
    logger.debug("activities loaded")

    logger.info("all data batches loaded")