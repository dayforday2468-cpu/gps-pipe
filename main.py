from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import polars as pl

from modules.config import DATA_DIR, SOURCE_PATH
from modules.dataload import *
from modules.datastore import *
from modules.datafilter import *
from modules.logger import get_logger
from modules.schema import *
from modules.datafilter import filter_points, filter_intervals

logger = get_logger(__name__)


def init_data_directory() -> None:
    data_dir = Path(DATA_DIR)
    source_path = Path(SOURCE_PATH).resolve()

    for path in data_dir.iterdir():
        if path.resolve() == source_path:
            continue

        if path.is_file():
            path.unlink()


DATASETS = {
    "raw_positions": {
        "loader": load_raw_positions_batches,
        "schema": RawPositionSchema,
    },
    "timeline_paths": {
        "loader": load_timeline_paths_batches,
        "schema": TimelinePathSchema,
    },
    "visits": {
        "loader": load_visits_batches,
        "schema": VisitSchema,
    },
    "activities": {
        "loader": load_activities_batches,
        "schema": ActivitySchema,
    },
}


@dataclass
class DataBatches:
    raw_positions: Iterator[pl.DataFrame]
    timeline_paths: Iterator[pl.DataFrame]
    visits: Iterator[pl.DataFrame]
    activities: Iterator[pl.DataFrame]


def extract_and_save() -> None:
    for name, config in DATASETS.items():
        batches = config["loader"](SOURCE_PATH)

        save_batches(
            batches,
            f"{DATA_DIR}/{name}.csv",
            config["schema"],
        )

        logger.debug("%s saved", name)

    logger.info("all data batches saved")


def load_saved_batches() -> DataBatches:
    loaded = {}

    for name, config in DATASETS.items():
        loaded[name] = load_csv_batches(
            f"{DATA_DIR}/{name}.csv",
            config["schema"],
        )

        logger.debug("%s loader initialized", name)

    logger.info("all data batch loaders initialized")

    return DataBatches(**loaded)


if __name__ == "__main__":
    # 프로젝트 초기화
    init_data_directory()
    logger.info("data directory initialized")

    # 원본 데이터를 표준 스키마의 CSV로 추출 및 저장
    extract_and_save()

    # 저장된 CSV를 배치 제너레이터로 다시 로딩
    batches = load_saved_batches()

    # 시간 필터 테스트
    start = datetime(2026, 8, 10, 0, 0)
    end = datetime(2026, 8, 11, 0, 0)

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    timeline_filtered = filter_points(
        batches.timeline_paths,
        start,
        end,
    )

    visits_filtered = filter_intervals(
        batches.visits,
        start,
        end,
    )

    activities_filtered = filter_intervals(
        batches.activities,
        start,
        end,
    )

    for df in raw_filtered:
        print(df)

    for df in timeline_filtered:
        print(df)

    for df in visits_filtered:
        print(df)

    for df in activities_filtered:
        print(df)