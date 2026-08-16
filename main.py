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
from modules.visualization import GPSVisualizer

logger = get_logger(__name__)


def init_data_directory() -> None:
    data_dir = Path(DATA_DIR)
    source_path = Path(SOURCE_PATH).resolve()

    for path in data_dir.iterdir():
        if path.resolve() == source_path:
            continue

        if path.is_file():
            path.unlink()

@dataclass
class DataBatches:
    @property
    def raw_positions(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(
            f"{DATA_DIR}/raw_positions.csv",
            RawPositionSchema,
        )

    @property
    def timeline_paths(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(
            f"{DATA_DIR}/timeline_paths.csv",
            TimelinePathSchema,
        )

    @property
    def visits(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(
            f"{DATA_DIR}/visits.csv",
            VisitSchema,
        )

    @property
    def activities(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(
            f"{DATA_DIR}/activities.csv",
            ActivitySchema,
        )


def extract_and_save() -> None:
    save_batches(
        load_raw_positions_batches(SOURCE_PATH),
        f"{DATA_DIR}/raw_positions.csv",
        RawPositionSchema,
    )
    logger.debug("raw positions saved")

    save_batches(
        load_timeline_paths_batches(SOURCE_PATH),
        f"{DATA_DIR}/timeline_paths.csv",
        TimelinePathSchema,
    )
    logger.debug("timeline paths saved")

    save_batches(
        load_visits_batches(SOURCE_PATH),
        f"{DATA_DIR}/visits.csv",
        VisitSchema,
    )
    logger.debug("visits saved")

    save_batches(
        load_activities_batches(SOURCE_PATH),
        f"{DATA_DIR}/activities.csv",
        ActivitySchema,
    )
    logger.debug("activities saved")

    logger.info("all data batches saved")

if __name__ == "__main__":
    # 프로젝트 초기화
    init_data_directory()
    logger.info("data directory initialized")

    # 원본 데이터를 표준 스키마의 CSV로 추출 및 저장
    extract_and_save()

    # 저장된 CSV를 배치 제너레이터로 다시 로딩
    batches = DataBatches()

        # 하루치 raw position 필터링
    start = datetime(2026, 8, 10, 0, 0)
    end = datetime(2026, 8, 11, 0, 0)

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    # 하루치 raw position 시각화
    visualizer = GPSVisualizer(
        title="Raw GPS Positions - 2026-08-10",
        show_legend=True,
    )

    visualizer.add_batches(
            raw_filtered,
            label="Raw GPS",
            point_size=5,
            point_color="black",
            show_line=False,
        )

    visualizer.show()