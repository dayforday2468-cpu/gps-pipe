from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import polars as pl

from modules.primitives.config import DATA_DIR, SOURCE_PATH
from modules.primitives.dataload import (
    load_activities_batches,
    load_raw_positions_batches,
    load_timeline_paths_batches,
    load_visits_batches,
)
from modules.primitives.datafilter import filter_intervals, filter_points
from modules.primitives.datastore import load_csv_batches, save_batches
from modules.primitives.logger import get_logger
from modules.primitives.schema import (
    ActivitySchema,
    RawPositionSchema,
    TimelinePathSchema,
    VisitSchema,
)
from modules.primitives.visualization import GPSVisualizer

logger = get_logger(__name__)


def init_data_directory() -> None:
    data_dir = Path(DATA_DIR)
    source_path = Path(SOURCE_PATH).resolve()

    for path in data_dir.iterdir():
        if path.resolve() == source_path:
            continue

        if path.is_file():
            path.unlink()

    logger.debug("initialize data directory")


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


def initialize_pipeline() -> DataBatches:
    # 데이터 폴더 초기화
    init_data_directory()

    # 원본 데이터를 표준 스키마의 CSV로 추출 및 저장
    extract_and_save()

    # 저장된 CSV를 배치 제너레이터로 다시 로딩
    return DataBatches()


if __name__ == "__main__":
    batches = initialize_pipeline()

    # 하루치 데이터 필터링
    start = datetime(2026, 8, 11, 0, 0)
    end = datetime(2026, 8, 12, 0, 0)

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

    # activity 시작점을 시각화용 latitude/longitude로 변환
    activities_points = (
        df.select(
            pl.col("start_latitude").alias("latitude"),
            pl.col("start_longitude").alias("longitude"),
        )
        for df in activities_filtered
    )

    # GPS 데이터 비교 시각화
    visualizer = GPSVisualizer(
        title="GPS Data Comparison - 2026-08-11",
        show_legend=True,
    )

    visualizer.add_batches(
        raw_filtered,
        label="Raw GPS",
        point_size=5,
        point_color="black",
        show_line=True,
        line_color="black",
        line_width=0.8,
        alpha=0.5,
    )

    visualizer.add_batches(
        timeline_filtered,
        label="Timeline Path",
        point_size=7,
        point_color="blue",
        show_line=True,
        line_color="blue",
        line_width=1.5,
        alpha=0.8,
    )

    visualizer.add_batches(
        visits_filtered,
        label="Visit",
        point_size=40,
        point_color="red",
        show_line=False,
        alpha=0.9,
    )

    visualizer.add_batches(
        activities_points,
        label="Activity",
        point_size=30,
        point_color="green",
        show_line=False,
        alpha=0.9,
    )

    visualizer.show()
