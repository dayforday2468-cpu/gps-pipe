from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

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
from modules.haversine import haversine_distance
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

    start = datetime(2026, 8, 10, 0, 0)
    end = datetime(2026, 8, 11, 0, 0)

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    raw_with_distance = haversine_distance(raw_filtered)

    distances = (
        raw_with_distance
        .select("distance_to_next")
        .drop_nulls()
    )

    print("=== Distance Summary ===")
    print(distances.describe())

    print("\n=== Largest Distances ===")
    print(
        raw_with_distance
        .select(
            "timestamp",
            "latitude",
            "longitude",
            "distance_to_next",
        )
        .drop_nulls()
        .sort("distance_to_next", descending=True)
        .head(20)
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        distances["distance_to_next"].to_numpy(),
        bins=50,
    )

    plt.title("Distribution of Distance Between Consecutive GPS Points")
    plt.xlabel("Distance to Next Point (m)")
    plt.ylabel("Count")

    plt.show()