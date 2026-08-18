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
from modules.haversine import haversine_distance
from modules.sudden_position_jump import remove_sudden_position_jumps
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

    # 하루치 raw position 필터링
    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    # 연속 GPS point 간 거리 계산
    raw_with_distance = haversine_distance(raw_filtered)

    # sudden position jump 제거
    cleaned_positions = remove_sudden_position_jumps(
        raw_with_distance,
        jump_thres=500,  # 거리 분포를 보고 조정
        same_place_thres=50,  # 앞/뒤 segment 평균 위치가 같은 장소인지 판단
        max_jump_points=3,  # 중간 segment 최대 point 수
    )

    # 제거된 point 추출
    removed_positions = raw_filtered.join(
        cleaned_positions.select("timestamp"),
        on="timestamp",
        how="anti",
    )

    print(f"before:  {raw_filtered.height}")
    print(f"after:   {cleaned_positions.height}")
    print(f"removed: {removed_positions.height}")

    # clean 결과 시각화
    visualizer = GPSVisualizer(
        title="Sudden Position Jump Removal - 2026-08-11",
        show_legend=True,
    )

    # 원본 trajectory - 배경처럼 연하게 표시
    visualizer.add(
        raw_filtered,
        label="Raw GPS",
        point_size=2,
        point_color="gray",
        show_line=True,
        line_color="gray",
        line_width=0.5,
        alpha=0.15,
    )

    # clean 후 trajectory
    visualizer.add(
        cleaned_positions,
        label="Cleaned GPS",
        point_size=7,
        point_color="blue",
        show_line=True,
        line_color="blue",
        line_width=1.2,
        alpha=0.8,
    )

    # 실제 제거된 jump point
    visualizer.add(
        removed_positions,
        label="Removed Jump",
        point_size=50,
        point_color="red",
        show_line=False,
        alpha=1.0,
    )

    # visualizer.animate(interval=50, repeat=True)
    visualizer.show()
