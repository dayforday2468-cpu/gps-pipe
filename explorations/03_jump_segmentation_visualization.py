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

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    raw_with_distance = haversine_distance(raw_filtered)

    jump_thres = 300

    # jump threshold를 기준으로 segment 분할
    segmented = raw_with_distance.with_columns(
        (
            pl.col("distance_to_next")
            .shift(1)
            .fill_null(0)
            .gt(jump_thres)
            .cast(pl.Int64)
            .cum_sum()
        ).alias("segment_id")
    )

    # segment별 평균 위치 계산
    segment_means = (
        segmented.group_by(
            "segment_id",
            maintain_order=True,
        )
        .agg(
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
            pl.len().alias("point_count"),
        )
        .sort("segment_id")
    )

    visualizer = GPSVisualizer(
        title="Sudden Position Jump Exploration",
        show_legend=True,
    )

    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
        "magenta",
    ]

    segment_ids = (
        segmented["segment_id"]
        .unique(maintain_order=True)
        .to_list()
    )

    print("\n=== Distance Between Previous and Next Segment Means ===")

    for i in range(1, len(segment_ids) - 1):
        prev_id = segment_ids[i - 1]
        current_id = segment_ids[i]
        next_id = segment_ids[i + 1]

        prev_mean = segment_means.filter(
            pl.col("segment_id") == prev_id
        )

        next_mean = segment_means.filter(
            pl.col("segment_id") == next_id
        )

        mean_pair = pl.concat(
            [
                prev_mean.select("latitude", "longitude"),
                next_mean.select("latitude", "longitude"),
            ]
        )

        distance = haversine_distance(mean_pair)["distance_to_next"][0]

        current_point_count = (
            segment_means
            .filter(pl.col("segment_id") == current_id)
            ["point_count"][0]
        )

        print(
            f"segment {current_id}: "
            f"prev={prev_id}, next={next_id}, "
            f"point_count={current_point_count}, "
            f"prev-next distance={distance:.2f} m"
        )

    # 평균 위치는 timestamp가 없으므로
    # time mode에서도 처음부터 모두 표시됨
    for i, segment_id in enumerate(segment_ids):
        color = colors[i % len(colors)]

        mean_df = segment_means.filter(
            pl.col("segment_id") == segment_id
        )

        visualizer.add(
            mean_df,
            label="Segment mean" if i == 0 else None,
            point_size=120,
            point_color=color,
            show_line=False,
            alpha=1.0,
        )

    # segment와 segment 사이 연결선 추가
    for i, segment_id in enumerate(segment_ids):
        color = colors[i % len(colors)]

        segment_df = segmented.filter(
            pl.col("segment_id") == segment_id
        )

        # 현재 segment
        visualizer.add(
            segment_df.select(
                "latitude",
                "longitude",
                "timestamp",
            ),
            label=f"Segment {segment_id}",
            point_size=14,
            point_color=color,
            show_line=True,
            line_color=color,
            line_style="-",
            line_width=1.5,
            alpha=0.75,
        )

        # 다음 segment가 있으면 둘 사이 연결선 추가
        if i < len(segment_ids) - 1:
            next_segment_id = segment_ids[i + 1]

            next_segment_df = segmented.filter(
                pl.col("segment_id") == next_segment_id
            )

            connector = pl.concat(
                [
                    segment_df.tail(1).select(
                        "latitude",
                        "longitude",
                        "timestamp",
                    ),
                    next_segment_df.head(1).select(
                        "latitude",
                        "longitude",
                        "timestamp",
                    ),
                ]
            )

            visualizer.add(
                connector,
                label=None,
                point_size=0,
                point_color=color,
                show_line=True,
                line_color=color,
                line_style="--",
                line_width=1.2,
                alpha=0.8,
            )

    visualizer.animate(
        interval=50,
        repeat=False,
        mode="time",
    )
    # visualizer.show()
