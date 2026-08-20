from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import shutil

import polars as pl

from modules.primitives.config import DATA_DIR, SOURCE_PATH
from modules.primitives.dataload import (
    load_activities_batches,
    load_raw_positions_batches,
    load_timeline_paths_batches,
    load_visits_batches,
)
from modules.primitives.datastore import load_csv_batches, save_batches
from modules.primitives.logger import get_logger
from modules.primitives.schema import (
    ActivitySchema,
    RawPositionSchema,
    TimelinePathSchema,
    VisitSchema,
)

logger = get_logger(__name__)


def init_data_directory() -> None:
    data_dir = Path(DATA_DIR)
    source_path = Path(SOURCE_PATH).resolve()

    for path in data_dir.iterdir():
        if path.resolve() == source_path:
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    logger.debug("initialize data directory")


@dataclass
class DataBatches:
    @property
    def raw_positions(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(f"{DATA_DIR}/raw_positions.csv", RawPositionSchema)

    @property
    def timeline_paths(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(f"{DATA_DIR}/timeline_paths.csv", TimelinePathSchema)

    @property
    def visits(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(f"{DATA_DIR}/visits.csv", VisitSchema)

    @property
    def activities(self) -> Iterator[pl.DataFrame]:
        return load_csv_batches(f"{DATA_DIR}/activities.csv", ActivitySchema)


def extract_and_save() -> None:
    save_batches(
        load_raw_positions_batches(SOURCE_PATH),
        f"{DATA_DIR}/raw_positions.csv",
        RawPositionSchema,
    )

    save_batches(
        load_timeline_paths_batches(SOURCE_PATH),
        f"{DATA_DIR}/timeline_paths.csv",
        TimelinePathSchema,
    )

    save_batches(
        load_visits_batches(SOURCE_PATH),
        f"{DATA_DIR}/visits.csv",
        VisitSchema,
    )

    save_batches(
        load_activities_batches(SOURCE_PATH),
        f"{DATA_DIR}/activities.csv",
        ActivitySchema,
    )


def initialize_pipeline() -> DataBatches:
    init_data_directory()
    extract_and_save()
    return DataBatches()
