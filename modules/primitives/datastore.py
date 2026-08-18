from collections.abc import Iterator
from pathlib import Path
import polars as pl
from pydantic import BaseModel

from modules.primitives.decorators import measure_time, measure_generator_time
from modules.primitives.config import BATCH_SIZE


def _validate_batch(df: pl.DataFrame, schema: type[BaseModel]) -> None:
    expected_columns = list(schema.model_fields.keys())

    if df.columns != expected_columns:
        raise ValueError(
            f"Schema column mismatch\n"
            f"expected: {expected_columns}\n"
            f"actual:   {df.columns}"
        )

    for row in df.iter_rows(named=True):
        schema.model_validate(row, strict=True)


@measure_time
def save_batches(
    batches: Iterator[pl.DataFrame],
    path: str,
    schema: type[BaseModel],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_batch = True

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        for df in batches:
            _validate_batch(df, schema)
            df.write_csv(f, include_header=first_batch)

            first_batch = False


@measure_time
def save_dataframe(
    df: pl.DataFrame,
    path: str,
    schema: type[BaseModel],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_batch(df, schema)
    df.write_csv(output_path)


@measure_generator_time
def load_csv_batches(
    path: str, schema: type[BaseModel], batch_size: int = BATCH_SIZE
) -> Iterator[pl.DataFrame]:
    batches = pl.scan_csv(
        path,
        try_parse_dates=True,
    ).collect_batches(
        chunk_size=batch_size,
    )

    for df in batches:
        _validate_batch(df, schema)
        yield df
