import polars as pl

from modules.primitives.config import EARTH_RADIUS
from modules.primitives.decorators import measure_time


def _validate_input(df: pl.DataFrame) -> None:
    required_columns = {"latitude", "longitude"}

    if not required_columns.issubset(df.columns):
        raise ValueError("latitude and longitude columns are required")

    if not df["latitude"].is_between(-90, 90).all():
        raise ValueError("latitude must be between -90 and 90")

    if not df["longitude"].is_between(-180, 180).all():
        raise ValueError("longitude must be between -180 and 180")


def haversine_expr(
    lat1: pl.Expr,
    lon1: pl.Expr,
    lat2: pl.Expr,
    lon2: pl.Expr,
) -> pl.Expr:
    lat1 = lat1.radians()
    lon1 = lon1.radians()
    lat2 = lat2.radians()
    lon2 = lon2.radians()

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (delta_lat / 2).sin().pow(2) + lat1.cos() * lat2.cos() * (
        delta_lon / 2
    ).sin().pow(2)

    c = 2 * pl.arctan2(
        a.sqrt(),
        (1 - a).sqrt(),
    )

    return EARTH_RADIUS * c


@measure_time
def haversine_distance(df: pl.DataFrame) -> pl.DataFrame:
    _validate_input(df)

    return df.with_columns(
        haversine_expr(
            pl.col("latitude"),
            pl.col("longitude"),
            pl.col("latitude").shift(-1),
            pl.col("longitude").shift(-1),
        ).alias("distance_to_next")
    )
