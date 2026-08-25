import math
import polars as pl

from modules.primitives.config import EARTH_RADIUS
from modules.primitives.decorators import measure_time


def _validate_input(df: pl.DataFrame) -> None:
    required_columns = {"latitude", "longitude"}

    if not required_columns.issubset(df.columns):
        raise ValueError("latitude and longitude columns are required")


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return EARTH_RADIUS * c


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
