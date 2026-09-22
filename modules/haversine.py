import math
import polars as pl

from modules.primitives.config import EARTH_RADIUS
from modules.primitives.schema import GeographicPositionSchema, validate_schema_columns


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


def haversine_distance(df: pl.DataFrame) -> pl.DataFrame:
    validate_schema_columns(df, GeographicPositionSchema)

    return df.with_columns(
        haversine_expr(
            pl.col("latitude"),
            pl.col("longitude"),
            pl.col("latitude").shift(-1),
            pl.col("longitude").shift(-1),
        ).alias("distance_to_next")
    )
