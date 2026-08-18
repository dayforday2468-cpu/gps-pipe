import polars as pl

from modules.primitives.decorators import measure_time


EARTH_RADIUS_M = 6_371_000


def _validate_input(df: pl.DataFrame) -> None:
    required_columns = {"latitude", "longitude"}

    if not required_columns.issubset(df.columns):
        raise ValueError("latitude and longitude columns are required")

    if not df["latitude"].is_between(-90, 90).all():
        raise ValueError("latitude must be between -90 and 90")

    if not df["longitude"].is_between(-180, 180).all():
        raise ValueError("longitude must be between -180 and 180")


@measure_time
def haversine_distance(df: pl.DataFrame) -> pl.DataFrame:
    _validate_input(df)

    lat1 = pl.col("latitude").radians()
    lon1 = pl.col("longitude").radians()

    lat2 = lat1.shift(-1)
    lon2 = lon1.shift(-1)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        (delta_lat / 2).sin().pow(2)
        + lat1.cos()
        * lat2.cos()
        * (delta_lon / 2).sin().pow(2)
    )

    c = 2 * pl.arctan2(
        a.sqrt(),
        (1 - a).sqrt(),
    )

    return df.with_columns(
        (EARTH_RADIUS_M * c).alias("distance_to_next")
    )