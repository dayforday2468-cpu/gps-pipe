import numpy as np
import polars as pl

from modules.haversine import haversine_expr


def calculate_k_distances(
    df: pl.DataFrame,
    k: int,
) -> pl.Series:
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")

    if len(df) <= k:
        raise ValueError("number of points must be greater than k")

    distances = []

    for i in range(len(df)):
        point = df.row(i, named=True)

        distance_df = df.with_columns(
            haversine_expr(
                pl.lit(point["latitude"]),
                pl.lit(point["longitude"]),
                pl.col("latitude"),
                pl.col("longitude"),
            ).alias("distance")
        )

        k_distance = (
            distance_df
            .get_column("distance")
            .sort()[k]
        )

        distances.append(k_distance)

    return pl.Series("k_distance", distances).sort(descending=True)