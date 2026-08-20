from collections.abc import Callable

import polars as pl

from modules.haversine import haversine_expr


DistanceExpr = Callable[[dict], pl.Expr]


def _calculate_k_distances(
    df: pl.DataFrame,
    k: int,
    distance_expr: DistanceExpr,
) -> pl.Series:
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")

    if len(df) <= k:
        raise ValueError("number of points must be greater than k")

    distances = []

    for i in range(len(df)):
        point = df.row(i, named=True)

        distance_df = df.with_columns(
            distance_expr(point).alias("distance")
        )

        k_distance = (
            distance_df
            .get_column("distance")
            .sort()[k]
        )

        distances.append(k_distance)

    return pl.Series("k_distance", distances).sort(descending=True)


def calculate_spatial_k_distances(
    df: pl.DataFrame,
    k: int,
) -> pl.Series:
    def spatial_distance(point: dict) -> pl.Expr:
        return haversine_expr(
            pl.lit(point["latitude"]),
            pl.lit(point["longitude"]),
            pl.col("latitude"),
            pl.col("longitude"),
        )

    return _calculate_k_distances(
        df,
        k,
        spatial_distance,
    )


def calculate_temporal_k_distances(
    df: pl.DataFrame,
    k: int,
) -> pl.Series:
    def temporal_distance(point: dict) -> pl.Expr:
        return (
            pl.col("timestamp") - pl.lit(point["timestamp"])
        ).abs().dt.total_seconds()

    return _calculate_k_distances(
        df,
        k,
        temporal_distance,
    )

def find_knee(
    values: pl.Series,
    normalize: bool = True,
) -> float:
    if values.is_empty():
        raise ValueError("values must not be empty")

    points = pl.DataFrame(
        {
            "index": pl.int_range(0, len(values), eager=True),
            "value": values,
        }
    )

    if normalize:
        points = points.with_columns(
            (
                pl.col("index")
                / (pl.col("index").max() - pl.col("index").min())
            ).alias("x"),
            (
                (pl.col("value") - pl.col("value").min())
                / (pl.col("value").max() - pl.col("value").min())
            ).alias("y"),
        )
    else:
        points = points.with_columns(
            pl.col("index").alias("x"),
            pl.col("value").alias("y"),
        )

    knee = (
        points
        .with_columns((pl.col("x") + pl.col("y")).alias("score"))
        .sort("score")
        .row(0, named=True)
    )

    return knee["value"]