from collections.abc import Callable

import geopandas as gpd
import polars as pl
from shapely.geometry import Point


from modules.haversine import haversine_expr, haversine_distance
from modules.primitives.config import JUMP_RATE
from modules.primitives.decorators import measure_time

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

        distance_df = df.with_columns(distance_expr(point).alias("distance"))

        k_distance = distance_df.get_column("distance").sort()[k]

        distances.append(k_distance)

    return pl.Series("k_distance", distances).sort(descending=True)


@measure_time
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


@measure_time
def calculate_temporal_k_distances(
    df: pl.DataFrame,
    k: int,
) -> pl.Series:
    def temporal_distance(point: dict) -> pl.Expr:
        return (
            (pl.col("timestamp") - pl.lit(point["timestamp"])).abs().dt.total_seconds()
        )

    return _calculate_k_distances(
        df,
        k,
        temporal_distance,
    )


@measure_time
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
            (pl.col("index") / (pl.col("index").max() - pl.col("index").min())).alias(
                "x"
            ),
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
        points.with_columns((pl.col("x") + pl.col("y")).alias("score"))
        .sort("score")
        .row(0, named=True)
    )

    return knee["value"]


@measure_time
def estimate_jump_threshold(df: pl.DataFrame) -> float:
    distances = (
        haversine_distance(df)
        .get_column("distance_to_next")
        .drop_nulls()
        .sort(descending=True)
    )

    return find_knee(distances)


@measure_time
def estimate_same_place_threshold(
    segments: pl.DataFrame,
) -> float:
    same_place_distances = segments.filter(
        pl.col("prev_next_distance").is_not_null()
    ).get_column("prev_next_distance")

    return same_place_distances.quantile(JUMP_RATE)


@measure_time
def calculate_road_k_distances(
    positions: pl.DataFrame,
    edges: gpd.GeoDataFrame,
    k: int,
) -> pl.Series:
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")

    if len(edges) < k:
        raise ValueError("number of edges must be greater than or equal to k")

    distances = []

    for position in positions.iter_rows(named=True):
        point = Point(
            position["x"],
            position["y"],
        )

        road_distances = edges.geometry.distance(point).sort_values()

        k_distance = road_distances.iloc[k - 1]

        distances.append(k_distance)

    return pl.Series(
        f"{k}_nearest_road_distance",
        distances,
    ).sort(descending=True)
