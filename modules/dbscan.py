import polars as pl

from modules.haversine import haversine_expr
from modules.primitives.decorators import measure_time
from modules.primitives.schema import RawPositionSchema, validate_schema_columns

UNASSIGNED = -1
NOISE = 0


def _retrieve_neighbors(
    df: pl.DataFrame,
    point_idx: int,
    eps_space: float,
    eps_time: float,
) -> list[int]:
    point = df.row(point_idx, named=True)

    neighbors = (
        df.with_row_index("_index")
        .with_columns(
            haversine_expr(
                pl.lit(point["latitude"]),
                pl.lit(point["longitude"]),
                pl.col("latitude"),
                pl.col("longitude"),
            ).alias("_spatial_distance"),
            (pl.col("timestamp") - pl.lit(point["timestamp"]))
            .abs()
            .dt.total_seconds()
            .alias("_temporal_distance"),
        )
        .filter(
            (pl.col("_spatial_distance") <= eps_space)
            & (pl.col("_temporal_distance") <= eps_time)
        )
        .get_column("_index")
        .to_list()
    )

    return neighbors


def _expand_cluster(
    df: pl.DataFrame,
    labels: list[int],
    neighbors: list[int],
    cluster_id: int,
    eps_space: float,
    eps_time: float,
    min_pts: int,
) -> None:
    queue = list(neighbors)

    for idx in neighbors:
        labels[idx] = cluster_id

    while queue:
        current_idx = queue.pop()

        current_neighbors = _retrieve_neighbors(
            df,
            current_idx,
            eps_space,
            eps_time,
        )

        if len(current_neighbors) < min_pts:
            continue

        for neighbor_idx in current_neighbors:
            if labels[neighbor_idx] == UNASSIGNED:
                labels[neighbor_idx] = cluster_id
                queue.append(neighbor_idx)

            elif labels[neighbor_idx] == NOISE:
                labels[neighbor_idx] = cluster_id


def _create_movements(
    clustered: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    clustered = clustered.with_columns(
        (pl.col("cluster_id") == NOISE).alias("_is_moving"),
    ).with_columns(
        (pl.col("_is_moving") & ~pl.col("_is_moving").shift(1).fill_null(False))
        .cast(pl.Int64)
        .cum_sum()
        .sub(1)
        .alias("_movement_id"),
        pl.col("position_id").shift(1).alias("_prev_position_id"),
        pl.col("position_id").shift(-1).alias("_next_position_id"),
    )

    position_clusters = clustered.select(
        "position_id",
        "cluster_id",
        pl.when(pl.col("_is_moving"))
        .then(pl.col("_movement_id"))
        .otherwise(None)
        .alias("movement_id"),
    )

    movements = (
        clustered.filter(pl.col("_is_moving"))
        .group_by("_movement_id", maintain_order=True)
        .agg(
            pl.col("position_id").first().alias("head_position_id"),
            pl.col("position_id").last().alias("tail_position_id"),
            pl.col("_prev_position_id").first().alias("prev_stay_position_id"),
            pl.col("_next_position_id").last().alias("next_stay_position_id"),
            pl.len().alias("point_count"),
        )
        .rename({"_movement_id": "movement_id"})
    )

    return position_clusters, movements


@measure_time
def st_dbscan(
    df: pl.DataFrame,
    eps_space: float,
    eps_time: float,
    min_pts: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    validate_schema_columns(df, RawPositionSchema)

    if eps_space <= 0:
        raise ValueError("eps_space must be greater than 0")

    if eps_time <= 0:
        raise ValueError("eps_time must be greater than 0")

    if min_pts < 1:
        raise ValueError("min_pts must be greater than or equal to 1")

    labels = [UNASSIGNED] * len(df)
    cluster_id = 0

    for point_idx in range(len(df)):
        if labels[point_idx] != UNASSIGNED:
            continue

        neighbors = _retrieve_neighbors(
            df,
            point_idx,
            eps_space,
            eps_time,
        )

        if len(neighbors) < min_pts:
            labels[point_idx] = NOISE
            continue

        cluster_id += 1

        _expand_cluster(
            df,
            labels,
            neighbors,
            cluster_id,
            eps_space,
            eps_time,
            min_pts,
        )

    clustered = df.with_columns(
        pl.Series(
            "cluster_id",
            labels,
        )
    )

    return _create_movements(clustered)
