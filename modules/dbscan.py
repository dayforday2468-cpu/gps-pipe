import polars as pl

from modules.haversine import haversine_expr
from modules.primitives.decorators import measure_time


UNASSIGNED = -1
NOISE = 0


def _validate_input(
    df: pl.DataFrame,
    eps_space: float,
    eps_time: float,
    min_pts: int,
) -> None:
    required_columns = {"timestamp", "latitude", "longitude"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"required columns are missing: {missing_columns}")

    if not df["latitude"].is_between(-90, 90).all():
        raise ValueError("latitude must be between -90 and 90")

    if not df["longitude"].is_between(-180, 180).all():
        raise ValueError("longitude must be between -180 and 180")

    if eps_space <= 0:
        raise ValueError("eps_space must be greater than 0")

    if eps_time <= 0:
        raise ValueError("eps_time must be greater than 0")

    if min_pts < 1:
        raise ValueError("min_pts must be greater than or equal to 1")


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
            (
                pl.col("timestamp") - pl.lit(point["timestamp"])
            )
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


@measure_time
def st_dbscan(
    df: pl.DataFrame,
    eps_space: float,
    eps_time: float,
    min_pts: int,
) -> pl.DataFrame:
    _validate_input(
        df,
        eps_space,
        eps_time,
        min_pts,
    )

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

    return df.with_columns(
        pl.Series("cluster_id", labels)
    )