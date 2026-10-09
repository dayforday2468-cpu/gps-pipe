import polars as pl

from modules.haversine import haversine_expr
from modules.primitives.decorators import measure_time
from modules.primitives.schema import (
    RawPositionSchema,
    PositionSegmentSchema,
    PositionState,
    validate_schema_columns,
)

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


def _create_position_segments(
    clustered: pl.DataFrame,
) -> pl.DataFrame:
    clustered = clustered.with_columns(
        (pl.col("cluster_id") == NOISE).alias("_is_moving")
    )

    clustered = clustered.with_columns(
        (pl.col("_is_moving") & ~pl.col("_is_moving").shift(1).fill_null(False))
        .cast(pl.Int64)
        .cum_sum()
        .alias("_movement_id")
    )

    return clustered.select(
        "position_id",
        pl.when(pl.col("_is_moving"))
        .then(pl.col("_movement_id"))
        .otherwise(pl.col("cluster_id"))
        .alias("segment_id"),
        pl.when(pl.col("_is_moving"))
        .then(pl.lit(PositionState.MOVEMENT.value))
        .otherwise(pl.lit(PositionState.STAY.value))
        .alias("state"),
    )


def _mark_jump_noise_removed(
    position_segments: pl.DataFrame,
) -> pl.DataFrame:
    validate_schema_columns(
        position_segments,
        PositionSegmentSchema,
    )

    position_segments = position_segments.with_columns(
        pl.when(pl.col("state") == PositionState.STAY.value)
        .then(pl.col("segment_id"))
        .otherwise(None)
        .forward_fill()
        .alias("_prev_stay_id"),
        pl.when(pl.col("state") == PositionState.STAY.value)
        .then(pl.col("segment_id"))
        .otherwise(None)
        .backward_fill()
        .alias("_next_stay_id"),
    )

    is_jump_noise = (
        (pl.col("state") == PositionState.MOVEMENT.value)
        & pl.col("_prev_stay_id").is_not_null()
        & (pl.col("_prev_stay_id") == pl.col("_next_stay_id"))
    )

    return position_segments.with_columns(
        pl.when(is_jump_noise)
        .then(0)
        .otherwise(pl.col("segment_id"))
        .alias("segment_id"),
        pl.when(is_jump_noise)
        .then(pl.lit(PositionState.REMOVED.value))
        .otherwise(pl.col("state"))
        .alias("state"),
    ).select(
        "position_id",
        "segment_id",
        "state",
    )


def _create_segment_summaries(
    position_segments: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    validate_schema_columns(
        position_segments,
        PositionSegmentSchema,
    )

    clusters = (
        position_segments.filter(pl.col("state") == PositionState.STAY.value)
        .group_by("segment_id", maintain_order=True)
        .agg(
            pl.col("position_id").first().alias("head_position_id"),
            pl.col("position_id").last().alias("tail_position_id"),
            pl.len().alias("point_count"),
        )
        .rename({"segment_id": "cluster_id"})
    )

    movements = (
        position_segments.filter(pl.col("state") == PositionState.MOVEMENT.value)
        .group_by("segment_id", maintain_order=True)
        .agg(
            pl.col("position_id").first().alias("head_position_id"),
            pl.col("position_id").last().alias("tail_position_id"),
            pl.len().alias("point_count"),
        )
        .rename({"segment_id": "movement_id"})
    )

    return clusters, movements


@measure_time
def st_dbscan(
    df: pl.DataFrame,
    eps_space: float,
    eps_time: float,
    min_pts: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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

    clustered = df.with_columns(pl.Series("cluster_id", labels))

    position_segments = _create_position_segments(
        clustered,
    )

    position_segments = _mark_jump_noise_removed(
        position_segments,
    )

    clusters, movements = _create_segment_summaries(
        position_segments,
    )

    return position_segments, clusters, movements
