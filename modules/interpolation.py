import math

import networkx as nx
import polars as pl

from modules.primitives.decorators import measure_time
from modules.primitives.schema import (
    CandidatePositionSchema,
    MatchedPathPointSchema,
    MovementSchema,
    validate_schema_columns,
)
from modules.road_network import find_shortest_road_path


def _calculate_distance_to_node(
    graph: nx.MultiGraph,
    candidate: CandidatePositionSchema,
    node_id: int,
) -> float:
    edge = graph.edges[
        candidate.edge_u,
        candidate.edge_v,
        candidate.edge_key,
    ]

    edge_length = edge["geometry"].length

    if node_id == candidate.edge_u:
        return candidate.distance_along_edge

    if node_id == candidate.edge_v:
        return edge_length - candidate.distance_along_edge

    raise ValueError(f"node {node_id} is not an endpoint of candidate edge")


def _get_edge_length(
    graph: nx.MultiGraph,
    u: int,
    v: int,
) -> float:
    return min(edge["length"] for edge in graph[u][v].values())


@measure_time
def interpolate_matched_paths(
    graph: nx.MultiGraph,
    movements: pl.DataFrame,
    matched_positions: pl.DataFrame,
) -> pl.DataFrame:
    validate_schema_columns(
        movements,
        MovementSchema,
    )

    validate_schema_columns(
        matched_positions,
        CandidatePositionSchema,
    )

    matched_path_points = []

    for movement_row in movements.iter_rows(named=True):
        movement = MovementSchema(**movement_row)

        movement_matched_positions = matched_positions.filter(
            pl.col("position_id").is_between(
                movement.head_position_id,
                movement.tail_position_id,
            )
        ).sort("position_id")

        if movement_matched_positions.height < 2:
            continue

        movement_candidates = [
            CandidatePositionSchema(**row)
            for row in movement_matched_positions.iter_rows(named=True)
        ]

        for candidate_a, candidate_b in zip(
            movement_candidates[:-1],
            movement_candidates[1:],
        ):
            total_distance, path = find_shortest_road_path(
                graph,
                candidate_a,
                candidate_b,
            )

            if math.isinf(total_distance):
                continue

            if not path:
                continue

            cumulative_distance = _calculate_distance_to_node(
                graph,
                candidate_a,
                path[0],
            )

            for sequence, node_id in enumerate(path):
                node = graph.nodes[node_id]

                if total_distance > 0:
                    path_progress = cumulative_distance / total_distance
                else:
                    path_progress = 0.0

                matched_path_points.append(
                    MatchedPathPointSchema(
                        start_position_id=candidate_a.position_id,
                        end_position_id=candidate_b.position_id,
                        sequence=sequence,
                        x=node["x"],
                        y=node["y"],
                        path_progress=path_progress,
                    )
                )

                if sequence < len(path) - 1:
                    cumulative_distance += _get_edge_length(
                        graph,
                        path[sequence],
                        path[sequence + 1],
                    )

    return pl.DataFrame([point.model_dump() for point in matched_path_points])
