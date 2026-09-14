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

        movement_matched_positions = (
            matched_positions.filter(
                pl.col("position_id").is_between(
                    movement.head_position_id,
                    movement.tail_position_id,
                )
            )
            .sort("position_id")
        )

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
            distance, path = find_shortest_road_path(
                graph,
                candidate_a,
                candidate_b,
            )

            if math.isinf(distance):
                continue

            for sequence, node_id in enumerate(path):
                node = graph.nodes[node_id]

                matched_path_points.append(
                    MatchedPathPointSchema(
                        start_position_id=candidate_a.position_id,
                        end_position_id=candidate_b.position_id,
                        sequence=sequence,
                        x=node["x"],
                        y=node["y"],
                    )
                )

    return pl.DataFrame(
        [
            point.model_dump()
            for point in matched_path_points
        ]
    )