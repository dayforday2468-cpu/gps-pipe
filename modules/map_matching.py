import geopandas as gpd
import numpy as np
import math
import networkx as nx
import polars as pl
from shapely.geometry import Point

from modules.primitives.config import MAX_CANDIDATES
from modules.primitives.decorators import measure_time
from modules.primitives.schema import (
    CandidatePositionSchema,
    ProjectedPositionSchema,
    validate_schema_columns,
)
from modules.projection import project_point_to_edge


def _find_candidate_positions(
    position: ProjectedPositionSchema,
    edges: gpd.GeoDataFrame,
    search_radius: float,
) -> list[CandidatePositionSchema]:
    point = Point(
        position.x,
        position.y,
    )

    candidate_indices = edges.sindex.query(
        point,
        predicate="dwithin",
        distance=search_radius,
    )

    candidates = []

    for (u, v, key), edge in edges.iloc[candidate_indices].iterrows():
        projection = project_point_to_edge(
            position.x,
            position.y,
            edge["geometry"],
        )

        if projection is None:
            continue

        projected_x, projected_y, distance, distance_along_edge = projection

        candidates.append(
            CandidatePositionSchema(
                position_id=position.position_id,
                edge_u=u,
                edge_v=v,
                edge_key=key,
                x=projected_x,
                y=projected_y,
                distance=distance,
                distance_along_edge=distance_along_edge,
            )
        )

    candidates.sort(key=lambda candidate: candidate.distance)

    return candidates[:MAX_CANDIDATES]


@measure_time
def generate_candidate_positions(
    positions: pl.DataFrame,
    edges: gpd.GeoDataFrame,
    search_radius: float,
) -> pl.DataFrame:
    validate_schema_columns(
        positions,
        ProjectedPositionSchema,
    )

    if search_radius < 0:
        raise ValueError("search_radius must be greater than or equal to 0")

    candidates = []

    for row in positions.iter_rows(named=True):
        position = ProjectedPositionSchema(**row)

        candidates.extend(
            _find_candidate_positions(
                position,
                edges,
                search_radius=search_radius,
            )
        )

    return pl.DataFrame([candidate.model_dump() for candidate in candidates])


def _calculate_shortest_road_distance(
    graph: nx.MultiGraph,
    candidate_a: CandidatePositionSchema,
    candidate_b: CandidatePositionSchema,
) -> float:
    same_edge = (
        candidate_a.edge_u == candidate_b.edge_u
        and candidate_a.edge_v == candidate_b.edge_v
        and candidate_a.edge_key == candidate_b.edge_key
    )

    if same_edge:
        return abs(candidate_b.distance_along_edge - candidate_a.distance_along_edge)

    edge_a = graph.edges[
        candidate_a.edge_u,
        candidate_a.edge_v,
        candidate_a.edge_key,
    ]
    edge_b = graph.edges[
        candidate_b.edge_u,
        candidate_b.edge_v,
        candidate_b.edge_key,
    ]

    edge_a_length = edge_a["geometry"].length
    edge_b_length = edge_b["geometry"].length

    distance_a_to_u = candidate_a.distance_along_edge
    distance_a_to_v = edge_a_length - candidate_a.distance_along_edge

    distance_u_to_b = candidate_b.distance_along_edge
    distance_v_to_b = edge_b_length - candidate_b.distance_along_edge

    endpoint_pairs = [
        (
            candidate_a.edge_u,
            candidate_b.edge_u,
            distance_a_to_u,
            distance_u_to_b,
        ),
        (
            candidate_a.edge_u,
            candidate_b.edge_v,
            distance_a_to_u,
            distance_v_to_b,
        ),
        (
            candidate_a.edge_v,
            candidate_b.edge_u,
            distance_a_to_v,
            distance_u_to_b,
        ),
        (
            candidate_a.edge_v,
            candidate_b.edge_v,
            distance_a_to_v,
            distance_v_to_b,
        ),
    ]

    shortest_distance = math.inf

    for source, target, source_distance, target_distance in endpoint_pairs:
        try:
            network_distance = nx.shortest_path_length(
                graph,
                source=source,
                target=target,
                weight="length",
            )
        except nx.NetworkXNoPath:
            continue

        total_distance = source_distance + network_distance + target_distance

        shortest_distance = min(
            shortest_distance,
            total_distance,
        )

    return shortest_distance


def calculate_emission_probabilities(
    candidates: pl.DataFrame,
    sigma_z: float,
) -> np.ndarray:
    validate_schema_columns(
        candidates,
        CandidatePositionSchema,
    )

    if sigma_z <= 0:
        raise ValueError("sigma_z must be greater than 0")

    candidate_models = [
        CandidatePositionSchema(**row) for row in candidates.iter_rows(named=True)
    ]

    emission_probabilities = np.empty(
        len(candidate_models),
        dtype=float,
    )

    for i, candidate in enumerate(candidate_models):
        emission_probabilities[i] = (
            1
            / (math.sqrt(2 * math.pi) * sigma_z)
            * math.exp(-0.5 * (candidate.distance / sigma_z) ** 2)
        )

    return emission_probabilities


def _calculate_transition_probability(
    graph: nx.MultiGraph,
    candidate_a: CandidatePositionSchema,
    candidate_b: CandidatePositionSchema,
    observed_distance: float,
    beta: float,
) -> float:
    if observed_distance < 0:
        raise ValueError("observed_distance must be greater than or equal to 0")

    route_distance = _calculate_shortest_road_distance(
        graph,
        candidate_a,
        candidate_b,
    )

    if math.isinf(route_distance):
        return 0.0

    distance_difference = abs(observed_distance - route_distance)

    return 1 / beta * math.exp(-distance_difference / beta)


def calculate_transition_matrix(
    graph: nx.MultiGraph,
    projected_position_a: ProjectedPositionSchema,
    projected_position_b: ProjectedPositionSchema,
    candidates_a: pl.DataFrame,
    candidates_b: pl.DataFrame,
    beta: float,
) -> np.ndarray:
    validate_schema_columns(
        candidates_a,
        CandidatePositionSchema,
    )

    validate_schema_columns(
        candidates_b,
        CandidatePositionSchema,
    )

    if beta <= 0:
        raise ValueError("beta must be greater than 0")

    observed_distance = math.hypot(
        projected_position_b.x - projected_position_a.x,
        projected_position_b.y - projected_position_a.y,
    )

    candidate_models_a = [
        CandidatePositionSchema(**row) for row in candidates_a.iter_rows(named=True)
    ]

    candidate_models_b = [
        CandidatePositionSchema(**row) for row in candidates_b.iter_rows(named=True)
    ]

    transition_matrix = np.empty(
        (
            len(candidate_models_a),
            len(candidate_models_b),
        ),
        dtype=float,
    )

    for i, candidate_a in enumerate(candidate_models_a):
        for j, candidate_b in enumerate(candidate_models_b):
            transition_matrix[i, j] = _calculate_transition_probability(
                graph,
                candidate_a,
                candidate_b,
                observed_distance,
                beta,
            )

    return transition_matrix


def _viterbi_forward(
    graph: nx.MultiGraph,
    projected_positions: list[ProjectedPositionSchema],
    candidate_groups: list[pl.DataFrame],
    sigma_z: float,
    beta: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    previous_scores = calculate_emission_probabilities(
        candidate_groups[0],
        sigma_z,
    )

    backpointers = []

    for index in range(1, len(candidate_groups)):
        transition_matrix = calculate_transition_matrix(
            graph,
            projected_positions[index - 1],
            projected_positions[index],
            candidate_groups[index - 1],
            candidate_groups[index],
            beta,
        )

        emission_probabilities = calculate_emission_probabilities(
            candidate_groups[index],
            sigma_z,
        )

        transition_scores = previous_scores[:, None] * transition_matrix

        best_previous = np.argmax(
            transition_scores,
            axis=0,
        )

        current_scores = (
            transition_scores[
                best_previous,
                np.arange(len(emission_probabilities)),
            ]
            * emission_probabilities
        )

        max_score = current_scores.max()

        if max_score > 0:
            current_scores /= max_score

        backpointers.append(best_previous)

        previous_scores = current_scores

    return previous_scores, backpointers


@measure_time
def viterbi_map_matching(
    graph: nx.MultiGraph,
    movements: pl.DataFrame,
    projected_positions: pl.DataFrame,
    candidate_positions: pl.DataFrame,
    sigma_z: float,
    beta: float,
) -> pl.DataFrame:
    validate_schema_columns(
        projected_positions,
        ProjectedPositionSchema,
    )

    validate_schema_columns(
        candidate_positions,
        CandidatePositionSchema,
    )

    if sigma_z <= 0:
        raise ValueError("sigma_z must be greater than 0")

    if beta <= 0:
        raise ValueError("beta must be greater than 0")

    position_ids = projected_positions.get_column("position_id").to_list()

    position_indices = {
        position_id: index for index, position_id in enumerate(position_ids)
    }

    projected_models = {
        row["position_id"]: ProjectedPositionSchema(**row)
        for row in projected_positions.iter_rows(named=True)
    }

    candidate_groups = {
        group.get_column("position_id")[0]: group
        for group in candidate_positions.partition_by(
            "position_id",
            maintain_order=True,
        )
    }

    matched_candidates = []

    for movement_row in movements.iter_rows(named=True):
        head_position_id = movement_row["head_position_id"]
        tail_position_id = movement_row["tail_position_id"]

        head_index = position_indices[head_position_id]
        tail_index = position_indices[tail_position_id]

        movement_position_ids = position_ids[head_index : tail_index + 1]

        movement_position_ids = [
            position_id
            for position_id in movement_position_ids
            if position_id in candidate_groups
        ]

        if not movement_position_ids:
            continue

        movement_projected_positions = [
            projected_models[position_id] for position_id in movement_position_ids
        ]

        movement_candidate_groups = [
            candidate_groups[position_id] for position_id in movement_position_ids
        ]

        final_scores, backpointers = _viterbi_forward(
            graph,
            movement_projected_positions,
            movement_candidate_groups,
            sigma_z,
            beta,
        )

        best_index = int(np.argmax(final_scores))

        selected_indices = [best_index]

        for backpointer in reversed(backpointers):
            best_index = int(backpointer[best_index])

            selected_indices.append(best_index)

        selected_indices.reverse()

        for candidate_group, candidate_index in zip(
            movement_candidate_groups,
            selected_indices,
        ):
            matched_candidates.append(
                candidate_group.row(
                    candidate_index,
                    named=True,
                )
            )

    return pl.DataFrame(matched_candidates)
