import geopandas as gpd
import numpy as np
import math
import networkx as nx
import polars as pl
from shapely.geometry import Point

from modules.haversine import haversine
from modules.primitives.config import MAX_CANDIDATES
from modules.primitives.decorators import measure_time
from modules.primitives.schema import (
    CandidatePositionSchema,
    ProjectedPositionSchema,
    RawPositionSchema,
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


def calculate_shortest_road_distance(
    graph: nx.MultiDiGraph,
    candidate_a: CandidatePositionSchema,
    candidate_b: CandidatePositionSchema,
) -> float:
    same_edge = (
        candidate_a.edge_u == candidate_b.edge_u
        and candidate_a.edge_v == candidate_b.edge_v
        and candidate_a.edge_key == candidate_b.edge_key
    )

    # 같은 edge에서는 GPS 오차에 의한 후진을 허용한다.
    if same_edge:
        return abs(candidate_b.distance_along_edge - candidate_a.distance_along_edge)

    edge_a = graph.edges[
        candidate_a.edge_u,
        candidate_a.edge_v,
        candidate_a.edge_key,
    ]

    edge_a_length = edge_a["geometry"].length

    distance_a_to_v = edge_a_length - candidate_a.distance_along_edge

    distance_u_to_b = candidate_b.distance_along_edge

    try:
        network_distance = nx.shortest_path_length(
            graph,
            source=candidate_a.edge_v,
            target=candidate_b.edge_u,
            weight="length",
        )
    except nx.NetworkXNoPath:
        return math.inf

    return distance_a_to_v + network_distance + distance_u_to_b


def _calculate_emission_probability(
    candidate: CandidatePositionSchema,
    sigma_z: float,
) -> float:
    if sigma_z <= 0:
        raise ValueError("sigma_z must be greater than 0")

    return (
        1
        / (math.sqrt(2 * math.pi) * sigma_z)
        * math.exp(-0.5 * (candidate.distance / sigma_z) ** 2)
    )


def _calculate_transition_probability(
    graph: nx.MultiDiGraph,
    candidate_a: CandidatePositionSchema,
    candidate_b: CandidatePositionSchema,
    observed_distance: float,
    beta: float,
) -> float:
    if observed_distance < 0:
        raise ValueError("observed_distance must be greater than or equal to 0")

    route_distance = calculate_shortest_road_distance(
        graph,
        candidate_a,
        candidate_b,
    )

    if math.isinf(route_distance):
        return 0.0

    distance_difference = abs(observed_distance - route_distance)

    return 1 / beta * math.exp(-distance_difference / beta)


@measure_time
def calculate_transition_matrix(
    graph: nx.MultiDiGraph,
    raw_position_a: RawPositionSchema,
    raw_position_b: RawPositionSchema,
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

    observed_distance = haversine(
        raw_position_a.latitude,
        raw_position_a.longitude,
        raw_position_b.latitude,
        raw_position_b.longitude,
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
