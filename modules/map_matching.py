import geopandas as gpd
import polars as pl
from shapely.geometry import Point

from modules.primitives.decorators import measure_time
from modules.projection import project_point_to_edge


def _find_candidate_positions(
    position_id: int,
    x: float,
    y: float,
    edges: gpd.GeoDataFrame,
    search_radius: float,
    max_candidates: int,
) -> list[dict]:
    if search_radius < 0:
        raise ValueError("search_radius must be greater than or equal to 0")

    if max_candidates < 1:
        raise ValueError("max_candidates must be greater than or equal to 1")

    point = Point(x, y)

    candidate_indices = edges.sindex.query(
        point,
        predicate="dwithin",
        distance=search_radius,
    )

    candidates = []

    for (u, v, key), edge in edges.iloc[candidate_indices].iterrows():
        projection = project_point_to_edge(
            x,
            y,
            edge["geometry"],
        )

        if projection is None:
            continue

        projected_x, projected_y, distance, distance_along_edge = projection

        candidates.append(
            {
                "position_id": position_id,
                "edge_u": u,
                "edge_v": v,
                "edge_key": key,
                "x": projected_x,
                "y": projected_y,
                "distance": distance,
                "distance_along_edge": distance_along_edge,
            }
        )

    candidates.sort(key=lambda candidate: candidate["distance"])

    return candidates[:max_candidates]


@measure_time
def generate_candidate_positions(
    positions: pl.DataFrame,
    edges: gpd.GeoDataFrame,
    search_radius: float,
    max_candidates: int,
) -> pl.DataFrame:
    candidates = []

    for position in positions.iter_rows(named=True):
        candidates.extend(
            _find_candidate_positions(
                position["position_id"],
                position["x"],
                position["y"],
                edges,
                search_radius=search_radius,
                max_candidates=max_candidates,
            )
        )

    return pl.DataFrame(candidates)
