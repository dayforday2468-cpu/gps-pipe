import geopandas as gpd
import polars as pl
from shapely.geometry import Point

from modules.projection import project_point_to_edge


def find_candidate_edges(
    x: float,
    y: float,
    edges: gpd.GeoDataFrame,
    search_radius: float,
) -> gpd.GeoDataFrame:
    if search_radius < 0:
        raise ValueError("search_radius must be greater than or equal to 0")

    point = Point(x, y)

    candidate_indices = edges.sindex.query(
        point,
        predicate="dwithin",
        distance=search_radius,
    )

    return edges.iloc[candidate_indices]


def generate_candidate_positions(
    positions: pl.DataFrame,
    edges: gpd.GeoDataFrame,
    search_radius: float,
) -> pl.DataFrame:
    candidates = []

    for position in positions.iter_rows(named=True):
        candidate_edges = find_candidate_edges(
            position["x"],
            position["y"],
            edges,
            search_radius=search_radius,
        )

        for (u, v, key), edge in candidate_edges.iterrows():
            projection = project_point_to_edge(
                position["x"],
                position["y"],
                edge["geometry"],
            )

            if projection is None:
                continue

            projected_x, projected_y, distance = projection

            candidates.append(
                {
                    "position_id": position["position_id"],
                    "edge_u": u,
                    "edge_v": v,
                    "edge_key": key,
                    "x": projected_x,
                    "y": projected_y,
                    "distance": distance,
                }
            )

    return pl.DataFrame(candidates)
