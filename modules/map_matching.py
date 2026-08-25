import geopandas as gpd

from shapely.geometry import Point


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
