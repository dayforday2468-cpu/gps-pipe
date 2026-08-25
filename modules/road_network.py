import json
import math
from pathlib import Path

import networkx as nx
import osmnx as ox
import polars as pl
from shapely.geometry import LineString

from modules.primitives.config import (
    EARTH_RADIUS,
    ROAD_NETWORK_CACHE_MARGIN,
    ROAD_NETWORK_DIR,
)
from modules.primitives.decorators import measure_time

GRAPH_PATH = Path(ROAD_NETWORK_DIR) / "road_network.graphml"
METADATA_PATH = Path(ROAD_NETWORK_DIR) / "metadata.json"


def calculate_bounds(
    positions: pl.DataFrame,
) -> tuple[float, float, float, float]:
    if positions.is_empty():
        raise ValueError("positions must not be empty")

    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(positions.columns)

    if missing_columns:
        raise ValueError(f"required columns are missing: {missing_columns}")

    west = positions.get_column("longitude").min()
    south = positions.get_column("latitude").min()
    east = positions.get_column("longitude").max()
    north = positions.get_column("latitude").max()

    return west, south, east, north


def expand_bounds(
    bounds: tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float]:
    west, south, east, north = bounds

    center_latitude = (south + north) / 2

    angular_margin = margin / EARTH_RADIUS
    latitude_margin = math.degrees(angular_margin)
    longitude_margin = math.degrees(
        angular_margin / math.cos(math.radians(center_latitude))
    )

    return (
        west - longitude_margin,
        south - latitude_margin,
        east + longitude_margin,
        north + latitude_margin,
    )


def contains_bounds(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    outer_west, outer_south, outer_east, outer_north = outer
    inner_west, inner_south, inner_east, inner_north = inner

    return (
        outer_west <= inner_west
        and outer_south <= inner_south
        and outer_east >= inner_east
        and outer_north >= inner_north
    )


def ensure_edge_geometries(
    graph: nx.MultiDiGraph,
) -> None:
    for u, v, edge in graph.edges(data=True):
        if edge.get("geometry") is not None:
            continue

        u_node = graph.nodes[u]
        v_node = graph.nodes[v]

        edge["geometry"] = LineString(
            [
                (u_node["x"], u_node["y"]),
                (v_node["x"], v_node["y"]),
            ]
        )


@measure_time
def load_road_network(
    positions: pl.DataFrame,
    margin: float,
) -> nx.MultiDiGraph:
    if margin < 0:
        raise ValueError("margin must be greater than or equal to 0")

    if margin > ROAD_NETWORK_CACHE_MARGIN:
        raise ValueError("margin must not exceed ROAD_NETWORK_CACHE_MARGIN")

    bounds = calculate_bounds(positions)

    requested_bounds = expand_bounds(
        bounds,
        margin=margin,
    )

    if GRAPH_PATH.exists() and METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

        cached_bounds = (
            metadata["west"],
            metadata["south"],
            metadata["east"],
            metadata["north"],
        )

        if contains_bounds(
            cached_bounds,
            requested_bounds,
        ):
            cached_graph = ox.load_graphml(GRAPH_PATH)

            return ox.truncate.truncate_graph_bbox(
                cached_graph,
                requested_bounds,
                truncate_by_edge=True,
            )

    cache_bounds = expand_bounds(
        bounds,
        margin=ROAD_NETWORK_CACHE_MARGIN,
    )

    graph = ox.graph_from_bbox(
        cache_bounds,
        network_type="drive",
    )

    ensure_edge_geometries(graph)

    Path(ROAD_NETWORK_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.save_graphml(
        graph,
        filepath=GRAPH_PATH,
    )

    west, south, east, north = cache_bounds

    metadata = {
        "west": west,
        "south": south,
        "east": east,
        "north": north,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return ox.truncate.truncate_graph_bbox(
        graph,
        requested_bounds,
        truncate_by_edge=True,
    )
