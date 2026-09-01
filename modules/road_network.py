import json
import math
from pathlib import Path
from typing import NamedTuple

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


class Bounds(NamedTuple):
    west: float
    south: float
    east: float
    north: float


def calculate_bounds(
    positions: pl.DataFrame,
) -> Bounds:
    if positions.is_empty():
        raise ValueError("positions must not be empty")

    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(positions.columns)

    if missing_columns:
        raise ValueError(f"required columns are missing: {missing_columns}")

    return Bounds(
        west=positions.get_column("longitude").min(),
        south=positions.get_column("latitude").min(),
        east=positions.get_column("longitude").max(),
        north=positions.get_column("latitude").max(),
    )


def expand_bounds(
    bounds: Bounds,
    margin: float,
) -> Bounds:
    center_latitude = (bounds.south + bounds.north) / 2

    angular_margin = margin / EARTH_RADIUS
    latitude_margin = math.degrees(angular_margin)
    longitude_margin = math.degrees(
        angular_margin / math.cos(math.radians(center_latitude))
    )

    return Bounds(
        west=bounds.west - longitude_margin,
        south=bounds.south - latitude_margin,
        east=bounds.east + longitude_margin,
        north=bounds.north + latitude_margin,
    )


def contains_bounds(
    outer: Bounds,
    inner: Bounds,
) -> bool:
    return (
        outer.west <= inner.west
        and outer.south <= inner.south
        and outer.east >= inner.east
        and outer.north >= inner.north
    )


def _ensure_edge_geometries(
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

def _prepare_road_network(
    graph: nx.MultiDiGraph,
    bounds: Bounds,
) -> nx.MultiGraph:
    truncated_graph = ox.truncate.truncate_graph_bbox(
        graph,
        bounds,
        truncate_by_edge=True,
    )

    return ox.convert.to_undirected(
        truncated_graph,
    )


@measure_time
def load_road_network(
    positions: pl.DataFrame,
    margin: float,
) -> nx.MultiGraph:
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
        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        cached_bounds = Bounds(
            west=metadata["west"],
            south=metadata["south"],
            east=metadata["east"],
            north=metadata["north"],
        )

        if contains_bounds(
            cached_bounds,
            requested_bounds,
        ):
            cached_graph = ox.load_graphml(
                GRAPH_PATH,
            )

            return _prepare_road_network(
                cached_graph,
                requested_bounds,
            )

    cache_bounds = expand_bounds(
        bounds,
        margin=ROAD_NETWORK_CACHE_MARGIN,
    )

    graph = ox.graph_from_bbox(
        cache_bounds,
        network_type="drive",
    )

    _ensure_edge_geometries(graph)

    Path(ROAD_NETWORK_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.save_graphml(
        graph,
        filepath=GRAPH_PATH,
    )

    metadata = {
        "west": cache_bounds.west,
        "south": cache_bounds.south,
        "east": cache_bounds.east,
        "north": cache_bounds.north,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return _prepare_road_network(
        graph,
        requested_bounds,
    )