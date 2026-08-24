import polars as pl
from pyproj import Transformer
from shapely.geometry import LineString, Point


def project_positions(
    positions: pl.DataFrame,
    target_crs,
) -> pl.DataFrame:
    transformer = Transformer.from_crs(
        "EPSG:4326",
        target_crs,
        always_xy=True,
    )

    x, y = transformer.transform(
        positions["longitude"].to_numpy(),
        positions["latitude"].to_numpy(),
    )

    return pl.DataFrame(
        {
            "position_id": positions["position_id"],
            "x": x,
            "y": y,
        }
    )

def project_point_to_edge(
    x: float,
    y: float,
    geometry: LineString,
) -> tuple[float, float, float]:
    point = Point(x, y)

    distance_along_edge = geometry.project(point)
    projected_point = geometry.interpolate(distance_along_edge)

    distance = point.distance(projected_point)

    return (
        projected_point.x,
        projected_point.y,
        distance,
    )