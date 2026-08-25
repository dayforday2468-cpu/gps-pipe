import polars as pl
from pyproj import Transformer
from shapely.geometry import LineString, Point

from modules.primitives.decorators import measure_time


@measure_time
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
) -> tuple[float, float, float, float] | None:
    point = Point(x, y)

    distance_along_edge = geometry.project(point)

    if not 0 < distance_along_edge < geometry.length:
        return None

    projected_point = geometry.interpolate(distance_along_edge)

    return (
        projected_point.x,
        projected_point.y,
        point.distance(projected_point),
        distance_along_edge,
    )
