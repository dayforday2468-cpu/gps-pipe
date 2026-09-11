import polars as pl
from pyproj import Transformer
from shapely.geometry import LineString, Point

from modules.primitives.decorators import measure_time
from modules.primitives.schema import (
    RawPositionSchema,
    ProjectedPositionSchema,
    validate_schema_columns
)

@measure_time
def project_positions(
    positions: pl.DataFrame,
    target_crs,
) -> pl.DataFrame:
    validate_schema_columns(
        positions,
        RawPositionSchema,
    )

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

@measure_time
def unproject_positions(
    positions: pl.DataFrame,
    source_crs,
) -> pl.DataFrame:
    validate_schema_columns(
        positions,
        ProjectedPositionSchema,
    )

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )

    longitude, latitude = transformer.transform(
        positions["x"].to_numpy(),
        positions["y"].to_numpy(),
    )

    return pl.DataFrame(
        {
            "position_id": positions["position_id"],
            "latitude": latitude,
            "longitude": longitude,
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
