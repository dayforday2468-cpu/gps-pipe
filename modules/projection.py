import polars as pl
from pyproj import Transformer


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
