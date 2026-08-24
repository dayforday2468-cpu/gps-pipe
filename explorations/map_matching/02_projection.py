from datetime import datetime

import osmnx as ox

from modules.primitives.config import ROAD_NETWORK_VIEW_MARGIN
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.primitives.visualization import GPSVisualizer
from modules.projection import project_positions
from modules.road_network import load_road_network

if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)
    time_range = f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    road_network = load_road_network(
        raw_filtered,
        margin=ROAD_NETWORK_VIEW_MARGIN,
    )

    projected_road_network = ox.project_graph(road_network)

    projected_positions = project_positions(
        raw_filtered,
        projected_road_network.graph["crs"],
    )

    print("=== Projected Road Network ===")
    print(f"CRS: {projected_road_network.graph['crs']}")
    print(projected_positions.head())

    visualizer = GPSVisualizer(
        title=f"Projected GPS with Road Network - {time_range}",
        show_legend=True,
    )

    visualizer.add_road_network(projected_road_network)

    visualizer.add(
        projected_positions,
        label="Projected GPS",
        point_size=3,
        point_color="red",
        show_line=True,
        line_color="blue",
        line_width=1.0,
        alpha=0.8,
    )

    visualizer.show()
