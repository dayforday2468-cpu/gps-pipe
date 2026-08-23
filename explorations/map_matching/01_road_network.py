from datetime import datetime

from modules.primitives.config import ROAD_NETWORK_VIEW_MARGIN
from modules.primitives.datafilter import filter_points
from modules.primitives.pipeline import initialize_pipeline
from modules.road_network import load_road_network

if __name__ == "__main__":
    batches = initialize_pipeline()

    start = datetime(2026, 8, 1, 7, 0)
    end = datetime(2026, 8, 2, 0, 0)

    raw_filtered = filter_points(
        batches.raw_positions,
        start,
        end,
    )

    graph = load_road_network(
        raw_filtered,
        margin=ROAD_NETWORK_VIEW_MARGIN,
    )

    print("=== Road Network ===")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"CRS: {graph.graph['crs']}")
