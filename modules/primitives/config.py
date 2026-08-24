# Data loading
BATCH_SIZE = 10_000

# Data paths
DATA_DIR = "data"
PROCESSED_DIR = f"{DATA_DIR}/processed"
SOURCE_PATH = f"{DATA_DIR}/timeline.json"
ROAD_NETWORK_DIR = f"{DATA_DIR}/road_network"

# Geographic constants
EARTH_RADIUS = 6_371_000  # meters

# Sudden position jump removal
JUMP_RATE = 0.05
MAX_JUMP_POINTS = 3

# Road network margins
ROAD_NETWORK_CACHE_MARGIN = 5_000  # meters
ROAD_NETWORK_VIEW_MARGIN = 1_000  # meters
