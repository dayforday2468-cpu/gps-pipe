from modules.dataload import *

from modules.logger import get_logger

logger = get_logger(__name__)
data_path = "data/timeline.json"

if __name__=="__main__":
    # 데이터 배치 로더 생성v
    raw_positions_batches = load_raw_positions_batches(data_path)
    logger.info("raw position loader initialized")

    timeline_paths_batches = load_timeline_paths_batches(data_path)
    logger.info("timeline path loader initialized")

    visits_batches = load_visits_batches(data_path)
    logger.info("visit loader initialized")

    activities_batches = load_activities_batches(data_path)
    logger.info("activity loader initialized")
