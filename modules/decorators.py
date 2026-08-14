from functools import wraps
from time import perf_counter

from modules.logger import get_logger


def measure_time(func):
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()

        result = func(*args, **kwargs)

        elapsed = perf_counter() - start

        logger.info(
            "%s completed in %.4f sec",
            func.__name__,
            elapsed,
        )

        return result

    return wrapper