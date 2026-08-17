import inspect
from functools import wraps
from time import perf_counter

from modules.primitives.logger import get_logger


def measure_time(func):
    logger = get_logger(f"{func.__module__}.{func.__name__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()

        result = func(*args, **kwargs)

        elapsed = perf_counter() - start

        logger.info(
            "completed in %.4f sec",
            elapsed,
        )

        return result

    return wrapper


def measure_generator_time(func):
    logger = get_logger(f"{func.__module__}.{func.__name__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        generator = func(*args, **kwargs)

        total_elapsed = 0.0

        while True:
            start = perf_counter()

            try:
                value = next(generator)

            except StopIteration:
                total_elapsed += perf_counter() - start
                break

            total_elapsed += perf_counter() - start

            yield value

        logger.info(
            "completed in %.4f sec",
            total_elapsed,
        )

    return wrapper
