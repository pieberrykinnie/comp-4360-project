import os
import sys
import logging
import functools
from pathlib import Path
from termcolor import colored


DATE_FMT: str = "%Y-%m-%d %H:%M:%S"


# Cache results of function calls with the same arguments.
@functools.lru_cache()
def create_logger(output_dir: Path, dist_rank: int = 0, name: str = "") -> logging.Logger:
    """
    Return a logger for the program.

    Args:
        output_dir: The directory where log files are stored.
        dist_rank: The "rank" of the distributed process running this logger.
        name: The name of the logger.

    Return: A logger for the current process.
    """
    # Create logger
    logger: logging.Logger = logging.getLogger(name)

    # Detailed logging
    logger.setLevel(logging.DEBUG)

    # Do not propagate messages of child loggers to parent (prevent duplicate messages)
    logger.propagate = False

    # Format log strings
    fmt: str = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt: str = colored("[%(asctime)s %(name)s]", "green") + \
        colored("(%(filename)s %(lineno)d)", "yellow") + \
        ": %(levelname)s %(message)s"

    # Create console handlers for main process
    # Other distributed processes will have ranks != 0, so the terminal isn't clogged
    if dist_rank == 0:
        # Print the logs to stdout
        console_handler: logging.StreamHandler = logging.StreamHandler(
            sys.stdout)

        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(color_fmt, DATE_FMT)
        )

        # Add the handler to logging
        logger.addHandler(console_handler)

    # Create file handlers
    # All processes will have log files
    file_handler: logging.FileHandler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}.txt"), "a")

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, DATE_FMT))

    logger.addHandler(file_handler)

    return logger
