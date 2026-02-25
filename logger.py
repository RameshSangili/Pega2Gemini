import logging
from logging.handlers import RotatingFileHandler

_LOGGER = None

def get_logger():
    global _LOGGER
    if _LOGGER:
        return _LOGGER

    logger = logging.getLogger("mcp-server")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # avoid duplicate logs

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File logs (DEBUG+), rotate at 5MB, keep 5 backups
    file_handler = RotatingFileHandler(
        "mcp.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console logs (INFO+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _LOGGER = logger
    return logger
