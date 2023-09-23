import logging

def setup_logger():
    """Setup basic logging for the application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
    return logging.getLogger("ModelComparator")

logger = setup_logger()
