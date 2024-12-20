# traffic_counter/config.py

import yaml
import logging
import sys

def setup_logging(log_file: str, level: str = "INFO"):
    """
    Sets up logging to both console and file.

    Args:
        log_file (str): Path to the log file.
        level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), "INFO"))

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(getattr(logging, level.upper(), "INFO"))

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), "INFO"))

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers if not already present
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
