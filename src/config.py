"""Configuration module for the E-commerce Data Pipeline.

Centralizes file paths, synthetic generation scaling defaults,
PySpark execution settings, and logging configuration across the project.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Optional


# ==============================================================================
# Path Configuration
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

# Additional directories
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Raw dataset file paths (CSV)
CUSTOMERS_FILE: Path = DATA_RAW_DIR / "customers.csv"
PRODUCTS_FILE: Path = DATA_RAW_DIR / "products.csv"
ORDERS_FILE: Path = DATA_RAW_DIR / "orders.csv"

# Raw dataset file paths (Parquet)
CUSTOMERS_PARQUET_FILE: Path = DATA_RAW_DIR / "customers.parquet"
PRODUCTS_PARQUET_FILE: Path = DATA_RAW_DIR / "products.parquet"
ORDERS_PARQUET_FILE: Path = DATA_RAW_DIR / "orders.parquet"

# Processed analytical output paths
KPI_SUMMARY_OUTPUT: Path = DATA_PROCESSED_DIR / "kpi_summary"
TOP_PRODUCTS_OUTPUT: Path = DATA_PROCESSED_DIR / "top_products"
CATEGORY_METRICS_OUTPUT: Path = DATA_PROCESSED_DIR / "category_metrics"
CUSTOMER_METRICS_OUTPUT: Path = DATA_PROCESSED_DIR / "customer_metrics"
MONTHLY_SALES_OUTPUT: Path = DATA_PROCESSED_DIR / "monthly_sales"


# ==============================================================================
# Pipeline & Spark Settings
# ==============================================================================
@dataclass(frozen=True)
class PipelineConfig:
    """Settings for synthetic data generation and Spark processing."""

    # Default synthetic generation scale: 100K customers, 10K products, 1M orders
    num_customers: int = 100_000
    num_products: int = 10_000
    num_orders: int = 1_000_000
    random_seed: int = 42
    pareto_alpha: float = 1.38

    # PySpark session configuration
    spark_app_name: str = "EcommerceDataPipeline"
    spark_master: str = "local[*]"
    spark_log_level: str = "WARN"
    spark_shuffle_partitions: int = 4


DEFAULT_CONFIG: PipelineConfig = PipelineConfig()
config: PipelineConfig = DEFAULT_CONFIG


# ==============================================================================
# Logging Configuration
# ==============================================================================
def setup_logger(
    name: str = "ecommerce_pipeline",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Args:
        name: Name of the logger, typically the module or pipeline name.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to write log output to a file.

    Returns:
        logging.Logger: A configured logger instance ready for debugging.
    """
    logger: logging.Logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def ensure_directories() -> None:
    """Ensure that all required project directories exist on disk."""
    logger = setup_logger("config.ensure_directories")
    for directory in (DATA_RAW_DIR, DATA_PROCESSED_DIR, NOTEBOOKS_DIR, TESTS_DIR):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: %s", directory)


ensure_directories()
