"""Main execution script for E-Commerce Synthetic Data Generation.

Orchestrates the generation of synthetic customers, product catalog listings,
and order transactions, saving the outputs as compressed Parquet files
in the data/raw/ repository. Reports detailed generation benchmarks,
timings, and final file sizes.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import (
    CUSTOMERS_PARQUET_FILE,
    DATA_RAW_DIR,
    ORDERS_PARQUET_FILE,
    PRODUCTS_PARQUET_FILE,
    config,
    setup_logger,
)
from src.data_generator import SyntheticDataGenerator

logger: logging.Logger = setup_logger("main", level=logging.INFO)


def format_file_size(size_in_bytes: int) -> str:
    """Format a byte count into a human-readable string (B, KB, MB, GB).

    Args:
        size_in_bytes: Size in bytes.

    Returns:
        str: Human-readable size string (e.g. '12.45 MB').
    """
    size: float = float(size_in_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} B"


def save_as_parquet(
    df: pd.DataFrame,
    file_path: Path,
    dataset_name: str,
    compression: str = "snappy",
) -> Tuple[float, int]:
    """Save a pandas DataFrame to disk in Parquet format and return stats.

    Args:
        df: DataFrame to save.
        file_path: Target Path on disk.
        dataset_name: Descriptive name for logging.
        compression: Parquet compression codec (defaults to 'snappy').

    Returns:
        Tuple[float, int]: (save_duration_seconds, file_size_bytes)
    """
    logger.info("Writing %s to Parquet: %s...", dataset_name, file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    start_time: float = time.time()
    df.to_parquet(file_path, index=False, engine="pyarrow", compression=compression)
    save_duration: float = time.time() - start_time

    file_size_bytes: int = file_path.stat().st_size
    logger.info(
        "Successfully saved %s (%s rows) to %s [%s in %.2fs]",
        dataset_name,
        f"{len(df):,}",
        file_path.name,
        format_file_size(file_size_bytes),
        save_duration,
    )
    return save_duration, file_size_bytes


def run_pipeline(
    num_customers: int,
    num_products: int,
    num_orders: int,
    seed: Optional[int] = 42,
) -> Dict[str, Dict[str, object]]:
    """Execute the synthetic data generation and export to Parquet files.

    Args:
        num_customers: Target customer count.
        num_products: Target product count.
        num_orders: Target order count.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict[str, Dict[str, object]]: Summary metrics and benchmarks per dataset.

    Raises:
        Exception: Re-raises any critical pipeline failure after logging.
    """
    logger.info("==================================================================")
    logger.info("       E-Commerce Data Pipeline: Parquet Generation Job          ")
    logger.info("==================================================================")
    logger.info(
        "Target Generation Specs: Customers=%s | Products=%s | Orders=%s",
        f"{num_customers:,}",
        f"{num_products:,}",
        f"{num_orders:,}",
    )
    logger.info("Target Raw Directory: %s", DATA_RAW_DIR)

    pipeline_start: float = time.time()
    results: Dict[str, Dict[str, object]] = {}

    try:
        # Initialize the generator with target parameters
        generator = SyntheticDataGenerator(
            num_customers=num_customers,
            num_products=num_products,
            num_orders=num_orders,
            seed=seed,
        )

        # ----------------------------------------------------------------------
        # 1. Customers Generation & Parquet Export
        # ----------------------------------------------------------------------
        logger.info("\n--- Phase 1: Generating Customers ---")
        t0 = time.time()
        customers_df: pd.DataFrame = generator.generate_customers(count=num_customers)
        cust_gen_time = time.time() - t0

        cust_save_time, cust_size = save_as_parquet(
            customers_df,
            CUSTOMERS_PARQUET_FILE,
            dataset_name="Customers",
        )
        results["Customers"] = {
            "rows": len(customers_df),
            "gen_time": cust_gen_time,
            "save_time": cust_save_time,
            "file_size": cust_size,
            "file_path": CUSTOMERS_PARQUET_FILE,
        }

        # ----------------------------------------------------------------------
        # 2. Products Generation & Parquet Export
        # ----------------------------------------------------------------------
        logger.info("\n--- Phase 2: Generating Products ---")
        t0 = time.time()
        products_df: pd.DataFrame = generator.generate_products(count=num_products)
        prod_gen_time = time.time() - t0

        prod_save_time, prod_size = save_as_parquet(
            products_df,
            PRODUCTS_PARQUET_FILE,
            dataset_name="Products",
        )
        results["Products"] = {
            "rows": len(products_df),
            "gen_time": prod_gen_time,
            "save_time": prod_save_time,
            "file_size": prod_size,
            "file_path": PRODUCTS_PARQUET_FILE,
        }

        # ----------------------------------------------------------------------
        # 3. Orders Generation (Pareto Distribution) & Parquet Export
        # ----------------------------------------------------------------------
        logger.info("\n--- Phase 3: Generating Orders (Pareto Distribution) ---")
        t0 = time.time()
        orders_df: pd.DataFrame = generator.generate_orders(
            customers_df=customers_df,
            products_df=products_df,
            count=num_orders,
        )
        ord_gen_time = time.time() - t0

        ord_save_time, ord_size = save_as_parquet(
            orders_df,
            ORDERS_PARQUET_FILE,
            dataset_name="Orders",
        )
        results["Orders"] = {
            "rows": len(orders_df),
            "gen_time": ord_gen_time,
            "save_time": ord_save_time,
            "file_size": ord_size,
            "file_path": ORDERS_PARQUET_FILE,
        }

        total_duration: float = time.time() - pipeline_start

        # ----------------------------------------------------------------------
        # Print Final Benchmark Summary
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("               SYNTHETIC DATA GENERATION BENCHMARK REPORT               ")
        print("=" * 80)
        print(f"{'Dataset':<12} | {'Rows':<12} | {'Gen Time':<10} | {'Save Time':<10} | {'Parquet Size':<12} | {'File Name'}")
        print("-" * 80)

        total_bytes: int = 0
        for name, data in results.items():
            total_bytes += int(data["file_size"])
            print(
                f"{name:<12} | "
                f"{data['rows']:<12,d} | "
                f"{data['gen_time']:<8.2f}s | "
                f"{data['save_time']:<8.2f}s | "
                f"{format_file_size(int(data['file_size'])):<12} | "
                f"{Path(str(data['file_path'])).name}"
            )

        print("-" * 80)
        print(f"Total Pipeline Execution Time: {total_duration:.2f} seconds")
        print(f"Total Raw Parquet Storage   : {format_file_size(total_bytes)}")
        print("=" * 80 + "\n")

        return results

    except KeyboardInterrupt:
        logger.warning("\nPipeline execution interrupted by user (SIGINT). Exiting safely...")
        sys.exit(130)

    except Exception as exc:
        logger.exception("A critical error occurred during pipeline execution: %s", exc)
        raise exc


def main() -> None:
    """CLI Entry point with argument parsing and top-level error handling."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic e-commerce datasets and export to Parquet."
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=config.num_customers,
        help=f"Number of customers to generate (default: {config.num_customers:,})",
    )
    parser.add_argument(
        "--products",
        type=int,
        default=config.num_products,
        help=f"Number of products to generate (default: {config.num_products:,})",
    )
    parser.add_argument(
        "--orders",
        type=int,
        default=config.num_orders,
        help=f"Number of orders to generate (default: {config.num_orders:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.random_seed,
        help=f"Random seed for deterministic reproducibility (default: {config.random_seed})",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Execute a quick preview run (1,000 customers, 100 products, 5,000 orders)",
    )

    args = parser.parse_args()

    try:
        if args.fast:
            logger.info("Executing fast preview run (1,000 customers, 100 products, 5,000 orders)...")
            run_pipeline(
                num_customers=1_000,
                num_products=100,
                num_orders=5_000,
                seed=args.seed,
            )
        else:
            run_pipeline(
                num_customers=args.customers,
                num_products=args.products,
                num_orders=args.orders,
                seed=args.seed,
            )

    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

