"""PySpark Sales Analytics Execution Script.

Executes the SalesAnalytics pipeline:
  1. Configures and initializes an optimized Apache Spark session.
  2. Ingests raw Parquet datasets (customers, products, and orders).
  3. Executes top customer revenue ranking, category sales aggregation,
     and month-over-month revenue growth trend analysis.
  4. Displays analytical results in structured tabular format.
  5. Measures and reports per-operation execution benchmarks.
  6. Safely terminates the SparkSession.
"""

import logging
from pathlib import Path
import sys
import time
from typing import Dict

from pyspark.sql import DataFrame, SparkSession

from src.config import (
    CUSTOMERS_PARQUET_FILE,
    ORDERS_PARQUET_FILE,
    PRODUCTS_PARQUET_FILE,
    setup_logger,
)
from src.spark_analytics import SalesAnalytics

logger: logging.Logger = setup_logger("run_analytics", level=logging.INFO)


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a human-friendly string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        str: Formatted duration (e.g. '1.42s' or '125ms').
    """
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f}ms"
    return f"{seconds:.2f}s"


def run_analytics_job() -> Dict[str, float]:
    """Execute all three analytics operations, measuring per-step durations.

    Returns:
        Dict[str, float]: Dictionary mapping operation names to execution times in seconds.
    """
    timings: Dict[str, float] = {}
    job_start: float = time.time()

    logger.info("==================================================================")
    logger.info("            Starting E-Commerce PySpark Sales Analytics           ")
    logger.info("==================================================================")

    # --------------------------------------------------------------------------
    # Step 1 & 2: Create Spark Session & Initialize SalesAnalytics
    # --------------------------------------------------------------------------
    logger.info("\n>>> [1/5] Initializing SparkSession (4GB, AQE=ON, Kryo=ON)...")
    t0 = time.time()
    spark: SparkSession = SalesAnalytics.create_spark_session(
        app_name="EcommerceSalesAnalyticsJob"
    )
    analytics = SalesAnalytics(spark=spark)
    timings["Spark Initialization"] = time.time() - t0
    logger.info(
        "SparkSession created successfully in %s",
        format_duration(timings["Spark Initialization"]),
    )

    try:
        # ----------------------------------------------------------------------
        # Step 3: Load Parquet datasets from data/raw/
        # ----------------------------------------------------------------------
        logger.info("\n>>> [2/5] Loading Parquet datasets from data/raw/...")
        t0 = time.time()

        customers_df: DataFrame = analytics.load_parquet(CUSTOMERS_PARQUET_FILE)
        products_df: DataFrame = analytics.load_parquet(PRODUCTS_PARQUET_FILE)
        orders_df: DataFrame = analytics.load_parquet(ORDERS_PARQUET_FILE)

        # Trigger quick count / caching metadata to measure ingestion
        n_cust: int = customers_df.count()
        n_prod: int = products_df.count()
        n_ord: int = orders_df.count()

        timings["Load Parquet Data"] = time.time() - t0
        logger.info(
            "Loaded Parquet datasets in %s (Customers: %s, Products: %s, Orders: %s)",
            format_duration(timings["Load Parquet Data"]),
            f"{n_cust:,}",
            f"{n_prod:,}",
            f"{n_ord:,}",
        )

        # ----------------------------------------------------------------------
        # Step 4: Top Customers by Revenue (Pareto Spenders)
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("                  TOP 10 CUSTOMERS BY REVENUE                   ")
        print("=" * 80)
        logger.info(">>> [3/5] Computing Top 10 Customers by Revenue...")
        t0 = time.time()

        top_customers_df: DataFrame = analytics.top_customers_by_revenue(
            orders_df=orders_df,
            products_df=products_df,
            n=10,
        )
        top_customers_df.show(truncate=False)
        timings["Top Customers by Revenue"] = time.time() - t0
        logger.info(
            "Top Customers operation completed in %s",
            format_duration(timings["Top Customers by Revenue"]),
        )

        # ----------------------------------------------------------------------
        # Step 5: Sales by Product Category
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("                  SALES BY PRODUCT CATEGORY                     ")
        print("=" * 80)
        logger.info(">>> [4/5] Computing Sales by Product Category...")
        t0 = time.time()

        category_sales_df: DataFrame = analytics.sales_by_category(
            orders_df=orders_df,
            products_df=products_df,
        )
        category_sales_df.show(truncate=False)
        timings["Sales by Category"] = time.time() - t0
        logger.info(
            "Sales by Category operation completed in %s",
            format_duration(timings["Sales by Category"]),
        )

        # ----------------------------------------------------------------------
        # Step 6: Monthly Trends & MoM Growth (%) via Window Functions
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("            MONTHLY REVENUE TRENDS & MOM GROWTH (%)             ")
        print("=" * 80)
        logger.info(">>> [5/5] Computing Monthly Trends and MoM Growth (Window LAG)...")
        t0 = time.time()

        monthly_trends_df: DataFrame = analytics.monthly_trends(
            orders_df=orders_df,
            products_df=products_df,
        )
        monthly_trends_df.show(n=36, truncate=False)
        timings["Monthly Trends (MoM Growth)"] = time.time() - t0
        logger.info(
            "Monthly Trends operation completed in %s",
            format_duration(timings["Monthly Trends (MoM Growth)"]),
        )

        total_job_time: float = time.time() - job_start
        timings["Total Execution Time"] = total_job_time

        # ----------------------------------------------------------------------
        # Display Execution Benchmark Summary
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("                 PYSPARK OPERATIONS BENCHMARK REPORT                    ")
        print("=" * 80)
        print(f"{'Operation':<35} | {'Execution Time':<18} | {'Status'}")
        print("-" * 80)
        for op_name, duration in timings.items():
            if op_name != "Total Execution Time":
                print(f"{op_name:<35} | {format_duration(duration):<18} | SUCCESS")
        print("-" * 80)
        print(f"{'Total Job Duration':<35} | {format_duration(total_job_time):<18} | FINISHED")
        print("=" * 80 + "\n")

        return timings

    except Exception as exc:
        logger.exception("An error occurred during PySpark analytics execution: %s", exc)
        raise exc

    finally:
        # ----------------------------------------------------------------------
        # Step 7: Stop SparkSession
        # ----------------------------------------------------------------------
        logger.info("Stopping SparkSession...")
        analytics.stop()


def main() -> None:
    """CLI entry point for running the analytics job."""
    try:
        run_analytics_job()
    except KeyboardInterrupt:
        logger.warning("\nExecution cancelled by user. Exiting...")
        sys.exit(130)
    except Exception as exc:
        logger.error("Job execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

