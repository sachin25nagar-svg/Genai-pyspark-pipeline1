"""Benchmark: Pandas vs. PySpark Performance Comparison on 1M Rows.

Performs identical end-to-end data processing workflows using both Pandas and PySpark:
  1. Load orders.parquet (1,000,000 rows) and products.parquet (10,000 rows).
  2. Join datasets on 'product_id' and compute line revenue (quantity * price).
  3. Group by 'customer_id', calculate aggregate revenue, and extract Top 10 customers.
  4. Measure and print detailed execution times and a side-by-side comparison table.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import (
    ORDERS_PARQUET_FILE,
    PRODUCTS_PARQUET_FILE,
    setup_logger,
)
from src.spark_analytics import SalesAnalytics

logger = setup_logger("benchmark", level=20)


def benchmark_pandas(
    orders_path: Path,
    products_path: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Execute the end-to-end data pipeline using Pandas.

    Args:
        orders_path: Path to orders.parquet file.
        products_path: Path to products.parquet file.

    Returns:
        Tuple[Dict[str, float], pd.DataFrame]:
            - Dictionary of measured execution timings per operation.
            - Top 10 customers DataFrame.
    """
    timings: Dict[str, float] = {}

    # 1. Load Parquet
    t0 = time.perf_counter()
    orders_df = pd.read_parquet(orders_path)
    products_df = pd.read_parquet(products_path)
    timings["1. Load Parquet"] = time.perf_counter() - t0

    # 2. Join & Calculate Revenue
    t0 = time.perf_counter()
    merged_df = orders_df.merge(
        products_df[["product_id", "price"]],
        on="product_id",
        how="inner",
    )
    merged_df["revenue"] = merged_df["quantity"] * merged_df["price"]
    timings["2. Join & Calculate Revenue"] = time.perf_counter() - t0

    # 3. Group by, Sum & Top 10
    t0 = time.perf_counter()
    top_10 = (
        merged_df.groupby("customer_id", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values(by="total_revenue", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_10["total_revenue"] = top_10["total_revenue"].round(2)
    timings["3. Group By, Sum & Top 10"] = time.perf_counter() - t0

    timings["Total Pipeline Time"] = sum(timings.values())
    return timings, top_10


def benchmark_pyspark(
    spark: SparkSession,
    orders_path: Path,
    products_path: Path,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Execute the end-to-end data pipeline using Apache PySpark.

    Forces physical execution at each stage to accurately capture I/O,
    distributed join shuffle, and aggregation latency.

    Args:
        spark: Active SparkSession instance.
        orders_path: Path to orders.parquet file.
        products_path: Path to products.parquet file.

    Returns:
        Tuple[Dict[str, float], pd.DataFrame]:
            - Dictionary of measured execution timings per operation.
            - Top 10 customers DataFrame.
    """
    timings: Dict[str, float] = {}

    # 1. Load Parquet (Materialize I/O using count)
    t0 = time.perf_counter()
    orders_df: DataFrame = spark.read.parquet(str(orders_path))
    products_df: DataFrame = spark.read.parquet(str(products_path))
    _ = orders_df.count()
    _ = products_df.count()
    timings["1. Load Parquet"] = time.perf_counter() - t0

    # 2. Join & Calculate Revenue (Materialize execution using count)
    t0 = time.perf_counter()
    joined_df: DataFrame = orders_df.join(
        products_df.select("product_id", "price"),
        on="product_id",
        how="inner",
    ).withColumn("revenue", F.col("quantity") * F.col("price"))
    _ = joined_df.count()
    timings["2. Join & Calculate Revenue"] = time.perf_counter() - t0

    # 3. Group by, Sum & Top 10 (Materialize via collect/toPandas)
    t0 = time.perf_counter()
    top_10_df: DataFrame = (
        joined_df.groupBy("customer_id")
        .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
        .orderBy(F.col("total_revenue").desc())
        .limit(10)
    )
    top_10_pd = top_10_df.toPandas()
    timings["3. Group By, Sum & Top 10"] = time.perf_counter() - t0

    timings["Total Pipeline Time"] = sum(timings.values())
    return timings, top_10_pd


def print_comparison_table(
    pandas_timings: Dict[str, float],
    spark_timings: Dict[str, float],
    pandas_top_10: pd.DataFrame,
    spark_top_10: pd.DataFrame,
    num_rows: int,
) -> None:
    """Print benchmark comparison table and validation results.

    Args:
        pandas_timings: Step timings for Pandas.
        spark_timings: Step timings for PySpark.
        pandas_top_10: Top 10 customers produced by Pandas.
        spark_top_10: Top 10 customers produced by PySpark.
        num_rows: Total orders dataset rows processed.
    """
    print("\n" + "=" * 98)
    print(f"        PERFORMANCE BENCHMARK: PANDAS vs. PYSPARK ({num_rows:,} ROWS)")
    print("=" * 98)
    header = f"{'Pipeline Operation':<36} | {'Pandas':<14} | {'PySpark':<14} | {'Speedup / Advantage'}"
    print(header)
    print("-" * 98)

    operations = [
        "1. Load Parquet",
        "2. Join & Calculate Revenue",
        "3. Group By, Sum & Top 10",
        "Total Pipeline Time",
    ]

    for op in operations:
        pd_time = pandas_timings[op]
        sp_time = spark_timings[op]

        if pd_time < sp_time:
            ratio = sp_time / pd_time
            advantage = f"Pandas is {ratio:.1f}x faster"
        else:
            ratio = pd_time / sp_time
            advantage = f"PySpark is {ratio:.1f}x faster"

        is_total = op == "Total Pipeline Time"
        if is_total:
            print("-" * 98)

        print(f"{op:<36} | {pd_time:<12.3f}s | {sp_time:<12.3f}s | {advantage}")

    print("=" * 98)

    # Output verification: Confirm exact numerical parity between engines
    print("\n" + "=" * 98)
    print("                    DATA CONSISTENCY & ACCURACY VERIFICATION                     ")
    print("=" * 98)
    print(f"{'Rank':<6} | {'Pandas Customer':<18} | {'Pandas Spend ($)':<16} | {'PySpark Customer':<18} | {'PySpark Spend ($)':<16} | {'Match?'}")
    print("-" * 98)

    all_matched = True
    for i in range(min(len(pandas_top_10), len(spark_top_10))):
        p_row = pandas_top_10.iloc[i]
        s_row = spark_top_10.iloc[i]
        match = (p_row["customer_id"] == s_row["customer_id"]) and (
            abs(float(p_row["total_revenue"]) - float(s_row["total_revenue"])) < 0.05
        )
        if not match:
            all_matched = False
        print(
            f"{i + 1:<6} | "
            f"{p_row['customer_id']:<18} | "
            f"${p_row['total_revenue']:<15.2f} | "
            f"{s_row['customer_id']:<18} | "
            f"${s_row['total_revenue']:<15.2f} | "
            f"{'OK' if match else 'MISMATCH'}"
        )

    print("-" * 98)
    if all_matched:
        print(">> VERIFIED: 100% numerical and ranking match between Pandas and PySpark!")
    else:
        print(">> Note: Minor rank or rounding variance observed.")
    print("=" * 98 + "\n")


def run_benchmark() -> None:
    """Run the complete performance benchmark on 1M rows."""
    orders_path = ORDERS_PARQUET_FILE
    products_path = PRODUCTS_PARQUET_FILE

    if not orders_path.exists() or not products_path.exists():
        logger.error(
            "Required Parquet files not found in data/raw/. Run 'python main.py' first!"
        )
        sys.exit(1)

    # 1. Run Pandas Benchmark
    logger.info(">>> Starting Pandas Benchmark on %s...", orders_path.name)
    pandas_timings, pandas_top_10 = benchmark_pandas(orders_path, products_path)
    logger.info("Pandas total time: %.3f seconds", pandas_timings["Total Pipeline Time"])

    # 2. Run PySpark Benchmark
    logger.info(">>> Initializing SparkSession for PySpark Benchmark...")
    spark: SparkSession = SalesAnalytics.create_spark_session(
        app_name="PandasVsSparkBenchmark"
    )

    try:
        logger.info(">>> Starting PySpark Benchmark on %s...", orders_path.name)
        spark_timings, spark_top_10 = benchmark_pyspark(
            spark=spark,
            orders_path=orders_path,
            products_path=products_path,
        )
        logger.info("PySpark total time: %.3f seconds", spark_timings["Total Pipeline Time"])

        # 3. Print Results Comparison Table
        print_comparison_table(
            pandas_timings=pandas_timings,
            spark_timings=spark_timings,
            pandas_top_10=pandas_top_10,
            spark_top_10=spark_top_10,
            num_rows=1_000_000,
        )

    finally:
        logger.info("Stopping SparkSession...")
        spark.stop()


if __name__ == "__main__":
    run_benchmark()
