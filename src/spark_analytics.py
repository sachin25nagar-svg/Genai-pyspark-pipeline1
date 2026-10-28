"""PySpark Sales Analytics Module.

Provides the `SalesAnalytics` class to run enterprise-scale distributed analytics
over e-commerce Parquet datasets. Implements windowed month-over-month growth metrics,
product category profitability, and customer lifetime revenue attribution.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import (
    CATEGORY_METRICS_OUTPUT,
    CUSTOMER_METRICS_OUTPUT,
    CUSTOMERS_PARQUET_FILE,
    DEFAULT_CONFIG,
    KPI_SUMMARY_OUTPUT,
    MONTHLY_SALES_OUTPUT,
    ORDERS_PARQUET_FILE,
    PRODUCTS_PARQUET_FILE,
    TOP_PRODUCTS_OUTPUT,
    setup_logger,
)

logger: logging.Logger = setup_logger("spark_analytics", level=logging.INFO)


class SalesAnalytics:
    """PySpark Sales Analytics Engine.

    Configures an optimized SparkSession for local execution with 4GB memory,
    Adaptive Query Execution (AQE), and Kryo serialization. Provides analytics
    methods for revenue aggregation, customer ranking, category breakdown,
    and windowed month-over-month growth calculations.

    Attributes:
        spark (SparkSession): Active Apache Spark session.
    """

    def __init__(self, spark: Optional[SparkSession] = None) -> None:
        """Initialize SalesAnalytics with an active or newly created SparkSession.

        Args:
            spark: Existing SparkSession. If None, one is configured and initialized
                using `create_spark_session()`.
        """
        self.spark: SparkSession = spark or self.create_spark_session()

    @staticmethod
    def create_spark_session(
        app_name: str = "EcommerceSalesAnalytics",
        master: str = "local[*]",
    ) -> SparkSession:
        """Configure and return an optimized SparkSession for local analytics.

        Configured settings:
          - 4GB Driver & Executor Memory (`spark.driver.memory="4g"`, `spark.executor.memory="4g"`)
          - Adaptive Query Execution enabled (`spark.sql.adaptive.enabled="true"`)
          - Kryo Serializer enabled for high performance (`spark.serializer="org.apache.spark.serializer.KryoSerializer"`)
          - Local multi-threaded execution (`master="local[*]"`)
          - Loopback driver host binding (`127.0.0.1`)

        Args:
            app_name: Application name displayed in Spark UI and metrics.
            master: Master execution target (defaults to 'local[*]').

        Returns:
            SparkSession: Configured active SparkSession instance.
        """
        cfg = DEFAULT_CONFIG
        logger.info(
            "Initializing SparkSession '%s' (Master: %s, Memory: 4GB, AQE: ON, Kryo: ON)...",
            app_name,
            master,
        )

        spark: SparkSession = (
            SparkSession.builder.appName(app_name)
            .master(master)
            # Memory configuration (4GB)
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            # Adaptive Query Execution (AQE)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # High-performance Kryo Serialization
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            # Local mode networking and shuffle tuning
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(cfg.spark_shuffle_partitions))
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

        # Set log level to WARN to reduce console verbosity
        spark.sparkContext.setLogLevel(cfg.spark_log_level)
        logger.info("SparkSession initialized successfully (Spark version: %s)", spark.version)
        return spark

    def load_parquet(self, path: Union[str, Path]) -> DataFrame:
        """Load Parquet files or directories into a PySpark DataFrame.

        Args:
            path: Absolute or relative path to the Parquet source dataset.

        Returns:
            DataFrame: PySpark DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the specified path does not exist on disk.
        """
        resolved_path = Path(path).resolve()
        if not resolved_path.exists():
            error_msg = f"Parquet source path not found: {resolved_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading Parquet dataset from: %s", resolved_path)
        df: DataFrame = self.spark.read.parquet(str(resolved_path))
        logger.info("Loaded '%s' with %d partitions", resolved_path.name, df.rdd.getNumPartitions())
        return df

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Join orders with products, calculate total spend per customer, and return top N.

        Computes:
          - Total gross revenue spent ($)
          - Distinct order transaction count
          - Total product units purchased
          - Average spend per order ($)

        Args:
            orders_df: Orders DataFrame containing 'order_id', 'customer_id',
                'product_id', and 'quantity'.
            products_df: Products catalog DataFrame containing 'product_id' and 'price'.
            n: Number of leading customers to return (default: 10).

        Returns:
            DataFrame: Top N customers sorted descending by total revenue.
        """
        logger.info("Calculating top %d customers by revenue...", n)

        # Join orders with products to compute line-level spend: quantity * price
        joined_df: DataFrame = orders_df.join(
            products_df.select("product_id", "price"),
            on="product_id",
            how="inner",
        ).withColumn("line_revenue", F.col("quantity") * F.col("price"))

        top_customers_df: DataFrame = (
            joined_df.groupBy("customer_id")
            .agg(
                F.round(F.sum("line_revenue"), 2).alias("total_revenue"),
                F.countDistinct("order_id").alias("total_orders"),
                F.sum("quantity").alias("total_units_purchased"),
                F.round(F.avg("line_revenue"), 2).alias("avg_order_spend"),
            )
            .orderBy(F.col("total_revenue").desc())
            .limit(n)
        )

        return top_customers_df

    def sales_by_category(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Group by product category and sum total revenue and units sold.

        Computes:
          - Total revenue per category ($)
          - Total quantity of units sold
          - Total distinct orders placed
          - Average product catalog price in category ($)

        Args:
            orders_df: Orders DataFrame containing 'product_id' and 'quantity'.
            products_df: Products catalog DataFrame containing 'product_id',
                'category', and 'price'.

        Returns:
            DataFrame: Category sales aggregated and ordered descending by revenue.
        """
        logger.info("Computing sales breakdown by product category...")

        joined_df: DataFrame = orders_df.join(
            products_df.select("product_id", "category", "price"),
            on="product_id",
            how="inner",
        ).withColumn("line_revenue", F.col("quantity") * F.col("price"))

        category_sales_df: DataFrame = (
            joined_df.groupBy("category")
            .agg(
                F.round(F.sum("line_revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("total_units_sold"),
                F.count("order_id").alias("total_orders"),
                F.round(F.avg("price"), 2).alias("avg_product_price"),
            )
            .orderBy(F.col("total_revenue").desc())
        )

        return category_sales_df

    def monthly_trends(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Calculate month-over-month (MoM) revenue growth percentage using Window functions.

        Groups transactions by calendar month (yyyy-MM), computes monthly revenue,
        and applies a Window specification with `F.lag` to compute the revenue delta
        and percentage growth rate relative to the preceding month.

        Formula:
            MoM Growth (%) = ((Current Month Revenue - Previous Month Revenue) / Previous Month Revenue) * 100

        Args:
            orders_df: Orders DataFrame containing 'order_date', 'product_id', and 'quantity'.
            products_df: Products catalog DataFrame containing 'product_id' and 'price'.

        Returns:
            DataFrame: Chronological monthly summary containing:
              - order_month: Month formatted as 'yyyy-MM'
              - monthly_revenue: Total revenue in month ($)
              - previous_month_revenue: Revenue from prior month ($)
              - revenue_growth_pct: Month-over-month growth percentage (%)
              - total_units_sold: Units sold in month
              - total_orders: Total orders placed
        """
        logger.info("Computing monthly trends and month-over-month growth percentages...")

        # 1. Join orders and products, extract calendar year-month, compute line total
        joined_df: DataFrame = (
            orders_df.join(
                products_df.select("product_id", "price"),
                on="product_id",
                how="inner",
            )
            .withColumn("line_revenue", F.col("quantity") * F.col("price"))
            .withColumn(
                "order_month",
                F.date_format(F.to_date("order_date", "yyyy-MM-dd"), "yyyy-MM"),
            )
        )

        # 2. Monthly aggregates
        monthly_agg: DataFrame = (
            joined_df.groupBy("order_month")
            .agg(
                F.round(F.sum("line_revenue"), 2).alias("monthly_revenue"),
                F.sum("quantity").alias("total_units_sold"),
                F.count("order_id").alias("total_orders"),
            )
        )

        # 3. Window specification ordered chronologically by month
        window_spec: Window = Window.orderBy("order_month")

        # 4. Lag function to retrieve previous month revenue
        prev_month_revenue_col = F.lag("monthly_revenue", 1).over(window_spec)

        # 5. Month-over-Month growth percentage calculation
        growth_pct_col = F.when(
            prev_month_revenue_col.isNull(),
            F.lit(None).cast("double"),
        ).otherwise(
            F.round(
                ((F.col("monthly_revenue") - prev_month_revenue_col) / prev_month_revenue_col) * 100.0,
                2,
            )
        )

        trends_df: DataFrame = (
            monthly_agg.withColumn("previous_month_revenue", prev_month_revenue_col)
            .withColumn("revenue_growth_pct", growth_pct_col)
            .orderBy("order_month")
        )

        return trends_df

    def stop(self) -> None:
        """Safely terminate the active SparkSession."""
        if self.spark:
            logger.info("Stopping SparkSession...")
            self.spark.stop()


# ==============================================================================
# Helper Functions & Pipeline Orchestration
# ==============================================================================
def save_dataframe(
    df: DataFrame,
    output_path: Path,
    single_file: bool = True,
    file_format: str = "csv",
) -> None:
    """Save a PySpark DataFrame to disk as CSV or Parquet.

    Attempts native distributed write first; falls back to pandas-based export
    if native Hadoop Windows binaries (winutils.exe) are absent.

    Args:
        df: PySpark DataFrame to export.
        output_path: Target directory path for output.
        single_file: If True, coalesces partitions to 1 for unified export.
        file_format: Export format ('csv' or 'parquet').
    """
    logger.info("Saving analytical results to: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        target_df = df.coalesce(1) if single_file else df
        if file_format == "parquet":
            target_df.write.mode("overwrite").parquet(str(output_path))
        else:
            target_df.write.mode("overwrite").option("header", "true").csv(str(output_path))
        logger.debug("Successfully saved to %s via Spark native writer", output_path)
    except Exception as exc:
        logger.warning(
            "Spark native write encountered an environment limitation (%s). "
            "Falling back to pandas export for Windows compatibility...",
            exc,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        pdf = df.toPandas()
        if file_format == "parquet":
            pdf.to_parquet(output_path / "results.parquet", index=False, engine="pyarrow")
        else:
            pdf.to_csv(output_path / "results.csv", index=False)
        logger.info("Successfully exported analytical results via pandas fallback to %s", output_path)


def run_pipeline() -> None:
    """Execute the complete SalesAnalytics PySpark Pipeline over Parquet data."""
    logger.info("==================================================================")
    logger.info("          Starting PySpark SalesAnalytics Pipeline               ")
    logger.info("==================================================================")

    analytics = SalesAnalytics()

    try:
        # 1. Load Parquet datasets (created by main.py)
        # Fallback to CSV if Parquet files do not yet exist
        if ORDERS_PARQUET_FILE.exists() and PRODUCTS_PARQUET_FILE.exists():
            orders_df: DataFrame = analytics.load_parquet(ORDERS_PARQUET_FILE)
            products_df: DataFrame = analytics.load_parquet(PRODUCTS_PARQUET_FILE)
        else:
            from src.config import ORDERS_FILE, PRODUCTS_FILE
            logger.warning("Parquet files not found. Falling back to CSVs...")
            orders_df = analytics.spark.read.option("header", "true").option("inferSchema", "true").csv(str(ORDERS_FILE))
            products_df = analytics.spark.read.option("header", "true").option("inferSchema", "true").csv(str(PRODUCTS_FILE))

        # 2. Top Customers by Revenue (Pareto spenders)
        print("\n" + "=" * 80)
        print("                  TOP 10 CUSTOMERS BY REVENUE                   ")
        print("=" * 80)
        top_cust_df = analytics.top_customers_by_revenue(orders_df, products_df, n=10)
        top_cust_df.show(truncate=False)
        save_dataframe(top_cust_df, CUSTOMER_METRICS_OUTPUT)

        # 3. Sales by Category
        print("\n" + "=" * 80)
        print("                  SALES BY PRODUCT CATEGORY                     ")
        print("=" * 80)
        cat_sales_df = analytics.sales_by_category(orders_df, products_df)
        cat_sales_df.show(truncate=False)
        save_dataframe(cat_sales_df, CATEGORY_METRICS_OUTPUT)

        # 4. Monthly Trends & MoM Growth
        print("\n" + "=" * 80)
        print("            MONTHLY REVENUE TRENDS & MOM GROWTH (%)             ")
        print("=" * 80)
        trends_df = analytics.monthly_trends(orders_df, products_df)
        trends_df.show(truncate=False)
        save_dataframe(trends_df, MONTHLY_SALES_OUTPUT)

        logger.info("==================================================================")
        logger.info("      SalesAnalytics PySpark Job Completed Successfully!         ")
        logger.info("==================================================================")

    finally:
        analytics.stop()


if __name__ == "__main__":
    run_pipeline()
