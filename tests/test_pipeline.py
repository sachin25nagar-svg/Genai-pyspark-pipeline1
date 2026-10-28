"""Unit tests for the E-Commerce Data Pipeline & SyntheticDataGenerator.

Validates:
  - SyntheticDataGenerator class instantiation & defaults
  - Customer schema, Faker data quality, and Normal age distribution
  - Product catalog schema, categories, pricing, and ratings
  - Order schema and Pareto 80/20 customer distribution
"""

from typing import List
import unittest

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    PROJECT_ROOT,
)
from src.data_generator import (
    PRODUCT_CATEGORIES,
    SyntheticDataGenerator,
)


class TestConfig(unittest.TestCase):
    """Test configuration paths and defaults."""

    def test_project_directories_exist(self) -> None:
        """Verify essential project directory structure exists."""
        self.assertTrue(PROJECT_ROOT.exists(), "Project root directory must exist.")
        self.assertTrue(DATA_RAW_DIR.exists(), "Raw data directory must exist.")
        self.assertTrue(DATA_PROCESSED_DIR.exists(), "Processed data directory must exist.")


class TestSyntheticDataGenerator(unittest.TestCase):
    """Unit test suite for SyntheticDataGenerator."""

    def setUp(self) -> None:
        """Initialize a deterministic generator instance for unit testing."""
        self.generator = SyntheticDataGenerator(
            num_customers=500,
            num_products=50,
            num_orders=2_000,
            seed=42,
        )

    def test_generator_initialization_defaults(self) -> None:
        """Verify default parameters for 100K customers, 10K products, 1M orders."""
        default_gen = SyntheticDataGenerator()
        self.assertEqual(default_gen.num_customers, 100_000)
        self.assertEqual(default_gen.num_products, 10_000)
        self.assertEqual(default_gen.num_orders, 1_000_000)

    def test_generate_customers_schema_and_distribution(self) -> None:
        """Verify customer table schema, non-null values, and normal age distribution."""
        count: int = 1_000
        df: pd.DataFrame = self.generator.generate_customers(count=count)

        # 1. Row count and columns
        self.assertEqual(len(df), count)
        expected_cols: List[str] = [
            "customer_id",
            "name",
            "email",
            "age",
            "city",
            "country",
            "registration_date",
        ]
        self.assertEqual(list(df.columns), expected_cols)

        # 2. Key uniqueness and format
        self.assertTrue(df["customer_id"].is_unique)
        self.assertTrue(df["customer_id"].str.startswith("CUST_").all())
        self.assertFalse(df.isnull().values.any(), "Customer DataFrame should contain no nulls.")

        # 3. Age statistical distribution (normal around 35)
        mean_age = df["age"].mean()
        self.assertTrue(32.0 <= mean_age <= 38.0, f"Mean age {mean_age} should be near 35")
        self.assertTrue((df["age"] >= 18).all(), "All ages should be >= 18")
        self.assertTrue((df["age"] <= 80).all(), "All ages should be <= 80")

        # 4. Email validation
        self.assertTrue(df["email"].str.contains("@").all(), "All emails must contain '@'")

    def test_generate_products_schema_and_constraints(self) -> None:
        """Verify product catalog schema, price boundaries (10-500), and ratings (1-5)."""
        count: int = 200
        df: pd.DataFrame = self.generator.generate_products(count=count)

        self.assertEqual(len(df), count)
        expected_cols: List[str] = [
            "product_id",
            "name",
            "category",
            "price",
            "stock",
            "rating",
        ]
        self.assertEqual(list(df.columns), expected_cols)

        # Product IDs
        self.assertTrue(df["product_id"].is_unique)
        self.assertTrue(df["product_id"].str.startswith("PROD_").all())

        # Category constraint
        self.assertTrue(set(df["category"]).issubset(set(PRODUCT_CATEGORIES)))

        # Price range (10 - 500)
        self.assertTrue((df["price"] >= 10.0).all(), "All prices must be >= 10.0")
        self.assertTrue((df["price"] <= 500.0).all(), "All prices must be <= 500.0")

        # Stock range
        self.assertTrue((df["stock"] >= 0).all(), "Stock cannot be negative")

        # Rating range (1 - 5)
        self.assertTrue((df["rating"] >= 1.0).all(), "Ratings must be >= 1.0")
        self.assertTrue((df["rating"] <= 5.0).all(), "Ratings must be <= 5.0")

    def test_generate_orders_pareto_distribution(self) -> None:
        """Verify order generation and Pareto distribution (20% customers ~ 80% orders)."""
        cust_df = self.generator.generate_customers(count=500)
        prod_df = self.generator.generate_products(count=50)
        orders_count = 5_000

        orders_df: pd.DataFrame = self.generator.generate_orders(
            customers_df=cust_df,
            products_df=prod_df,
            count=orders_count,
        )

        self.assertEqual(len(orders_df), orders_count)
        expected_cols: List[str] = [
            "order_id",
            "customer_id",
            "product_id",
            "quantity",
            "order_date",
        ]
        self.assertEqual(list(df_col := list(orders_df.columns)), expected_cols)

        # Referential integrity
        valid_cust_ids = set(cust_df["customer_id"])
        valid_prod_ids = set(prod_df["product_id"])
        self.assertTrue(set(orders_df["customer_id"]).issubset(valid_cust_ids))
        self.assertTrue(set(orders_df["product_id"]).issubset(valid_prod_ids))

        # Quantity constraint (1 - 10)
        self.assertTrue((orders_df["quantity"] >= 1).all())
        self.assertTrue((orders_df["quantity"] <= 10).all())

        # Validate Pareto concentration (Top 20% should account for 70% - 90% of orders)
        order_counts = orders_df["customer_id"].value_counts().to_numpy()
        top_20_pct_cust = int(0.20 * len(cust_df))
        top_20_share = (order_counts[:top_20_pct_cust].sum() / orders_count) * 100.0

        self.assertTrue(
            70.0 <= top_20_share <= 90.0,
            f"Expected Pareto 80/20 share near 80%, got {top_20_share:.2f}%",
        )


if __name__ == "__main__":
    unittest.main()
