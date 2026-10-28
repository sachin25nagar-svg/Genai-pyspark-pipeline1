"""Synthetic E-Commerce Data Generator module.

Provides the `SyntheticDataGenerator` class to generate enterprise-scale,
statistically realistic mock e-commerce datasets using Faker for demographic text
and NumPy for statistical modeling (Normal distributions for ages, Pareto
distributions for the 80/20 customer ordering pattern).
"""

from datetime import datetime
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple, Union

from faker import Faker
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    CUSTOMERS_FILE,
    DATA_RAW_DIR,
    DEFAULT_CONFIG,
    ORDERS_FILE,
    PRODUCTS_FILE,
    setup_logger,
)

logger: logging.Logger = setup_logger("data_generator", level=logging.INFO)

# Supported product catalog categories
PRODUCT_CATEGORIES: List[str] = [
    "Electronics",
    "Clothing",
    "Home",
    "Sports",
    "Books",
]

# Vocabulary templates for generating diverse and realistic product names
PRODUCT_NAME_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "Electronics": {
        "adjectives": ["Wireless", "Smart", "Ultra", "4K", "Bluetooth", "Noise-Canceling", "Portable", "Pro", "Mechanical", "Ergonomic"],
        "nouns": ["Headphones", "Earbuds", "Keyboard", "Gaming Mouse", "Smartwatch", "Power Bank", "USB Hub", "Speaker", "Monitor", "Webcam"],
    },
    "Clothing": {
        "adjectives": ["Classic", "Slim-Fit", "Cotton", "Breathable", "Water-Resistant", "Fleece", "Merino Wool", "Casual", "Athletic", "Vintage"],
        "nouns": ["T-Shirt", "Denim Jeans", "Zip Hoodie", "Running Shoes", "Rain Jacket", "Knit Sweater", "Polo Shirt", "Track Pants", "Sneakers", "Shorts"],
    },
    "Home": {
        "adjectives": ["Stainless Steel", "Cast Iron", "Ceramic", "Non-Stick", "Microfiber", "Programmable", "Digital", "Ergonomic", "Organic", "Modern"],
        "nouns": ["Chef Knife", "Dutch Oven", "Cookware Set", "Coffee Maker", "Blender", "Air Purifier", "Bed Sheet Set", "Desk Lamp", "Storage Bins", "Diffuser"],
    },
    "Sports": {
        "adjectives": ["Non-Slip", "Adjustable", "Insulated", "Elastic", "Speed", "High-Density", "Heavy-Duty", "Breathable", "Pro-Grip", "All-Weather"],
        "nouns": ["Yoga Mat", "Dumbbell Set", "Water Bottle", "Resistance Bands", "Jump Rope", "Foam Roller", "Kettlebell", "Gym Bag", "Gloves", "Running Belt"],
    },
    "Books": {
        "adjectives": ["Mastering", "The Art of", "Modern", "Essential", "Complete Guide to", "Designing", "Practical", "Hands-On", "Advanced", "The Lean"],
        "nouns": ["Data Engineering", "Machine Learning", "System Design", "Cloud Computing", "Clean Code", "Python Mastery", "DevOps", "AI Architectures", "Startup Manual", "Algorithms"],
    },
}


class SyntheticDataGenerator:
    """High-performance synthetic e-commerce data generator.

    Generates relational datasets for customers, products, and customer orders.
    Employs NumPy statistical distributions to simulate real-world behaviors:
      - Normal distribution for customer ages (centered around 35).
      - Pareto distribution for order generation (Pareto 80/20 principle:
        approximately 20% of customers generate 80% of orders).

    Attributes:
        num_customers (int): Default number of customer profiles (100,000).
        num_products (int): Default number of product catalog entries (10,000).
        num_orders (int): Default number of order transactions (1,000,000).
        seed (Optional[int]): Random seed for deterministic reproducibility.
        pareto_alpha (float): Shape parameter for Pareto distribution (~1.38 yields ~80/20 split).
    """

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: Optional[int] = 42,
        pareto_alpha: float = 1.38,
    ) -> None:
        """Initialize the synthetic generator with dataset sizes and random seeds.

        Args:
            num_customers: Default customer count to generate (100K by default).
            num_products: Default product count to generate (10K by default).
            num_orders: Default orders count to generate (1M by default).
            seed: Random seed for reproducibility.
            pareto_alpha: Shape parameter for Pareto customer order distribution.
        """
        self.num_customers: int = num_customers
        self.num_products: int = num_products
        self.num_orders: int = num_orders
        self.seed: Optional[int] = seed
        self.pareto_alpha: float = pareto_alpha

        self.fake: Faker = Faker("en_US")
        if self.seed is not None:
            self._set_seeds(self.seed)

        logger.info(
            "SyntheticDataGenerator initialized (Defaults: %s customers, %s products, %s orders, seed=%s)",
            f"{self.num_customers:,}",
            f"{self.num_products:,}",
            f"{self.num_orders:,}",
            self.seed,
        )

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds across Python, Faker, and NumPy.

        Args:
            seed: Integer seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

    def generate_customers(self, count: Optional[int] = None) -> pd.DataFrame:
        """Generate a DataFrame of synthetic customer profiles.

        Attributes generated:
          - customer_id: Formatted unique ID string (e.g., 'CUST_0000001').
          - name: Realistically generated full name via Faker.
          - email: Unique email address corresponding to customer name via Faker.
          - age: Normal distribution centered around 35 years (std=10, clipped [18, 80]).
          - city: Urban/city name generated via Faker.
          - country: Country name generated via Faker.
          - registration_date: Realistic historical signup date.

        Args:
            count: Number of customers to generate. Defaults to `self.num_customers`.

        Returns:
            pd.DataFrame: Table of customer profiles.
        """
        n: int = count if count is not None else self.num_customers
        logger.info("Starting generation of %s customers...", f"{n:,}")

        # 1. Statistical age distribution: Normal(loc=35, scale=10), clipped to [18, 80]
        ages: np.ndarray = np.clip(
            np.random.normal(loc=35.0, scale=10.0, size=n).round().astype(int),
            18,
            80,
        )

        # 2. Vectorized Registration Dates (sampled between 2021-01-01 and 2025-12-31)
        start_ts: int = pd.Timestamp("2021-01-01").value
        end_ts: int = pd.Timestamp("2025-12-31").value
        random_timestamps: np.ndarray = np.random.randint(
            start_ts, end_ts, size=n, dtype=np.int64
        )
        registration_dates: List[str] = (
            pd.to_datetime(random_timestamps).strftime("%Y-%m-%d").tolist()
        )

        # 3. ID generation
        customer_ids: List[str] = [f"CUST_{i:07d}" for i in range(1, n + 1)]

        # 4. Generate Names and Emails via Faker with progress bar
        batch_size: int = 10_000
        names: List[str] = []
        emails: List[str] = []
        cities: List[str] = []
        countries: List[str] = []

        # Pre-seed geographic pools for fast realistic sampling across large scale
        geo_pool_size: int = min(2_000, n)
        city_pool: List[str] = [self.fake.city() for _ in range(geo_pool_size)]
        country_pool: List[str] = [self.fake.country() for _ in range(50)]
        email_domains: List[str] = [
            "gmail.com",
            "yahoo.com",
            "outlook.com",
            "icloud.com",
            "hotmail.com",
            "proton.me",
        ]

        with tqdm(total=n, desc="Customers [Faker Names & Emails]", unit="cust", file=sys.stdout) as pbar:
            for start_idx in range(0, n, batch_size):
                chunk_len: int = min(batch_size, n - start_idx)
                for _ in range(chunk_len):
                    first = self.fake.first_name()
                    last = self.fake.last_name()
                    names.append(f"{first} {last}")
                    domain = random.choice(email_domains)
                    clean_first = first.lower().replace("'", "").replace(" ", "")
                    clean_last = last.lower().replace("'", "").replace(" ", "")
                    emails.append(f"{clean_first}.{clean_last}{random.randint(10, 9999)}@{domain}")

                pbar.update(chunk_len)

        # Sample cities and countries using NumPy
        sampled_cities: np.ndarray = np.random.choice(city_pool, size=n)
        sampled_countries: np.ndarray = np.random.choice(country_pool, size=n)

        customers_df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "name": names,
                "email": emails,
                "age": ages,
                "city": sampled_cities,
                "country": sampled_countries,
                "registration_date": registration_dates,
            }
        )

        logger.info(
            "Completed customers generation: %s rows, Mean Age: %.1f",
            f"{len(customers_df):,}",
            customers_df["age"].mean(),
        )
        return customers_df

    def generate_products(self, count: Optional[int] = None) -> pd.DataFrame:
        """Generate a DataFrame of synthetic catalog products.

        Attributes generated:
          - product_id: Formatted unique ID string (e.g., 'PROD_00001').
          - name: Descriptive product name matching category vocabulary.
          - category: One of ['Electronics', 'Clothing', 'Home', 'Sports', 'Books'].
          - price: Continuous float price between $10.00 and $500.00.
          - stock: Available inventory quantity (0 - 1,000 units).
          - rating: Customer satisfaction rating (1.0 - 5.0 stars).

        Args:
            count: Number of products to generate. Defaults to `self.num_products`.

        Returns:
            pd.DataFrame: Table of catalog products.
        """
        n: int = count if count is not None else self.num_products
        logger.info("Starting generation of %s products...", f"{n:,}")

        # 1. Product IDs
        product_ids: List[str] = [f"PROD_{i:05d}" for i in range(1, n + 1)]

        # 2. Categories
        categories: np.ndarray = np.random.choice(PRODUCT_CATEGORIES, size=n)

        # 3. Prices (Uniform float between 10.00 and 500.00)
        prices: np.ndarray = np.round(np.random.uniform(10.0, 500.0, size=n), 2)

        # 4. Stock quantities (Integer between 0 and 1000)
        stocks: np.ndarray = np.random.randint(0, 1001, size=n)

        # 5. Ratings (1.0 to 5.0 stars with normal distribution skewed towards 4.1)
        raw_ratings = np.random.normal(loc=4.1, scale=0.6, size=n)
        ratings: np.ndarray = np.round(np.clip(raw_ratings, 1.0, 5.0), 1)

        # 6. Generate Descriptive Names by Category with progress tracking
        names: List[str] = []
        with tqdm(total=n, desc="Products [Catalog Assembly]", unit="prod", file=sys.stdout) as pbar:
            for i, cat in enumerate(categories):
                templates = PRODUCT_NAME_TEMPLATES[cat]
                adj = random.choice(templates["adjectives"])
                noun = random.choice(templates["nouns"])
                # Disambiguate name with ID suffix if catalog is large
                suffix = f" {i + 1}" if n > 1_000 else ""
                names.append(f"{adj} {noun}{suffix}")
                pbar.update(1)

        products_df = pd.DataFrame(
            {
                "product_id": product_ids,
                "name": names,
                "category": categories,
                "price": prices,
                "stock": stocks,
                "rating": ratings,
            }
        )

        logger.info(
            "Completed products generation: %s rows across %d categories (Mean price: $%.2f)",
            f"{len(products_df):,}",
            len(PRODUCT_CATEGORIES),
            products_df["price"].mean(),
        )
        return products_df

    def generate_orders(
        self,
        customers_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate e-commerce order transactions modeling the 80/20 Pareto rule.

        Simulates purchasing behavior where approximately 20% of customers
        are responsible for ~80% of total transactions using NumPy's Pareto distribution.

        Attributes generated:
          - order_id: Formatted unique ID string (e.g., 'ORD_00000001').
          - customer_id: Customer ID selected according to Pareto probability.
          - product_id: Product ID purchased in this transaction.
          - quantity: Units purchased (integer between 1 and 10).
          - order_date: Order timestamp (YYYY-MM-DD).

        Args:
            customers_df: DataFrame of customers to reference (generated if None).
            products_df: DataFrame of products to reference (generated if None).
            count: Number of orders to generate. Defaults to `self.num_orders`.

        Returns:
            pd.DataFrame: Table of order transactions.
        """
        n_orders: int = count if count is not None else self.num_orders

        # Ensure valid customer and product references exist
        if customers_df is None:
            logger.info("No customers DataFrame supplied; generating %s default customers...", f"{self.num_customers:,}")
            customers_df = self.generate_customers()

        if products_df is None:
            logger.info("No products DataFrame supplied; generating %s default products...", f"{self.num_products:,}")
            products_df = self.generate_products()

        n_customers: int = len(customers_df)
        n_products: int = len(products_df)
        logger.info("Starting generation of %s orders (Pareto Alpha=%.2f)...", f"{n_orders:,}", self.pareto_alpha)

        # ----------------------------------------------------------------------
        # Pareto Distribution: 20% of customers make ~80% of orders
        # ----------------------------------------------------------------------
        # Draw Pareto random variables: shape parameter calibrated for 80/20 split
        pareto_samples: np.ndarray = np.random.pareto(a=self.pareto_alpha, size=n_customers)
        customer_probabilities: np.ndarray = pareto_samples / pareto_samples.sum()

        with tqdm(total=4, desc="Orders [Pareto Sampling & Indexing]", file=sys.stdout) as pbar:
            # Step 1: Sample customer IDs using Pareto distribution weights
            sampled_cust_indices: np.ndarray = np.random.choice(
                n_customers,
                size=n_orders,
                p=customer_probabilities,
            )
            customer_id_array: np.ndarray = customers_df["customer_id"].to_numpy()
            sampled_customer_ids: np.ndarray = customer_id_array[sampled_cust_indices]
            pbar.update(1)

            # Step 2: Sample product IDs uniformly across catalog
            sampled_prod_indices: np.ndarray = np.random.randint(0, n_products, size=n_orders)
            product_id_array: np.ndarray = products_df["product_id"].to_numpy()
            sampled_product_ids: np.ndarray = product_id_array[sampled_prod_indices]
            pbar.update(1)

            # Step 3: Quantities (1 to 10 with realistic exponential weight towards smaller quantities)
            qty_weights: List[float] = [0.45, 0.22, 0.12, 0.08, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005]
            quantities: np.ndarray = np.random.choice(
                np.arange(1, 11),
                size=n_orders,
                p=qty_weights,
            )
            pbar.update(1)

            # Step 4: Vectorized Order Dates (sampled between 2024-01-01 and 2026-06-30)
            start_ts: int = pd.Timestamp("2024-01-01").value
            end_ts: int = pd.Timestamp("2026-06-30").value
            random_timestamps: np.ndarray = np.random.randint(
                start_ts, end_ts, size=n_orders, dtype=np.int64
            )
            order_dates: np.ndarray = (
                pd.to_datetime(random_timestamps).strftime("%Y-%m-%d").to_numpy()
            )
            pbar.update(1)

        # ----------------------------------------------------------------------
        # Validate Pareto 80/20 concentration
        # ----------------------------------------------------------------------
        order_counts_per_cust: np.ndarray = np.bincount(sampled_cust_indices, minlength=n_customers)
        sorted_counts: np.ndarray = np.sort(order_counts_per_cust)[::-1]
        top_20_pct_cust_count: int = max(1, int(0.20 * n_customers))
        top_20_orders_share: float = (
            sorted_counts[:top_20_pct_cust_count].sum() / n_orders
        ) * 100.0

        logger.info(
            "Pareto distribution verification: Top 20%% of customers placed %.2f%% of all orders.",
            top_20_orders_share,
        )

        # Order IDs: 'ORD_00000001'
        order_ids: List[str] = [f"ORD_{i:08d}" for i in range(1, n_orders + 1)]

        orders_df = pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": sampled_customer_ids,
                "product_id": sampled_product_ids,
                "quantity": quantities,
                "order_date": order_dates,
            }
        )

        logger.info("Completed orders generation: %s rows", f"{len(orders_df):,}")
        return orders_df

    def generate_all(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        save_csv: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full generation pipeline and optionally persist to disk.

        Args:
            output_dir: Directory where CSV files should be saved. Defaults to config raw data dir.
            save_csv: Whether to write datasets to CSV files.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                (customers_df, products_df, orders_df)
        """
        target_dir: Path = Path(output_dir) if output_dir is not None else DATA_RAW_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("==================================================================")
        logger.info("           Synthetic E-Commerce Data Generation Pipeline          ")
        logger.info("==================================================================")
        logger.info(
            "Configuration: Customers=%s | Products=%s | Orders=%s",
            f"{self.num_customers:,}",
            f"{self.num_products:,}",
            f"{self.num_orders:,}",
        )

        # 1. Customers
        customers_df = self.generate_customers()
        if save_csv:
            cust_path = target_dir / "customers.csv"
            logger.info("Writing customers to %s...", cust_path)
            customers_df.to_csv(cust_path, index=False)

        # 2. Products
        products_df = self.generate_products()
        if save_csv:
            prod_path = target_dir / "products.csv"
            logger.info("Writing products to %s...", prod_path)
            products_df.to_csv(prod_path, index=False)

        # 3. Orders
        orders_df = self.generate_orders(
            customers_df=customers_df,
            products_df=products_df,
        )
        if save_csv:
            orders_path = target_dir / "orders.csv"
            logger.info("Writing orders to %s (this may take a few seconds)...", orders_path)
            orders_df.to_csv(orders_path, index=False)

        logger.info("==================================================================")
        logger.info("   Generation Complete: All datasets generated successfully!     ")
        logger.info("==================================================================")
        return customers_df, products_df, orders_df


# ==============================================================================
# Functional Compatibility Wrappers (for external script and test integration)
# ==============================================================================
def generate_customers(num_records: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate customer profiles using SyntheticDataGenerator."""
    generator = SyntheticDataGenerator(num_customers=num_records, seed=seed)
    return generator.generate_customers(count=num_records)


def generate_products(num_records: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate product catalog using SyntheticDataGenerator."""
    generator = SyntheticDataGenerator(num_products=num_records, seed=seed)
    return generator.generate_products(count=num_records)


def generate_orders(
    customers_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
    num_records: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate order transactions using SyntheticDataGenerator."""
    generator = SyntheticDataGenerator(num_orders=num_records, seed=seed)
    return generator.generate_orders(
        customers_df=customers_df,
        products_df=products_df,
        count=num_records,
    )


def generate_and_save_all(
    num_customers: int = 100_000,
    num_products: int = 10_000,
    num_orders: int = 1_000_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute pipeline with custom counts and persist to disk."""
    generator = SyntheticDataGenerator(
        num_customers=num_customers,
        num_products=num_products,
        num_orders=num_orders,
        seed=seed,
    )
    return generator.generate_all()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic E-Commerce Data Generator")
    parser.add_argument(
        "--customers",
        type=int,
        default=100_000,
        help="Number of customers to generate (default: 100,000)",
    )
    parser.add_argument(
        "--products",
        type=int,
        default=10_000,
        help="Number of products to generate (default: 10,000)",
    )
    parser.add_argument(
        "--orders",
        type=int,
        default=1_000_000,
        help="Number of orders to generate (default: 1,000,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run quick lightweight preview test (1,000 cust, 100 prod, 5,000 ord)",
    )

    args = parser.parse_args()

    if args.fast:
        logger.info("Executing fast preview run (1,000 customers, 100 products, 5,000 orders)...")
        gen = SyntheticDataGenerator(
            num_customers=1_000,
            num_products=100,
            num_orders=5_000,
            seed=args.seed,
        )
    else:
        gen = SyntheticDataGenerator(
            num_customers=args.customers,
            num_products=args.products,
            num_orders=args.orders,
            seed=args.seed,
        )

    gen.generate_all()
