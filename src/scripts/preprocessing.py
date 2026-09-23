"""
Data preprocessing script for California Housing dataset.
Splits data into train, validation, and test sets.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    logger.info("Splitting data into train, validation, and test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    val_size_adjusted = args.validation_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=args.random_state
    )

    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Validation set size: {X_val.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")

    feature_names = housing.feature_names

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["target"] = y_train

    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df["target"] = y_val

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["target"] = y_test

    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")

    logger.info(f"Saving train data to {train_path}")
    train_df.to_csv(train_path, index=False, header=False)

    logger.info(f"Saving validation data to {val_path}")
    val_df.to_csv(val_path, index=False, header=False)

    logger.info(f"Saving test data to {test_path}")
    test_df.to_csv(test_path, index=False, header=False)

    logger.info("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
