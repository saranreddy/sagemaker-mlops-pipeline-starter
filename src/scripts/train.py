"""
Training script for XGBoost model on California Housing dataset.
"""
import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--num-round", type=int, default=100)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    args = parser.parse_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    validation_dir = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    logger.info("Loading training data...")
    train_df = pd.read_csv(os.path.join(train_dir, "train.csv"), header=None)
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    logger.info("Loading validation data...")
    val_df = pd.read_csv(os.path.join(validation_dir, "validation.csv"), header=None)
    X_val = val_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, -1].values

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": "rmse",
    }

    logger.info(f"Training with params: {params}")

    evallist = [(dtrain, "train"), (dval, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    train_pred = model.predict(dtrain)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    val_pred = model.predict(dval)
    val_mse = mean_squared_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)

    logger.info(f"Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    logger.info(f"Validation MSE: {val_mse:.4f}, R²: {val_r2:.4f}")

    model_path = os.path.join(model_dir, "xgboost-model")
    logger.info(f"Saving model to {model_path}")
    model.save_model(model_path)

    metrics = {
        "train_mse": float(train_mse),
        "train_r2": float(train_r2),
        "validation_mse": float(val_mse),
        "validation_r2": float(val_r2),
    }

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
