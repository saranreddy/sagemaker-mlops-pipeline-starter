"""
Model evaluation script for test dataset.
"""

import argparse
import json
import logging
import os
import tarfile

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mse-threshold", type=float, default=0.5)
    args = parser.parse_args()

    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.csv"
    output_dir = "/opt/ml/processing/evaluation"

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Extracting model from {model_path}")
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall("/tmp/model")

    model = xgb.Booster()
    model.load_model("/tmp/model/xgboost-model")

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, header=None)
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)

    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test R²: {r2:.4f}")

    pass_threshold = mse < args.mse_threshold
    logger.info(f"MSE threshold: {args.mse_threshold}")
    logger.info(f"Pass threshold: {pass_threshold}")

    evaluation_result = {
        "regression_metrics": {
            "mse": {"value": float(mse)},
            "rmse": {"value": float(rmse)},
            "mae": {"value": float(mae)},
            "r2": {"value": float(r2)},
        },
        "threshold_metrics": {
            "mse_threshold": {"value": float(args.mse_threshold)},
            "pass_threshold": {"value": pass_threshold},
        },
    }

    output_path = os.path.join(output_dir, "evaluation.json")
    logger.info(f"Writing evaluation results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(evaluation_result, f, indent=2)

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
