"""Configuration utilities for SageMaker Pipeline."""
import os
from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or environment variables.

    Args:
        config_path: Path to config YAML file. If None, uses config/config.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Copy config/config.example.yaml to config/config.yaml and customize it."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["aws_region"] = os.environ.get("AWS_REGION", config["aws_region"])
    config["sagemaker_role_arn"] = os.environ.get("SAGEMAKER_ROLE_ARN", config["sagemaker_role_arn"])
    config["s3_bucket"] = os.environ.get("S3_BUCKET", config["s3_bucket"])

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_fields = ["aws_region", "sagemaker_role_arn", "s3_bucket", "pipeline_name", "model_package_group_name"]

    for field in required_fields:
        if field not in config or not config[field]:
            raise ValueError(f"Required configuration field '{field}' is missing or empty")

    if "123456789012" in config["sagemaker_role_arn"]:
        raise ValueError(
            "Please update the SageMaker role ARN in config.yaml with your actual AWS account ID. "
            "You can get this from the Terraform outputs after running 'terraform apply'."
        )

    if "123456789012" in config["s3_bucket"]:
        raise ValueError(
            "Please update the S3 bucket name in config.yaml with your actual AWS account ID. "
            "You can get this from the Terraform outputs after running 'terraform apply'."
        )
