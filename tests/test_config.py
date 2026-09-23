"""Tests for pipeline configuration."""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline.config import load_config, validate_config


def test_load_config_from_file():
    """Test loading configuration from YAML file."""
    config_data = {
        "aws_region": "us-west-2",
        "sagemaker_role_arn": "arn:aws:iam::999999999999:role/TestRole",
        "s3_bucket": "test-bucket",
        "pipeline_name": "test-pipeline",
        "model_package_group_name": "test-models",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = load_config(config_path)
        assert config["aws_region"] == "us-west-2"
        assert config["pipeline_name"] == "test-pipeline"
    finally:
        os.unlink(config_path)


def test_validate_config_success():
    """Test configuration validation with valid config."""
    config = {
        "aws_region": "us-west-2",
        "sagemaker_role_arn": "arn:aws:iam::999999999999:role/TestRole",
        "s3_bucket": "test-bucket",
        "pipeline_name": "test-pipeline",
        "model_package_group_name": "test-models",
    }
    validate_config(config)


def test_validate_config_missing_field():
    """Test configuration validation with missing required field."""
    config = {
        "aws_region": "us-west-2",
        "s3_bucket": "test-bucket",
    }

    with pytest.raises(ValueError, match="Required configuration field"):
        validate_config(config)


def test_validate_config_placeholder_values():
    """Test configuration validation rejects placeholder values."""
    config = {
        "aws_region": "us-west-2",
        "sagemaker_role_arn": "arn:aws:iam::123456789012:role/TestRole",
        "s3_bucket": "test-bucket",
        "pipeline_name": "test-pipeline",
        "model_package_group_name": "test-models",
    }

    with pytest.raises(ValueError, match="Please update the SageMaker role ARN"):
        validate_config(config)
