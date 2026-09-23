#!/usr/bin/env python3
"""
Script to deploy a model from the Model Registry to a real-time endpoint.
"""
import argparse
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import load_config, validate_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_latest_approved_model_package(sm_client, model_package_group_name):
    """Get the latest approved model package from the Model Registry."""
    try:
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
        )

        if not response["ModelPackageSummaryList"]:
            logger.warning("No approved model packages found. Looking for PendingManualApproval...")
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="PendingManualApproval",
                SortBy="CreationTime",
                SortOrder="Descending",
            )

        if not response["ModelPackageSummaryList"]:
            raise ValueError(f"No model packages found in group: {model_package_group_name}")

        model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
        approval_status = response["ModelPackageSummaryList"][0]["ModelApprovalStatus"]

        logger.info(f"Found model package: {model_package_arn}")
        logger.info(f"Approval status: {approval_status}")

        return model_package_arn

    except ClientError as e:
        raise Exception(f"Failed to get model package: {e}")


def create_model(sm_client, model_name, model_package_arn, role_arn):
    """Create a SageMaker model from a model package."""
    try:
        logger.info(f"Creating model: {model_name}")
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": model_package_arn},
            ExecutionRoleArn=role_arn,
        )
        logger.info(f"✓ Model created: {model_name}")
        return model_name
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException" and "already exists" in str(e):
            logger.info(f"Model {model_name} already exists, using existing model")
            return model_name
        raise Exception(f"Failed to create model: {e}")


def create_endpoint_config(sm_client, endpoint_config_name, model_name, instance_type, instance_count):
    """Create an endpoint configuration."""
    try:
        logger.info(f"Creating endpoint config: {endpoint_config_name}")
        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": instance_count,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1,
                }
            ],
        )
        logger.info(f"✓ Endpoint config created: {endpoint_config_name}")
        return endpoint_config_name
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException" and "already exists" in str(e):
            logger.info(f"Endpoint config {endpoint_config_name} already exists, using existing config")
            return endpoint_config_name
        raise Exception(f"Failed to create endpoint config: {e}")


def create_or_update_endpoint(sm_client, endpoint_name, endpoint_config_name):
    """Create or update an endpoint."""
    try:
        try:
            response = sm_client.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} exists, updating...")
            sm_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
            logger.info(f"✓ Endpoint update initiated: {endpoint_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.info(f"Creating new endpoint: {endpoint_name}")
                sm_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
                logger.info(f"✓ Endpoint creation initiated: {endpoint_name}")
            else:
                raise

        logger.info("\nEndpoint is being created/updated. This may take 5-10 minutes.")
        logger.info("Monitor progress in AWS Console → SageMaker → Endpoints")
        return endpoint_name

    except ClientError as e:
        raise Exception(f"Failed to create/update endpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker endpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--model-package-arn",
        type=str,
        help="Model package ARN (optional, defaults to latest approved model)",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading configuration...")
        config = load_config(args.config)
        validate_config(config)

        sm_client = boto3.client("sagemaker", region_name=config["aws_region"])

        if args.model_package_arn:
            model_package_arn = args.model_package_arn
        else:
            logger.info("Fetching latest approved model package...")
            model_package_arn = get_latest_approved_model_package(sm_client, config["model_package_group_name"])

        import time

        timestamp = int(time.time())
        model_name = f"{config['pipeline_name']}-model-{timestamp}"
        endpoint_config_name = f"{config['pipeline_name']}-config-{timestamp}"
        endpoint_name = config["endpoint_name"]

        create_model(sm_client, model_name, model_package_arn, config["sagemaker_role_arn"])

        create_endpoint_config(
            sm_client,
            endpoint_config_name,
            model_name,
            config["endpoint_instance_type"],
            config["endpoint_instance_count"],
        )

        create_or_update_endpoint(sm_client, endpoint_name, endpoint_config_name)

        logger.info("\n✓ Deployment initiated successfully!")
        logger.info(f"Endpoint name: {endpoint_name}")
        logger.info(f"Model package: {model_package_arn}")

    except Exception as e:
        logger.error(f"✗ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
