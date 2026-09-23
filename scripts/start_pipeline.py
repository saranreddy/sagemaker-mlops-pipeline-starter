#!/usr/bin/env python3
"""
Script to start a SageMaker Pipeline execution.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import CaliforniaHousingPipeline, load_config, validate_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Start SageMaker Pipeline execution")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading configuration...")
        config = load_config(args.config)

        logger.info("Validating configuration...")
        validate_config(config)

        logger.info("Creating pipeline definition...")
        pipeline_manager = CaliforniaHousingPipeline(config)

        logger.info("Starting pipeline execution...")
        execution = pipeline_manager.start_pipeline()

        logger.info("✓ Pipeline execution started successfully!")
        logger.info(f"Execution ARN: {execution.arn}")
        logger.info(f"Execution ID: {execution._execution_id}")

        logger.info("\nTo monitor the execution:")
        logger.info("  1. Go to AWS Console → SageMaker → Pipelines")
        logger.info(f"  2. Select pipeline: {config['pipeline_name']}")
        logger.info(f"  3. View execution: {execution._execution_id}")

    except Exception as e:
        logger.error(f"✗ Failed to start pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
