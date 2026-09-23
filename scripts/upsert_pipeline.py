#!/usr/bin/env python3
"""
Script to upsert (create or update) the SageMaker Pipeline.
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
    parser = argparse.ArgumentParser(description="Upsert SageMaker Pipeline")
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

        logger.info("Upserting pipeline to SageMaker...")
        pipeline = pipeline_manager.upsert_pipeline()

        logger.info("✓ Pipeline upserted successfully!")
        logger.info(f"Pipeline name: {config['pipeline_name']}")
        logger.info(f"Region: {config['aws_region']}")

        logger.info("\nTo start the pipeline, run:")
        logger.info(f"  python scripts/start_pipeline.py --config {args.config}")

    except Exception as e:
        logger.error(f"✗ Failed to upsert pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
