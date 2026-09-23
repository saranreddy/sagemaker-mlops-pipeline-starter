"""Tests for pipeline construction."""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.pipeline import CaliforniaHousingPipeline


@pytest.fixture
def mock_config():
    """Fixture providing test configuration."""
    return {
        "aws_region": "us-west-2",
        "sagemaker_role_arn": "arn:aws:iam::999999999999:role/TestRole",
        "s3_bucket": "test-bucket",
        "pipeline_name": "test-pipeline",
        "model_package_group_name": "test-models",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "mse_threshold": 0.5,
        "framework_version": "1.7-1",
    }


@patch("src.pipeline.pipeline.boto3.Session")
@patch("src.pipeline.pipeline.sagemaker.Session")
@patch("src.pipeline.pipeline.PipelineSession")
@patch("src.pipeline.pipeline.sagemaker.image_uris.retrieve")
def test_pipeline_initialization(mock_retrieve, mock_pipeline_session, mock_session, mock_boto_session, mock_config):
    """Test pipeline initialization."""
    mock_retrieve.return_value = "mock-image-uri"

    pipeline = CaliforniaHousingPipeline(mock_config)

    assert pipeline.config == mock_config
    assert pipeline.region == "us-west-2"
    assert pipeline.role == "arn:aws:iam::999999999999:role/TestRole"
    assert pipeline.bucket == "test-bucket"
    assert pipeline.pipeline_name == "test-pipeline"


@patch("src.pipeline.pipeline.boto3.Session")
@patch("src.pipeline.pipeline.sagemaker.Session")
@patch("src.pipeline.pipeline.PipelineSession")
@patch("src.pipeline.pipeline.sagemaker.image_uris.retrieve")
def test_create_parameters(mock_retrieve, mock_pipeline_session, mock_session, mock_boto_session, mock_config):
    """Test pipeline parameter creation."""
    mock_retrieve.return_value = "mock-image-uri"

    pipeline = CaliforniaHousingPipeline(mock_config)
    parameters = pipeline.create_parameters()

    assert "instance_type" in parameters
    assert "instance_count" in parameters
    assert "mse_threshold" in parameters


@patch("src.pipeline.pipeline.boto3.Session")
@patch("src.pipeline.pipeline.sagemaker.Session")
@patch("src.pipeline.pipeline.PipelineSession")
@patch("src.pipeline.pipeline.sagemaker.image_uris.retrieve")
@patch("src.pipeline.pipeline.Pipeline")
def test_create_pipeline(
    mock_pipeline_cls, mock_retrieve, mock_pipeline_session, mock_session, mock_boto_session, mock_config
):
    """Test complete pipeline creation."""
    mock_retrieve.return_value = "mock-image-uri"
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    pipeline_manager = CaliforniaHousingPipeline(mock_config)
    pipeline = pipeline_manager.create_pipeline()

    mock_pipeline_cls.assert_called_once()
    call_kwargs = mock_pipeline_cls.call_args[1]
    assert call_kwargs["name"] == "test-pipeline"
    assert len(call_kwargs["steps"]) == 4
