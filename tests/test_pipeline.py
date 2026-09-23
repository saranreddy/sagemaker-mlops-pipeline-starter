"""Tests for pipeline construction."""

from unittest.mock import MagicMock, PropertyMock, patch

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


def configure_session_mocks(mock_pipeline_session_cls, mock_session_cls):
    """Configure session mocks with proper sagemaker_config."""
    mock_pipeline_session_instance = MagicMock()
    type(mock_pipeline_session_instance).sagemaker_config = PropertyMock(return_value={})
    mock_pipeline_session_cls.return_value = mock_pipeline_session_instance

    mock_session_instance = MagicMock()
    type(mock_session_instance).sagemaker_config = PropertyMock(return_value={})
    mock_session_cls.return_value = mock_session_instance


@patch("src.pipeline.pipeline.boto3.Session")
@patch("src.pipeline.pipeline.sagemaker.Session")
@patch("src.pipeline.pipeline.PipelineSession")
@patch("src.pipeline.pipeline.sagemaker.image_uris.retrieve")
def test_pipeline_initialization(mock_retrieve, mock_pipeline_session, mock_session, mock_boto_session, mock_config):
    """Test pipeline initialization."""
    mock_retrieve.return_value = "mock-image-uri"
    configure_session_mocks(mock_pipeline_session, mock_session)

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
    configure_session_mocks(mock_pipeline_session, mock_session)

    pipeline = CaliforniaHousingPipeline(mock_config)
    parameters = pipeline.create_parameters()

    assert "instance_type" in parameters
    assert "instance_count" in parameters
    assert "mse_threshold" in parameters


@patch("src.pipeline.pipeline.ConditionStep")
@patch("src.pipeline.pipeline.ModelStep")
@patch("src.pipeline.pipeline.TrainingStep")
@patch("src.pipeline.pipeline.ProcessingStep")
@patch("src.pipeline.pipeline.boto3.Session")
@patch("src.pipeline.pipeline.sagemaker.Session")
@patch("src.pipeline.pipeline.PipelineSession")
@patch("src.pipeline.pipeline.sagemaker.image_uris.retrieve")
@patch("src.pipeline.pipeline.Pipeline")
def test_create_pipeline(
    mock_pipeline_cls,
    mock_retrieve,
    mock_pipeline_session,
    mock_session,
    mock_boto_session,
    mock_processing_step,
    mock_training_step,
    mock_model_step,
    mock_condition_step,
    mock_config,
):
    """Test complete pipeline creation."""
    mock_retrieve.return_value = "mock-image-uri"
    configure_session_mocks(mock_pipeline_session, mock_session)

    mock_step_process = MagicMock()
    mock_step_process.name = "PreprocessData"
    mock_processing_step.return_value = mock_step_process

    mock_step_train = MagicMock()
    mock_step_train.name = "TrainModel"
    mock_training_step.return_value = mock_step_train

    mock_step_eval = MagicMock()
    mock_step_eval.name = "EvaluateModel"
    mock_processing_step.side_effect = [mock_step_process, mock_step_eval]

    mock_step_register = MagicMock()
    mock_step_register.name = "RegisterModel"
    mock_model_step.return_value = mock_step_register

    mock_step_cond = MagicMock()
    mock_step_cond.name = "CheckMseThreshold"
    mock_condition_step.return_value = mock_step_cond

    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    pipeline_manager = CaliforniaHousingPipeline(mock_config)
    result_pipeline = pipeline_manager.create_pipeline()

    assert result_pipeline == mock_pipeline_instance
    mock_pipeline_cls.assert_called_once()
    call_kwargs = mock_pipeline_cls.call_args[1]
    assert call_kwargs["name"] == "test-pipeline"
    assert len(call_kwargs["steps"]) == 4
