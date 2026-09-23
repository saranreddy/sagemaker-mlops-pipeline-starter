"""
SageMaker Pipeline definition for California Housing price prediction.
"""

import logging
from typing import Any, Dict

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaliforniaHousingPipeline:
    """SageMaker Pipeline for California Housing price prediction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.

        Args:
            config: Configuration dictionary containing pipeline parameters
        """
        self.config = config
        self.region = config["aws_region"]
        self.role = config["sagemaker_role_arn"]
        self.bucket = config["s3_bucket"]
        self.pipeline_name = config["pipeline_name"]

        self.sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=self.region))
        self.pipeline_session = PipelineSession(boto_session=boto3.Session(region_name=self.region))

        self.base_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=self.region,
            version=config.get("framework_version", "1.7-1"),
            py_version="py3",
        )

    def create_parameters(self):
        """Create pipeline parameters."""
        instance_type = ParameterString(
            name="InstanceType",
            default_value=self.config.get("instance_type", "ml.m5.xlarge"),
        )

        instance_count = ParameterInteger(name="InstanceCount", default_value=self.config.get("instance_count", 1))

        mse_threshold = ParameterFloat(
            name="MseThreshold",
            default_value=self.config.get("mse_threshold", 0.5),
        )

        return {
            "instance_type": instance_type,
            "instance_count": instance_count,
            "mse_threshold": mse_threshold,
        }

    def create_processing_step(self, parameters):
        """Create data processing step."""
        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            instance_type=parameters["instance_type"],
            instance_count=1,
            base_job_name="california-housing-processing",
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

        step_process = ProcessingStep(
            name="PreprocessData",
            processor=sklearn_processor,
            code="src/scripts/preprocessing.py",
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{self.bucket}/{self.pipeline_name}/data/train",
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{self.bucket}/{self.pipeline_name}/data/validation",
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{self.bucket}/{self.pipeline_name}/data/test",
                ),
            ],
        )

        return step_process

    def create_training_step(self, step_process, parameters):
        """Create model training step."""
        xgb_estimator = Estimator(
            image_uri=self.base_image_uri,
            role=self.role,
            instance_type=parameters["instance_type"],
            instance_count=parameters["instance_count"],
            output_path=f"s3://{self.bucket}/{self.pipeline_name}/models",
            base_job_name="california-housing-training",
            sagemaker_session=self.pipeline_session,
            hyperparameters={
                "max_depth": 5,
                "eta": 0.1,
                "num_round": 100,
                "objective": "reg:squarederror",
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        )

        step_train = TrainingStep(
            name="TrainModel",
            estimator=xgb_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                    content_type="text/csv",
                ),
            },
        )

        return step_train

    def create_evaluation_step(self, step_train, step_process, parameters):
        """Create model evaluation step."""
        script_eval = ScriptProcessor(
            image_uri=self.base_image_uri,
            command=["python3"],
            instance_type=parameters["instance_type"],
            instance_count=1,
            base_job_name="california-housing-eval",
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
        )

        step_eval = ProcessingStep(
            name="EvaluateModel",
            processor=script_eval,
            code="src/scripts/evaluate.py",
            inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"s3://{self.bucket}/{self.pipeline_name}/evaluation",
                ),
            ],
            job_arguments=["--mse-threshold", str(parameters["mse_threshold"].default_value)],
            property_files=[evaluation_report],
        )

        return step_eval, evaluation_report

    def create_model_registration_step(self, step_train, step_eval, evaluation_report):
        """Create model registration step."""
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"s3://{self.bucket}/{self.pipeline_name}/evaluation/evaluation.json",
                content_type="application/json",
            )
        )

        model = Model(
            image_uri=self.base_image_uri,
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            sagemaker_session=self.pipeline_session,
            role=self.role,
        )

        step_register = ModelStep(
            name="RegisterModel",
            step_args=model.register(
                content_types=["text/csv"],
                response_types=["text/csv"],
                inference_instances=["ml.t2.medium", "ml.m5.large"],
                transform_instances=["ml.m5.large"],
                model_package_group_name=self.config["model_package_group_name"],
                approval_status="PendingManualApproval",
                model_metrics=model_metrics,
            ),
        )

        return step_register

    def create_condition_step(self, step_eval, step_register, evaluation_report, parameters):
        """Create conditional step based on evaluation metrics."""
        cond_lte = ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=step_eval.name,
                property_file=evaluation_report,
                json_path="regression_metrics.mse.value",
            ),
            right=parameters["mse_threshold"],
        )

        step_cond = ConditionStep(
            name="CheckMseThreshold",
            conditions=[cond_lte],
            if_steps=[step_register],
            else_steps=[],
        )

        return step_cond

    def create_pipeline(self):
        """Create and return the complete SageMaker Pipeline."""
        logger.info(f"Creating pipeline: {self.pipeline_name}")

        parameters = self.create_parameters()

        step_process = self.create_processing_step(parameters)
        step_train = self.create_training_step(step_process, parameters)
        step_eval, evaluation_report = self.create_evaluation_step(step_train, step_process, parameters)
        step_register = self.create_model_registration_step(step_train, step_eval, evaluation_report)
        step_cond = self.create_condition_step(step_eval, step_register, evaluation_report, parameters)

        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                parameters["instance_type"],
                parameters["instance_count"],
                parameters["mse_threshold"],
            ],
            steps=[step_process, step_train, step_eval, step_cond],
            sagemaker_session=self.pipeline_session,
        )

        return pipeline

    def upsert_pipeline(self):
        """Create or update the pipeline."""
        pipeline = self.create_pipeline()

        logger.info("Upserting pipeline...")
        pipeline.upsert(role_arn=self.role)

        logger.info(f"Pipeline '{self.pipeline_name}' upserted successfully!")

        return pipeline

    def start_pipeline(self):
        """Start pipeline execution."""
        pipeline = self.create_pipeline()

        logger.info(f"Starting pipeline execution: {self.pipeline_name}")
        execution = pipeline.start()

        logger.info(f"Pipeline execution started: {execution.arn}")
        logger.info(f"Execution ID: {execution._execution_id}")

        return execution
