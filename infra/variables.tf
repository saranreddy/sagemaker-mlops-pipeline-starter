variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "sagemaker-mlops"
}

variable "pipeline_name" {
  description = "SageMaker Pipeline name"
  type        = string
  default     = "california-housing-pipeline"
}

variable "model_package_group_name" {
  description = "SageMaker Model Package Group name"
  type        = string
  default     = "california-housing-models"
}
