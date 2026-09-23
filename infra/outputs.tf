output "s3_bucket_name" {
  description = "S3 bucket for SageMaker artifacts"
  value       = aws_s3_bucket.artifacts.id
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "model_package_group_name" {
  description = "SageMaker Model Package Group name"
  value       = aws_sagemaker_model_package_group.models.model_package_group_name
}

output "model_package_group_arn" {
  description = "SageMaker Model Package Group ARN"
  value       = aws_sagemaker_model_package_group.models.arn
}
