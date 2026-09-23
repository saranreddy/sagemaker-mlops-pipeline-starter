resource "aws_sagemaker_model_package_group" "models" {
  model_package_group_name        = var.model_package_group_name
  model_package_group_description = "California Housing price prediction models"

  tags = {
    Name      = var.model_package_group_name
    ManagedBy = "terraform"
  }
}
