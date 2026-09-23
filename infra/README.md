# Infrastructure Setup

This directory contains Terraform configuration for provisioning AWS resources required by the SageMaker MLOps pipeline.

## Resources Created

- **S3 Bucket**: For storing SageMaker artifacts (models, data, outputs)
  - Versioning enabled
  - Server-side encryption (AES256)
  - Public access blocked

- **IAM Role**: SageMaker execution role with least-privilege permissions
  - S3 access (read/write to artifacts bucket)
  - ECR access (for pulling container images)
  - CloudWatch Logs (for job logging)
  - SageMaker service permissions
  - PassRole for SageMaker

- **Model Package Group**: For organizing model versions in Model Registry

## Prerequisites

- AWS CLI configured with credentials
- Terraform 1.0 or later
- Appropriate AWS permissions to create IAM roles, S3 buckets, and SageMaker resources

## Usage

### Initialize

```bash
terraform init
```

### Plan

```bash
terraform plan
```

### Apply

```bash
terraform apply
```

When prompted, type `yes` to confirm.

### Get Outputs

After applying, retrieve the values you'll need for pipeline configuration:

```bash
terraform output
```

Copy these values to `config/config.yaml`:
- `sagemaker_role_arn`
- `s3_bucket_name`

### Destroy

When you're done and want to clean up all resources:

```bash
terraform destroy
```

**Warning**: This will delete the S3 bucket and all its contents. Make sure to back up any important data first.

## Customization

Edit `terraform.tfvars` (copy from `terraform.tfvars.example`) to customize:

```hcl
aws_region = "us-west-2"
project_name = "my-mlops-project"
pipeline_name = "my-pipeline"
model_package_group_name = "my-models"
```

## Cost Estimate

Infrastructure resources (S3, IAM, Model Registry) have minimal costs:
- **S3**: Pay only for storage and requests (negligible for small datasets)
- **IAM**: No charge
- **Model Package Group**: No charge

Primary costs come from running SageMaker jobs (processing, training) and endpoints.

## Security Notes

- IAM role follows least-privilege principle
- S3 bucket blocks all public access
- Server-side encryption enabled by default
- No hardcoded credentials or secrets

## Outputs

After `terraform apply`, you'll see:

```
s3_bucket_name = "sagemaker-mlops-artifacts-123456789012"
sagemaker_role_arn = "arn:aws:iam::123456789012:role/sagemaker-mlops-execution-role"
model_package_group_name = "california-housing-models"
model_package_group_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/california-housing-models"
```

Use these values in your pipeline configuration.
