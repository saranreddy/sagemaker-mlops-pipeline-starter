resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project_name}-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Name      = "${var.project_name}-execution-role"
    ManagedBy = "terraform"
  }
}

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role_policy" "sagemaker_execution" {
  name   = "${var.project_name}-execution-policy"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_execution.json
}

data "aws_iam_policy_document" "sagemaker_execution" {
  statement {
    sid    = "S3Access"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*"
    ]
  }

  statement {
    sid    = "ECRAccess"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ]
    resources = ["*"]
  }

  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = ["arn:${data.aws_partition.current.partition}:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"]
  }

  statement {
    sid    = "SageMakerAccess"
    effect = "Allow"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:CreateProcessingJob",
      "sagemaker:CreateModel",
      "sagemaker:CreateModelPackage",
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:DescribeModel",
      "sagemaker:DescribeModelPackage",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:CreateEndpoint",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DescribeEndpoint",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:InvokeEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:ListModelPackages"
    ]
    resources = ["*"]
  }

  statement {
    sid    = "PassRole"
    effect = "Allow"
    actions = [
      "iam:PassRole"
    ]
    resources = [aws_iam_role.sagemaker_execution.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}
