data "aws_caller_identity" "current" {}

locals {
  artifact_access_log_bucket_name = "${var.artifact_bucket_name}-access-logs"
  base_tags = merge(
    {
      Project     = lookup(var.tags, "Project", "mlops-platform")
      ManagedBy   = "Terraform"
      Environment = var.environment
    },
    var.tags
  )
  resolved_training_image_uri = coalesce(var.training_image_uri, "${var.ecr_repository_url}:latest")
}

data "aws_iam_policy_document" "artifacts_kms" {
  statement {
    sid    = "EnableRootPermissions"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }

    actions = ["kms:*"]
    # KMS key policies use "*" to refer to the key resource itself.
    resources = ["*"]
  }

  statement {
    sid    = "AllowCICDServiceRoles"
    effect = "Allow"

    principals {
      type = "AWS"
      identifiers = [
        var.codebuild_role_arn,
        var.codepipeline_role_arn
      ]
    }

    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:GenerateDataKeyWithoutPlaintext",
      "kms:ReEncryptFrom",
      "kms:ReEncryptTo"
    ]
    # KMS key policies use "*" to refer to the key resource itself.
    resources = ["*"]
  }
}

resource "aws_kms_key" "artifacts" {
  description             = "KMS key for CI/CD artifacts (${var.environment})"
  deletion_window_in_days = var.kms_deletion_window_in_days
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.artifacts_kms.json
  tags                    = local.base_tags
}

resource "aws_kms_alias" "artifacts" {
  name          = "alias/cicd/${var.environment}/artifacts"
  target_key_id = aws_kms_key.artifacts.key_id
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = var.artifact_bucket_name
  force_destroy = var.force_destroy
  tags          = merge(local.base_tags, { Purpose = "codepipeline-artifacts" })
}

resource "aws_s3_bucket" "artifacts_log" {
  bucket        = local.artifact_access_log_bucket_name
  force_destroy = var.force_destroy
  tags          = merge(local.base_tags, { Purpose = "codepipeline-artifact-access-logs" })
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_ownership_controls" "artifacts_log" {
  bucket = aws_s3_bucket.artifacts_log.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "artifacts_log" {
  bucket = aws_s3_bucket.artifacts_log.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "artifacts_log" {
  bucket = aws_s3_bucket.artifacts_log.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.artifacts.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts_log" {
  bucket = aws_s3_bucket.artifacts_log.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.artifacts.arn
    }
    bucket_key_enabled = true
  }
}

data "aws_iam_policy_document" "artifact_bucket_tls" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "artifact_bucket_tls" {
  bucket = aws_s3_bucket.artifacts.id
  policy = data.aws_iam_policy_document.artifact_bucket_tls.json
}

data "aws_iam_policy_document" "artifact_bucket_log_delivery" {
  statement {
    sid    = "S3ServerAccessLogsPolicy"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["logging.s3.amazonaws.com"]
    }

    actions = ["s3:PutObject"]
    resources = [
      "${aws_s3_bucket.artifacts_log.arn}/access-logs/*"
    ]

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = [aws_s3_bucket.artifacts.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_s3_bucket_policy" "artifact_bucket_log_delivery" {
  bucket = aws_s3_bucket.artifacts_log.id
  policy = data.aws_iam_policy_document.artifact_bucket_log_delivery.json
}

resource "aws_s3_bucket_logging" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  target_bucket = aws_s3_bucket.artifacts_log.id
  target_prefix = "access-logs/"

  depends_on = [aws_s3_bucket_policy.artifact_bucket_log_delivery]
}

resource "aws_codebuild_project" "build_and_push" {
  name         = "${var.environment}-mlops-build-image"
  description  = "Build and push inference image to ECR"
  service_role = var.codebuild_role_arn

  artifacts {
    type = "CODEPIPELINE"
  }

  source {
    type      = "CODEPIPELINE"
    buildspec = <<-EOT
      version: 0.2
      phases:
        pre_build:
          commands:
            - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin ${split("/", var.ecr_repository_url)[0]}
            - IMAGE_TAG=$${CODEBUILD_RESOLVED_SOURCE_VERSION}
            - IMAGE_TAG=$${IMAGE_TAG:0:7}
        build:
          commands:
            - docker build -t ${var.ecr_repository_url}:$${IMAGE_TAG} .
            - docker tag ${var.ecr_repository_url}:$${IMAGE_TAG} ${var.ecr_repository_url}:latest
        post_build:
          commands:
            - docker push ${var.ecr_repository_url}:$${IMAGE_TAG}
            - docker push ${var.ecr_repository_url}:latest
      artifacts:
        files:
          - '**/*'
    EOT
  }

  environment {
    compute_type                = var.codebuild_compute_type
    image                       = var.codebuild_image
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = true
  }

  encryption_key = aws_kms_key.artifacts.arn
  tags           = local.base_tags
}

resource "aws_codebuild_project" "trigger_training" {
  name         = "${var.environment}-mlops-trigger-training"
  description  = "Trigger SageMaker training after image build"
  service_role = var.codebuild_role_arn

  artifacts {
    type = "CODEPIPELINE"
  }

  source {
    type      = "CODEPIPELINE"
    buildspec = <<-EOT
      version: 0.2
      phases:
        build:
          commands:
            - TRAINING_JOB_NAME=${var.environment}-training-$(date +%s)
            - |
              aws sagemaker create-training-job \
                --region $AWS_DEFAULT_REGION \
                --training-job-name "$${TRAINING_JOB_NAME}" \
                --algorithm-specification TrainingImage=${local.resolved_training_image_uri},TrainingInputMode=File \
                --role-arn ${var.sagemaker_training_role_arn} \
                --input-data-config '[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":"${var.training_input_s3_uri}","S3DataDistributionType":"FullyReplicated"}}}]' \
                --output-data-config '{"S3OutputPath":"${var.training_output_s3_uri}"}' \
                --resource-config '{"InstanceType":"${var.training_instance_type}","InstanceCount":${var.training_instance_count},"VolumeSizeInGB":${var.training_volume_size_gb}}' \
                --stopping-condition '{"MaxRuntimeInSeconds":${var.training_max_runtime_seconds}}'
      artifacts:
        files:
          - '**/*'
    EOT
  }

  environment {
    compute_type                = var.codebuild_compute_type
    image                       = var.codebuild_image
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode             = false
  }

  encryption_key = aws_kms_key.artifacts.arn
  tags           = local.base_tags
}

resource "aws_codepipeline" "this" {
  name     = coalesce(var.pipeline_name, "${var.environment}-mlops-pipeline")
  role_arn = var.codepipeline_role_arn

  artifact_store {
    location = aws_s3_bucket.artifacts.bucket
    type     = "S3"

    encryption_key {
      id   = aws_kms_key.artifacts.arn
      type = "KMS"
    }
  }

  stage {
    name = "Source"

    action {
      name             = "SourceFromGitHub"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      output_artifacts = ["SourceOutput"]

      configuration = {
        ConnectionArn    = var.code_connection_arn
        FullRepositoryId = var.github_full_repository_id
        BranchName       = var.github_branch
        DetectChanges    = "true"
      }
    }
  }

  stage {
    name = "Build"

    action {
      name             = "BuildAndPushImage"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      input_artifacts  = ["SourceOutput"]
      output_artifacts = ["BuildOutput"]

      configuration = {
        ProjectName = aws_codebuild_project.build_and_push.name
      }
    }
  }

  stage {
    name = "Train"

    action {
      name            = "TriggerSageMakerTraining"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      input_artifacts = ["BuildOutput"]

      configuration = {
        ProjectName = aws_codebuild_project.trigger_training.name
      }
    }
  }

  tags = local.base_tags
}
