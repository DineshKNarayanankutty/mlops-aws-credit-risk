variable "environment" {
  description = "Environment name"
  type        = string
}

variable "artifact_bucket_name" {
  description = "S3 bucket name for pipeline artifacts"
  type        = string
}

variable "codebuild_role_arn" {
  description = "IAM role ARN for CodeBuild projects"
  type        = string
}

variable "codepipeline_role_arn" {
  description = "IAM role ARN for CodePipeline"
  type        = string
}

variable "code_connection_arn" {
  description = "CodeStar connection ARN for GitHub"
  type        = string
}

variable "github_full_repository_id" {
  description = "GitHub repo in owner/repo format"
  type        = string
}

variable "github_branch" {
  description = "GitHub branch for source stage"
  type        = string
  default     = "main"
}

variable "ecr_repository_url" {
  description = "ECR repository URL for docker image pushes"
  type        = string
}

variable "sagemaker_training_role_arn" {
  description = "IAM role ARN used by SageMaker training jobs"
  type        = string
}

variable "training_input_s3_uri" {
  description = "S3 URI for SageMaker training input data"
  type        = string
}

variable "training_output_s3_uri" {
  description = "S3 URI for SageMaker training output"
  type        = string
}

variable "training_image_uri" {
  description = "Optional explicit training image URI. Defaults to <ecr_repository_url>:latest"
  type        = string
  default     = null
}

variable "training_instance_type" {
  description = "SageMaker training instance type"
  type        = string
  default     = "ml.m5.large"
}

variable "training_instance_count" {
  description = "Number of instances for SageMaker training"
  type        = number
  default     = 1
}

variable "training_volume_size_gb" {
  description = "Training volume size in GiB"
  type        = number
  default     = 30
}

variable "training_max_runtime_seconds" {
  description = "Max SageMaker training runtime"
  type        = number
  default     = 3600
}

variable "codebuild_compute_type" {
  description = "CodeBuild compute type"
  type        = string
  default     = "BUILD_GENERAL1_MEDIUM"
}

variable "codebuild_image" {
  description = "CodeBuild build image"
  type        = string
  default     = "aws/codebuild/standard:7.0"
}

variable "pipeline_name" {
  description = "Optional pipeline name override"
  type        = string
  default     = null
}

variable "kms_deletion_window_in_days" {
  description = "Deletion window for module-managed KMS key"
  type        = number
  default     = 30
}

variable "force_destroy" {
  description = "Allow artifact bucket destruction when non-empty"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
