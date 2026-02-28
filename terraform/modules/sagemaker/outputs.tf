output "model_package_group_name" {
  value = try(aws_sagemaker_model_package_group.this[0].model_package_group_name, null)
}

output "model_package_group_arn" {
  value = try(aws_sagemaker_model_package_group.this[0].arn, null)
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

output "model_artifacts_bucket_name" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "model_artifacts_bucket_arn" {
  value = aws_s3_bucket.model_artifacts.arn
}

output "model_artifacts_kms_key_arn" {
  value = aws_kms_key.model_artifacts.arn
}
