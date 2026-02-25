output "region" {
  value = var.region
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "default_vpc_id" {
  value = data.aws_vpc.default.id
}

output "model_artifacts_bucket_name" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "ecr_repository_name" {
  value = aws_ecr_repository.credit_risk_api.name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.credit_risk_api.repository_url
}

output "eks_cluster_name" {
  value = aws_eks_cluster.this.name
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.this.endpoint
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}

output "codebuild_role_arn" {
  value = aws_iam_role.codebuild_role.arn
}

output "sagemaker_model_package_group_name" {
  value = aws_sagemaker_model_package_group.credit_risk.model_package_group_name
}
