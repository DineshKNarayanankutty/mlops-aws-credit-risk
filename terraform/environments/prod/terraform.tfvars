region                          = "ap-south-1"
environment                     = "prod"
cluster_name                    = "mlops-eks-prod"
cluster_version                 = "1.35"
node_instance_type              = "t3.large"
inference_node_instance_type    = "t3.large"
cluster_endpoint_public_access  = true
cluster_endpoint_private_access = true

model_package_group_name              = "mlops-prod-model-package-group"
sagemaker_model_artifacts_bucket_name = "mlops-prod-model-artifacts-bucket777"
cicd_artifact_bucket_name             = "mlops-prod-cicd-artifacts-bucket777"

codestar_connection_arn   = "REPLACE_WITH_CODESTAR_CONNECTION_ARN"
github_full_repository_id = "REPLACE_WITH_GITHUB_ORG_REPO"
github_branch             = "main"

training_input_s3_uri  = "s3://mlops-prod-artifacts-bucket777/training/input/"
training_output_s3_uri = "s3://mlops-prod-artifacts-bucket777/training/output/"
