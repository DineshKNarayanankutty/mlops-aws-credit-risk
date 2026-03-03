region                          = "ap-south-1"
environment                     = "dev"
cluster_name                    = "mlops-eks-dev"
cluster_version                 = "1.35"
node_instance_type              = "t3.small"
inference_node_instance_type    = "t3.small"
cluster_endpoint_public_access  = false
cluster_endpoint_private_access = true

model_package_group_name              = "mlops-dev-model-package-group"
sagemaker_model_artifacts_bucket_name = "mlops-dev-model-artifacts-bucket777"
cicd_artifact_bucket_name             = "mlops-dev-cicd-artifacts-bucket777"

code_connection_arn       = "REPLACE_WITH_code_connection_arn"
github_full_repository_id = "REPLACE_WITH_GITHUB_ORG_REPO"
github_branch             = "main"

training_input_s3_uri  = "s3://mlops-dev-artifacts-bucket777/training/input/"
training_output_s3_uri = "s3://mlops-dev-artifacts-bucket777/training/output/"

