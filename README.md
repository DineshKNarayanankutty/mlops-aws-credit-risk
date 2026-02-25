# MLOps AWS Credit Risk

End-to-end cloud-native MLOps project for credit risk prediction on AWS using SageMaker, S3, ECR, EKS, CloudWatch, Prometheus, CodeBuild, and CodePipeline, fully provisioned with Terraform.

## Architecture

```text
                         +----------------------+
                         |       GitHub         |
                         +----------+-----------+
                                    |
                                    v
                         +----------+-----------+
                         |     CodePipeline     |
                         +----------+-----------+
                                    |
                                    v
                         +----------+-----------+
                         |      CodeBuild       |
                         |  (buildspec.yml)     |
                         +----+-----------+-----+
                              |           |
                    pull model.pkl        | docker build/push
                              |           v
                              |   +-------+--------+
                              |   |      ECR       |
                              |   +-------+--------+
                              |           |
                              |           v
+---------------------+       |   +-------+--------+       +----------------------+
| SageMaker Training  +-------+--->  EKS Deployment +------>  ALB Ingress / API   |
|  (training/train.py)| model.pkl   | FastAPI on Pods |      | /health /predict    |
+----------+----------+ in S3       +-------+--------+       +----------------------+
           |                                |
           | model.tar.gz + metrics         | /metrics
           v                                v
+----------+----------+            +--------+-----------------+
|   S3 Model Bucket   |            | Prometheus + CloudWatch  |
+----------+----------+            | Container Insights        |
           |                       +---------------------------+
           v
+----------+----------+
| SageMaker Model     |
| Registry Group      |
+---------------------+
```

## What This Deploys

- SageMaker-compatible training script (`training/train.py`) with sklearn `RandomForestClassifier`
- Model artifact storage in S3 (`model.pkl`)
- SageMaker Model Package Group (Model Registry)
- FastAPI inference service with:
  - `GET /health`
  - `POST /predict`
  - `GET /metrics` (Prometheus)
- Docker image for inference (`python:3.10-slim`)
- EKS deployment with:
  - Deployment (2 replicas)
  - Service (ClusterIP)
  - ALB Ingress
  - HPA @ 70% CPU
- CloudWatch Container Insights via EKS addon
- CI/CD pipeline:
  - GitHub source
  - CodeBuild build/push/deploy
  - ECR push + EKS rolling image update
- Terraform-managed infra

## Prerequisites

- AWS CLI configured with admin-capable credentials
- Terraform >= 1.6
- Docker
- kubectl
- Python 3.10+
- `sagemaker` Python SDK for training orchestration
- `eksctl` (recommended for IAM mapping convenience)
- Existing default VPC in target region (this project intentionally uses default VPC to keep cost and setup minimal)

## 1) Provision Core Infrastructure

```bash
terraform -chdir=terraform init
terraform -chdir=terraform apply -auto-approve \
  -var "region=us-east-1" \
  -var "project_name=mlops-aws-credit-risk"
```

Useful outputs:

```bash
terraform -chdir=terraform output
```

## 2) Run SageMaker Training Job

This runs `training/train.py` as a SageMaker training job and writes:
- SageMaker output artifact (`model.tar.gz`) to S3
- Direct `model.pkl` to `s3://<bucket>/approved/model.pkl` for deployment image builds

```bash
export AWS_REGION="$(terraform -chdir=terraform output -raw region)"
export SAGEMAKER_ROLE_ARN="$(terraform -chdir=terraform output -raw sagemaker_execution_role_arn)"
export MODEL_BUCKET="$(terraform -chdir=terraform output -raw model_artifacts_bucket_name)"
```

```bash
python - <<'PY'
import os
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["MODEL_BUCKET"]

session = sagemaker.Session()
estimator = SKLearn(
    entry_point="train.py",
    source_dir="training",
    role=role,
    framework_version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{bucket}/training-output/",
    hyperparameters={
        "n-features": 20,
        "n-estimators": 300,
        "model-s3-uri": f"s3://{bucket}/approved/model.pkl",
    },
    sagemaker_session=session,
)
estimator.fit(wait=True)
print("Training complete.")
print("Model tar artifact:", estimator.model_data)
PY
```

## 3) Register Model in SageMaker Model Registry

```bash
export MODEL_GROUP_NAME="$(terraform -chdir=terraform output -raw sagemaker_model_package_group_name)"
export ECR_IMAGE_URI="$(terraform -chdir=terraform output -raw ecr_repository_url):latest"
export TRAINING_ARTIFACT_URI="$(aws s3 ls s3://$MODEL_BUCKET/training-output/ --recursive | awk '/model.tar.gz/{print "s3://'$MODEL_BUCKET'/"$4}' | tail -n1)"
```

```bash
python - <<'PY'
import os
import boto3

sm = boto3.client("sagemaker")

response = sm.create_model_package(
    ModelPackageGroupName=os.environ["MODEL_GROUP_NAME"],
    ModelPackageDescription="Credit risk model package from SageMaker training",
    InferenceSpecification={
        "Containers": [
            {
                "Image": os.environ["ECR_IMAGE_URI"],
                "ModelDataUrl": os.environ["TRAINING_ARTIFACT_URI"],
            }
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",
)
print("Model package ARN:", response["ModelPackageArn"])
PY
```

## 4) Docker Build (Local, Optional)

```bash
mkdir -p model
aws s3 cp "s3://$MODEL_BUCKET/approved/model.pkl" model/model.pkl
docker build -f docker/Dockerfile -t credit-risk-api:local .
docker run --rm -p 8000:8000 credit-risk-api:local
```

Test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.2,0.4,0.6,0.8,1.2,1.4,1.6,1.8,2.0,2.2]}'
```

## 5) Deploy to EKS (kubectl)

Configure kubectl:

```bash
export EKS_CLUSTER_NAME="$(terraform -chdir=terraform output -raw eks_cluster_name)"
aws eks update-kubeconfig --region "$AWS_REGION" --name "$EKS_CLUSTER_NAME"
```

Apply manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

Notes:
- Install AWS Load Balancer Controller in cluster before applying `k8s/ingress.yaml`.
- HPA requires metrics server; install if not already available in the cluster.

## 6) Provision CI/CD Pipeline

Get infra outputs:

```bash
export ECR_REPO_NAME="$(terraform -chdir=terraform output -raw ecr_repository_name)"
export EKS_CLUSTER_NAME="$(terraform -chdir=terraform output -raw eks_cluster_name)"
export CODEBUILD_ROLE_ARN="$(terraform -chdir=terraform output -raw codebuild_role_arn)"
export MODEL_ARTIFACT_S3_URI="s3://$(terraform -chdir=terraform output -raw model_artifacts_bucket_name)/approved/model.pkl"
```

Apply pipeline terraform:

```bash
terraform -chdir=pipeline init
terraform -chdir=pipeline apply -auto-approve \
  -var "region=$AWS_REGION" \
  -var "project_name=mlops-aws-credit-risk" \
  -var "github_owner=$(git config --get remote.origin.url | sed -E 's#.*github.com[:/]([^/]+)/([^/.]+)(.git)?#\1#')" \
  -var "github_repo=$(git config --get remote.origin.url | sed -E 's#.*github.com[:/]([^/]+)/([^/.]+)(.git)?#\2#')" \
  -var "codebuild_role_arn=$CODEBUILD_ROLE_ARN" \
  -var "ecr_repository_name=$ECR_REPO_NAME" \
  -var "eks_cluster_name=$EKS_CLUSTER_NAME" \
  -var "model_artifact_s3_uri=$MODEL_ARTIFACT_S3_URI"
```

After creation, complete the CodeStar connection handshake in AWS Console (Developer Tools > Connections).

Authorize CodeBuild role in EKS (once):

```bash
eksctl create iamidentitymapping \
  --cluster "$EKS_CLUSTER_NAME" \
  --region "$AWS_REGION" \
  --arn "$CODEBUILD_ROLE_ARN" \
  --group system:masters \
  --username codebuild
```

## Monitoring

### CloudWatch Container Insights

Enabled by Terraform EKS addon `amazon-cloudwatch-observability`.

Check:

```bash
aws eks describe-addon --cluster-name "$EKS_CLUSTER_NAME" --addon-name amazon-cloudwatch-observability --region "$AWS_REGION"
```

### Prometheus Metrics

Application exposes metrics at `/metrics`.

Example scrape config:

```yaml
scrape_configs:
  - job_name: "kubernetes-pods-credit-risk"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: credit-risk-api
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

## Cleanup (Cost Control)

Destroy pipeline first, then core infra:

```bash
terraform -chdir=pipeline destroy -auto-approve
terraform -chdir=terraform destroy -auto-approve
```

Also verify no billable leftovers:
- Running SageMaker training jobs
- SageMaker endpoints (if you create any)
- Detached EBS volumes
- ALB/NLB resources
- CloudWatch log groups you want removed
