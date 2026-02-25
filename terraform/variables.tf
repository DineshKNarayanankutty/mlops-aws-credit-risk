variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "mlops-aws-credit-risk"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "cluster_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "node_instance_type" {
  description = "EKS worker node instance type"
  type        = string
  default     = "t3.small"
}

variable "node_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 4
}

variable "model_package_group_name" {
  description = "Optional explicit SageMaker model package group name"
  type        = string
  default     = ""
}

variable "model_bucket_force_destroy" {
  description = "Allow force-destroy of model artifact bucket"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
