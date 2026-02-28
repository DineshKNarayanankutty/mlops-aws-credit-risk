variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for controller configuration"
  type        = string
}

variable "oidc_provider_arn" {
  description = "OIDC provider ARN used by IRSA role"
  type        = string
}

variable "oidc_provider_url" {
  description = "OIDC provider URL used by IRSA role"
  type        = string
}

variable "namespace" {
  description = "Namespace for AWS Load Balancer Controller"
  type        = string
  default     = "kube-system"
}

variable "service_account_name" {
  description = "Service account name for AWS Load Balancer Controller"
  type        = string
  default     = "aws-load-balancer-controller"
}

variable "chart_version" {
  description = "Helm chart version for AWS Load Balancer Controller"
  type        = string
  default     = "1.11.0"
}

variable "create_namespace" {
  description = "Whether to create namespace with Terraform"
  type        = bool
  default     = false
}

variable "create_irsa_role" {
  description = "Whether to create IRSA role in this module"
  type        = bool
  default     = true
}

variable "existing_irsa_role_arn" {
  description = "Existing IRSA role ARN when create_irsa_role=false"
  type        = string
  default     = null

  validation {
    condition     = var.create_irsa_role || var.existing_irsa_role_arn != null
    error_message = "existing_irsa_role_arn must be provided when create_irsa_role is false."
  }
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}

variable "backend_security_group_id" {
  description = "Security group ID used by ALBs as shared backend SG"
  type        = string
}
