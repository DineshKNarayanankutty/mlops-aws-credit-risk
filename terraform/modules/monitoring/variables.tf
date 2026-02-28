variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "namespace" {
  description = "Namespace for monitoring stack"
  type        = string
  default     = "monitoring"
}

variable "create_namespace" {
  description = "Whether to create monitoring namespace"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Whether to install Prometheus"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Whether to install Grafana"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Whether to install metrics-server"
  type        = bool
  default     = true
}

variable "enable_container_insights" {
  description = "Whether to enable Amazon CloudWatch Container Insights add-on"
  type        = bool
  default     = false
}

variable "log_retention_in_days" {
  description = "Retention period for monitoring-related CloudWatch log groups"
  type        = number
  default     = 30
}

variable "prometheus_chart_version" {
  description = "Helm chart version for Prometheus"
  type        = string
  default     = "25.30.0"
}

variable "grafana_chart_version" {
  description = "Helm chart version for Grafana"
  type        = string
  default     = "8.10.0"
}

variable "metrics_server_chart_version" {
  description = "Helm chart version for metrics-server"
  type        = string
  default     = "3.13.0"
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
