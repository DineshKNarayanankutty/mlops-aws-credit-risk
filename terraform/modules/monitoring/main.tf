locals {
  base_tags = merge(
    {
      Project     = lookup(var.tags, "Project", "mlops-platform")
      Environment = lookup(var.tags, "Environment", "unknown")
      ManagedBy   = "Terraform"
    },
    var.tags
  )
  container_insights_log_group_suffixes = ["application", "dataplane", "host", "performance"]
}

resource "aws_cloudwatch_log_group" "container_insights" {
  count = var.enable_container_insights ? length(local.container_insights_log_group_suffixes) : 0

  name              = "/aws/containerinsights/${var.cluster_name}/${local.container_insights_log_group_suffixes[count.index]}"
  retention_in_days = var.log_retention_in_days
  tags              = local.base_tags
}

resource "aws_eks_addon" "container_insights" {
  count = var.enable_container_insights ? 1 : 0

  cluster_name                = var.cluster_name
  addon_name                  = "amazon-cloudwatch-observability"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
  tags                        = local.base_tags

  depends_on = [aws_cloudwatch_log_group.container_insights]
}

resource "kubernetes_namespace" "monitoring" {
  count = var.create_namespace && (var.enable_prometheus || var.enable_grafana) ? 1 : 0

  metadata {
    name = var.namespace
  }
}

resource "helm_release" "prometheus" {
  count = var.enable_prometheus ? 1 : 0

  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "prometheus"
  version          = var.prometheus_chart_version
  namespace        = var.namespace
  create_namespace = false
  wait             = true
  timeout          = 600

  set {
    name  = "server.persistentVolume.enabled"
    value = "true"
  }

  set {
    name  = "alertmanager.enabled"
    value = "true"
  }

  depends_on = [kubernetes_namespace.monitoring]
}

resource "helm_release" "grafana" {
  count = var.enable_grafana ? 1 : 0

  name             = "grafana"
  repository       = "https://grafana.github.io/helm-charts"
  chart            = "grafana"
  version          = var.grafana_chart_version
  namespace        = var.namespace
  create_namespace = false
  wait             = true
  timeout          = 600

  set {
    name  = "persistence.enabled"
    value = "true"
  }

  set {
    name  = "service.type"
    value = "LoadBalancer"
  }

  set {
    name  = "service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-scheme"
    value = "internal"
  }

  set {
    name  = "service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-internal"
    value = "true"
  }

  depends_on = [kubernetes_namespace.monitoring]
}

resource "helm_release" "metrics_server" {
  count = var.enable_metrics_server ? 1 : 0

  name             = "metrics-server"
  repository       = "https://kubernetes-sigs.github.io/metrics-server/"
  chart            = "metrics-server"
  version          = var.metrics_server_chart_version
  namespace        = "kube-system"
  create_namespace = false
  wait             = true
  timeout          = 600

  # Required for HPA metrics on EKS-managed kubelet endpoints.
  set {
    name  = "rbac.create"
    value = "true"
  }

  set {
    name  = "apiService.create"
    value = "true"
  }

  set {
    name  = "args[0]"
    value = "--kubelet-insecure-tls"
  }

  set {
    name  = "args[1]"
    value = "--kubelet-preferred-address-types=InternalIP,Hostname"
  }

  set {
    name  = "args[2]"
    value = "--metric-resolution=15s"
  }
}
