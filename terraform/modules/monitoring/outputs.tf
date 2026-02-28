output "cloudwatch_container_insights_addon" {
  value = try(aws_eks_addon.container_insights[0].addon_name, null)
}

output "prometheus_release_name" {
  value = try(helm_release.prometheus[0].name, null)
}

output "grafana_release_name" {
  value = try(helm_release.grafana[0].name, null)
}

output "metrics_server_release_name" {
  value = try(helm_release.metrics_server[0].name, null)
}
