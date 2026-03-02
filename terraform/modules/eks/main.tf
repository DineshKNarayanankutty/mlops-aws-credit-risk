data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_region" "current" {}

locals {
  base_tags = merge(
    {
      Project     = lookup(var.tags, "Project", "mlops-platform")
      Environment = lookup(var.tags, "Environment", "unknown")
      ManagedBy   = "Terraform"
    },
    var.tags
  )

  node_group_profile = lookup(
    {
      dev = {
        system_instance_type       = "t3.small"
        inference_instance_type    = "t3.small"
        create_inference           = false
        system_min_size            = 1
        system_max_size            = 1
        system_desired_size        = 1
        inference_min_size         = 0
        inference_max_size         = 0
        inference_desired_size     = 0
        system_root_volume_size    = 20
        inference_root_volume_size = 20
        cluster_log_retention_days = 3
      }
      stage = {
        system_instance_type       = "t3.medium"
        inference_instance_type    = "t3.medium"
        create_inference           = true
        system_min_size            = 1
        system_max_size            = 2
        system_desired_size        = 1
        inference_min_size         = 1
        inference_max_size         = 2
        inference_desired_size     = 1
        system_root_volume_size    = 30
        inference_root_volume_size = 40
        cluster_log_retention_days = 7
      }
      prod = {
        system_instance_type       = var.node_instance_type
        inference_instance_type    = var.inference_node_instance_type
        create_inference           = true
        system_min_size            = 2
        system_max_size            = 5
        system_desired_size        = 2
        inference_min_size         = 2
        inference_max_size         = 5
        inference_desired_size     = 2
        system_root_volume_size    = var.system_node_root_volume_size
        inference_root_volume_size = var.inference_node_root_volume_size
        cluster_log_retention_days = 30
      }
    },
    var.environment,
    {
      system_instance_type       = var.node_instance_type
      inference_instance_type    = var.inference_node_instance_type
      create_inference           = true
      system_min_size            = var.system_node_group_min_size
      system_max_size            = var.system_node_group_max_size
      system_desired_size        = var.system_node_group_desired_size
      inference_min_size         = var.inference_node_group_min_size
      inference_max_size         = var.inference_node_group_max_size
      inference_desired_size     = var.inference_node_group_desired_size
      system_root_volume_size    = var.system_node_root_volume_size
      inference_root_volume_size = var.inference_node_root_volume_size
      cluster_log_retention_days = 30
    }
  )

  create_cluster_autoscaler    = var.environment == "prod"
  system_node_group_subnet_ids = var.environment == "dev" ? [var.private_subnets[0]] : null
}

resource "aws_security_group" "alb_backend" {
  name_prefix = "${var.cluster_name}-alb-backend-"
  description = "Shared backend security group used by AWS Load Balancer Controller"
  vpc_id      = var.vpc_id

  # ALBs require outbound access to health-check and forward traffic to node/pod targets.
  egress {
    description = "Allow ALB backend connectivity to targets"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    {
      Name = "${var.cluster_name}-alb-backend-sg"
    },
    local.base_tags
  )
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  vpc_id     = var.vpc_id
  subnet_ids = var.private_subnets

  enable_irsa = true

  cluster_endpoint_public_access         = var.cluster_endpoint_public_access
  cluster_endpoint_private_access        = var.cluster_endpoint_private_access
  cluster_enabled_log_types              = var.cluster_enabled_log_types
  cloudwatch_log_group_retention_in_days = local.node_group_profile.cluster_log_retention_days

  create_kms_key  = true
  kms_key_aliases = ["eks/${var.cluster_name}/secrets"]

  cluster_encryption_config = {
    resources = ["secrets"]
  }

  cluster_additional_security_group_ids = var.cluster_additional_security_group_ids

  # Restrict worker-node ingress paths to cluster control-plane rules + ALB backend SG.
  node_security_group_additional_rules = {
    ingress_alb_backend_nodeports = {
      description              = "ALB backend SG to worker NodePort range"
      protocol                 = "tcp"
      from_port                = 30000
      to_port                  = 32767
      type                     = "ingress"
      source_security_group_id = aws_security_group.alb_backend.id
    }
  }

  eks_managed_node_groups = merge(
    {
      system = {
        instance_types = [local.node_group_profile.system_instance_type]

        ami_type      = "AL2023_x86_64_STANDARD"
        capacity_type = "ON_DEMAND"

        min_size     = local.node_group_profile.system_min_size
        max_size     = local.node_group_profile.system_max_size
        desired_size = local.node_group_profile.system_desired_size
        subnet_ids   = local.system_node_group_subnet_ids

        block_device_mappings = {
          xvda = {
            device_name = "/dev/xvda"
            ebs = {
              volume_size           = local.node_group_profile.system_root_volume_size
              volume_type           = "gp3"
              encrypted             = true
              kms_key_id            = aws_kms_key.node_ebs.arn
              delete_on_termination = true
            }
          }
        }

        labels = {
          "node-role" = "system"
        }

        tags = local.create_cluster_autoscaler ? {
          "k8s.io/cluster-autoscaler/enabled"             = "true"
          "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        } : {}
      }
    },
    local.node_group_profile.create_inference ? {
      inference = {
        instance_types = [local.node_group_profile.inference_instance_type]

        ami_type      = "AL2023_x86_64_STANDARD"
        capacity_type = "ON_DEMAND"

        min_size     = local.node_group_profile.inference_min_size
        max_size     = local.node_group_profile.inference_max_size
        desired_size = local.node_group_profile.inference_desired_size

        block_device_mappings = {
          xvda = {
            device_name = "/dev/xvda"
            ebs = {
              volume_size           = local.node_group_profile.inference_root_volume_size
              volume_type           = "gp3"
              encrypted             = true
              kms_key_id            = aws_kms_key.node_ebs.arn
              delete_on_termination = true
            }
          }
        }

        labels = {
          "workload" = "inference"
        }

        taints = [
          {
            key    = "workload"
            value  = "inference"
            effect = "NO_SCHEDULE"
          }
        ]

        tags = local.create_cluster_autoscaler ? {
          "k8s.io/cluster-autoscaler/enabled"             = "true"
          "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        } : {}
      }
    } : {}
  )

  tags = merge(
    local.base_tags
  )
}

resource "aws_kms_key" "node_ebs" {
  description             = "KMS key for EKS node EBS volumes (${var.cluster_name})"
  deletion_window_in_days = var.kms_deletion_window_in_days
  enable_key_rotation     = true

  tags = merge(
    {
      Name = "${var.cluster_name}-node-ebs-kms"
    },
    local.base_tags
  )
}

resource "aws_kms_alias" "node_ebs" {
  name          = "alias/eks/${var.cluster_name}/node-ebs"
  target_key_id = aws_kms_key.node_ebs.key_id
}

data "aws_iam_policy_document" "cluster_autoscaler_assume_role" {
  count = local.create_cluster_autoscaler ? 1 : 0

  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.oidc_provider, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.oidc_provider, "https://", "")}:sub"
      values   = ["system:serviceaccount:${var.autoscaler_service_account_namespace}:${var.autoscaler_service_account_name}"]
    }
  }
}

resource "aws_iam_role" "cluster_autoscaler" {
  count = local.create_cluster_autoscaler ? 1 : 0

  name               = "${var.cluster_name}-cluster-autoscaler"
  assume_role_policy = data.aws_iam_policy_document.cluster_autoscaler_assume_role[0].json

  tags = local.base_tags
}

data "aws_iam_policy_document" "cluster_autoscaler" {
  count = local.create_cluster_autoscaler ? 1 : 0

  statement {
    sid    = "DescribePermissions"
    effect = "Allow"

    actions = [
      "autoscaling:DescribeAutoScalingGroups",
      "autoscaling:DescribeAutoScalingInstances",
      "autoscaling:DescribeLaunchConfigurations",
      "autoscaling:DescribeScalingActivities",
      "autoscaling:DescribeTags",
      "ec2:DescribeImages",
      "ec2:DescribeInstanceTypes",
      "ec2:DescribeLaunchTemplateVersions",
      "ec2:DescribeSubnets",
      "ec2:DescribeAvailabilityZones",
      "eks:DescribeNodegroup"
    ]

    # autoscaling/ec2/eks Describe APIs do not support resource-level permissions.
    resources = ["*"]
  }

  statement {
    sid    = "ScalePermissions"
    effect = "Allow"

    actions = [
      "autoscaling:SetDesiredCapacity",
      "autoscaling:TerminateInstanceInAutoScalingGroup"
    ]

    resources = [
      "arn:${data.aws_partition.current.partition}:autoscaling:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:autoScalingGroup:*:autoScalingGroupName/*"
    ]

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/k8s.io/cluster-autoscaler/enabled"
      values   = ["true"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/k8s.io/cluster-autoscaler/${var.cluster_name}"
      values   = ["owned"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/kubernetes.io/cluster/${var.cluster_name}"
      values   = ["owned", "shared"]
    }
  }
}

resource "aws_iam_policy" "cluster_autoscaler" {
  count = local.create_cluster_autoscaler ? 1 : 0

  name   = "${var.cluster_name}-cluster-autoscaler-policy"
  policy = data.aws_iam_policy_document.cluster_autoscaler[0].json

  tags = local.base_tags
}

resource "aws_iam_role_policy_attachment" "cluster_autoscaler" {
  count = local.create_cluster_autoscaler ? 1 : 0

  role       = aws_iam_role.cluster_autoscaler[0].name
  policy_arn = aws_iam_policy.cluster_autoscaler[0].arn
}

