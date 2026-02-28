terraform {
  backend "s3" {
    bucket       = "mlops-aws-credit-risk-s3"
    key          = "dev/terraform.tfstate"
    region       = "ap-south-1"
    use_lockfile = true
    encrypt      = true
  }
}
