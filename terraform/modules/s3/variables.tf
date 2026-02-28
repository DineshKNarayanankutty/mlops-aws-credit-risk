variable "bucket_name" {
  description = "mlops-tf-platform-bucket"
  type        = string
}

variable "versioning_enabled" {
  description = "Enable versioning"
  type        = bool
  default     = true
}

variable "kms_key_arn" {
  description = "Optional existing KMS key ARN for bucket encryption"
  type        = string
  default     = null
}

variable "kms_deletion_window_in_days" {
  description = "Deletion window for module-managed KMS key"
  type        = number
  default     = 30
}

variable "access_log_bucket_name" {
  description = "Optional explicit access log bucket name. Defaults to <bucket_name>-logs"
  type        = string
  default     = null
}

variable "access_log_prefix" {
  description = "Prefix for S3 server access logs"
  type        = string
  default     = "access-logs/"
}

variable "lifecycle_enabled" {
  description = "Enable bucket lifecycle policy"
  type        = bool
  default     = true
}

variable "noncurrent_version_expiration_days" {
  description = "Days to retain non-current object versions"
  type        = number
  default     = 30
}

variable "abort_incomplete_multipart_upload_days" {
  description = "Days before aborting incomplete multipart uploads"
  type        = number
  default     = 7
}

variable "force_destroy" {
  description = "Allow bucket deletion even when non-empty"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
