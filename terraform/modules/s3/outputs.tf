output "bucket_name" {
  value = aws_s3_bucket.this.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.this.arn
}

output "access_log_bucket_name" {
  value = aws_s3_bucket.log.bucket
}

output "access_log_bucket_arn" {
  value = aws_s3_bucket.log.arn
}

output "kms_key_arn" {
  value = local.s3_kms_key_arn
}
