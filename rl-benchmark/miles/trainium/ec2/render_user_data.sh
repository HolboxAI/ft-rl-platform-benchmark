#!/usr/bin/env bash
# Render ec2/user_data.sh into MIME multipart user-data that cloud-init runs on EVERY boot.
#   ec2/render_user_data.sh <out_file>     (reads ec2/ec2.env for the placeholders)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../ec2/ec2.env
source "$HERE/../ec2/ec2.env"
out=$1
{
  printf 'Content-Type: multipart/mixed; boundary="//"\nMIME-Version: 1.0\n\n'
  printf -- '--//\nContent-Type: text/cloud-config; charset="us-ascii"\nMIME-Version: 1.0\n'
  printf 'Content-Transfer-Encoding: 7bit\nContent-Disposition: attachment; filename="cloud-config.txt"\n\n'
  printf '#cloud-config\ncloud_final_modules:\n- [scripts-user, always]\n\n'
  printf -- '--//\nContent-Type: text/x-shellscript; charset="us-ascii"\nMIME-Version: 1.0\n'
  printf 'Content-Transfer-Encoding: 7bit\nContent-Disposition: attachment; filename="userdata.txt"\n\n'
  sed -e "s#__S3_ROOT__#$S3_ROOT#g" -e "s#__AWS_REGION__#$AWS_REGION#g" \
      -e "s#__HARD_LIMIT_HOURS__#$HARD_LIMIT_HOURS#g" "$HERE/user_data.sh"
  printf -- '\n--//--\n'
} > "$out"
