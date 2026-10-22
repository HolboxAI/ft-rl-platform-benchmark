#!/bin/bash
# cloud-init user-data script. ec2/render_user_data.sh wraps it in a MIME multipart with
# `cloud_final_modules: [scripts-user, always]`, so it runs on EVERY boot (AWS's documented
# pattern). Each boot it (re)writes the runner and starts it once, so a new stage = upload a
# new job and start/reboot the instance (no SSH or SSM needed). render_user_data.sh fills
# the __PLACEHOLDERS__.
set -euxo pipefail

mkdir -p /data/control /data/runs /data/state
cat > /etc/skyrl-runner.env <<EOF
S3_ROOT=__S3_ROOT__
AWS_DEFAULT_REGION=__AWS_REGION__
HARD_LIMIT_HOURS=__HARD_LIMIT_HOURS__
EOF

cat > /usr/local/bin/skyrl-runner.sh <<'RUNNER'
#!/bin/bash
# Per-boot runner: fetch job -> run it -> heartbeat status + log to S3 -> halt.
set -uo pipefail
source /etc/skyrl-runner.env
export AWS_DEFAULT_REGION

# Hard limit, re-armed on every boot. Shutdown behaviour is "stop", so the disk survives.
shutdown -h "+$((HARD_LIMIT_HOURS * 60))" "skyrl: hard time limit" || true

W=/data/control
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
BOOT=$(date -u +%FT%TZ)
RUN_ID=""

put_status() {
  local stage
  stage=$(cat /data/state/current_stage 2>/dev/null || true)
  printf '{"state": "%s", "run_id": "%s", "stage": "%s", "instance_id": "%s", "boot": "%s", "updated": "%s"}\n' \
    "$1" "$RUN_ID" "$stage" "$IID" "$BOOT" "$(date -u +%FT%TZ)" > "$W/status.json"
  aws s3 cp --only-show-errors "$W/status.json" "$S3_ROOT/control/status.json"
  if [ -n "$RUN_ID" ]; then
    aws s3 cp --only-show-errors "$W/status.json" "$S3_ROOT/runs/$RUN_ID/status.json"
  fi
}

if ! aws s3 cp --only-show-errors "$S3_ROOT/control/job.sh" "$W/job.sh" \
   || ! aws s3 cp --only-show-errors "$S3_ROOT/control/job.env" "$W/job.env"; then
  put_status NO_JOB
  shutdown -h now
  exit 0
fi
set -a; source "$W/job.env"; set +a

LOG=/data/runs/$RUN_ID.job.log
echo "===== boot $BOOT on $IID =====" >> "$LOG"
put_status RUNNING

( while sleep 120; do
    aws s3 cp --only-show-errors "$LOG" "$S3_ROOT/runs/$RUN_ID/job.log"
    put_status RUNNING
  done ) &
HEARTBEAT=$!

timeout "$((HARD_LIMIT_HOURS * 3600 - 600))s" bash "$W/job.sh" >> "$LOG" 2>&1
rc=$?

kill "$HEARTBEAT" 2>/dev/null || true
aws s3 cp --only-show-errors "$LOG" "$S3_ROOT/runs/$RUN_ID/job.log"
if [ "$rc" -eq 0 ]; then put_status SUCCEEDED; else put_status "FAILED rc=$rc"; fi
sleep 5
shutdown -h now
RUNNER
chmod +x /usr/local/bin/skyrl-runner.sh

cat > /etc/systemd/system/skyrl-runner.service <<'UNIT'
[Unit]
Description=SkyRL CholecT50 GPU-control job runner (S3-driven)
Wants=network-online.target docker.service
After=network-online.target docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/skyrl-runner.sh
TimeoutStartSec=0
UNIT

# Not enabled: an enabled unit ordered after cloud-final forms a boot ordering cycle that
# systemd breaks by dropping the runner (seen on 2026-10-16). This script runs every boot
# from cloud-final instead and starts the unit once, without waiting on it.
systemctl disable skyrl-runner.service 2>/dev/null || true
systemctl daemon-reload
systemctl start --no-block skyrl-runner.service
