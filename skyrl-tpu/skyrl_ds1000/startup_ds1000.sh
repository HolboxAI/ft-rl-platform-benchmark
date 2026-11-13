#!/bin/bash
# TPU VM startup script (runs as root at boot): fetch the code from GCS and start the DS-1000 pipeline, so a
# queued spot VM starts working the moment it exists, with no laptop SSH. Runs once per VM.
# Passed to `gcloud compute tpus queued-resources create --metadata-from-file startup-script=...`
# (queue_ds1000.sh); the code is uploaded first by `queue_ds1000.sh upload`.
exec >> /var/log/ds1000_startup.log 2>&1
[ -f /root/.ds1000_started ] && { echo "already started"; exit 0; }
touch /root/.ds1000_started
export HOME=/root
# startup scripts get the system's low open-file limit; TorchTPU aborts with "Too many open files" (2026-10-26)
ulimit -n 1048576
cd /root
CODE=gs://tpu-rl-run-cholec/code
for f in ds1000-tpu_rl_benchmark.tgz ds1000-SkyRL-tpu.tgz; do
  for i in 1 2 3 4 5; do gcloud storage cp -q "$CODE/$f" "/root/$f" && break; sleep 10; done
done
tar xzf /root/ds1000-tpu_rl_benchmark.tgz -C /root
mkdir -p /root/SkyRL-tpu && tar xzf /root/ds1000-SkyRL-tpu.tgz -C /root/SkyRL-tpu
cd /root/tpu_rl_benchmark/skyrl-tpu && mkdir -p logs
gcloud storage cp -q /var/log/ds1000_startup.log gs://tpu-rl-run-cholec/runs/ds1000_single_vm/startup.log
nohup bash skyrl_ds1000/single_vm_ds1000.sh > logs/ds1000_single_vm.out 2>&1 < /dev/null &
echo "pipeline started $(date -u)"
