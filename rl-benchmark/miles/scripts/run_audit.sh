#!/bin/bash
# Static porting audit of Miles and SkyRL (runs on a laptop; nothing is executed).
#   bash miles/scripts/run_audit.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC=${SRC:-/tmp/accel-audit-src}
mkdir -p "$SRC"
[ -d "$SRC/miles" ] || git clone --depth 1 https://github.com/radixark/miles "$SRC/miles"
[ -d "$SRC/skyrl" ] || git clone --depth 1 https://github.com/NovaSky-AI/SkyRL "$SRC/skyrl"
python3 -m common.porting_audit "$SRC/miles" --name miles --out miles/metrics/porting_audit.json
python3 -m common.porting_audit "$SRC/skyrl" --name skyrl --out skyrl/metrics/porting_audit.json
python3 -m common.porting_audit "$SRC/skyrl/skyrl/tx" --name skyrl-tx-jax --out skyrl/metrics/porting_audit_tx_jax.json
