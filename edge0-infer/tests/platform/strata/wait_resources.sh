#!/usr/bin/env bash
# Functional checks only: wait for the selected GPU and sufficient available RAM.
# Other CPU workloads do not block. This script never stops another process.
set -eu
mode=${1:-direct}
gpu=${2:-0}
case "$gpu" in ''|*[!0-9]*) exit 2 ;; esac
case "$mode" in direct) required_gib=70 ;; ram) required_gib=110 ;; *) exit 2 ;; esac
ready() {
    local processes memory available
    date -Is
    processes=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
    memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    available=$(LC_ALL=C free -b | awk '$1 == "Mem:" {print $7}')
    nvidia-smi -i "$gpu"
    LC_ALL=C free -b
    test -z "$processes" && test "$memory" -lt 1024 && test "$available" -ge "$((required_gib * 1024 * 1024 * 1024))"
}
attempt=0
until ready; do
    test "$attempt" -lt 30 || { echo "GPU $gpu or available RAM still unavailable after 30 minutes"; exit 3; }
    attempt=$((attempt + 1))
    sleep 60
done
