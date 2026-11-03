#!/bin/sh
set -eu
run_id="${ROUTESCALE_RUN_ID:-run-$(date -u +%Y%m%dT%H%M%S)-$$}"
case "$run_id" in *[!a-zA-Z0-9_-]*) echo "Invalid ROUTESCALE_RUN_ID" >&2; exit 2;; esac
exec python3 scripts/framework_service.py run --run-dir "/state/$run_id"
