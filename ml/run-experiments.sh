#!/usr/bin/env bash
# Chạy hàng đợi các run PhoBERT (xem ml/experiment_queue.py): `make experiments` hoặc
#   bash ml/run-experiments.sh [--dry-run] [--explore] [--epochs N] [--retrain-mismatched] ...
# Logic nằm trong Python để kiểm thử được; script này chỉ chọn thư mục và nạp .env.
set -euo pipefail
cd "$(dirname "$0")/.."

env_args=()
if [ -f .env ]; then
  env_args=(--env-file .env)
fi
# ${arr[@]+...}: bash 3.2 của macOS coi mảng rỗng là biến chưa đặt khi bật `set -u`.
exec uv run ${env_args[@]+"${env_args[@]}"} python -m ml.experiment_queue "$@"
