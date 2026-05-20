#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sim_eval_root="$(cd "${script_dir}/.." && pwd)"
cd "${sim_eval_root}"

EPISODES="${EPISODES:-1}"
SCENE="${SCENE:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OPEN_LOOP_HORIZON="${OPEN_LOOP_HORIZON:-8}"
OPENPI_ACTION_MODE="${OPENPI_ACTION_MODE:-joint_position}"
OPENPI_CONTROL_DT="${OPENPI_CONTROL_DT:-0.06666666666666667}"
POLICY_CLIENT="${POLICY_CLIENT:-jointpos}"
HEADLESS="${HEADLESS:-1}"

client_args=(
  --episodes "${EPISODES}"
  --scene "${SCENE}"
  --remote-host "${HOST}"
  --remote-port "${PORT}"
  --open-loop-horizon "${OPEN_LOOP_HORIZON}"
  --openpi-action-mode "${OPENPI_ACTION_MODE}"
  --openpi-control-dt "${OPENPI_CONTROL_DT}"
  --policy-client "${POLICY_CLIENT}"
)

if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" ]]; then
  client_args+=(--headless)
fi

echo "[sim-evals run_openpi_pi05_eval] remote=${HOST}:${PORT}"
echo "[sim-evals run_openpi_pi05_eval] EPISODES=${EPISODES} SCENE=${SCENE} HEADLESS=${HEADLESS}"
echo "[sim-evals run_openpi_pi05_eval] POLICY_CLIENT=${POLICY_CLIENT}"
echo "[sim-evals run_openpi_pi05_eval] OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON}"
echo "[sim-evals run_openpi_pi05_eval] OPENPI_ACTION_MODE=${OPENPI_ACTION_MODE} OPENPI_CONTROL_DT=${OPENPI_CONTROL_DT}"

uv run --no-sync run_eval.py "${client_args[@]}" "$@"
