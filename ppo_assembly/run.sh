#!/usr/bin/env bash
set -eo pipefail
ppo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$ppo_dir/.." && pwd)"
ppo_env_prefix="${PPO_CONDA_ENV:-/home/cjh/anaconda3/envs/isaaclab}"
conda_setup="${PPO_CONDA_SETUP:-/home/cjh/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$conda_setup" || ! -x "$ppo_env_prefix/bin/python" ]]; then
  echo "Isaac Lab conda environment unavailable: $ppo_env_prefix" >&2
  exit 2
fi
source "$conda_setup"
conda activate "$ppo_env_prefix"
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$ppo_dir/.cache/matplotlib"
case "${1:-}" in
  prepare) shift; exec "$ppo_env_prefix/bin/python" -B "$ppo_dir/prepare.py" "$@" ;;
  force-probe) shift; exec "$ppo_env_prefix/bin/python" -B "$ppo_dir/force_probe.py" "$@" ;;
  train|play|check) exec "$ppo_env_prefix/bin/python" -B "$ppo_dir/main.py" "$@" ;;
  curves) shift; exec "$ppo_env_prefix/bin/python" -B "$ppo_dir/curves.py" "$@" ;;
  test) shift; exec "$ppo_env_prefix/bin/python" -B -m unittest ppo_assembly.test_task_math "$@" ;;
  dashboard) shift; exec "$ppo_env_prefix/bin/python" -B -m tensorboard.main --logdir "$ppo_dir/runs" --host 127.0.0.1 "$@" ;;
  *) echo "Usage: ./ppo_assembly/run.sh {prepare|force-probe|train|play|check|curves|dashboard|test} [options]" >&2; exit 2 ;;
esac
