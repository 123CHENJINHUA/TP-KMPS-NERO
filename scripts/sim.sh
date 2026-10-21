#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_dir="${ISAAC_SIM_PATH:-/home/cjh/Downloads/isaacsim/_build/linux-x86_64/release}"
if [[ ! -x "$isaac_dir/python.sh" ]]; then
    echo 'Set ISAAC_SIM_PATH to the Isaac Sim 5.1 installation directory.' >&2
    exit 1
fi
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$isaac_dir/python.sh" "$project_dir/sim_bridge/app.py" "$@"
