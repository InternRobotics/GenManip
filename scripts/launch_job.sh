#!/bin/bash
# save as: start.sh
# Notes for users:
# - If your model server is external (or required by your policy), start it separately before running this script.
# - If you have a custom client, replace standalone_tools/client.py and update the call below.

# ========================= CLI =========================
GENMANIP_PATH="${GENMANIP_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
CLIENT_PYTHON="${CLIENT_PYTHON:-python}"
GMP_BIN="${GMP_BIN:-gmp}"
NUM_CLIENTS="${NUM_CLIENTS:-2}"
RUN_ID="${RUN_ID:-eval_$(date -u +%Y%m%dT%H%M%SZ)}"
SUBMIT_TARGET="${SUBMIT_TARGET:-ebench}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-p GENMANIP_PATH] [-n NUM_CLIENTS] [-r RUN_ID] [-s SUBMIT_TARGET]

  -p  Path to GenManip-Sim (default: $GENMANIP_PATH)
  -n  Number of clients to launch (default: $NUM_CLIENTS)
  -r  Run id for gmp submit (default: $RUN_ID)
  -s  Submit target for gmp submit (default: $SUBMIT_TARGET)
EOF
}

while getopts ":p:n:r:s:h" opt; do
    case "$opt" in
        p) GENMANIP_PATH="$OPTARG" ;;
        n) NUM_CLIENTS="$OPTARG" ;;
        r) RUN_ID="$OPTARG" ;;
        s) SUBMIT_TARGET="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 1 ;;
        :) echo "Missing argument for -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

if ! [[ "$NUM_CLIENTS" =~ ^[0-9]+$ ]] || [ "$NUM_CLIENTS" -lt 1 ]; then
    echo "NUM_CLIENTS must be a positive integer. Got: $NUM_CLIENTS" >&2
    exit 1
fi

# ========================= Start Server =========================
cd "$GENMANIP_PATH" || exit 1

LOG_DIR="$GENMANIP_PATH/logs/servers"
mkdir -p "$LOG_DIR"

export MDL_SYSTEM_PATH="${MDL_SYSTEM_PATH:-/isaac-sim/materials/:$GENMANIP_PATH/saved/assets/scene_usds/ebench/simple_pnp/task3/SubUSDs/materials:$GENMANIP_PATH/saved/assets/miscs/mdl/ebench/mdl}"

echo "Installing Isaac Sim requirements..."
"$ISAAC_PYTHON" -m pip install -r requirements.txt

echo "Starting Ray server..."
RANK="${RANK:-unknown}"
SERVER_LOG="$LOG_DIR/server_${RANK}.log"
setsid "$ISAAC_PYTHON" ray_eval_server.py --no_save_process >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

sleep 25


# ========================= Start Client =========================
echo "Installing client package..."
"$CLIENT_PYTHON" -m pip install -e "standalone_tools/packages/genmanip_client[full_numpy1]"

echo "Submitting job: $SUBMIT_TARGET"
"$GMP_BIN" submit "$SUBMIT_TARGET" --run_id "$RUN_ID"

sleep 15

echo "Starting $NUM_CLIENTS clients..."
CLIENT_PIDS=()
for ((i=0; i<NUM_CLIENTS; i++)); do
    "$CLIENT_PYTHON" standalone_tools/client.py -a r5a -g lift2 --worker_ids "$i" --chunk_size 50 &
    CLIENT_PIDS+=("$!")
done

check_clients() {
    for pid in "${CLIENT_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

while check_clients; do
    sleep 1
done

# Kill the server since all clients are dead
kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
