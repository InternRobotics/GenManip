#!/bin/bash

ISAAC_PYTHON="/isaac-sim/python.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT" || exit 1

ASSETS_DIR="${REPO_ROOT}/saved/assets"
export MDL_SYSTEM_PATH="/isaac-sim/materials/:${ASSETS_DIR}/miscs/mdl/ebench/mdl:${ASSETS_DIR}/scene_usds/ebench/simple_pnp_v0.2/task3/SubUSDs/materials"

CONFIG_BASE="configs/tasks/ebench15/POC/desk/train"
LOG_DIR="logs/ebench15_POC_train"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CONVERT_TO_LEROBOT="${CONVERT_TO_LEROBOT:-1}"
LEROBOT_HOME="${LEROBOT_HOME:-${REPO_ROOT}/saved/lerobot_dataset/ebench15_lerobot/desk_v2}"

if [[ -z "${RANK:-}" ]]; then
    echo "RANK is not set. Set RANK to the zero-based config index before running this script."
    echo "Example: RANK=0 bash scripts/data_gen_dlc.sh"
    exit 1
fi

mkdir -p "$LOG_DIR"

mapfile -t CONFIG_FILES < <(find "$CONFIG_BASE" -maxdepth 1 -type f -name "*.yml" | sort)

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "No config files found in ${CONFIG_BASE}"
    exit 1
fi

for idx in "${!CONFIG_FILES[@]}"; do
    CONFIG_FILE="${CONFIG_FILES[$idx]}"
    TASK_NAME="$(basename "$CONFIG_FILE" .yml)"
    TASK_DATA_PATH="$(awk '/^[[:space:]]*task_name:/ {print $2; exit}' "$CONFIG_FILE")"
    RENDER_DATA_PATH="saved/demonstrations/${TASK_DATA_PATH}/render"

    if [[ $((RANK)) -ne "$idx" ]]; then
        echo "RANK ${RANK} -> Skipping ${TASK_NAME} (required RANK: ${idx})"
        continue
    fi

    echo "=========================================="
    echo "Processing ${TASK_NAME}"
    echo "Config: $CONFIG_FILE"
    echo "=========================================="

    echo "Running demogen for ${TASK_NAME} ..."
    "$ISAAC_PYTHON" demogen.py -cfg "$CONFIG_FILE" > "logs/${TASK_NAME}_demogen.log" 2>&1

    echo "Running render (no mdl variants) for ${TASK_NAME} ..."
    "$ISAAC_PYTHON" render.py -cfg "$CONFIG_FILE" > "logs/${TASK_NAME}_render_base.log" 2>&1

    if [[ "$CONVERT_TO_LEROBOT" == "1" ]]; then
        echo "Converting render data to LeRobot for ${TASK_NAME} ..."
        mkdir -p "$LEROBOT_HOME"
        export HF_LEROBOT_HOME="$LEROBOT_HOME"
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook)"
        elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
            source "/opt/conda/etc/profile.d/conda.sh"
        fi
        conda activate "$CONDA_ENV"
        echo "LeRobot output root: ${HF_LEROBOT_HOME}"
        python standalone_tools/genmanip2lerobot.py \
            --data_path "$RENDER_DATA_PATH" \
            --robot lift2 \
            --overwrite > "logs/${TASK_NAME}_lerobot.log" 2>&1
    fi

    # "$ISAAC_PYTHON" scripts/duplicate_trajectory_with_mdl_variants.py \
    #     --trajectory-dir "saved/demonstrations/ebench/simple_pnp_v0.2/${TASK_NAME}/trajectory/" \
    #     --replace-json "configs/tasks/ebench/simple_pnp_v0.2/train_replace.json" \
    #     --task-name "task${i}" \
    #     --rewrite-target visual > "logs/${TASK_NAME}_variants.log" 2>&1

    # echo "Running render for ${TASK_NAME} ..."
    # "$ISAAC_PYTHON" render.py -cfg "$CONFIG_FILE" > "${LOG_DIR}/${TASK_NAME}_render.log" 2>&1
done

echo "All tasks completed!"
