#!/usr/bin/env bash
# Build an EXPERT dataset (scripted RoboTwin planner):
#   1) collect demos:  third_party/robotwin/data/<task>/<task_config>/   (episodes = episode_num in the task config)
#   2) process:        ALOHA-format hdf5 with each task's fixed prompt from third_party/robotwin/task_prompts.json
#   3) merge:          examples/robotwin/processed_data/<dataset>/
#   4) convert + push: LeRobot dataset on the HF hub
#
# Usage: bash build_expert_dataset_pipeline.sh <task_config> <episodes> <dataset_name> <hf_repo_id> <task> [<task> ...]
#   e.g. bash build_expert_dataset_pipeline.sh demo_clean 50 stage2 <hf_user>/stack_blocks \
#            stack_blocks_two_green_on_red stack_blocks_two_red_on_green
# Optional env vars: GPU (default 0), START_SEED (default 0), CONVERT_PY (python with lerobot, default: python)
set -euo pipefail

TASK_CONFIG=${1:?}; N=${2:?}; DATASET=${3:?}; REPO_ID=${4:?}; shift 4
TASKS=("$@")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROBOTWIN="$SCRIPT_DIR/../../third_party/robotwin"
set -x

for task in "${TASKS[@]}"; do
  (cd "$ROBOTWIN" && bash collect_data.sh "$task" "$TASK_CONFIG" "${GPU:-0}" ${START_SEED:-})
done

mkdir -p "$SCRIPT_DIR/processed_data/$DATASET"
for task in "${TASKS[@]}"; do
  (cd "$SCRIPT_DIR" && python process_expert_data.py "$task" "$TASK_CONFIG" "$N" --fixed-instruction \
    --data-dir "$ROBOTWIN/data/$task/$TASK_CONFIG")
  rm -rf "$SCRIPT_DIR/processed_data/$DATASET/$task-$TASK_CONFIG-$N"
  mv "$SCRIPT_DIR/processed_data/$task-$TASK_CONFIG-$N" "$SCRIPT_DIR/processed_data/$DATASET/"
done

"${CONVERT_PY:-python}" "$SCRIPT_DIR/convert_aloha_data_to_lerobot_robotwin.py" \
  --raw-dir "$SCRIPT_DIR/processed_data/$DATASET/" --repo-id "$REPO_ID" --push-to-hub
