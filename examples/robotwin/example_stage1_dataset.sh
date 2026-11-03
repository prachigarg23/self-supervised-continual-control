#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROBOTWIN="$SCRIPT_DIR/../../third_party/robotwin"
OUT_SUBDIR=multi_task
REPO_ID=prachigarg23/robotwin_mid_training_expert

eval "$(conda shell.bash hook)"
conda activate RoboTwin

# Tasks. Each task's fixed prompt comes from third_party/robotwin/task_prompts.json.
declare -a TASKS=(
  place_dual_shoes
  put_bottles_dustbin
  beat_block_hammer
  handover_mic
  lift_pot
  open_microwave
  pick_dual_bottles
  place_bread_basket
  place_empty_cup
  place_container_plate
)

n=${#TASKS[@]}

cd "$ROBOTWIN"
for ((i = 0; i < n; i++)); do
  bash collect_data.sh "${TASKS[i]}" demo_clean 0
done

cd "$SCRIPT_DIR"
for ((i = 0; i < n; i++)); do
  python process_expert_data.py "${TASKS[i]}" demo_clean 50 --fixed-instruction
done

mkdir -p "$SCRIPT_DIR/processed_data/$OUT_SUBDIR"
for ((i = 0; i < n; i++)); do
  src="$SCRIPT_DIR/processed_data/${TASKS[i]}-demo_clean-50"
  dest="$SCRIPT_DIR/processed_data/$OUT_SUBDIR/$(basename "$src")"
  rm -rf "$dest"
  mv "$src" "$SCRIPT_DIR/processed_data/$OUT_SUBDIR/"
done

cd "$SCRIPT_DIR"
uv run python convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir "$SCRIPT_DIR/processed_data/$OUT_SUBDIR/" \
  --repo-id "$REPO_ID" \
  --push-to-hub 2>&1 | tee conversion.log
