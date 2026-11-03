#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROBOTWIN="$SCRIPT_DIR/../../third_party/robotwin"

eval "$(conda shell.bash hook)"
conda activate RoboTwin

cd "$ROBOTWIN"
bash collect_data.sh stack_blocks_two_green_on_red demo_clean 0
bash collect_data.sh stack_blocks_two_red_on_green demo_clean 0

cd "$SCRIPT_DIR"
# Fixed prompts come from third_party/robotwin/task_prompts.json.
python process_expert_data.py stack_blocks_two_green_on_red demo_clean 50 --fixed-instruction
python process_expert_data.py stack_blocks_two_red_on_green demo_clean 50 --fixed-instruction

mkdir -p "$SCRIPT_DIR/processed_data/stack_blocks_stage2_mini"
mv "$SCRIPT_DIR/processed_data/stack_blocks_two_green_on_red-demo_clean-50" "$SCRIPT_DIR/processed_data/stack_blocks_stage2_mini/"
mv "$SCRIPT_DIR/processed_data/stack_blocks_two_red_on_green-demo_clean-50" "$SCRIPT_DIR/processed_data/stack_blocks_stage2_mini/"

cd "$SCRIPT_DIR"
uv run python convert_aloha_data_to_lerobot_robotwin.py \
  --raw-dir "$SCRIPT_DIR/processed_data/stack_blocks_stage2_mini/" \
  --repo-id prachigarg23/robotwin_post_training_stack_blocks_expert \
  --push-to-hub
