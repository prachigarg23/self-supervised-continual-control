#!/usr/bin/env bash
# Build a SELF-DEMO dataset (policy rollouts of a checkpoint on the 10 Stage-1 tasks, seeds from 5000;
# successes and failures are both kept):
#   1) generate:       third_party/robotwin/S1_policy_generated_data/<run>/<task>/episode_{i}.hdf5
#   2) process:        third_party/robotwin/S1_policy_generated_data/processed_<run>/  (ALOHA format, fixed task prompts)
#   3) convert + push: LeRobot dataset on the HF hub
#
# Usage: bash build_selfdemo_dataset_pipeline.sh <ckpt_step_dir> <train_config> <run_name> <rollouts_per_task> <hf_repo_id> \
#            [task_config (default demo_clean_eval; demo_clean_eval_topview for top view)] [exec_horizon (default 25)]
#   e.g. bash build_selfdemo_dataset_pipeline.sh checkpoints/pi05_FAST_robotwin_mid_training/paper/10000 \
#            pi05_FAST_robotwin_mid_training S1_rollouts_50 50 <hf_user>/robotwin_500_policy_rollouts
# Run in the env with openpi + RoboTwin + lerobot. Pick the GPU with CUDA_VISIBLE_DEVICES.
set -euo pipefail

CKPT=$(realpath "${1:?}"); CFG=${2:?}; RUN=${3:?}; N=${4:?}; REPO_ID=${5:?}
TASK_CONFIG=${6:-demo_clean_eval}; EXEC_HORIZON=${7:-25}
cd "$(dirname "$0")/../../third_party/robotwin"
set -x

python generate_selfdemos.py --ckpt "$CKPT" --cfg "$CFG" --log_dir "logdir_policy_generations/$RUN" \
  --test_num "$N" --hdf5_save_dir "$RUN" --task_config "$TASK_CONFIG" --exec_horizon "$EXEC_HORIZON"

python process_selfdemo_data.py --base-dir "S1_policy_generated_data/$RUN" \
  --save-dir "S1_policy_generated_data/processed_$RUN" --num-episodes "$N" --repo-id "$REPO_ID"
