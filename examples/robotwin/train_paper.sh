#!/bin/bash
set -euo pipefail

train() {
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py "$1" --exp-name="$2" --overwrite
}

case "${1:-}" in
    mid_training)
        python scripts/compute_norm_stats.py --config-name pi05_FAST_robotwin_mid_training
        train pi05_FAST_robotwin_mid_training paper
        ;;
    post_training_ES)
        train pi05_FAST_robotwin_post_training_ES paper
        ;;
    post_training_ES_SS)
        train pi05_FAST_robotwin_post_training_ES_SS paper
        ;;
    *)
        echo "Usage: bash examples/robotwin/train_paper.sh {mid_training|post_training_ES|post_training_ES_SS}"
        exit 1
        ;;
esac
