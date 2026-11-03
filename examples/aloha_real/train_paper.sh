#!/bin/bash
set -euo pipefail

train() {
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py "$1" --exp-name="$2" --overwrite
}

case "${1:-}" in
    pick_up_ES)
        train pi05_FAST_aloha_pick_up_ES paper
        ;;
    pick_up_ES_SS)
        train pi05_FAST_aloha_pick_up_ES_SS paper
        ;;
    gear_insertion_ES)
        train pi05_FAST_aloha_gear_insertion_ES paper
        ;;
    # gear_insertion_ES_SS)
    #     train pi05_FAST_aloha_gear_insertion_ES_SS paper
    #     ;;
    *)
        echo "Usage: bash examples/aloha_real/train_paper.sh {pick_up_ES|pick_up_ES_SS|gear_insertion_ES}"
        exit 1
        ;;
esac
