# ALOHA (real robot)

Teleoperation, recording and policy evaluation on a real ALOHA. This folder is the `aloha` ROS package
(launch files, camera publishers); the robot client is in [examples/aloha_real](../../examples/aloha_real).
For hardware assembly see the [original ALOHA repo](https://github.com/tonyzhaozh/aloha).

## Setup

1. Install the ALOHA hardware and Interbotix software (`~/interbotix_ws`) as in the

   [original ALOHA repo](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation).
2. Build this `aloha` package in its own catkin workspace:

```bash
mkdir -p ~/openpi_aloha_ws/src
ln -s <path to self-supervised-continual-control>/third_party/aloha ~/openpi_aloha_ws/src/aloha
cd ~/openpi_aloha_ws
catkin_make
```

3. RealSense cameras:

```bash
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev
pip3 install pyrealsense2==2.55.0.4583
```

Set your camera serial numbers in `aloha_scripts/realsense_publisher.py`.

4. Virtual env for the policy server and robot client (from the repo root):

```bash
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client
```

## Recording demonstrations

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source ~/openpi_aloha_ws/devel/setup.bash
roslaunch aloha 4arms_teleop.launch
```

Terminal 2. `record_episodes.py` is from the original ALOHA repo. First add the task to `TASK_CONFIGS` in
`aloha_scripts/constants.py` next to it (our tasks are listed in this repo's
[aloha_scripts/constants.py](aloha_scripts/constants.py)):

```python
TASK_CONFIGS = {
    '<task>': {
        'dataset_dir': DATA_DIR + '/<task>',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
    },
}
```

```bash
cd ~/interbotix_ws/src/aloha/aloha_scripts
conda activate aloha
python record_episodes.py --episode_idx <idx> --task_name <task>
```

## Evaluation

Download a checkpoint (see [Checkpoints](../../README.md#checkpoints)), e.g.:

```bash
hf download prachigarg23/pi05_FAST_aloha_gear_insertion_ES --local-dir checkpoints/pi05_FAST_aloha_gear_insertion_ES
```

Terminal 1, robot and cameras:

```bash
source /opt/ros/noetic/setup.bash
source ~/openpi_aloha_ws/devel/setup.bash
roslaunch aloha inference.launch
```

Terminal 2, policy server (from the repo root):

```bash
source examples/aloha_real/.venv/bin/activate
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_FAST_aloha_gear_insertion_ES \
    --policy.dir=checkpoints/pi05_FAST_aloha_gear_insertion_ES

# zero-shot π0.5 base
python scripts/serve_policy.py --env ALOHA
```

Terminal 3, robot client (from the repo root):

```bash
source /opt/ros/noetic/setup.bash
source ~/interbotix_ws/devel/setup.bash
source ~/openpi_aloha_ws/devel/setup.bash
source examples/aloha_real/.venv/bin/activate
python -m examples.aloha_real.main --enable_raw_video_saver --model_str <name> --test_set <test_set> \
    --prompt "Pick up the purple gear and place it on the purple gear shaft using the arm. Use the other arm to stabilize the caterpillar base."
```

With `--enable_raw_video_saver`, videos are written to `raw_videos_aligned/<test_set>/<model_str>/`.

## A/B testing

[scripts/ab_testing_policy.py](../../scripts/ab_testing_policy.py) evaluates several checkpoints one after another, in a random order. It starts the policy server and the robot client itself; only
Terminal 1 (`inference.launch`) needs to be running. Set the checkpoint dir for each config in `POLICY_DIR_MAP` at the
top of the script, then (from the repo root, in the policy server's env):

```bash
python scripts/ab_testing_policy.py \
    --policy-names pi05_FAST_aloha_gear_insertion_ES pi05_FAST_aloha_gear_insertion_ES_SS \
    --num-experiments 20 \
    --client-prompt "Pick up the purple gear and place it on the purple gear shaft using the arm. Use the other arm to stabilize the caterpillar base."
```

Press Enter to start each rollout, then grade it: 2 = success, 1 = partial, 0 = fail. Grades are appended to
`ab_testing_results/secret_results_log.csv`.

## Troubleshooting

- Cameras or arms not found after replugging: `sudo udevadm control --reload && sudo udevadm trigger`
- `The robot 'master_left' is not discoverable` / `timeout exceeded while waiting for service /master_left/set_operating_modes`: reboot that motor in Dynamixel Wizard.
