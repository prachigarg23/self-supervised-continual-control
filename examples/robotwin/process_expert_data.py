import sys

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def decode_image(img):
    if img.ndim == 3:  # already (H, W, C) raw array
        return img
    return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path, fixed_instruction=None, desc_type="seen"):
    begin = 0
    floders = os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fixed = (fixed_instruction or "").strip()
    if fixed:
        print(f"Using fixed instruction for all episodes ({len(fixed)} chars); ignoring {path}/instructions/episode*.json")

    for i in range(episode_num):

        if fixed:
            instructions = [fixed]
        else:
            instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
            with open(instruction_data_path, "r") as f_instr:
                instruction_dict = json.load(f_instr)
            instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        underscore_path = os.path.join(path, f"episode_{i}.hdf5")
        legacy_path = os.path.join(path, "data", f"episode{i}.hdf5")
        hdf5_input_path = underscore_path if os.path.exists(underscore_path) else legacy_path
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(hdf5_input_path)
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])  # joints angle

            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                cam_high.append(cv2.resize(decode_image(image_dict["head_camera"][j]), (640, 480)))
                cam_right_wrist.append(cv2.resize(decode_image(image_dict["right_camera"][j]), (640, 480)))
                cam_left_wrist.append(cv2.resize(decode_image(image_dict["left_camera"][j]), (640, 480)))

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--fixed-instruction",
        type=str,
        nargs="?",
        const=None,
        default=None,
        help="Default mode: every episode gets the task's single prompt from third_party/robotwin/task_prompts.json. "
        "Optionally pass a string to use that prompt instead.",
    )
    parser.add_argument(
        "--random-prompts",
        action="store_true",
        help="Original RoboTwin behaviour: per-episode prompts from instructions/episode*.json.",
    )
    parser.add_argument(
        "--instruction-key",
        type=str,
        default="seen",
        choices=("seen", "unseen"),
        help="Which key to read from episode instruction JSON with --random-prompts.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override input data directory. HDF5 files should be episode_{i}.hdf5 directly inside this dir.",
    )
    args = parser.parse_args()

    if args.random_prompts and args.fixed_instruction:
        parser.error("--random-prompts and --fixed-instruction are mutually exclusive")
    if args.random_prompts:
        fixed_instruction = None
        print(f"Prompt mode: random RoboTwin prompts (instructions/episode*.json, key '{args.instruction_key}')")
    elif args.fixed_instruction:
        fixed_instruction = args.fixed_instruction
        print(f"Prompt mode: fixed, custom string: {fixed_instruction!r}")
    else:
        prompts_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                     "third_party", "robotwin", "task_prompts.json"))
        with open(prompts_path, "r") as f:
            task_prompts = json.load(f)
        if args.task_name not in task_prompts:
            parser.error(f"no prompt for '{args.task_name}' in {prompts_path}; "
                         "add it there, pass --fixed-instruction \"<prompt>\", or use --random-prompts")
        fixed_instruction = task_prompts[args.task_name]
        print(f"Prompt mode: fixed (task_prompts.json): {fixed_instruction!r}")

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    if args.data_dir:
        load_dir = args.data_dir
    else:
        load_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "robotwin", "data", str(task_name), str(setting))

    begin = 0
    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        fixed_instruction=fixed_instruction,
        desc_type=args.instruction_key,
    )
