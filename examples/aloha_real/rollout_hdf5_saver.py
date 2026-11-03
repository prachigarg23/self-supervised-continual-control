# usage: not run directly; enabled by `main.py --enable_hdf5_saver` -> rollouts_hdf5/<model_str>/episode_NN.hdf5 + task_instructions.json
import pathlib
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

import time
import h5py
import einops
import json

# usb_cam topic  →  openpi topic
_TOPIC_MAP = {
    "cam_high": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist"
}

class RolloutHDF5Saver(_subscriber.Subscriber):
    def __init__(self, out_dir: pathlib.Path, subsample: int = 1, prompt: str = "") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample
        self._prompt = prompt
        
        existing_files = list(out_dir.glob("episode_*.hdf5"))
        if not existing_files:
            self._episode_counter = 0
        else:
            indices = [
                int(f.stem.split("_", 1)[1])
                for f in existing_files
                if f.stem.startswith("episode_") and f.stem.split("_", 1)[1].isdigit()
            ]
            self._episode_counter = max(indices) + 1 if indices else 0

    @override
    def on_episode_start(self) -> None:
        self.data_dict = {
            '/observations/qpos': [],
            '/action': [],
        }
        for cam_name in _TOPIC_MAP:
            self.data_dict[f'/observations/images/{cam_name}'] = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        qpos = observation['state']
        action = action['actions']
        self.data_dict["/observations/qpos"].append(qpos)
        self.data_dict["/action"].append(action)
        for cam_name in observation['images_orig']:
            self.data_dict[f'/observations/images/{cam_name}'].append(
                einops.rearrange(observation["images_orig"][cam_name], "c h w -> h w c")
            )
                
    @override
    def on_episode_end(self) -> None:
        file_path = self._out_dir / f"episode_{self._episode_counter}.hdf5"
        txt_path = self._out_dir / f"prompts.txt" # Path for the prompt
        json_path = self._out_dir / "task_instructions.json"
        
        print(f"Saving episode {self._episode_counter} to {file_path}")
        
        print(f"Give score of rollout(0=full fail, 1=grasp fail, 2=success): ", end="")
        try:
            human_score = int(input())
        except ValueError:
            human_score = -1
        print()
        with open(txt_path, 'a') as f:
            f.write(f"{file_path} - {self._prompt} - {human_score}\n")

        tasks_data = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    tasks_data = json.load(f)
            except json.JSONDecodeError:
                tasks_data = {}

        tasks_data[str(self._episode_counter)] = self._prompt

        with open(json_path, 'w') as f:
            json.dump(tasks_data, f, indent=2)
            
        num_timesteps = len(self.data_dict['/action'])
        
        t0 = time.time()
        with h5py.File(file_path, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in _TOPIC_MAP:
                _ = image.create_dataset(cam_name, (num_timesteps, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3))
            _ = obs.create_dataset('qpos', (num_timesteps, 14))
            _ = root.create_dataset('action', (num_timesteps, 14))
            
            for name, array in self.data_dict.items():
                root[name][...] = array
                
        print(f'Saving complete: {time.time() - t0:.1f} secs')
        self._episode_counter += 1