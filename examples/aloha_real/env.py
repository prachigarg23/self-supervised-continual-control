from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        save_height: int = 480,
        save_width: int = 640,
        prompt: str | None = None,
    ) -> None:
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height
        self._render_width = render_width
        self._save_height = save_height
        self._save_width = save_width
        self._prompt = prompt

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]
        
        obs["images_orig"] = {
            "cam_high": [],
            "cam_left_wrist": [],
            "cam_right_wrist": []
        }

        for cam_name in obs["images"]:
            img_orig = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._save_height, self._save_width)
            )
            obs["images_orig"][cam_name] = einops.rearrange(img_orig, "h w c -> c h w")
            
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")


        result = {
            "state": obs["qpos"],
            "images": obs["images"],
            "images_orig": obs["images_orig"],
        }
        if self._prompt is not None:
            result["prompt"] = self._prompt
        return result

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
