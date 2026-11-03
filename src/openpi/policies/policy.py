from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

import cv2

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        # import pdb; pdb.set_trace()

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        self._record_step += 1

        return results

# GPT generated
class VideoRecorder(_base_policy.BasePolicy):
    """
    Records side-by-side camera observations used by the policy into mp4 videos.

    Output:
      policy_records/<exp_name>/episode_XXX.mp4

    Cameras (left → right):
      cam_high | cam_left_wrist | cam_right_wrist
    """

    CAM_ORDER = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __init__(self, policy, record_dir="policy_records", exp_name="default", fps=10):
        self._policy = policy

        self.dir = pathlib.Path(record_dir) / exp_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.timestamps = []


        # infer next episode index (never overwrite)
        existing = sorted(self.dir.glob("episode_*.mp4"))
        self.episode = len(existing)

        self.frames: list[np.ndarray] = []
        self.fps = fps
        self._closed = False

    def _start_episode(self):
        self.frames = []
        self._closed = False

    def _end_episode(self):
        if not self.frames:
            return

        out_path = self.dir / f"episode_{self.episode:03d}.mp4"

        h, w, _ = self.frames[0].shape
        duration = self.timestamps[-1] - self.timestamps[0]
        fps = max(len(self.frames) / max(duration, 1e-6), 1.0)

        vw = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not vw.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        for f in self.frames:
            vw.write(f[..., ::-1])  # RGB → BGR

        vw.release()

        self.episode += 1
        self.frames = []
        self.timestamps = []

    def infer(self, obs):
        # start episode lazily on first frame
        if not self.frames:
            self._start_episode()

        images = obs.get("images", {})
        imgs = []

        for cam in self.CAM_ORDER:
            img = images.get(cam)
            if img is None:
                continue

            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # CHW → HWC
            img = img.transpose(1, 2, 0)
            imgs.append(img)

        if imgs:
            frame = np.concatenate(imgs, axis=1)
            prompt = obs.get("prompt")
            if prompt:
                frame = np.ascontiguousarray(frame)
                text = str(prompt)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.9
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
                x, y = 15, 15 + th
                # semi-transparent black background for legibility
                pad = 8
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x - pad, y - th - pad),
                    (x + tw + pad, y + baseline + pad),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            self.frames.append(frame)
            self.timestamps.append(time.monotonic())

        return self._policy.infer(obs)

    def close(self):
        # ensure exactly one flush per websocket connection
        if self._closed:
            return
        self._closed = True
        self._end_episode()
