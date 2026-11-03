### Starts raw_video_saver.py
# usage: not run directly; enabled by `main.py --enable_raw_video_saver` -> raw_videos_aligned/<test_set>/<model_str>/episode_NN.{mp4,png,txt} (H.264 crf16)
import pathlib
import subprocess

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2

from examples.aloha_real.prompt_overlay import overlay_prompt

# usb_cam topic  →  openpi topic. Only cam_high is collected/saved to keep
# per-rollout I/O minimal; wrist topics are left here (commented) for reference.
_TOPIC_MAP = {
    "/usb_cam_high/image_raw": "cam_high",
    # "/usb_cam_left_wrist/image_raw": "cam_left_wrist",
    # "/usb_cam_right_wrist/image_raw": "cam_right_wrist",
}


class RawVideoSaver(_subscriber.Subscriber):
    """Saves episode data from /usb_cam_view/image_raw."""

    def __init__(self, parent_dir: pathlib.Path, subsample: int = 1, prompt: str = "") -> None:
        parent_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = parent_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample
        self._prompt = prompt or ""
        # Pick next episode index by scanning existing episode_NN.mp4 files.
        existing = [int(p.stem.split("_")[1]) for p in parent_dir.glob("episode_*.mp4")
                    if p.stem.split("_")[1].isdigit()]
        self._counter = (max(existing) + 1) if existing else 0
        self._tag = ""

    @override
    def on_episode_start(self) -> None:
        self._tag = f"episode_{self._counter:02d}"
        self._images = {
            _TOPIC_MAP[cam_topic]: [] for cam_topic in _TOPIC_MAP
        }
        for cam_topic in _TOPIC_MAP:
            rospy.Subscriber(cam_topic, Image, self.recv_rostopic_img_callback, callback_args=_TOPIC_MAP[cam_topic])
        self._bridge = CvBridge()

    def recv_rostopic_img_callback(self, msg, cam_name) -> None:
        cv2_image = self._bridge.imgmsg_to_cv2(msg)
        self._images[cam_name].append(cv2_image)

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        ...
        
    @override
    def on_episode_end(self) -> None:
        # save prompt sidecar (consumed by grid_first_frames.py)
        if self._prompt:
            (self._out_dir / f"{self._tag}.txt").write_text(self._prompt + "\n")

        # save clean first frame of cam_high (no overlay) for grid composition
        high_frames = self._images.get("cam_high") or []
        if high_frames:
            first_bgr = cv2.cvtColor(np.asarray(high_frames[0]), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self._out_dir / f"{self._tag}.png"), first_bgr)

        # save cam_high singular view (with prompt overlay)
        cam_name = "cam_high"
        if cam_name in self._images and self._images[cam_name]:
            out_path = self._out_dir / f"{self._tag}.mp4"
            print(f"Saving RAW video to {out_path}")
            fps = 50 // max(1, self._subsample)
            height, width, _ = self._images[cam_name][0].shape

            # H.264 so the clips play in browsers; encoded from raw frames.
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
                 "-i", "-",
                 "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(out_path)],
                stdin=subprocess.PIPE,
            )

            for img in self._images[cam_name][::self._subsample]:
                bgr_frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                bgr_frame = overlay_prompt(bgr_frame, self._prompt)
                proc.stdin.write(np.ascontiguousarray(bgr_frame).tobytes())

            proc.stdin.close()
            proc.wait()

        self._counter += 1
### Ends raw_video_saver.py