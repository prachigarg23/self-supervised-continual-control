# usage: python -m examples.aloha_real.main --enable_raw_video_saver --model_str <name> --test_set <name> --prompt "<text>" [--enable_hdf5_saver] [--num_episodes N] [--max_episode_steps N]
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env
import pathlib
from datetime import datetime

from examples.aloha_real.rollout_hdf5_saver import RolloutHDF5Saver
from examples.aloha_real.raw_video_saver import RawVideoSaver

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000
    prompt: str | None = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Videos are written to raw_videos_aligned/<test_set>/<model_str>/episode_NN.*
    out_dir: str = ""
    enable_raw_video_saver: bool = False
    test_set: str = ""  # e.g. "test_set_A"

    model_str: str = "DEFAULT"
    enable_hdf5_saver: bool = False
    hdf5_dir: str = "rollouts_hdf5"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()

    subscribers = []
    if args.enable_hdf5_saver:
        subscribers.append(
            RolloutHDF5Saver(pathlib.Path(args.hdf5_dir) / args.model_str, prompt=args.prompt)
        )
    if args.enable_raw_video_saver:
        if args.out_dir:
            raw_dir = pathlib.Path(args.out_dir)
        else:
            raw_dir = pathlib.Path("raw_videos_aligned")
            if args.test_set:
                raw_dir = raw_dir / args.test_set
            raw_dir = raw_dir / args.model_str
        subscribers.append(RawVideoSaver(raw_dir, prompt=args.prompt or ""))

    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(
            reset_position=metadata.get("reset_pose"),
            prompt=args.prompt,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=subscribers,
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args: Args = tyro.cli(Args)
    main(args)
