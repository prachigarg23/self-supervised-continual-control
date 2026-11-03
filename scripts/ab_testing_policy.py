import dataclasses
import enum
import logging
import socket

import tyro
import time 
import pandas as pd
import numpy as np

import pathlib
import sys
import os
root_path = pathlib.Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import subprocess
from multiprocessing import Process
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

import random

# from examples.aloha_real.main import Args as ClientArgs
# from examples.aloha_real.main import main as client_main

POLICY_DIR_MAP = {
    "<config_name_A>": "checkpoints/<config_name_A>/<exp_name>/<step>",
    "<config_name_B>": "checkpoints/<config_name_B>/<exp_name>/<step>",
    "pi05_FAST_aloha": "gs://openpi-assets/checkpoints/pi05_base"
}

HIDDEN_POLICY_MAP = {
}

@dataclasses.dataclass
class Args:
    policy_names: list[str] = dataclasses.field(default_factory=list)
    out_dir: str = "ab_testing_results/"
    row_first: bool = False
    
    client_prompt: str | None = None
    num_experiments: int = 1
    verbose: bool = False
    
@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""

@dataclasses.dataclass
class PolicyArgs:
    policy_name: str | None = None
    record: bool = False
    record_name: str | None = None
    port: int = 8000

def create_policy(ab_policy_args: PolicyArgs) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(ab_policy_args.policy_name), POLICY_DIR_MAP[ab_policy_args.policy_name]
    )
    
def start_server_process(ab_policy_args: PolicyArgs):
    policy = create_policy(ab_policy_args)
    policy_metadata = policy.metadata
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=ab_policy_args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()

def run_n_inference(n: int, ab_policy_args: PolicyArgs, args: Args) -> None:
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    secret_log_path = pathlib.Path(args.out_dir) / "secret_results_log.csv"
    
    p = Process(target=start_server_process, args=(ab_policy_args,))
    p.start()
    logging.info(f"Waiting for model {ab_policy_args.policy_name} to load...")
    time.sleep(10)

    try:
        client_command = (
            "source ~/interbotix_ws/devel/setup.bash && "
            "source ~/openpi_aloha_ws/devel/setup.bash && "
            f"source {root_path}/examples/aloha_real/.venv/bin/activate && "
            f"export PYTHONPATH=$PYTHONPATH:{root_path} && "
            f"python {root_path}/examples/aloha_real/main.py "
            f"--prompt '{args.client_prompt}' --port {ab_policy_args.port}"
        )

        for i in range(n):
            logging.info(f"\n[READY] Model: {ab_policy_args.policy_name} | Exp: {i+1}/{n}")
            user_input = input("Press Enter to start experiment (or type 'q' to stop this model): ").strip().lower()
            
            if user_input == 'q':
                logging.info(f"Terminating remaining experiments for {ab_policy_args.policy_name}")
                break

            print(f"\n===\tRunning experiment #{i+1}\t===")
            result = subprocess.run(
                client_command,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logging.error(f"Client failed: {result.stderr}")
            
            while True:
                try:
                    val = input(f"Enter full success(2) / partial(1) / fail(0) for {ab_policy_args.policy_name}: ").strip()
                    episode_human_grade = int(val)
                    if episode_human_grade in [0, 1, 2]: break
                except ValueError:
                    print("Invalid input. Please enter 0, 1, or 2.")

            data = {
                "timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")],
                "model_name": [ab_policy_args.policy_name],
                "experiment_number": [i + 1],
                "result": [episode_human_grade],
                "prompt": [args.client_prompt]
            }
            df_row = pd.DataFrame(data)
            df_row.to_csv(secret_log_path, mode='a', header=not secret_log_path.exists(), index=False)
            
            time.sleep(1)
            
    finally:
        p.terminate()
        p.join()
        logging.info(f"Server for {ab_policy_args.policy_name} terminated.")

def randomly_assign_models(args: Args) -> list[str]:
    """Randomizes the list of models provided in args."""
    shuffled_policies = list(args.policy_names)
    random.shuffle(shuffled_policies)
    return shuffled_policies

def row_wise_ab(args: Args) -> None: # TODO: too slow to load / unload each model constantly
    ...

def col_wise_ab(args: Args) -> None:
    shuffled_models = randomly_assign_models(args)
    logging.info(f"===> Randomized model order: {shuffled_models}")

    for policy_name in shuffled_models:
        run_n_inference(
            n=args.num_experiments,
            ab_policy_args=PolicyArgs(policy_name=policy_name),
            args=args
        )

def main(args: Args) -> None:
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, 
        format="%(levelname)s: %(message)s",
        force=True
    )
    if not args.policy_names:
        logging.error("No policy names provided!")
        return

    print(f"===\n\tRunning AB Test on: {args.policy_names}\n===")
    
    if args.row_first:
        row_wise_ab(args)
    else:
        col_wise_ab(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))