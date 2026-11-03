import logging
import os
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber

_DEBUG_TIMING = os.environ.get("OPENPI_DEBUG_TIMING", "0") == "1"


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        if _DEBUG_TIMING:
            t_ep = time.time()
            logging.info("TIMING episode_start t=%.6f", t_ep)
            print(f"TIMING episode_start t={t_ep:.6f}")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            if _DEBUG_TIMING:
                loop_start = time.time()
                logging.info("TIMING loop_start step=%d t=%.6f", self._episode_steps, loop_start)
                print(f"TIMING loop_start step={self._episode_steps} t={loop_start:.6f}")
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                sleep_dur = step_time - dt
                if _DEBUG_TIMING:
                    logging.info(
                        "TIMING sleep_start step=%d t=%.6f dur=%.6f",
                        self._episode_steps,
                        now,
                        sleep_dur,
                    )
                    print(f"TIMING sleep_start step={self._episode_steps} t={now:.6f} dur={sleep_dur:.6f}")
                time.sleep(sleep_dur)
                if _DEBUG_TIMING:
                    after_sleep = time.time()
                    logging.info(
                        "TIMING sleep_end step=%d t=%.6f",
                        self._episode_steps,
                        after_sleep,
                    )
                    print(f"TIMING sleep_end step={self._episode_steps} t={after_sleep:.6f}")
                last_step_time = time.time()
            else:
                last_step_time = now

        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop."""
        if _DEBUG_TIMING:
            t0 = time.time()
            logging.info("TIMING get_observation_start t=%.6f", t0)
            print(f"TIMING get_observation_start t={t0:.6f}")
        observation = self._environment.get_observation()
        if _DEBUG_TIMING:
            t1 = time.time()
            logging.info("TIMING get_observation_end t=%.6f", t1)
            print(f"TIMING get_observation_end t={t1:.6f}")

        if _DEBUG_TIMING:
            t2 = time.time()
            logging.info("TIMING get_action_start t=%.6f", t2)
            print(f"TIMING get_action_start t={t2:.6f}")
        action = self._agent.get_action(observation)
        if _DEBUG_TIMING:
            t3 = time.time()
            logging.info("TIMING get_action_end t=%.6f", t3)
            print(f"TIMING get_action_end t={t3:.6f}")

        if _DEBUG_TIMING:
            t4 = time.time()
            logging.info("TIMING apply_action t=%.6f", t4)
            print(f"TIMING apply_action t={t4:.6f}")
        self._environment.apply_action(action)
        if _DEBUG_TIMING:
            t5 = time.time()
            logging.info("TIMING step_end t=%.6f", t5)
            print(f"TIMING step_end t={t5:.6f}")

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
