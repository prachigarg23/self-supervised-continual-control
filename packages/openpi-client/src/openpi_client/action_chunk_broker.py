from typing import Dict
import logging
import os
import time

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

_DEBUG_TIMING = os.environ.get("OPENPI_DEBUG_TIMING", "0") == "1"


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            if _DEBUG_TIMING:
                t0 = time.time()
                logging.info("TIMING chunk_infer_start t=%.6f", t0)
                print(f"TIMING chunk_infer_start t={t0:.6f}")
            self._last_results = self._policy.infer(obs)
            if _DEBUG_TIMING:
                t1 = time.time()
                logging.info("TIMING chunk_infer_end t=%.6f", t1)
                print(f"TIMING chunk_infer_end t={t1:.6f}")
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if _DEBUG_TIMING:
            t2 = time.time()
            logging.info(
                "TIMING chunk_step index=%d t=%.6f",
                self._cur_step - 1,
                t2,
            )
            print(f"TIMING chunk_step index={self._cur_step - 1} t={t2:.6f}")

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
