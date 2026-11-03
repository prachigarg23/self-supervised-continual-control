import asyncio
import http
import logging
import os
import time
import traceback
import numpy as _np

_ENABLE_VLM_PROPOSER = os.environ.get("OPENPI_VLM_PROPOSER", "0") == "1"
if _ENABLE_VLM_PROPOSER:
    from scripts.vlm_proposer import propose_task_from_images

    def _call_vlm_propose_task(obs_images: dict, current_prompt: str):
        try:
            return propose_task_from_images(obs_images, current_prompt)
        except Exception:
            logging.getLogger(__name__).exception("Local VLM proposer failed")
            return None


import websockets
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self.vlm_proposer_prompt = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # propose new task via VLM task proposer
                if _ENABLE_VLM_PROPOSER and obs.get("propose_new_task", False): # Set to False by Default
                    try:
                        obs_images = {
                            k: v for k, v in obs.items()
                            if "image" in k and isinstance(v, _np.ndarray)
                        }
                        cur_prompt = obs.get("prompt")
                        if obs_images and cur_prompt is not None:
                            proposed = _call_vlm_propose_task(obs_images, str(cur_prompt))
                            if proposed:
                                logger.info(f"Replacing prompt \n'{cur_prompt}' with VLM-proposed task:  \n'{proposed}'")
                                obs["prompt"] = proposed
                    except Exception:
                        logger.exception("Error while calling VLM task proposer; continuing")

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                action["prompt"] = obs["prompt"]
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            # TODO: check how to save and not have to restart each time during real-robot evals
            except websockets.exceptions.WebSocketException:
                logger.info(f"Connection from {websocket.remote_address} closed")

                if hasattr(self._policy, "close"):
                    self._policy.close()

                break

            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
