import logging

import websockets.sync.client
from openpi_client import msgpack_numpy


PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class WebsocketClientPolicy:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        logging.info(f"Waiting for server at {self._uri}...")
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: dict):
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: dict):
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        self._ws.send(data)
        return self._ws.recv()
