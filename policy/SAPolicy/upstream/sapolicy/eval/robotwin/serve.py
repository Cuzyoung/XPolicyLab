"""Standalone SAPolicy model server, wire-compatible with RoboTwin's ModelClient.

RoboTwin ships script/policy_model_server.py for this, but its module-level
`from envs._GLOBAL_CONFIGS import CONFIGS_PATH` pulls in envs/__init__.py, which
imports sapien -- so it cannot run in the policy env. This file speaks the exact
same protocol without importing anything from `envs`:

    request  : 4-byte big-endian length + JSON {"cmd": <model method>, "obs": <arg|null>}
    response : 4-byte big-endian length + JSON {"res": <return value>}
    numpy    : {"__numpy_array__": true, "data": <b64>, "dtype": ..., "shape": ...}

Run in the sa env:
    workspace=/home/jw/proj/workspace \
    PYTHONPATH=./policy:/path/to/SpatialAlignVLA \
    python policy/SAPolicy/serve.py --config policy/SAPolicy/deploy_policy.yml --port 9977
"""
import argparse
import base64
import json
import socket
import socketserver
import threading
import traceback
from typing import Any

import numpy as np
import yaml


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(text: str) -> Any:
    def hook(dct):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(text, object_hook=hook)


def _recv_exact(sock, n):
    chunks, remaining = [], n
    while remaining:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# 观测包有兆字节级(3 路相机 × 322×238 的 JSON),真实长度远小于此;超过就说明
# 长度头是从消息中间截出来的,即流已经错位。
_MAX_MSG = 256 << 20


def _recv_header(sock):
    """读满 4 字节长度头;连接在消息边界上正常关闭时返回 None。

    这里不能用 sock.recv(4) —— TCP 不保证一次读满。短读一次,长度就按错位的字节
    算出来,服务端于是等一个永远不会到的包、客户端等一个永远不会来的回复,两边
    各等各的。实测就是这样:一个分片跑满 50 个 episode,另一个在第 3 个上永久
    阻塞在 poll(),整轮被看门狗砍成部分结果。
    """
    chunks, remaining = [], 4
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if remaining == 4:
                return None          # 干净的 EOF,正常收尾
            raise ConnectionError("peer closed mid-header")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[serve] client connected: {self.client_address}", flush=True)
        while True:
            try:
                head = _recv_header(self.request)
                if head is None:
                    break
                size = int.from_bytes(head, "big")
                if size <= 0 or size > _MAX_MSG:
                    raise ConnectionError(f"implausible message length {size}; stream desynced")
                body = _recv_exact(self.request, size)
                data = json_to_numpy(body.decode("utf-8"))
                cmd, obs = data.get("cmd"), data.get("obs")
                method = getattr(self.server.model, cmd, None)
                if not callable(method):
                    raise AttributeError(f"no model method named {cmd!r}")
                result = method(obs) if obs is not None else method()
                payload = numpy_to_json({"res": result}).encode("utf-8")
            except ConnectionError:
                break
            except Exception as exc:
                traceback.print_exc()
                payload = numpy_to_json({"error": str(exc)}).encode("utf-8")
            self.request.sendall(len(payload).to_bytes(4, "big"))
            self.request.sendall(payload)
        print(f"[serve] client disconnected: {self.client_address}", flush=True)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    port = args.port or int(cfg.get("port", 9977))

    try:
        from SAPolicy import get_model  # RoboTwin tree: policy/SAPolicy is a symlink here
    except ImportError:
        from sapolicy.eval.robotwin import get_model  # launched from the SA repo

    print("[serve] loading policy ...", flush=True)
    model = get_model(cfg)

    server = Server((args.host, port), Handler)
    server.model = model
    print(f"[serve] listening on {args.host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
