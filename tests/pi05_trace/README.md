# Lightweight Pi05 trace contract tests

From the XPolicyLab checkout, create a separate Python 3.11 environment:

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python tests/pi05_trace/requirements.txt
.venv/bin/python -m pytest tests/unit/test_pi05_denoise_trace.py -q
```

`requirements.txt` pins every transitive package. The input pins can be resolved
again with:

```bash
uv pip compile --python-version 3.11 tests/pi05_trace/requirements.in \
  --output-file tests/pi05_trace/requirements.txt
```

These tests import the actual adapter, shared action pack/unpack helpers,
plain-RGB observation decoder, WebSocket handlers and msgpack codec. The protocol
tests call handlers in process and never bind a socket. Model initialization uses
synthetic configuration objects and a deterministic fake policy; no checkpoint or
robot is constructed.

The Policy tests also import the actual `openpi.policies.policy.Policy` and actual
transform composition. A sentinel output transform scales, offsets and crops
actions, making it observable if normalized traces were transformed incorrectly.
JAX array/tree/random interfaces use scoped NumPy-backed stubs, and the Flax JIT
wrapper is a recording stub. These tests verify the Python boundary, lazy wrapper
creation and call routing; they do **not** run or validate the native JAX sampler,
JIT compilation, model numerics, native PRNG behavior, GPU performance or training.

The optional `cv2` and `h5py` imports are unavailable stubs that fail if used.
Only already decoded RGB arrays are covered; encoded JPEG/HDF5 paths are outside
this lightweight suite. Import substitutions and paths are restored after each
fixture to avoid changing the dependency graph seen by other tests.

This environment intentionally contains no OpenPI backend, Torch, JAX, Flax,
camera/robot SDKs or model weights. Existing backend-dependent test modules need
their documented backend environment; a focused pass here is not a full-suite,
real-model or hardware validation claim.
