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

## Real model validation

`real_model_probe.py` runs the actual local Pi05 `Model`, observation adapter,
`Policy`, JAX sampler and output transforms. It does not start a WebSocket server,
import ManiMux, connect a robot or claim task success. Its `--help` works in the
lightweight environment, but **capture/verify require the real Pi05 environment**.
The CPU tests of the probe itself are separate:

```bash
.venv/bin/python tests/pi05_trace/real_model_probe.py --help
.venv/bin/python -m pytest tests/pi05_trace/test_real_model_probe.py -q
```

Prepare these inputs before running model inference:

- Two clean committed XPolicyLab checkouts, the old default backend and the trace
  patch. The script can load either checkout using `--checkout`, without editing it.
- One resolved standalone Pi_05 deployment YAML, used unchanged in both runs.
  Set a local `model_path`, the correct normalization assets, native ALOHA/YAM
  training config, joint action type and explicit dual 6+1 layout. Use one env.
  Do not pass an unresolved ManiMux experiment YAML.
- Existing local JAX checkpoint shards, normalization files and cached PaliGemma
  tokenizer. This probe prohibits OpenPI asset downloads or refreshes, returning
  existing local/cache paths only. It never installs dependencies. Cache location
  is `OPENPI_DATA_HOME`, or `~/.cache/openpi` when unset. Directory symlinks inside
  checkpoint/assets are rejected to prevent incomplete content fingerprints.
- A representative observation JSON and non-pickle NPZ. Images must already be
  decoded RGB arrays, as received by Model.update_obs. Encoded image transport is
  outside this model probe. For example, a native YAM JSON is:

```json
{
  "schema_version": 1,
  "source": "recorded observation from <session/request>, no robot dispatch",
  "observation": {
    "observation/state": {"array": "state"},
    "observation/image": {"array": "top_rgb"},
    "observation/left_wrist": {"array": "left_rgb"},
    "observation/right_wrist": {"array": "right_rgb"},
    "instruction": "put the bottle into the basket"
  }
}
```

The NPZ contains `state` (14 joints), `top_rgb`, `left_rgb`, `right_rgb` (HWC uint8),
and `noise` (float32 `[action_horizon, model_action_dim]`, normally `[50,32]`).
Use the same saved initial noise in both runs. An ALOHA observation can instead
use `state`, `images/cam_high`, `images/cam_left_wrist`, `images/cam_right_wrist`
and `instruction` in the JSON tree, with the same explicit `{"array": "key"}`
references. Object/pickle arrays, missing references, unused arrays, nonfinite
values and incompatible noise are rejected. A synthetic observation is allowed
only when its `source` describes that limitation accurately.

Use the **same probe file and model environment** for both phases. Capture calls
the old Policy without the unsupported `return_denoise_trace` keyword. It refuses
a trace-capable Policy as the old baseline. Replace the absolute example paths
and `TRACE_SHA` with the actual reviewed full commit before running:

```bash
PROBE=/path/to/trace/XPolicyLab/tests/pi05_trace/real_model_probe.py
MODEL_PYTHON=/path/to/pi05-model-environment/bin/python
OLD_XPL=/path/to/old/XPolicyLab
TRACE_XPL=/path/to/trace/XPolicyLab
OLD_SHA=02e74cbcfc36164cb0d3bbfddfc2b645a087d325
TRACE_SHA=REPLACE_WITH_FULL_REVIEWED_TRACE_COMMIT

"$MODEL_PYTHON" "$PROBE" capture \
  --checkout "$OLD_XPL" --expected-revision "$OLD_SHA" \
  --config /path/to/resolved-pi05.yaml \
  --observation /path/to/observation.json --arrays /path/to/observation.npz \
  --num-steps 1 10 20 --repeats 3 --output /path/to/artifacts/trace-baseline

"$MODEL_PYTHON" "$PROBE" verify \
  --checkout "$TRACE_XPL" --expected-revision "$TRACE_SHA" \
  --baseline /path/to/artifacts/trace-baseline --baseline-revision "$OLD_SHA" \
  --config /path/to/resolved-pi05.yaml \
  --observation /path/to/observation.json --arrays /path/to/observation.npz \
  --num-steps 1 10 20 --repeats 3 --output /path/to/artifacts/trace-verified
```

The script rejects dirty source, overwritten output, a baseline from the modified
checkout, changed probe code, changed inputs/configuration, different model shapes,
package/Python/device identities, or mismatched checkpoint/normalization contents.
Weights and assets are identified by SHA256 of actual file contents, not supplied
labels or file sizes. Hashing large checkpoints is an explicit one-time startup
cost, excluded from reported inference timing. The same resolved asset paths are
required for this controlled same-machine comparison.

`report.json` and `arrays.npz` preserve successful and failed evidence:

- Old backend versus trace-off uses exact action equality. Trace-on versus both
  uses explicit tolerances (`rtol=1e-5`, `atol=1e-6` by default), with maximum error
  recorded per comparison. Do not loosen tolerances just to make a failure pass.
- Actual trace must be finite float32 `[T,H,Dmodel]`. Its Euler reconstruction,
  followed by the real output transforms, is compared with the final actions.
  This checks trace/action alignment, not an independent per-step network oracle.
- Seeded consecutive calls without supplied noise compare trace-on/off/old actions
  and final real JAX RNG state. Repeated reset requires exact sequence replay.
- The real Model output is checked against Policy output for both trace and packed
  joint actions. No fake policy, JIT or model is used in capture/verify.
- Wall time surrounds complete infer plus host array materialization. First-call
  and subsequent timings are separate. First-call may include compilation, but
  existing compilation caches are not cleared. Trace reconstruction validation is
  excluded from that timing. Output array bytes are reported, not WS payload size.

A passing local model probe still leaves the encoded-image debug client, actual
WebSocket transport, complete ManiMux AHS loop, physical execution and memory/task
measurements outstanding. Missing model dependencies or assets produce a failure
report and nonzero exit, never a synthetic replacement or deployment-ready result.
