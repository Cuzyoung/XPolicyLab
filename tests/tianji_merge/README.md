# Tianji CPU merge regression suite

Run before and after merging upstream main, using the Pi05 Python environment:

```bash
policy/Pi_05/openpi/.venv/bin/python tests/tianji_merge/run_cpu.py
```

The runner starts fresh processes with CUDA hidden, JAX forced to CPU, offline
model downloads and two CPU math threads. It verifies the backend and source
checkout before collecting tests. It does not start services or load production
weights. An interpreter in another checkout may be used for a Git worktree.

Coverage:

- Synthetic checkpoint metadata: joint wrist/three-camera, Joint-FK, ratio and
  H50 identity, timing, normalization width and checksum failures.
- Actual joint model factory and initialization with an injected inference
  backend: arm ordering, RGB masks, gripper-only clipping, read-only RTC inputs,
  sampler forwarding, duplicate observation rejection and reset.
- H32/H50 TCP default/RTC round trips and the pass-ball profile's isolation.
- Tiny real Orbax checkpoint restoration in float32/bfloat16: ordinary versus
  low-memory value equality and handler registry isolation.
- Legacy UMI image parity and cropped letterbox RGB behavior. Four optional
  training-source comparisons require `CALIBWRIST_REFERENCE_ROOT`.
- WebSocket dispatch/metadata/reset fencing, Pi05 sampling capabilities, seed
  behavior and synthetic RTC/PAINT/AAC/AutoHorizon flow tests.

New upstream features should also run their own CPU unit tests after merging.
Compare failures to the pre-merge baseline; do not mask new failures by globally
excluding tests. A synthetic backend cannot establish real-weight prediction
quality, GPU compatibility, hardware timing or physical task success.
