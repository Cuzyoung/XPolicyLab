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
Use `--upstream` to include the DROID and trace restrictions, AutoHorizon context
isolation, upstream trace/serialization tests, LingBot capability tests and the
OpenPI policy wrapper's non-manual tests. Real-model manual tests stay deselected.
Pass `--only <test paths>` to run an explicit selection with the same CPU guards.
`XPOLICYLAB_TEST_EXTRA_PATH` can point to isolated test dependencies without
changing the production interpreter's packages.
Compare failures to the pre-merge baseline; do not mask new failures by globally
excluding tests. A synthetic backend cannot establish real-weight prediction
quality, GPU compatibility, hardware timing or physical task success.

## Main integration evidence (2026-11-06)

Local deployment baseline: `0a3286e`. Fetched upstream main: `ac9591d`.
Merge commit: `b30e5ed`. Baseline suite at `8fcb691`: 95 passed, 4 skipped.
After updating constructor-bypassing test fixtures for upstream capability guards:

```bash
<pi05-python> tests/tianji_merge/run_cpu.py --upstream
# 194 passed, 4 skipped, 2 deselected
<pi05-python> tests/tianji_merge/run_cpu.py --only tests/unit/test_openwam_rtc_cache.py
# 31 passed (requires the OpenWAM import dependencies)
```

The four skips need an external CalibWrist training reference. The two deselected
tests require production weights and are marked manual. OpenWAM checks exercise
CPU eager/compiled RTC, forward/VJP behavior, cache reset and mode switches.
The upstream LingBot compile-metadata test imports ManiMux; supply the parent
workspace package when using an isolated Git worktree.

No production code was altered to make tests pass. Legacy Pi05 synthetic fixtures
now declare their ALOHA observation profile and JAX backend, as normal model
construction already does. Local UMI preprocessing, pack-plate contracts/models,
and low-memory restoration files are byte-identical to the deployment baseline.

Integration concerns:

- Upstream withdraws LingBot RTC because it did not meet its H50/K32/30 Hz
  timing budget. Historical ManiMux LingBot RTC selections must not be treated
  as supported; their capability negotiation now rejects RTC.
- Upstream Cosmos3 pins `74c550f45fe536c080c0e161e43b80556d2035bd`, which its
  configured NVIDIA remote rejects with `not our ref`. The previous pin
  `c7e8d76b5da8aeae38cdac91c6cfd57185b2f6bc` is fetchable. The integration branch
  currently preserves the upstream pin, pending an explicit choice before
  applying this branch to the live checkout or publishing it.
