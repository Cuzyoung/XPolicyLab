# OpenLoopVLA assets

The policy needs two immutable assets outside Git:

1. A native exported OpenLoopVLA `.pt` checkpoint whose package root also
   contains `config.yaml` and `dataset_statistics.json`.
2. The verified HRM-Penguin V2 package referenced by
   `OPENLOOPVLA_V2_PACKAGE_ROOT`.

Keep model weights and V2 assets outside the repository. The adapter passes the
V2 path as the single `framework.v2_package_root` configuration override and
loads the trained checkpoint with strict native state-dict handling.

