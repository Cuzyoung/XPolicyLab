"""Run the merge contract suite in a fresh, CPU-only interpreter.

Invoke with the Pi05 environment's Python. Extra arguments go to pytest.
"""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
env = os.environ.copy()
env.update({
    'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'none',
    'JAX_PLATFORMS': 'cpu', 'JAX_PLATFORM_NAME': 'cpu',
    'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
})
# Pin imports to this checkout, including when using another worktree's venv.
env['PYTHONPATH'] = os.pathsep.join(map(str, [
    ROOT.parent, ROOT, ROOT / 'policy/Pi_05/openpi/src',
    ROOT / 'policy/Pi_05/openpi/packages/openpi-client/src',
]))
selected = [
    'test_tianji_merge_regression.py', 'test_pi05_pack_plate.py',
    'test_pi05_pack_plate_h50.py', 'test_pi05_pass_ball_zero_pose.py',
    'test_umi_image_preprocessing.py', 'test_ws_infer_sampling.py',
    'test_ws_reset_fence.py', 'test_pi05_rtc_adapter.py',
    'test_pi05_sampling_capabilities.py', 'test_pi05_inference_seed.py',
    'test_pi05_aac_adapter.py', 'test_pi05_paint_adapter.py',
    'test_pi05_autohorizon_adapter.py', 'test_pi05_autohorizon_selector.py',
    'test_flow_sampling_expansion.py',
]
probe = '''
from pathlib import Path
import jax, torch, openpi.models.model, XPolicyLab.policy.Pi_05.model
root = Path.cwd().resolve()
assert all(d.platform == 'cpu' for d in jax.devices())
assert not torch.cuda.is_available()
for module in (openpi.models.model, XPolicyLab.policy.Pi_05.model):
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
print('CPU-only backend and checkout import provenance verified', flush=True)
'''
subprocess.run([sys.executable, '-c', probe], cwd=ROOT, env=env, check=True)
raise SystemExit(subprocess.run([
    sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q', '--import-mode=importlib',
    *[str(ROOT / 'tests/unit' / name) for name in selected], *sys.argv[1:],
], cwd=ROOT, env=env).returncode)
