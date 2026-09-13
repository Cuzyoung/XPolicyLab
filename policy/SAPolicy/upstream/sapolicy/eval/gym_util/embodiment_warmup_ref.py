"""HDF5-only helpers for cross-embodiment Panda reference lookup.

Kept separate from embodiment_warmup.py so Panda eval paths never import
mimicgen (which re-imports robosuite and can break EGL on eval workers).
"""
import os

import h5py
import numpy as np

_CE_NON_PANDA_EMBODIMENTS = ("iiwa", "sawyer", "ur5e")


def derive_panda_reference_path(dataset_path):
    """Best-effort sibling lookup for nutassembly-cross-embodiment dataset_paths."""
    path = os.path.expanduser(dataset_path)
    for emb in _CE_NON_PANDA_EMBODIMENTS:
        if emb in path:
            candidate = path.replace(emb, "panda")
            if os.path.isfile(candidate):
                return candidate
    return None


def build_panda_reference_table(panda_hdf5_path):
    """Average frame-0 Panda eef positions across demos for warmup targeting."""
    with h5py.File(os.path.expanduser(panda_hdf5_path), "r") as f:
        demo_keys = list(f["data"].keys())
        eef_pos = np.zeros((len(demo_keys), 3))
        for i, key in enumerate(demo_keys):
            eef_pos[i] = f[f"data/{key}/obs/robot0_eef_pos"][0]

    return dict(eef_pos_mean=eef_pos.mean(axis=0), num_demos=len(demo_keys))
