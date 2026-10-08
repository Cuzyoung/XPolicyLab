"""Cross-embodiment distribution comparison: dataset-level TCP/action deltas and
model-predicted action deltas, extracted into one shared per-dimension array schema
so the same metrics/plotting code runs on both.

Two independent signals, both computed as consecutive-step deltas:
  - "action" : raw_relative_action (RobomimicHDF5Dataset with use_relative_actions=True),
               the SE(3) delta the action head is actually trained to predict.
               Layout: pos(3) + rotation-6d(6) + gripper(1).
  - "tcp"    : observation['tcp_pos']/['tcp_orn'] (camera-frame TCP visual supervision,
               see RobomimicHDF5Dataset._align_tcp_to_eef_site_frame), differenced
               between consecutive steps of the same demo.

Consumers: scripts/compare_embodiment_distributions.py (dataset-level, local, no GPU)
and scripts/probe_cross_embodiment.py --dump-raw-actions (model-level, GPU) both save
arrays via save_arrays() and load them back with the same schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import ks_2samp, wasserstein_distance

POS_DIMS = 3
ROT6D_DIMS = 6

# Fixed categorical assignment (entity -> color), not cycled by sort order. Picked from
# the dataviz skill's validated 8-hue set, skipping adjacent slots 2/4 (orange/yellow)
# which fail its all-pairs check -- every histogram subplot here overlays all groups
# at once, i.e. an all-pairs context.
EMBODIMENT_COLORS = {
    "panda": "#2a78d6",   # blue
    "sawyer": "#1baf7a",  # aqua
    "iiwa": "#4a3aa7",    # violet
    "ur5e": "#e34948",    # red
}
_FALLBACK_PALETTE = list(EMBODIMENT_COLORS.values())


def _rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Inverse of the flatten used to build raw_relative_action's rotation-6d field
    (robomimic_hdf5.py: rel_rotmat[:, :, :2].swapaxes(-1, -2).reshape(..., 6)) --
    Gram-Schmidt the two stored basis vectors back into a full rotation matrix.
    """
    r6d = np.asarray(r6d, dtype=np.float64)
    a1, a2 = r6d[..., 0:3], r6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def decompose_action_array(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """(N, >=10) pos3+rot6d6+gripper1 array -> per-dim scalar arrays plus aggregate
    pos_norm / rot_angle summaries (physically interpretable, unlike raw 6D components).
    """
    actions = np.asarray(actions, dtype=np.float64)
    min_dim = POS_DIMS + ROT6D_DIMS + 1
    if actions.ndim != 2 or actions.shape[-1] < min_dim:
        raise ValueError(f"Expected (N, >={min_dim}) action array, got shape {actions.shape}")
    pos = actions[:, :POS_DIMS]
    rot6d = actions[:, POS_DIMS:POS_DIMS + ROT6D_DIMS]
    gripper = actions[:, POS_DIMS + ROT6D_DIMS]

    rotmat = _rotation_6d_to_matrix(rot6d)
    rot_angle = Rotation.from_matrix(rotmat).magnitude()

    return {
        "pos_x": pos[:, 0],
        "pos_y": pos[:, 1],
        "pos_z": pos[:, 2],
        "pos_norm": np.linalg.norm(pos, axis=-1),
        "rot_angle": rot_angle,
        "gripper": gripper,
    }


def select_consecutive_indices(
    dataset,
    max_demos: Optional[int] = None,
    max_steps_per_demo: Optional[int] = None,
) -> List[int]:
    """Flat dataset indices covering whole demos in step order, for signals (like TCP
    deltas) that need genuinely adjacent steps -- a uniform global stride would mostly
    land on non-adjacent pairs and silently produce cross-trajectory garbage.

    Relies on ``dataset.episode_indices`` (list of {demo_key, step_idx, traj_len} in
    flat-index order), the same structure RobomimicHDF5Dataset.__getitem__ decodes the
    flat index against.
    """
    episode_indices = dataset.episode_indices
    demo_order: List[str] = []
    seen = set()
    for info in episode_indices:
        dk = info["demo_key"]
        if dk not in seen:
            seen.add(dk)
            demo_order.append(dk)
    if max_demos is not None:
        demo_order = demo_order[:max_demos]
    wanted_demos = set(demo_order)

    indices = []
    per_demo_count: Dict[str, int] = {}
    for flat_idx, info in enumerate(episode_indices):
        dk = info["demo_key"]
        if dk not in wanted_demos:
            continue
        count = per_demo_count.get(dk, 0)
        if max_steps_per_demo is not None and count >= max_steps_per_demo:
            continue
        indices.append(flat_idx)
        per_demo_count[dk] = count + 1
    return indices


def extract_action_deltas(
    dataset,
    indices: Optional[Iterable[int]] = None,
    horizon_step: int = 0,
) -> Dict[str, np.ndarray]:
    """Per-step SE(3) relative-action deltas from a dataset built with
    use_relative_actions=True. ``horizon_step`` picks which step of each sample's
    action window to read; 0 is the immediate next-step delta. Every dataset index
    already shifts the window by one step, so horizon_step=0 gives exactly one delta
    per trajectory step with no horizon-overlap duplication -- a plain uniform stride
    over ``indices`` is fine here (unlike extract_tcp_deltas, no adjacency needed).
    """
    idxs = list(indices) if indices is not None else range(len(dataset))
    rows = []
    for i in idxs:
        sample = dataset[i]
        raw = sample.get("raw_relative_action")
        if raw is None:
            raise KeyError(
                "Sample has no 'raw_relative_action'; build the dataset with "
                "use_relative_actions=True."
            )
        rows.append(np.asarray(raw)[horizon_step])
    return decompose_action_array(np.stack(rows, axis=0))


def _last_frame(value: np.ndarray, core_ndim: int) -> np.ndarray:
    """Strip any leading observation-history axis, keeping the current (most recent)
    frame. ``core_ndim`` is the number of trailing dims that make up one frame (1 for
    a (3,) position, 2 for a (3, 3) rotation matrix) -- anything beyond that is history."""
    value = np.asarray(value, dtype=np.float64)
    while value.ndim > core_ndim:
        value = value[-1]
    return value


def extract_tcp_deltas(dataset, camera: str, indices: Sequence[int]) -> Dict[str, np.ndarray]:
    """Frame-to-frame delta of the TCP visual signal (tcp_pos/tcp_orn, camera frame),
    independent of the action-head supervision pipeline. ``indices`` should come from
    select_consecutive_indices() -- pairs that aren't truly adjacent steps of the same
    demo (detected via episode_id/step_id) are dropped rather than diffed.
    """
    idxs = list(indices)
    pos_rows, orn_rows, episode_ids, step_ids = [], [], [], []
    for i in idxs:
        sample = dataset[i]
        obs = sample["observation"]
        tcp_pos = (obs.get("tcp_pos") or {}).get(camera)
        tcp_orn = (obs.get("tcp_orn") or {}).get(camera)
        if tcp_pos is None or tcp_orn is None:
            raise KeyError(f"Sample has no tcp_pos/tcp_orn for camera={camera!r}.")
        tcp_pos = _last_frame(tcp_pos, core_ndim=1)
        orn_core_ndim = 2 if np.asarray(tcp_orn).shape[-2:] == (3, 3) else 1
        tcp_orn = _last_frame(tcp_orn, core_ndim=orn_core_ndim)
        tcp_orn = tcp_orn.reshape(3, 3) if tcp_orn.size == 9 else tcp_orn
        pos_rows.append(tcp_pos)
        orn_rows.append(tcp_orn)
        episode_ids.append(sample["episode_id"])
        step_ids.append(int(sample["step_id"]))

    pos_deltas, rot_angles = [], []
    n_dropped = 0
    for j in range(1, len(idxs)):
        if episode_ids[j] != episode_ids[j - 1] or step_ids[j] != step_ids[j - 1] + 1:
            n_dropped += 1
            continue
        pos_deltas.append(pos_rows[j] - pos_rows[j - 1])
        rel_rot = Rotation.from_matrix(orn_rows[j - 1]).inv() * Rotation.from_matrix(orn_rows[j])
        rot_angles.append(rel_rot.magnitude())

    if not pos_deltas:
        raise RuntimeError(
            "No consecutive-step pairs found; pass indices from select_consecutive_indices()."
        )
    pos_deltas = np.stack(pos_deltas, axis=0)
    return {
        "pos_x": pos_deltas[:, 0],
        "pos_y": pos_deltas[:, 1],
        "pos_z": pos_deltas[:, 2],
        "pos_norm": np.linalg.norm(pos_deltas, axis=-1),
        "rot_angle": np.asarray(rot_angles),
        "_n_dropped_boundaries": n_dropped,
    }


def compute_distribution_metrics(baseline: np.ndarray, other: np.ndarray) -> dict:
    """1D Wasserstein distance (EMD, physical units, no bin/bandwidth choice, always
    finite) + a two-sample KS test, for one dimension against a baseline group."""
    baseline = np.asarray(baseline, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    ks = ks_2samp(baseline, other)
    return {
        "wasserstein": float(wasserstein_distance(baseline, other)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "n_baseline": int(baseline.size),
        "n_other": int(other.size),
    }


def compute_all_metrics(
    per_group: Dict[str, Dict[str, np.ndarray]],
    baseline_key: str,
) -> Dict[str, Dict[str, dict]]:
    """metrics[group][dim] for every group against per_group[baseline_key][dim],
    skipping internal bookkeeping keys (leading underscore) and dims missing from
    the baseline."""
    baseline = per_group[baseline_key]
    out: Dict[str, Dict[str, dict]] = {}
    for group_key, dims in per_group.items():
        out[group_key] = {
            dim: compute_distribution_metrics(baseline[dim], values)
            for dim, values in dims.items()
            if not dim.startswith("_") and dim in baseline
        }
    return out


def metrics_to_markdown(metrics: Dict[str, Dict[str, dict]], baseline_key: str) -> str:
    dims = sorted({d for g in metrics.values() for d in g})
    header = ["embodiment"] + [f"{d} (wasserstein / ks_stat / ks_p)" for d in dims]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for group_key, group_metrics in metrics.items():
        row = [group_key + (" (baseline)" if group_key == baseline_key else "")]
        for d in dims:
            m = group_metrics.get(d)
            row.append(
                "-" if m is None else f"{m['wasserstein']:.4g} / {m['ks_stat']:.3f} / {m['ks_p']:.3g}"
            )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_overlaid_histograms(
    per_group: Dict[str, Dict[str, np.ndarray]],
    out_path,
    bins: int = 50,
    dims: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    plot_style: str = "both",
) -> None:
    """One subplot per dimension, one series per group (embodiment), colors assigned by
    entity (EMBODIMENT_COLORS) so they stay consistent across figures.

    plot_style:
      hist - step histogram only (bin-sensitive, but shows the raw sample counts).
      kde  - gaussian_kde density curve only (smooth, but a bandwidth choice).
      both - faint filled histogram behind a solid KDE line (default): the histogram
             keeps the KDE honest about sample noise/multimodality, the KDE line makes
             the shape/overlap easy to read at a glance.
    """
    if plot_style not in ("hist", "kde", "both"):
        raise ValueError(f"plot_style must be one of hist/kde/both, got {plot_style!r}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    dims = list(dims) if dims is not None else sorted(
        {d for g in per_group.values() for d in g if not d.startswith("_")}
    )
    ncols = min(3, len(dims))
    nrows = -(-len(dims) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

    draw_hist = plot_style in ("hist", "both")
    draw_kde = plot_style in ("kde", "both")

    for ax_idx, dim in enumerate(dims):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        for gi, (group_key, values) in enumerate(per_group.items()):
            if dim not in values:
                continue
            data = values[dim]
            color = EMBODIMENT_COLORS.get(group_key, _FALLBACK_PALETTE[gi % len(_FALLBACK_PALETTE)])
            # In "both" mode the KDE line carries the legend label; the histogram is a
            # faint backdrop so there's only one legend entry per group.
            label = group_key if not draw_kde else None
            if draw_hist:
                ax.hist(
                    data, bins=bins, density=True,
                    histtype="stepfilled" if draw_kde else "step",
                    alpha=0.15 if draw_kde else 1.0,
                    linewidth=1.8, color=color, label=label,
                )
            if draw_kde:
                try:
                    kde = gaussian_kde(data)
                except (np.linalg.LinAlgError, ValueError):
                    # Degenerate (near-)constant data -- no density to smooth; the
                    # histogram backdrop (if enabled) still shows the spike.
                    continue
                pad = 0.05 * (data.max() - data.min() or 1.0)
                grid = np.linspace(data.min() - pad, data.max() + pad, 256)
                ax.plot(grid, kde(grid), color=color, linewidth=1.8, label=group_key)
        ax.set_title(dim, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    for ax_idx in range(len(dims), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_arrays(arrays: Dict[str, np.ndarray], out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in arrays.items() if not k.startswith("_")}
    np.savez(out_path, **clean)


def load_arrays(path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
