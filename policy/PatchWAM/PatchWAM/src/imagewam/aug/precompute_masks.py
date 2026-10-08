"""Offline foreground-mask precomputation for B-tier background replacement.

The LeRobot data has no segmentation masks, so B-tier needs masks generated
ahead of time. This CLI walks a directory of RGB frames and writes a binary
foreground mask (robot + manipulated objects + table kept as foreground) per
frame, in the layout expected by ``mask_provider.PrecomputedMaskProvider``:

    <out_root>/<episode_id>/<camera>_<frame_idx>.png    (or <frame_idx>.png)

Two segmenters are provided:

* ``sam``  -- Segment Anything (auto mask -> largest central instances). Best
  quality; requires ``segment-anything`` + a checkpoint. Run on a GPU node.
* ``heuristic`` -- cheap center-saliency + color/edge fallback. No deps beyond
  numpy/PIL. Coarse; use only to smoke-test the B-tier pipeline.

This is an offline preprocessing step (slow); it is NOT meant to run inside the
dataloader. Adapt ``iter_frames`` to your actual on-disk / video layout.
"""
from __future__ import annotations

import argparse
import os
from typing import Iterator, Tuple

import numpy as np


def segment_heuristic(rgb: np.ndarray) -> np.ndarray:
    """rgb: [H,W,3] uint8 -> mask [H,W] uint8 in {0,255}. Coarse center saliency."""
    h, w, _ = rgb.shape
    g = rgb.astype("float32").mean(-1) / 255.0
    # gradient magnitude (edges tend to sit on foreground objects/robot)
    gy = np.abs(np.gradient(g, axis=0))
    gx = np.abs(np.gradient(g, axis=1))
    edge = gx + gy
    edge = edge / (edge.max() + 1e-6)
    # radial prior: foreground is more likely near the image center
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    center = np.clip(1.0 - r, 0.0, 1.0)
    score = 0.6 * edge + 0.4 * center
    mask = (score > score.mean() + 0.5 * score.std()).astype("uint8") * 255
    return mask


def build_sam(checkpoint: str, model_type: str = "vit_h", device: str = "cuda"):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    return SamAutomaticMaskGenerator(sam)


def segment_sam(generator, rgb: np.ndarray) -> np.ndarray:
    """Keep the union of the most central / largest instances as foreground."""
    h, w, _ = rgb.shape
    anns = generator.generate(rgb)
    if not anns:
        return np.zeros((h, w), dtype="uint8")
    cy, cx = h / 2.0, w / 2.0
    scored = []
    for a in anns:
        m = a["segmentation"]
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        d = np.sqrt(((ys.mean() - cy) / cy) ** 2 + ((xs.mean() - cx) / cx) ** 2)
        scored.append((a["area"] * (1.2 - d), m))
    scored.sort(key=lambda t: t[0], reverse=True)
    keep = max(1, int(0.4 * len(scored)))
    out = np.zeros((h, w), dtype=bool)
    for _, m in scored[:keep]:
        out |= m
    return out.astype("uint8") * 255


def iter_frames(frames_root: str) -> Iterator[Tuple[str, str, np.ndarray]]:
    """Yield (episode_id, key, rgb[H,W,3]) for every PNG/JPG under frames_root.

    Expected layout: ``<frames_root>/<episode_id>/<key>.(png|jpg)``. ADAPT this
    to your dataset (e.g. decode videos and emit sampled frames).
    """
    from PIL import Image

    for ep in sorted(os.listdir(frames_root)):
        ep_dir = os.path.join(frames_root, ep)
        if not os.path.isdir(ep_dir):
            continue
        for fn in sorted(os.listdir(ep_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            key = os.path.splitext(fn)[0]
            rgb = np.asarray(Image.open(os.path.join(ep_dir, fn)).convert("RGB"))
            yield ep, key, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True, help="input RGB frames root")
    ap.add_argument("--out_root", required=True, help="output mask root")
    ap.add_argument("--segmenter", choices=["sam", "heuristic"], default="heuristic")
    ap.add_argument("--sam_checkpoint", default=None)
    ap.add_argument("--sam_model_type", default="vit_h")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from PIL import Image

    gen = None
    if args.segmenter == "sam":
        assert args.sam_checkpoint, "--sam_checkpoint required for SAM"
        gen = build_sam(args.sam_checkpoint, args.sam_model_type, args.device)

    n = 0
    for ep, key, rgb in iter_frames(args.frames_root):
        mask = segment_sam(gen, rgb) if gen is not None else segment_heuristic(rgb)
        out_dir = os.path.join(args.out_root, ep)
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(mask).save(os.path.join(out_dir, key + ".png"))
        n += 1
        if n % 200 == 0:
            print(f"[precompute_masks] {n} frames", flush=True)
    print(f"[precompute_masks] done, {n} masks -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
