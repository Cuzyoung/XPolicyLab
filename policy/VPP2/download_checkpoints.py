"""Download the paired evaluation weights and shared Wan encoders."""
import os
from pathlib import Path

from modelscope import snapshot_download


def main():
    snapshot_download(
        model_id="haodong123/VPP2_preview",
        local_dir=str(Path(__file__).resolve().parent),
        token=os.environ.get("MODELSCOPE_API_TOKEN"),
        allow_patterns=[
            "checkpoints/joint2b_s100000/*",
            "checkpoints/Wan2.1-I2V-14B-480P/*",
        ],
    )
    print("Downloaded VPP2 evaluation assets. Run launch_policy.py --dry-run to validate them.")


if __name__ == "__main__":
    main()
