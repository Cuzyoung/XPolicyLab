"""Compare deployment images with the captured native training image block."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from XPolicyLab.policy.FastWAM.preprocessing import prepare_robotwin_image
from XPolicyLab.policy.FastWAM.model import _standardize_rgb
from fastwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.datasets.lerobot.transforms.image import ToTensor

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'training/latest-models-local-20261013/fastwam-training-source/src/fastwam/datasets/lerobot/robot_video_dataset.py'


def native_training_image(sample):
    # Execute the actual pinned native block, not a second implementation.
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RobotVideoDataset')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get')
    begin = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n) == "image_is_pad = sample['image_is_pad']")
    end = next(i for i,n in enumerate(method.body) if isinstance(n, ast.Assign) and ast.unparse(n) == 'video = video.permute(1, 0, 2, 3)')
    block = ast.Module(body=method.body[begin:end+1], type_ignores=[])
    size = {'img_h':384,'img_w':320}
    namespace = {'torch':torch,'transforms_F':TF,'sample':sample,
                 'self':SimpleNamespace(lerobot_dataset=SimpleNamespace(presample_images=True),
                                        concat_multi_camera='robotwin',
                                        resize_transform=ResizeSmallestSideAspectPreserving(size),
                                        crop_transform=CenterCrop(size),
                                        normalize_transform=Normalize({'mean':0.5,'std':0.5}))}
    exec(compile(block, str(SOURCE), 'exec'), namespace)
    return namespace['video'].permute(1,0,2,3)


@pytest.mark.parametrize('hw', [(480,640),(240,320),(360,640)])
def test_training_pixels_match_exactly(hw):
    rng=np.random.default_rng(19)
    names=('head_camera','left_camera','right_camera')
    arrays=[rng.integers(0,256,(*hw,3),dtype=np.uint8) for _ in names]
    pipeline=[ToTensor(),transforms.Resize([240,320])]
    metas=[{'key':k,'shape':[3,240,320]} for k in names]
    processor=SimpleNamespace(shape_meta={'images':metas},val_transforms=pipeline)
    observation={'observation':{k:{'rgb':a} for k,a in zip(names,arrays)}}
    got=prepare_robotwin_image(observation,processor,[384,320])
    images=[]
    for a in arrays:
        t=torch.from_numpy(a).permute(2,0,1).unsqueeze(0)
        for transform in pipeline:t=transform(t)
        images.append(t)
    expected=native_training_image({'pixel_values':torch.stack(images),'image_is_pad':torch.zeros(1,dtype=torch.bool)})
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    assert got.shape==(1,3,384,320) and got.dtype==torch.float32


def test_raw_rgb_reaches_training_transforms_without_early_resize():
    a=np.random.default_rng(1).integers(0,256,(480,640,3),dtype=np.uint8)
    assert _standardize_rgb(a,resize=False) is a
    assert _standardize_rgb(a).shape==(240,320,3)
