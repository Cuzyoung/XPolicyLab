# imagewam.aug — 域随机化增广（clean-to-randomization）

为 C2R（只训 clean、测 clean+random）提升 **random 泛化**。合法边界：只对自己的
clean 数据自造视觉多样性，**不碰 RoboTwin randomized 测试分布**。

## A 档 · 光度域随机化（开箱即用，训练时生效，无需掩码）

`DomainRandomization` 是 `video_augmentation` 的 drop-in 替换（同签名
`forward(video)`，`video=[cam,T,C,H,W]` float[0,1]，逐相机、整段 clip 一致、无几何增广）。

在数据配置里把 `video_augmentation` 换成：

```yaml
video_augmentation:
  _target_: imagewam.aug.DomainRandomization
  p: 0.8
  photometric:
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4
    hue: 0.1
    gamma: [0.7, 1.4]
    exposure: [-0.5, 0.5]
    color_temperature: 0.15
    gaussian_noise_std: 0.03
    blur_sigma: [0.0, 1.5]
    blur_prob: 0.3
  background:
    enabled: false        # B 档默认关（需要掩码）
```

## B 档 · 掩码引导背景替换（需离线预计算掩码）

数据里没有分割掩码，需两步：

1. **离线生成掩码**（GPU 节点，慢）：
   ```bash
   python -m imagewam.aug.precompute_masks \
     --frames_root <RGB帧目录> --out_root <掩码输出目录> \
     --segmenter sam --sam_checkpoint <sam_vit_h.pth>
   ```
   输出布局 `<out_root>/<episode_id>/<camera>_<frame_idx>.png`，与
   `PrecomputedMaskProvider` 对齐。（`--segmenter heuristic` 可无依赖冒烟。）

2. **训练时接线**：`background.enabled=true`，并让
   `robot_video_dataset` 在调用 `video_augmentation(video)` 时把该 clip 的掩码
   一并传入 `forward(video, masks=...)`（掩码 `[cam,T,1,H,W]`，用
   `PrecomputedMaskProvider.get(episode_id, frame_indices, camera)` 取）。
   这一步需要在 dataset 里加几行把样本 id/帧号透传给 provider。

```yaml
  background:
    enabled: true
    p: 0.8
    asset_dirs: [<随机背景图/纹理目录>]   # 缺省则用程序化纹理(纯色/渐变/低频噪声)
    feather: 1.5
    distractor_prob: 0.3
```

## 建议
先只上 A 档跑一版看 random SR 涨多少；不够再投入做 B 档（补掩码 + dataset 透传）。
注意保留一部分原始样本（`p<1`），避免 clean SR 掉太多。

## C 档 · 风格随机化 AdaIN（开箱即用，无需掩码）

AdaIN「换纹理留形状」：对每帧做逐通道统计（mean/std）白化后，用随机采样的目标统计重上色，
保留空间结构。整段 clip 用同一组随机风格参数（时间一致）。纯 torch，无需外部权重/网络。

```yaml
  style:
    enabled: true
    p: 0.5
    target_mean: [0.2, 0.8]
    target_std: [0.05, 0.4]
    strength: [0.4, 1.0]   # 0=identity, 1=完全换风格
```

## D 档 · 频域幅度谱随机化（开箱即用，无需掩码）

对每帧 2D FFT，只随机扰动幅度谱、保留相位谱（相位携带物体形状/结构），再 iFFT 回空间域。
整段 clip 一致，计算轻。

```yaml
  fourier:
    enabled: true
    p: 0.5
    amp_jitter: [0.5, 1.5]         # 逐频率幅度乘性抖动
    envelope_strength: [0.0, 0.6]  # 向随机低频包络混合的强度
    envelope_grid: 8
```

## 顺序与向后兼容
`DomainRandomization._augment_clip` 顺序：photometric → style → fourier →（有 mask 时）background。
每档各自按其 `p` 独立触发。style/fourier/background 默认 `enabled: false`，不配置时行为与原来逐字节一致。
