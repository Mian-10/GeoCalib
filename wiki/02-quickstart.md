# 02 · 安装与运行方式

## 1. 环境要求

- Python >= 3.9
- PyTorch + torchvision（GPU 可选，CPU 可运行）
- `geocalib` 推理依赖：`torch`、`torchvision`、`opencv-python`、`kornia`、`matplotlib`
- `siclib` 训练/评估额外依赖：`omegaconf`、`hydra-core`、`albumentations>=1.4,<2`、`h5py`、`pandas`、`tqdm`、`tensorboard`、`wandb`

## 2. 安装

```bash
git clone https://github.com/cvg/GeoCalib.git && cd GeoCalib

# 推理包
python -m pip install -e .

# 训练 / 评估库
python -m pip install -e siclib
```

或者直接从 Git 安装推理包：

```bash
python -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
```

## 3. 最小推理示例

```python
import torch
from geocalib import GeoCalib

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)                 # 默认 pinhole 权重

image = model.load_image("path/to/image.jpg").to(device)   # [C,H,W], [0,1]
result = model.calibrate(image)

print(result["camera"])    # BaseCamera，含 f / vfov / K / k1 ...
print(result["gravity"])   # Gravity，含 roll / pitch / R / vec3d
```

`calibrate` 返回的字典键：

| 键 | 类型 | 说明 |
| --- | --- | --- |
| `camera` | `BaseCamera` 子类 | 已还原到原图尺度与裁剪 |
| `gravity` | `Gravity` | 相机坐标系下的重力方向 |
| `covariance` | `Tensor` | 参数协方差 |
| `up_field` / `latitude_field` | `Tensor` | 透视场（插值回原图尺寸） |
| `up_confidence` / `latitude_confidence` | `Tensor` | 逐像素置信度 |
| `roll_uncertainty` / `pitch_uncertainty` / `gravity_uncertainty` | `Tensor` | 角度不确定度（弧度） |
| `focal_uncertainty` / `vfov_uncertainty` | `Tensor` | 焦距 / vFoV 不确定度 |

## 4. 相机模型选择

```python
model = GeoCalib(weights="distorted")   # 或 "pinhole"（默认），或本地 .tar 路径
result = model.calibrate(image, camera_model="simple_radial")
```

| `camera_model` | 参数 | 适用场景 |
| --- | --- | --- |
| `pinhole` | `fx, fy` | 无畸变 / 轻微畸变（默认） |
| `simple_radial` | `fx, fy, k1` | 弱径向畸变 |
| `radial` | `fx, fy, k1, k2` | 较强径向畸变 |
| `simple_divisional` | `fx, fy, k1` | 强鱼眼畸变 |

> 主点始终假设在图像中心（`cx=W/2, cy=H/2`），不参与优化。

## 5. 部分标定（先验）

```python
# 已知焦距
result = model.calibrate(image, priors={"focal": focal_length_tensor})

# 已知重力方向
result = model.calibrate(image, priors={"gravity": gravity_direction_tensor})  # [3]
```

## 6. 多图共享内参

```python
# batch: list[Tensor[C,H,W]] 或 [B,C,H,W]
result = model.calibrate(batch, shared_intrinsics=True)
```

所有图像共享同一组内参（焦距），重力仍逐图估计。

## 7. 刚性多相机 rig

```python
# camera_R_rig[i]: 从 rig 坐标系到相机 i 的旋转
result = model.calibrate(batch, camera_R_rig=camera_R_rig)
# 若相机型号相同，再加 shared_intrinsics=True
result = model.calibrate(batch, camera_R_rig=camera_R_rig, shared_intrinsics=True)
```

此时所有相机共享**同一个 rig 坐标系下的重力方向**，通过已知相对旋转转换到各相机坐标系。

## 8. torch.hub

```python
model = torch.hub.load("cvg/GeoCalib", "GeoCalib", trust_repo=True)
```

`hubconf.py` 暴露 `model(*args, **kwargs)`，等价于 `GeoCalib(*args, **kwargs)`。

## 9. Demo

### 9.1 Gradio Web Demo

```bash
python gradio_app.py
```

- 上传或选择 `assets/` 中的示例图；
- 可选相机模型；
- 输出参数文本、up 场、latitude 场、置信度与去畸变图像；
- 使用 `@spaces.GPU` 装饰器，可直接部署到 Hugging Face Spaces。

### 9.2 摄像头交互 Demo

```bash
python -m geocalib.interactive_demo --camera_id 0
```

不传 `--camera_id` 时，会提示输入 [droidcam](https://droidcam.app) 的 IP。

按键：

| 键 | 功能 |
| --- | --- |
| `h` | 地平线 |
| `u` | up 向量场 |
| `l` | 纬度热力图 |
| `c` | 置信度热力图 |
| `d` | 去畸变图像 |
| `g` | 虚拟网格点 |
| `b` | 虚拟立方体 |
| `1/2/3` | pinhole / simple_radial / simple_divisional |
| `q` | 退出 |

### 9.3 Notebook

`demo.ipynb` 提供完整交互示例（含输出）。

## 10. 评估

结果写入 `outputs/results/<benchmark>/<tag>/`。

### 10.1 基准与命令

| Benchmark | 模块 | 数据目录 | 体积 |
| --- | --- | --- | --- |
| LaMAR | `siclib.eval.lamar2k` | `data/lamar2k` | ~400 MB |
| MegaDepth | `siclib.eval.megadepth2k` | `data/megadepth2k` | ~2.1 GB |
| MegaDepth（畸变） | `siclib.eval.megadepth2k_radial` | `data/megadepth2k-radial` | ~1.47 GB |
| TartanAir | `siclib.eval.tartanair` | `data/tartanair` | ~1.85 GB |
| Stanford2D3D | `siclib.eval.stanford2d3d` | `data/stanford2d3d` | ~885 MB |
| OpenPano | `siclib.eval.openpano` | `data/poly+maps+laval/poly+maps+laval` | 本地生成 |
| OpenPano（畸变） | `siclib.eval.openpano_radial` | `data/poly+maps+laval/pano_dataset_distorted` | 本地生成 |

```bash
# GeoCalib（pinhole）
python -m siclib.eval.lamar2k --conf geocalib-pinhole --tag geocalib --overwrite

# GeoCalib 在畸变数据上切 simple_radial
python -m siclib.eval.megadepth2k_radial --conf geocalib-pinhole --tag geocalib \
    --overwrite model.camera_model=simple_radial

# DeepCalib
python -m siclib.eval.lamar2k --conf deepcalib --tag deepcalib --overwrite

# 评估自己训练的模型
python -m siclib.eval.megadepth2k --checkpoint <experiment_name> --tag <eval_name> --overwrite
```

### 10.2 使用先验

```bash
python -m siclib.eval.<benchmark> --conf <config> --tag <tag> \
    data.use_prior_focal=true \
    data.use_prior_gravity=true \
    data.use_prior_k1=true
```

### 10.3 可视化检查

```bash
python -m siclib.eval.inspect lamar2k geocalib
python -m siclib.eval.inspect <benchmark> <tag1> <tag2> ...
```

## 11. 训练

### 11.1 准备 MSCAN-B 预训练骨干

```bash
mkdir -p weights
wget "https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/files/?p=%2Fmscan_b.pth&dl=1" \
    -O weights/mscan_b.pth
```

### 11.2 训练命令

```bash
# 多卡 DDP
python -m siclib.train geocalib-pinhole-openpano --conf geocalib --distributed

# 单卡、小 batch
python -m siclib.train geocalib-pinhole-openpano \
    --conf geocalib \
    data.train_batch_size=5

# 记录 tensorboard
python -m siclib.train geocalib-pinhole-openpano \
    --conf geocalib --distributed train.writer=tensorboard
```

- 默认 batch size 24，约需 2× RTX 4090（24GB）；
- checkpoint 写入 `outputs/training/<experiment>/`；
- 训练完成后可用实验名评估：

```bash
python -m siclib.eval.<benchmark> --checkpoint geocalib-pinhole-openpano --tag geocalib-retrained
```

### 11.3 畸变模型微调

```bash
python -m siclib.train geocalib-radial-openpano --conf geocalib-radial --distributed
```

`geocalib-radial.yaml` 使用 `openpano-radial` 数据、`simple_radial` 相机模型、`lr=1e-5`、200k steps。

## 12. OpenPano 数据集构建

```bash
# 1) 下载 Laval 数据集到 data/indoorDatasetCalibrated，并 tonemap
python -m siclib.datasets.utils.tonemapping \
    --hdr_dir data/indoorDatasetCalibrated --out_dir data/laval-tonemap

# 2) 下载 PolyHaven / HDRMAPS 全景图
python -m siclib.datasets.utils.download_openpano \
    --name openpano --laval_dir data/laval-tonemap

# 3) 生成透视图与 CSV
python -m siclib.datasets.create_dataset_from_pano --config-name openpano

# 多进程 + GPU 加速
python -m siclib.datasets.create_dataset_from_pano \
    --config-name openpano n_workers=10 device=cuda

# 畸变版本
python -m siclib.datasets.create_dataset_from_pano --config-name openpano_radial
```

生成结果：`data/openpano/openpano/{train,val,test}/` 与 `{train,val,test}.csv`，约 37k / 2.1k / 2.1k 张。

## 13. 目录与路径约定

`siclib/settings.py` 定义：

```python
DATA_PATH      = <repo>/data/
TRAINING_PATH  = <repo>/outputs/training/
EVAL_PATH      = <repo>/outputs/results/
```

如果仓库根目录存在 `settings.py`，会优先从中导入这三个路径，便于自定义。

## 14. 常见问题

- **图像尺寸必须是 32 的倍数**：推理时 `ImagePreprocessor` 会以 `crop` 方式裁到 32 的倍数；直接调用网络时需自行保证。
- **权重下载失败**：`GeoCalib(weights=...)` 从 GitHub Release 下载到 `torch.hub.get_dir()/geocalib/`，可手动放置。
- **显存不足**：LM 优化对 batch 敏感，训练时优先降低 `data.train_batch_size`。
- **CPU 运行慢**：`calibrate` 内部执行 30 步 LM，CPU 上实时性有限。
