# 03 · `geocalib` 推理包详解

`geocalib` 是最小可用推理包，目录 `/workspace/geocalib`。对外只需 `from geocalib import GeoCalib`。

## 1. 模块总览

| 文件 | 职责 | 关键对象 |
| --- | --- | --- |
| `__init__.py` | 导出模型、配置 logging | `GeoCalib`, `logger` |
| `extractor.py` | 用户接口：下载权重、预处理、调用网络、后处理 | `GeoCalib` |
| `geocalib.py` | 网络结构定义 | `LowLevelEncoder`, `UpDecoder`, `LatitudeDecoder`, `PerspectiveDecoder`, `GeoCalib` |
| `camera.py` | 相机模型与投影/畸变/雅可比 | `BaseCamera`, `Pinhole`, `SimpleRadial`, `Radial`, `SimpleDivisional`, `camera_models` |
| `gravity.py` | 重力向量及其 roll/pitch 转换 | `Gravity` |
| `perspective_fields.py` | 由相机 + 重力生成透视场及其雅可比 | `get_perspective_field`, `J_perspective_field`, `get_up_field`, `get_latitude_field`, `get_horizon_line` |
| `lm_optimizer.py` | 可微 LM 优化 | `LMOptimizer` |
| `misc.py` | 张量包装、流形、通用雅可比 | `TensorWrapper`, `SphericalManifold`, `EuclideanManifold`, `J_vecnorm`, `J_focal2fov`, `J_up_projection` |
| `modules.py` | 网络积木 | `MSCAN`, `LightHamHead`, `Hamburger`, `NMF2D`, `ConvModule`, ... |
| `utils.py` | 图像预处理、坐标/角度转换、设备选择 | `ImagePreprocessor`, `load_image`, `read_image`, `focal2fov`, `fov2focal`, `rad2rotmat`, `get_device`, `print_calibration` |
| `viz2d.py` | 绘图辅助 | 2D 点/线/箭头/热力图 |
| `interactive_demo.py` | 摄像头实时 Demo | `VideoCapture`, `InteractiveDemo` |

## 2. `extractor.GeoCalib` —— 对外推理接口

```python
class GeoCalib(nn.Module):
    def __init__(self, weights: str = "pinhole")
    def load_image(self, path: Path) -> torch.Tensor
    def calibrate(
        self,
        img: torch.Tensor,
        camera_model: str = "pinhole",
        priors: Optional[Dict[str, torch.Tensor]] = None,
        shared_intrinsics: bool = False,
        camera_R_rig: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]
```

### 2.1 权重加载

- `weights="pinhole"` 或 `"distorted"`：
  - 从 `https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-{weights}.tar` 下载；
  - 缓存到 `torch.hub.get_dir()/geocalib/`；
  - `self.model.flexible_load(state_dict["model"])`。
- `weights=<本地 .tar 路径>`：直接 `torch.load`，同样调用 `flexible_load`。
- 其他值：抛 `ValueError`。

> `geocalib` 推理包假定 checkpoint 的结构与内置网络一致；若自定义结构，请用 `siclib.models.extractor.GeoCalib`。

### 2.2 预处理与后处理

- `ImagePreprocessor({"resize": 320, "edge_divisible_by": 32})`：
  - 短边缩放到 320；
  - 裁剪使宽高都是 32 的倍数（`crop_pad` 记录裁剪量）；
  - 输出 `image / scales / crop_pad / image_size / transform / original_image_size`。
- `_post_process`：
  - `camera.undo_scale_crop(img_data)` 把内参还原到原图；
  - `up_field / latitude_field` 双线性插值回原图分辨率；
  - `up_confidence / latitude_confidence` 同样插值；
  - `focal_uncertainty` 乘以 `1/scales[1]`。

### 2.3 先验与批处理

| 输入 | 处理 |
| --- | --- |
| `img.shape == (C,H,W)` | 自动加 batch 维 |
| `priors["focal"]` | 乘以 resize scale 后作为 `prior_focal` |
| `priors["gravity"]` | 直接作为 `prior_gravity`（3D 向量） |
| `shared_intrinsics=True` | 传入优化器，批内共享焦距 |
| `camera_R_rig` | 传入优化器，用于多相机 rig 共享重力 |

`calibrate` 内部：

```python
self.model.optimizer.set_camera_model(camera_model)
self.model.optimizer.shared_intrinsics = shared_intrinsics
out = self.model(img_data | prior_values | {"camera_R_rig": camera_R_rig})
```

## 3. `geocalib.geocalib` —— 网络结构

### 3.1 `LowLevelEncoder`

```python
class LowLevelEncoder(nn.Module):
    in_channel = 3
    feat_dim = 64
    conv1 = ConvModule(3, 64, k=3, p=1)
    conv2 = ConvModule(64, 64, k=3, p=1)
```

输出 `{"features": c2}`，作为解码器的低层特征 `ll`。输入尺寸必须是 32 的倍数。

### 3.2 `UpDecoder`

```python
decoder = LightHamHead()
linear_pred_up = nn.Conv2d(decoder.out_channels, 2, 1)
```

- 输出 `up_field = F.normalize(linear_pred_up(x), dim=1)`（2 通道单位向量）；
- `up_confidence = sigmoid(log_confidence)`。

### 3.3 `LatitudeDecoder`

```python
linear_pred_latitude = nn.Conv2d(out_channels, 1, 1)
lat = asin(clamp(tanh(logits), -1+eps, 1-eps))
```

- 输出 `latitude_field`（弧度，范围约 `(-π/2, π/2)`）；
- `latitude_confidence = sigmoid(log_confidence)`。

### 3.4 `PerspectiveDecoder`

组合 `UpDecoder` 与 `LatitudeDecoder`，输出四元组：

```python
{"up_field", "up_confidence", "latitude_field", "latitude_confidence"}
```

### 3.5 `GeoCalib(nn.Module)`

```python
class GeoCalib(nn.Module):
    def __init__(self, **optimizer_options)
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
    def flexible_load(self, state_dict) -> None
```

`forward`：

```python
features = {
    "hl": self.backbone(data)["features"],   # MSCAN 高层特征
    "ll": self.ll_enc(data)["features"],     # 64 通道低层特征
}
out = self.perspective_decoder({"features": features})
out |= {k: data[k] for k in ["image","scales","prior_gravity","prior_focal","prior_dist","camera_R_rig"] if k in data}
out |= self.optimizer(out)
return out
```

`flexible_load` 支持：

- 完全匹配 → `strict=True`；
- 前缀不匹配 → 去掉第一段前缀重试；
- 允许缺失 BatchNorm running stats。

## 4. `geocalib.camera` —— 相机模型

### 4.1 张量布局

`BaseCamera._data` 形状 `(..., 8)`：

```
[w, h, fx, fy, cx, cy, k1, k2]
```

- 构造时允许 6/7/8 维，不足 8 维自动补零；
- `cx, cy` 默认 `w/2, h/2`；
- 支持 `f` 或 `vfov` 二选一初始化；支持 `k1_hat`（归一化畸变系数）。

### 4.2 主要属性/方法

| 成员 | 说明 |
| --- | --- |
| `size` | `(w, h)` |
| `f` | `(fx, fy)` |
| `c` | `(cx, cy)` |
| `vfov` / `hfov` | 垂直/水平视场角（弧度） |
| `K` | 3×3 内参矩阵 |
| `from_dict(param_dict)` | 由 `width/height/vfov/f/k1/k1_hat/dist/scales` 构造 |
| `pinhole()` | 去掉畸变，返回同类型 6 参数相机 |
| `scale(scales)` | 图像缩放后更新内参 |
| `crop(pad)` | 裁剪后更新尺寸与主点 |
| `undo_scale_crop(data)` | 逆操作 |
| `update_focal(delta, as_log=False)` | 更新焦距，clamp 到 vFoV 5°~150° |
| `update_dist(delta, dist_range)` | 更新畸变参数并 clamp |
| `project / J_project` | 3D → 归一化 2D |
| `distort / J_distort` | 加畸变 |
| `undistort / J_undistort` | 去畸变 |
| `denormalize / normalize` | 归一化坐标 ↔ 像素坐标 |
| `world2image / image2world` | 3D ↔ 像素，含有效性 mask |
| `undistort_image(img)` | 网格采样去畸变（batch=1） |
| `get_img_from_pano(pano, gravity, yaws, resize_factor)` | 从全景渲染透视图 |

### 4.3 具体模型

| 类 | `name()` | 畸变模型 | 参数 |
| --- | --- | --- | --- |
| `Pinhole` | `pinhole` | 无 | — |
| `SimpleRadial` | `simple_radial` | `1 + k1 r²`，逆为 `1 - k1 r²` | `k1` |
| `Radial` | `radial` | `1 + k1 r² + k2 r⁴`，逆为 `1 - k1 r² + (3k1²-k2) r⁴` | `k1, k2` |
| `SimpleDivisional` | `simple_divisional` | `(1 - sqrt(1 - 4k1r²)) / (2k1r²)`，逆为 `1/(1+k1r²)` | `k1` |

注册表：

```python
camera_models = {
    "pinhole": Pinhole,
    "simple_radial": SimpleRadial,
    "radial": Radial,
    "simple_divisional": SimpleDivisional,
}
```

## 5. `geocalib.gravity.Gravity`

```python
class Gravity(TensorWrapper):   # 内部 3D 单位向量
    def from_rp(roll, pitch)
    @property vec3d / x / y / z
    @property roll / pitch / rp
    @property R          # 由 roll/pitch 得到的旋转矩阵
    def J_roll() / J_pitch() / J_rp()
    def update(delta, spherical=False)
    def J_update(spherical=False)
```

`from_rp` 的定义：

```python
g = [-sin(roll)cos(pitch), -cos(roll)cos(pitch), sin(pitch)]
```

`roll`、`pitch` 由向量反解，`pitch = asin(z)`。`update` 在 `spherical=True` 时走 `SphericalManifold.plus`，否则在 `(roll, pitch)` 上做欧氏加法。

## 6. `geocalib.perspective_fields`

| 函数 | 说明 |
| --- | --- |
| `get_horizon_line(gravity, threshold=1e-5)` | 由重力求地平线（图像中 latitude=0 的直线） |
| `get_up_field(camera, gravity, use_log_focal=False)` | 逐像素 up 单位向量（2 通道） |
| `J_up_field(camera, gravity, ...)` | up 场对参数的雅可比 |
| `get_latitude_field(camera, gravity, use_log_focal=False)` | 逐像素纬度角（1 通道） |
| `J_latitude_field(camera, gravity, ...)` | 纬度场雅可比 |
| `get_perspective_field(camera, gravity, use_up=True, use_latitude=True, ...)` | 返回 `(up_field, latitude_field)` |
| `J_perspective_field(camera, gravity, spherical, use_log_focal, camera_R_rig=None)` | 返回两者的联合雅可比 |

实现要点：

1. 由 `camera.pixel_coordinates()` 生成像素网格；
2. `camera.image2world` → 归一化 bearing；
3. 通过重力旋转 `R`（以及可选的 rig 旋转）把 bearing 转到重力对齐坐标系；
4. up 向量取旋转后 bearing 的 x/y 分量并归一化；
5. latitude 取 `atan2(y, sqrt(x²+z²))`；
6. 雅可比通过链式法则组合相机投影/畸变/归一化与重力旋转的雅可比。

`use_log_focal` 时焦距参数化为 `log f`，雅可比相应乘 `f`。

## 7. `geocalib.lm_optimizer.LMOptimizer`

### 7.1 默认配置

```python
default_conf = {
    "camera_model": "pinhole",
    "shared_intrinsics": False,
    "num_steps": 30,
    "lambda_": 0.1,
    "fix_lambda": False,
    "early_stop": False,
    "atol": 1e-8, "rtol": 1e-8,
    "use_spherical_manifold": True,
    "use_log_focal": True,
    "loss_fn": "huber_loss",          # {"squared_loss", "huber_loss"}
    "up_loss_fn_scale": 1e-2,
    "lat_loss_fn_scale": 1e-2,
    "verbose": False,
}
```

### 7.2 主要方法

| 方法 | 说明 |
| --- | --- |
| `set_camera_model(name)` | 从 `camera_models` 选择模型 |
| `setup_optimization_and_priors(data, shared_intrinsics)` | 决定是否估计 gravity / focal / dist，设置先验 |
| `calculate_residuals(camera, gravity, data)` | 计算 up/latitude 残差，返回残差、雅可比、权重、loss |
| `calculate_costs(residuals, weights, loss_fn)` | 计算总代价 |
| `calculate_gradient_and_hessian(J, res, weights, loss_fn)` | 组装 `g = Jᵀ W r`、`H = Jᵀ W J` |
| `setup_system(...)` | 构建参数更新方程（共享内参/rig 有不同布局） |
| `optimizer_step(H, g, lambda_)` | Cholesky 求解 `(H+λI)Δ = -g` |
| `estimate_uncertainty(H)` | 由 `H⁻¹` 得到参数协方差 |
| `update_estimate(camera, gravity, delta)` | 应用更新（球面流形/对数焦距） |
| `optimize(data)` | 主循环：初始化 → 迭代 → 早停 → 不确定度 |
| `forward(data)` | 返回 `camera / gravity / covariance / uncertainty / field` |

### 7.3 迭代逻辑（简化）

```text
camera, gravity = initialize(...)
cost = calculate_cost(...)
for step in range(num_steps):
    residuals, J, weights, loss = calculate_residuals(...)
    H, g = calculate_gradient_and_hessian(...)
    delta = optimizer_step(H, g, lambda_)
    new_camera, new_gravity = update_estimate(delta)
    new_cost = calculate_cost(...)
    if new_cost < cost: 接受更新, λ /= factor
    else:               拒绝更新, λ *= factor
    if early_stop and allclose: break
uncertainty = estimate_uncertainty(H)
```

### 7.4 不确定度

由 `H` 的逆得到参数协方差，再通过链式雅可比转换为：

- `roll_uncertainty`
- `pitch_uncertainty`
- `gravity_uncertainty`
- `focal_uncertainty`
- `vfov_uncertainty`

这些量既用于结果输出，也用于下游位姿估计（如 `siclib/pose_estimation.py` 中按不确定度决定是否使用重力先验）。

## 8. `geocalib.misc`

| 对象 | 说明 |
| --- | --- |
| `autocast` | 装饰器：统一把张量转成 `float32` 并恢复 |
| `TensorWrapper` | 轻量张量子类，支持 `shape/dtype/device/to/float/new_zeros` 等 |
| `EuclideanManifold` | `plus(x, delta) = x + delta` |
| `SphericalManifold` | Householder 变换实现球面流形的 `plus` / `J_plus` |
| `J_vecnorm` | `vec/||vec||` 的雅可比 |
| `J_focal2fov` | `focal2fov` 的雅可比 |
| `J_up_projection` | up 向量投影的雅可比 |

`SphericalManifold` 采用 Hertzberg 等论文中的流形封装与 Golub 的 Householder 算法，避免 3D 单位向量更新后范数漂移。

## 9. `geocalib.modules`

### 9.1 MSCAN

```python
class MSCAN(nn.Module):
    embed_dims = [64, 128, 320, 512]
    depths = [3, 3, 12, 3]
```

- 4 个 stage，每 stage 由 `OverlapPatchEmbed` + 若干 `Block` 组成；
- `Block` = `AttentionModule`（`DWConv` + `SpatialAttention`）+ `Mlp`（`DWConv` + FC）；
- 输入图像会先 **RGB → BGR 且 ×255**（对齐 SegNeXt 预训练）；
- 输出 `{"features": [f1, f2, f3, f4], "padding": ...}`。

### 9.2 LightHamHead / Hamburger / NMF2D

- `LightHamHead(in_channels=[64,128,320,512], out_channels=64, ham_channels=512)`；
- 将多尺度特征投影到统一通道，拼接后经 `Hamburger` 做矩阵分解式全局建模（`NMF2D`），再上采样融合；
- 同时输出特征图与 `log_confidence`（用于置信度头）；
- 另有 `FeatureFusionBlock`、`ResidualConvUnit`、`ConvModule` 等辅助模块。

### 9.3 结构关系

```text
MSCAN (backbone)
  └─ features: [f1(64), f2(128), f3(320), f4(512)]
LowLevelEncoder
  └─ features: 64
        │
        ▼
PerspectiveDecoder
  ├─ UpDecoder:      LightHamHead → Conv2d(64→2) → normalize
  └─ LatitudeDecoder: LightHamHead → Conv2d(64→1) → tanh → asin
```

## 10. `geocalib.utils`

| 函数/类 | 说明 |
| --- | --- |
| `fit_to_multiple(x, multiple, mode, crop)` | 计算 pad 量 |
| `fit_features_to_multiple(features, multiple, crop)` | reflect pad 到倍数 |
| `ImagePreprocessor` | 缩放/方形裁剪/边长为倍数/缩放后端（kornia / torchvision） |
| `numpy_image_to_torch` | HWC uint8 → CHW float |
| `read_image` / `load_image` | 读图（可选灰度） |
| `skew_symmetric` | 反对称矩阵 |
| `rad2rotmat(roll, pitch, yaw)` | 欧拉角 → 旋转矩阵 |
| `fov2focal(fov, size)` / `focal2fov(focal, size)` | 视场角 ↔ 焦距 |
| `pitch2rho` / `rho2pitch` | pitch 与入射角转换 |
| `rad2deg` / `deg2rad` | 角度转换 |
| `get_device()` | 返回 `cuda / mps / cpu` |
| `print_calibration(result)` | 打印 roll/pitch/vFoV/focal 及不确定度 |

## 11. Demo 辅助

- `viz2d.py`：保存图像、绘制点/线/箭头/热力图。
- `interactive_demo.py`：
  - `VideoCapture` 用独立线程只保留最新帧，降低延迟；
  - `InteractiveDemo` 持有 `GeoCalib`，每帧 `calibrate`；
  - `render_frame` 叠加地平线、up 场、纬度热力图、置信度、去畸变图、虚拟网格/立方体；
  - `format_results` 输出 roll/pitch/vFoV/focal 与不确定度。
