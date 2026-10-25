# 11 · API 速查

> 本章汇总关键类与函数签名、参数与返回值，便于快速查阅。
> 更深入的实现说明见 [03](./03-geocalib-inference-package.md)、[04](./04-geometry-and-optimization.md)、[05](./05-siclib-model-zoo.md)、[06](./06-siclib-datasets.md)、[07](./07-siclib-evaluation.md)、[08](./08-siclib-training.md)。

---

## 1. `geocalib` 推理包

### 1.1 `GeoCalib`（对外接口，`geocalib/extractor.py`）

```python
class GeoCalib(nn.Module):
    def __init__(self, weights: str = "pinhole")
    def load_image(self, path: Path) -> torch.Tensor
    def calibrate(
        self,
        img: torch.Tensor,                       # (C,H,W) / (1,C,H,W) / (B,C,H,W)，RGB，[0,1]
        camera_model: str = "pinhole",
        priors: Optional[Dict[str, torch.Tensor]] = None,
        shared_intrinsics: bool = False,
        camera_R_rig: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]
```

`weights` 取值：`"pinhole"` / `"distorted"`（自动下载）或本地权重路径。
`priors` 支持 `{"focal": f, "gravity": g}`（张量或 batch 张量）。

**返回字典**

| 键 | 形状 / 类型 | 说明 |
| --- | --- | --- |
| `camera` | `BaseCamera` | 已还原到原始图像尺寸的相机参数 |
| `gravity` | `Gravity` | 相机坐标系重力向量 |
| `covariance` | `(B, D, D)` | LM 参数协方差 |
| `up_field` | `(B,2,H,W)` | 上方向场（原始分辨率） |
| `latitude_field` | `(B,1,H,W)` | 纬度场 |
| `up_confidence` / `latitude_confidence` | `(B,H,W)` | 逐像素置信度 |
| `focal_uncertainty` / `vfov_uncertainty` | `(B,)` | 焦距 / vFoV 不确定度 |
| `roll_uncertainty` / `pitch_uncertainty` / `gravity_uncertainty` | `(B,)` | 重力相关不确定度 |

> 返回值仅包含实际存在的键；未估计的参数不会出现。

### 1.2 内部网络（`geocalib/geocalib.py`）

| 类 | 签名 | 说明 |
| --- | --- | --- |
| `LowLevelEncoder` | `forward(data) -> {"features": (B,64,H/2,W/2)}` | 3→64→64，stride 2 |
| `UpDecoder` | `forward(data) -> {"up_field", "up_confidence"}` | LightHamHead 输出 |
| `LatitudeDecoder` | `forward(data) -> {"latitude_field", "latitude_confidence"}` | LightHamHead 输出 |
| `PerspectiveDecoder` | `forward(data) -> dict` | 组合 up + latitude |
| `GeoCalib` | `__init__(**optimizer_options)`；`forward(data) -> dict`；`flexible_load(state_dict)` | 完整推理网络 |

### 1.3 相机模型（`geocalib/camera.py`）

```python
camera_models = {
    "pinhole": Pinhole,
    "simple_radial": SimpleRadial,
    "radial": Radial,
    "simple_divisional": SimpleDivisional,
}
```

`BaseCamera` 内部数据布局：`[w, h, fx, fy, cx, cy, k1, k2]`。

| 成员 | 签名 | 说明 |
| --- | --- | --- |
| `from_dict` | `@classmethod from_dict(cls, param_dict) -> BaseCamera` | 从字典构造（键：`width,height,vfov,roll,pitch,k1,k2,…`） |
| `name` | `@classmethod name() -> str` | 模型名 |
| `size` | 属性 `(...,2)` | `[w, h]` |
| `f` | 属性 `(...,2)` | `[fx, fy]` |
| `vfov` / `hfov` | 属性 `(...,)` | 视场角（弧度） |
| `c` | 属性 `(...,2)` | 主点 |
| `K` | 属性 `(...,3,3)` | 内参矩阵 |
| `update_focal` | `(delta, as_log=False)` | 更新焦距（clamp 到 5°~150° vFoV） |
| `scale` / `crop` | `(scales)` / `(pad)` | 预处理后的几何同步 |
| `undo_scale_crop` | `(data: Dict) -> BaseCamera` | 还原到原图尺寸 |
| `project` / `J_project` | `(p3d)` | 归一化平面投影与雅可比 |
| `distort` / `undistort` | `(pts, return_scale=False)` / `(pts)` | 畸变/去畸变，返回 `(pts, valid)` |
| `J_distort` / `J_undistort` | `(p2d, wrt="pts")` | 数值/解析雅可比 |
| `world2image` / `image2world` | `(p3d)` / `(p2d)` | 3D→像素 / 像素→归一化方向 |
| `pixel_coordinates` | 属性 `(...,N,2)` | 全图像素网格 |
| `undistort_image` | `(img) -> torch.Tensor` | 基于 `cv2.remap` 的整图去畸变 |
| `get_img_from_pano` | `(pano_img, gravity, yaws, resize_factor=None)` | 从全景图渲染透视图像 |

畸变参数属性：`SimpleRadial.k1` / `k1_hat`，`Radial.k1/k2`，`SimpleDivisional.k1`；`update_dist(delta, dist_range)` 在合法范围内更新。

### 1.4 重力（`geocalib/gravity.py`）

```python
class Gravity(TensorWrapper):      # 内部 3 维单位向量
    @classmethod from_rp(cls, roll, pitch) -> Gravity
    vec3d / x / y / z               # 属性
    roll / pitch / rp               # 属性（弧度 / 拼接）
    J_roll / J_pitch / J_rp         # 雅可比
    R                               # 3x3 旋转矩阵
    J_R
    update(delta, spherical=False) -> Gravity
    J_update(spherical=False)
```

`from_rp` 公式：

```python
g = [-sin(roll)*cos(pitch), -cos(roll)*cos(pitch), sin(pitch)]
```

### 1.5 透视场（`geocalib/perspective_fields.py`）

| 函数 | 签名 | 返回 |
| --- | --- | --- |
| `get_horizon_line` | `(camera, gravity, relative=True)` | 地平线直线参数 |
| `get_up_field` | `(camera, gravity, normalize=True)` | `(B,2,H,W)` 上方向场 |
| `J_up_field` | `(camera, gravity, spherical=..., log_focal=..., camera_R_rig=None)` | 对参数的雅可比 |
| `get_latitude_field` | `(camera, gravity)` | `(B,1,H,W)` 纬度角（弧度） |
| `J_latitude_field` | 同上 | 雅可比 |
| `get_perspective_field` | `(camera, gravity, use_up=True, use_latitude=True, spherical=..., log_focal=..., camera_R_rig=None)` | `(up, latitude)` |
| `J_perspective_field` | `(camera, gravity, spherical=..., log_focal=..., camera_R_rig=None)` | 合并雅可比 |

### 1.6 LM 优化器（`geocalib/lm_optimizer.py`）

```python
class LMOptimizer(nn.Module):
    default_conf = {
        "camera_model": "pinhole", "shared_intrinsics": False,
        "num_steps": 30, "lambda_": 0.1, "fix_lambda": False,
        "early_stop": True, "atol": 1e-8, "rtol": 1e-8,
        "use_spherical_manifold": True, "use_log_focal": True,
        "up_loss_fn_scale": 1e-2, "lat_loss_fn_scale": 1e-2,
        "verbose": False,
    }

    def set_camera_model(self, camera_model: str) -> None
    def setup_optimization_and_priors(self, data=None, shared_intrinsics=False) -> None
    def calculate_residuals(self, data, estimate) -> Dict[str, Tensor]
    def calculate_costs(self, data, estimate) -> Dict[str, Tensor]
    def calculate_gradient_and_hessian(self, data, estimate) -> Tuple[Tensor, Tensor]
    def setup_system(self, data, estimate) -> Tuple[Tensor, Tensor]
    def estimate_uncertainty(self, ...) -> Dict[str, Tensor]
    def update_estimate(self, ...) -> Any
    def optimize(self, data) -> Tuple[BaseCamera, Gravity, Dict]
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]
```

模块级工具：

| 函数 | 说明 |
| --- | --- |
| `get_trivial_estimation(data, camera_model)` | 用图像中心/默认 vFoV 初始化 |
| `scaled_loss(x, loss_fn)` | 按尺度归一化后的损失 |
| `huber_loss(x)` | 返回 `(loss, weight, grad)` |
| `early_stop(new_cost, prev_cost, atol, rtol)` | 收敛判断 |
| `update_lambda(lam, rho)` | 自适应阻尼 |
| `optimizer_step(H, g, lam, ...)` | Cholesky 求解 + 更新 |
| `setup_system_rig_unshared_intrinsics` / `setup_system_shared_intrinsics` | 多图 / rig 正规方程 |
| `get_gravity_uncertainty(...)` | 由协方差提取重力不确定度 |
| `initialize_gravity_rig(gravity_init, camera_R_rig)` | rig 重力初始化 |

### 1.7 工具（`geocalib/utils.py`）

| 函数 | 说明 |
| --- | --- |
| `fit_to_multiple(x, multiple, mode="center", crop=False)` | 把尺寸调整为整数倍 |
| `fit_features_to_multiple(features, multiple, crop=False)` | 对字典中的 tensor 批量处理 |
| `ImagePreprocessor(conf)` | `__call__(img) -> {image, scales, image_size, transform, …}` |
| `numpy_image_to_torch(image)` / `torch_image_to_numpy(image)` | 类型转换 |
| `read_image(path, grayscale=False)` / `load_image(path, grayscale=False, return_tensor=True)` | 读图 |
| `write_image(img, path)` | 写图 |
| `skew_symmetric(v)` | 反对称矩阵 |
| `rad2rotmat(roll, pitch, yaw=0)` | 欧拉角→旋转矩阵 |
| `fov2focal(fov, size)` / `focal2fov(focal, size)` | vFoV ↔ 焦距 |
| `pitch2rho(pitch, f, h)` / `rho2pitch(rho, f, h)` | pitch ↔ ρ |
| `rad2deg(rad)` / `deg2rad(deg)` | 角度转换 |
| `get_device()` | 自动选择 `cuda/mps/cpu` |
| `print_calibration(results)` | 打印标定结果 |

---

## 2. `siclib/models`

### 2.1 模型注册

```python
from siclib.models import get_model
model = get_model("networks.geocalib")(conf)
```

支持完整路径或短名；优先找 `BaseModel` 子类，否则用模块级 `__main_model__`。

### 2.2 `BaseModel`（`models/base_model.py`）

```python
class BaseModel(nn.Module, metaclass=MetaModel):
    default_conf = {
        "name": None, "trainable": True,
        "freeze_batch_normalization": False,
        "timeit": False, "watch": False,
    }
    required_data_keys = []
    strict_conf = False

    def __init__(self, conf)
    def flexible_load(self, state_dict)
    def train(self, mode=True)
    def forward(self, data) -> Dict[str, Tensor]      # 校验 required_data_keys
    def _init(self, conf)                              # 子类实现
    def _forward(self, data) -> Dict[str, Tensor]      # 子类实现
    def loss(self, pred, data) -> Tuple[Dict, Dict]   # 子类实现，返回 (losses, metrics)
```

- `MetaModel` 在类创建时递归合并父类 `default_conf` → `base_default_conf`。
- `conf.weights` 非空时自动 `torch.load` 并 `flexible_load`（键兼容 `model` / `state_dict`，支持 `gravity` ↔ `up` 重命名）。

### 2.3 GeoCalib 训练模型（`networks/geocalib.py`）

```python
class GeoCalib(BaseModel):
    default_conf = {
        "backbone": {"name": "encoders.mscan"},
        "ll_enc": {"name": "encoders.low_level_encoder"},
        "perspective_decoder": {"name": "decoders.perspective_decoder"},
        "optimizer": {"name": "optimization.lm_optimizer"},
    }
    required_data_keys = ["image"]

    def _forward(self, data) -> Dict[str, Tensor]   # perspective_decoder + optimizer
    def loss(self, pred, data) -> Tuple[Dict, Dict] # 汇总 perspective_total + param_total
```

### 2.4 `GeoCalibPretrained`（`networks/geocalib_pretrained.py`）

包装 `geocalib` 推理包，供评估直接使用预训练权重：

```python
conf = {
    "name": "networks.geocalib_pretrained",
    "camera_model": "pinhole",       # 或 "simple_radial"
    "model_weights": "pinhole",      # 或 "distorted"
}
```

### 2.5 `DeepCalib`（`networks/deepcalib.py`）

DenseNet161 骨干 + 4 个分类头（`roll / rho / vfov / k1_hat`），用 NLL 分箱损失；`bounds` 控制各参数取值范围。

### 2.6 解码器

| 类 | 关键配置 | 说明 |
| --- | --- | --- |
| `PerspectiveDecoder` | `use_up`, `use_latitude`, `up_decoder`, `latitude_decoder` | 组合并汇总 loss |
| `UpDecoder` | `loss_type`（l1/l2/dot/cauchy/huber/classification）、`use_uncertainty_loss`、`recall_thresholds` | 输出 `up_field` / `up_confidence` |
| `LatitudeDecoder` | 同上 | 输出 `latitude_field` / `latitude_confidence` |
| `LightHamHead` | `predict_uncertainty` | NMF2D/Hamburger 解码 |
| `FPN` | — | 通用特征金字塔（备选） |

`UpDecoder.default_conf` 片段：

```python
{"loss_type": "l1", "use_uncertainty_loss": True,
 "recall_thresholds": [1, 3, 5, 10],
 "decoder": {"name": "decoders.light_hamburger", "predict_uncertainty": True}}
```

### 2.7 优化器（`models/optimization/`）

| 类 | 关键差异 |
| --- | --- |
| `LMOptimizer` | 训练用：`num_steps=10`、`loss_fn="squared_loss"`、`early_stop=False`、`init_conf={"name":"trivial"}` |
| `InferenceOptimizer` | 推理用：`num_steps=30`、`loss_fn="huber_loss"`、`early_stop=True` |

公共方法：`set_camera_model`、`setup_optimization_and_priors`、`forward(data)`。
损失函数（`losses.py`）：`scaled_loss`、`squared_loss`、`huber_loss`、`barron_loss`、`scaled_barron`。
初始化（`utils.py`）：`get_initial_estimation`、`get_heuristic_estimation`、`get_trivial_estimation`、`early_stop`、`update_lambda`、`optimizer_step`。

### 2.8 预测缓存（`models/cache_loader.py`）

```python
class CacheLoader(BaseModel):
    default_conf = {
        "path": "???",          # 支持 format string
        "data_keys": None, "device": None, "trainable": False,
        "add_data_path": True, "collate": True,
        "scale": ["keypoints"], "padding_fn": None,
        "padding_length": None, "numeric_type": "float32",
    }
    required_data_keys = ["name"]
```

`forward(data)` 按 `data["name"]` 从 h5 读取预测，可选反缩放、类型转换与批处理。

### 2.9 `siclib` 版推理接口（`models/extractor.py`）

```python
class GeoCalib(nn.Module):
    def __init__(self, weights="pinhole")
    def calibrate(self, img, camera_model="pinhole", priors=None,
                  shared_intrinsics=False) -> Dict[str, Tensor]
```

与 `geocalib` 版差异：**不支持 `camera_R_rig`**；加载自定义权重时从 checkpoint 的 `conf` 重建模型结构。

---

## 3. `siclib/datasets`

| API | 签名 | 说明 |
| --- | --- | --- |
| `get_dataset` | `(name) -> Type[BaseDataset]` | 动态导入 |
| `BaseDataset.__init__` | `(conf)` | 合并默认配置并调用 `_init` |
| `BaseDataset.get_data_loader` | `(split, shuffle=None, pinned=False, distributed=False) -> DataLoader` | 训练/验证/测试 loader |
| `BaseDataset.get_overfit_loader` | `(split) -> DataLoader` | 调试用 |
| `SimpleDataset` | `default_conf` 见 06 章 | CSV 驱动数据集 |
| `_SimpleDataset.__getitem__` | `(idx) -> Dict` | 返回样本字典 |
| `load_csv` | `(csv_file, img_root) -> (infos, params, gravity)` | 解析 CSV |
| `augmentations` | 字典 | `default/dark/perspective/deepcalib/geocalib/identity` |
| `BaseAugmentation.__call__` | `(image, return_tensor=False)` | 增强入口 |
| `ImagePreprocessor.__call__` | `(img, interpolation=None) -> Dict` | 缩放/裁剪 |

---

## 4. `siclib/eval`

| API | 签名 | 说明 |
| --- | --- | --- |
| `get_benchmark` | `(benchmark) -> Type[EvalPipeline]` | 动态导入 |
| `run_benchmark` | `(benchmark, eval_conf, experiment_dir, model=None)` | 训练中调用 |
| `EvalPipeline.run` | `(experiment_dir, model=None, overwrite=False, overwrite_eval=False)` | 完整评估 |
| `EvalPipeline.get_predictions` | `(experiment_dir, model=None, overwrite=False) -> Path` | 导出预测 |
| `EvalPipeline.run_eval` | `(loader, pred_file) -> (summaries, figures, results)` | 计算指标 |
| `SimplePipeline` | `default_conf` 见 07 章 | 通用实现 |
| `calculate_pixel_projection_error` | `(camera_pred, camera_gt, N=500, distortion_only=True)` | 像素投影误差 |
| `compute_camera_metrics` | `(camera_pred, camera_gt, thresholds)` | 相机指标 |
| `compute_gravity_metrics` | `(gravity_pred, gravity_gt)` | 重力指标 |
| `load_eval` / `save_eval` / `exists_eval` | `(dir, …)` | 结果读写 |
| `parse_eval_args` | `(benchmark, args, configs_path, default=None) -> (name, conf)` | 参数解析 |
| `load_model` | `(model_conf, checkpoint, get_last=False) -> nn.Module` | 加载模型 |

---

## 5. `siclib/train`

| 函数 | 签名 | 说明 |
| --- | --- | --- |
| `training` | `(rank, conf, output_dir, args)` | 单进程训练主循环 |
| `main_worker` | `(rank, conf, output_dir, args)` | rank 0 捕获日志 |
| `do_evaluation` | `(model, loader, device, loss_fn, conf, pbar=True)` | 验证集评估 |
| `get_lr_scheduler` | `(optimizer, conf)` | 构建调度器（含 SequentialLR） |
| `filter_parameters` | `(params, regexp)` | 按正则筛选可训练参数 |
| `pack_lr_parameters` | `(params, base_lr, lr_scaling)` | 参数分组与学习率缩放 |
| `loss_has_nan` | `(loss, distributed)` | NaN 检测（DDP 同步） |

Checkpoint：`save_experiment`、`load_experiment`、`get_last_checkpoint`、`get_best_checkpoint`、`list_checkpoints`、`delete_old_checkpoints`（`utils/experiments.py`）。

---

## 6. 指标与工具

### 6.1 指标（`siclib/models/utils/metrics.py`）

| 函数 | 签名 | 返回 |
| --- | --- | --- |
| `pitch_error` | `(pred_gravity, target_gravity)` | pitch 绝对误差（度） |
| `roll_error` | 同上 | roll 绝对误差（度） |
| `gravity_error` | 同上 | 3D 夹角误差（度） |
| `vfov_error` | `(pred_cam, target_cam)` | vFoV 绝对误差（度） |
| `dist_error` | `(pred_cam, target_cam)` | 畸变参数误差 |
| `latitude_error` | `(predictions, targets)` | 纬度场误差（逐像素） |
| `up_error` | `(predictions, targets)` | 上方向场误差（逐像素） |

### 6.2 聚合工具（`siclib/utils/tools.py`）

| 类 / 函数 | 说明 |
| --- | --- |
| `AverageMetric` / `FAverageMetric` / `MedianMetric` | 增量聚合 |
| `PRMetric` | PR 曲线 |
| `QuantileMetric` | 分位数 |
| `RecallMetric` | 多阈值召回 |
| `AUCMetric` | 多阈值 AUC |
| `compute_recall` / `compute_auc` | 无状态计算 |
| `Timer` | 计时 |
| `get_class` | 动态导入并断言类型 |
| `set_seed` / `get_random_state` / `set_random_state` / `fork_rng` | 随机性控制 |
| `set_num_threads` / `get_device` | 运行时设置 |

### 6.3 I/O 与转换（`siclib/utils/`）

| API | 说明 |
| --- | --- |
| `export_predictions(loader, model, path, keys, optional_keys, as_half, callback_fn)` | 导出预测到 h5 |
| `batch_to_device(data, device)` / `add_batch_dim(data)` | 张量批处理 |
| `fov2focal` / `focal2fov` / `rad2deg` / `deg2rad` | 转换 |
| `load_image` / `write_image` | 图像读写 |
| `capture_outputs(path)` | 日志重定向 |

---

## 7. 输出字段速查

### 7.1 推理返回（`geocalib.extractor.GeoCalib.calibrate`）

见 1.1 节表格。

### 7.2 模型预测（`siclib` `GeoCalib._forward`）

| 键 | 说明 |
| --- | --- |
| `up_field` / `up_confidence` | 上方向场与置信度 |
| `latitude_field` / `latitude_confidence` | 纬度场与置信度 |
| `camera` | 估计相机（`BaseCamera`） |
| `gravity` | 估计重力（`Gravity`） |
| `covariance` | 参数协方差 |
| `focal_uncertainty` / `vfov_uncertainty` | 焦距/vFoV 不确定度 |
| `roll_uncertainty` / `pitch_uncertainty` / `gravity_uncertainty` | 重力不确定度 |

### 7.3 训练损失键

| 层级 | 典型键 |
| --- | --- |
| 透视场 | `up_loss`、`latitude_loss`、`perspective_total` |
| 参数 | `roll_loss`、`pitch_loss`、`vfov_loss`、`k1_loss`、`param_total` |
| 总计 | `total` |

### 7.4 评估摘要键

`mean_<metric>`、`median_<metric>`、`auc_<metric>@<threshold>`、`*_gt`、`*_uncertainty`、`up_error` / `latitude_error` 及其 `_med_` / `_weighted_` 变体。

---

## 8. 快速调用示例

```python
import torch
from geocalib import GeoCalib

model = GeoCalib(weights="pinhole").cuda().eval()
img = model.load_image("assets/pinhole-church.jpg").cuda()
out  = model.calibrate(img, camera_model="pinhole")

print(out["camera"].vfov)          # 弧度
print(out["gravity"].rp)           # [roll, pitch]
print(out["vfov_uncertainty"])     # 不确定度
```

```python
from siclib.models import get_model
from siclib.models.base_model import BaseModel

model = get_model("networks.geocalib")(conf.model).eval()
pred = model(data)
losses, metrics = model.loss(pred, data)
```
