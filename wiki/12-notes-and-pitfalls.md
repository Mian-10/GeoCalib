# 12 · 注意事项、重复实现与已知问题

> 本文件汇总阅读源码时最容易被忽略的细节：两套并行实现的差异、可以复现的缺陷、路径/尺寸/版本等隐性约束，以及扩展代码时的入手点。
> 建议在动手改代码或复现论文结果之前先通读一遍。

---

## 12.1 两套并行实现（`geocalib` vs `siclib`）

仓库不是“一个库 + 一个脚本目录”，而是两个**可独立安装**的包，且几何与优化部分存在两份实现：

| 功能 | 推理包 `geocalib/` | 训练/评估库 `siclib/geometry/` + `siclib/models/optimization/` |
| --- | --- | --- |
| 相机模型 | [geocalib/camera.py](../geocalib/camera.py) | [siclib/geometry/camera.py](../siclib/geometry/camera.py) |
| 相机基类 | [geocalib/camera.py](../geocalib/camera.py) 内的 `BaseCamera` | [siclib/geometry/base_camera.py](../siclib/geometry/base_camera.py) |
| 重力表示 | [geocalib/gravity.py](../geocalib/gravity.py) | [siclib/geometry/gravity.py](../siclib/geometry/gravity.py) |
| 球面流形 | [geocalib/gravity.py](../geocalib/gravity.py) | [siclib/geometry/manifolds.py](../siclib/geometry/manifolds.py) |
| 透视场 | [geocalib/perspective_fields.py](../geocalib/perspective_fields.py) | [siclib/geometry/perspective_fields.py](../siclib/geometry/perspective_fields.py) |
| LM 优化器 | [geocalib/lm_optimizer.py](../geocalib/lm_optimizer.py)（`nn.Module` 风格） | [siclib/models/optimization/lm_optimizer.py](../siclib/models/optimization/lm_optimizer.py)（`BaseModel` 风格） |
| 损失函数 | [geocalib/lm_optimizer.py](../geocalib/lm_optimizer.py) 内 | [siclib/models/optimization/losses.py](../siclib/models/optimization/losses.py) |
| 网络模块 | [geocalib/modules.py](../geocalib/modules.py) | [siclib/models/networks/](../siclib/models/networks/) + `encoders/` + `decoders/` |
| 入口封装 | [geocalib/extractor.py](../geocalib/extractor.py) `GeoCalib.calibrate` | [siclib/models/networks/geocalib.py](../siclib/models/networks/geocalib.py) `GeoCalib`、[geocalib_pretrained.py](../siclib/models/networks/geocalib_pretrained.py) |

### 关键差异

| 维度 | `geocalib` 版 | `siclib` 版 |
| --- | --- | --- |
| 设计目标 | 单图/多图推理，接口极简 | 训练 + 评估，完全由配置驱动 |
| 配置方式 | 类内 `default_conf` 字典 + 构造参数 | Hydra/OmegaConf + `get_model` 动态导入 |
| 优化步数默认值 | 30（`LMOptimizer`）、`InferenceOptimizer` 也默认 30 | 10（训练时每步都跑 LM，故意压低） |
| 损失函数默认值 | `huber_loss`，`up/lat_loss_fn_scale=1e-2` | `squared_loss`（训练），`InferenceOptimizer` 用 `huber_loss` |
| 初始化 | 仅 `trivial` / 先验 | 额外支持 `heuristic` 与任意注册模型做 initializer |
| 输出 | `camera`、`gravity`、协方差、不确定度、up/lat 场与置信度 | 额外输出 `up_loss`/`latitude_loss`/`param_total` 等训练损失与指标 |
| 多图共享内参 | 支持 `shared_intrinsics=True` | 支持 `shared_intrinsics`（`setup_optimization_and_priors`） |
| 多相机 rig | 支持 `camera_R_rig`，共享重力 | **不支持** `camera_R_rig`（`siclib/models/extractor.py` 无此参数） |
| 先验畸变字段 | `prior_dist` | 初始化时读 `prior_k1`，开关判断时读 `prior_dist`（不一致，见 12.2） |
| 主点 | 固定图像中心 | 固定图像中心 |

### 选择建议

- **只想做推理 / 集成到自己的项目**：用 `geocalib`，`import geocalib; geocalib.calibrate(img)` 即可，依赖最少，支持 rig 与共享内参。
- **想复现论文训练 / 跑 benchmark**：用 `siclib`。评估默认走 `networks.geocalib_pretrained`，即实际调用的仍是 `geocalib` 包里的预训练模型。
- **修改几何或损失**：如果改动需要训练，请改 `siclib` 侧；如果只服务推理，改 `geocalib` 侧。**两边都要同步**，否则推理与训练行为会漂移。
- 注意：`siclib/models/networks/geocalib_pretrained.py` 直接 `from geocalib import GeoCalib`，说明两个包在评估路径上是**硬耦合**的（`siclib` 安装时 `geocalib` 必须可用）。

---

## 12.2 已知问题

### P0：`get_trivial_estimation` 的 `init_vfov` 未定义，训练路径必崩

位置：[siclib/models/optimization/utils.py](../siclib/models/optimization/utils.py#L108-L109)

```python
focal = data.get("prior_focal", 0.7 * torch.max(batch_h, batch_w))
init_vfov = init_vfov if focal is None else focal2fov(focal, h)   # init_vfov 未定义
```

- `init_vfov` 在本函数中从未赋值，第一次引用就抛 `UnboundLocalError`（无论 `focal` 是否为 `None`）。
- 调用链：`LMOptimizer._forward` → `get_initial_estimation(..., trivial_init=True)` → `get_trivial_estimation`。而 `LMOptimizer.default_conf["init_conf"]["name"]` 正是 `"trivial"`，`configs/model/geocalib.yaml` 也没有覆盖它，因此**默认训练配置会命中**。
- 对比正确实现：[geocalib/lm_optimizer.py](../geocalib/lm_optimizer.py#L43-L44) 中为 `init_vfov = focal2fov(focal, h)`。
- 修复建议：

```python
focal = data.get("prior_focal", 0.7 * torch.max(batch_h, batch_w))
init_vfov = focal2fov(focal, h)
```

- 规避方式（不改代码）：在 `optimizer` 配置里把 `init_conf.name` 改成 `heuristic`，此时走 `get_heuristic_estimation`，不会触发该分支。代价是初始化依赖 `up_field`/`latitude_field` 的可用性与质量。

### P1：`prior_k1` 与 `prior_dist` 语义不一致，`use_prior_k1=True` 不会关闭畸变估计

- 数据集侧：[siclib/datasets/simple_dataset.py](../siclib/datasets/simple_dataset.py#L161) 生成键 `prior_k1`。
- 初始化侧：[siclib/models/optimization/utils.py](../siclib/models/optimization/utils.py#L113) 读 `prior_k1` 构造相机。
- 开关侧：[siclib/models/optimization/lm_optimizer.py](../siclib/models/optimization/lm_optimizer.py#L119) 只检查 `prior_dist`，据此决定 `estimate_dist = False`。

结果：`use_prior_k1=True` 时初始化确实用了先验 k1，但优化阶段仍把畸变当作自由变量继续更新，**先验并没有被“锁住”**。若目标是固定畸变，需要同时提供 `prior_dist`（或把两侧键名统一）。

### P2：`SimpleDataset.p_rotate` 定义了但从未使用

[siclib/datasets/simple_dataset.py](../siclib/datasets/simple_dataset.py#L84) 注释写着“按 ±90° 旋转图像的概率”，但全仓库检索只有定义处命中。配置成非零值不会产生任何旋转增强。

### P2：`GeoCalibPretrained` 不支持 loss/metrics

[siclib/models/networks/geocalib_pretrained.py](../siclib/models/networks/geocalib_pretrained.py) 中 `metrics()` 与 `loss()` 直接 `raise NotImplementedError`。这意味着它只能用于**评估/推理**，不能放进训练循环。训练请用 `networks.geocalib`。

### P2：`siclib` 的 `GeoCalib` 不透传 `camera_R_rig`

[siclib/models/networks/geocalib.py](../siclib/models/networks/geocalib.py#L45-L49) 只透传 `image / scales / prior_gravity / prior_focal / prior_k1`。若要做刚性多相机 rig 的共享重力估计，必须走 `geocalib` 包的 `calibrate(..., camera_R_rig=...)`。

### P2：`strict_conf` 关闭时未知配置字段会被静默忽略

`BaseModel` 的 `strict_conf = False` 意味着 YAML 里拼错的键（如 `up_loss_scale`）不会报错，只会被丢掉，表现为“改了配置但没生效”。调参时优先打开严格模式或对照 [09-configuration.md](./09-configuration.md) 的字段表。

### 设计约束：主点固定为图像中心

`BaseCamera.from_dict` 在未显式给出 `px/py` 时默认 `cx=w/2, cy=h/2`，`camera_models` 的四个模型都不把主点作为优化变量。也就是说 GeoCalib 估计的是 **f（或 vFoV）+ 重力 + 可选径向畸变**，不做主点标定。裁剪后的图像必须重新计算中心，`crop()` 已处理该逻辑。

---

## 12.3 使用注意事项

### 安装与版本

| 注意点 | 说明 |
| --- | --- |
| `albumentations` 必须 `<2` | `siclib/requirements.txt` 限定 `>=1.4,<2`；2.x 改了 API，数据增强会报错 |
| `torch >= 2.0` | 训练用到 `torch.func` / `torch.compile`；旧版本会缺函数 |
| `weights_only=False` | `BaseModel.flexible_load` 以非安全模式加载，加载来源不明的权重需谨慎 |
| `pkg_resources` | `siclib` 内部有使用，Python 3.12+ 需确保 `setuptools` 仍安装 |
| `opencv` EXR | 读取 HDR 全景需要 `OPENCV_IO_ENABLE_OPENEXR=1` |
| 文件句柄 | 大数据集多 worker 时可能触及 `RLIMIT_NOFILE`，建议 `ulimit -n 4096` |

### 路径与工作目录

- `geocalib` 的 `weights` 参数（`pinhole` / `distorted` / 本地 `.tar` 路径）相对**当前工作目录**解析；从其他目录运行脚本时容易找不到权重。
- `siclib` 的 `weights/mscan_b.pth`、`weights/geocalib.tar`、`weights/deepcalib.tar` 同样是相对路径，仓库**不包含 `weights/` 目录**，需自行准备。可用 `settings.py` 中的 `TRAINING_PATH` / `DATA_PATH` 覆盖根路径。
- 数据集的 `data.dataset_dir` 默认 `data/openpano/...`，也是相对路径。
- 评估配置位于 [siclib/eval/configs/](../siclib/eval/configs/)，**不是** `siclib/configs/eval/`。传 `--conf` 时按模块名解析（如 `geocalib-pinhole`）。

### 图像尺寸

- 推理预处理器 `ImagePreprocessor({"resize": 320, "edge_divisible_by": 32})`：图像短边缩放到 320，并保证宽高是 32 的倍数。
- 网络下采样要求输入尺寸可被 32 整除；直接喂任意尺寸原图到 `siclib` 模型（绕过 preprocessor）可能因 padding/上采样对不齐而报错或精度下降。
- `square_crop` 默认行为会影响 vFoV 语义：裁剪后焦距不缩放的话角度会变，`crop()` 与 `undo_scale_crop()` 必须成对使用。

### 相机模型一致性

- 训练配置的 `optimizer.camera_model`、评估配置的 `model.camera_model`、推理时的 `camera_model` 参数三者必须一致。
- `megadepth2k` 用 pinhole，`megadepth2k_radial` 用 `SimpleRadial`；用 pinhole 模型去评 radial 数据集，`k1` 相关指标无意义。
- 相机模型名必须在 `camera_models` 字典中：`pinhole`、`radial`、`simple_radial`、`simple_divisional`。拼错会直接 assert 失败。

### 先验

- 先验键名是 `prior_focal`、`prior_gravity`、`prior_k1` / `prior_dist`。推理 API 的入口参数则是 `priors={"focal": ..., "gravity": ...}`，由 `extractor` 负责改名与缩放（`prior_focal * scales[1]`）。
- 给定 `prior_gravity` 后 `estimate_gravity=False`，LM 不再优化重力，但仍会用其计算透视场残差。
- 先验焦距必须与**输入到网络的缩放后图像**匹配，`extractor` 已自动乘 `scales[1]`；自己调用底层模型时要手动处理。

### 多图与 rig

- `shared_intrinsics=True`：batch 内所有图共享同一焦距，适合同一相机拍摄的多视角图。
- `camera_R_rig`：仅在 `geocalib` 包支持，表示 rig 到各相机的旋转；配合它可让多相机共享同一重力方向。
- 批量输入需保持同一尺寸，或分别调用后自行聚合。

### 训练

- `num_steps` 会覆盖 `epochs`：`epochs = num_steps // len(train_loader) + 1`，二者不要同时“精打细算”。
- 多卡必须用 `--distributed`，且各节点需能访问同一个 `output_dir`（`file://` 初始化锁）。
- `--overfit` 只用于调试，会在少量样本上反复迭代，指标不可用于结论。
- `--detect_anomaly` 会显著拖慢训练，仅在定位 NaN 时开启。
- 训练产物在 `outputs/training/`，代码快照会一并复制，便于回溯。

### 评估

- 三步流程（预测 → 评估 → 汇总）会产生 `predictions.h5` 缓存；改了模型或权重记得 `--overwrite`，否则会复用旧预测。
- `openpano` benchmark 的 `url=None`，需要本地已构建数据集；未构建时会报缺文件而不是自动下载。
- `stanford2d3d` 有独立的使用条款，需自行确认后再下载。
- 评估默认模型是 `networks.geocalib_pretrained`（即 `geocalib` 包的权重），不是 `siclib` 训练出的 checkpoint。要评自己的权重需改 `model` 配置或传 `--checkpoint`。

---

## 12.4 性能与显存

| 场景 | 建议 |
| --- | --- |
| 训练 pinhole | 默认 batch 24；显存不足时优先降 batch，再考虑降 `resize` |
| 训练 radial | 默认 batch 18（多一个畸变参数、优化更重） |
| 推理 | `resize=320` 已足够；`num_steps=30` 是精度/速度折中，降步数会掉精度 |
| 多图 batch | 显存随 batch 线性增长，`shared_intrinsics` 不减少显存 |
| 数据加载 | `num_workers=8`、`prefetch_factor=2`；I/O 瓶颈时先确认不是 `RLIMIT_NOFILE` |
| 编译加速 | `--compile` 首次编译较慢，小模型收益有限 |
| 半精度 | 训练用 `--mixed_precision`；推理默认 FP32，协方差对数值精度较敏感 |

---

## 12.5 复现性

- `siclib.train` 会把 `config.yaml` 与代码快照写入 `output_dir`，论文复现应保留这两项。
- 数据增强中随机数由 `set_seed` + `worker_init_fn` 控制；`reseed` 打开时每个 epoch 重新播种。
- LM 优化在 `torch.no_grad()`/`eval()` 下确定性较强，但 `early_stop` 与浮点归约会带来极小差异。
- 多卡训练时各进程的 batch 与 worker 数会均分，改变卡数会影响有效学习率与总步数对应关系，复现时请固定卡数。
- 数据集生成（`create_dataset_from_pano.py`）用字符串 seed 经 SHA-256 转整数，**同一 seed 在不同机器上可复现**；但 `images_per_pano`、`n_workers` 改变会改变划分。

---

## 12.6 扩展指南

| 想做的事 | 入手点 |
| --- | --- |
| 新增相机模型 | 继承 `BaseCamera`（`siclib/geometry/base_camera.py`），实现 `project/distort/undistort` 与必要雅可比；在 `camera.py` 的 `camera_models` 注册；如需训练，同步改 `geocalib/camera.py` |
| 新增数据集 | 继承 `BaseDataset` 或 `SimpleDataset`；准备 CSV（`width,height,vfov,roll,pitch,k1,k2`）；在 `siclib/configs/data/` 加 YAML |
| 新增 benchmark | 在 `siclib/eval/` 实现 pipeline（可继承 `SimplePipeline`），在 `siclib/eval/__init__.py` 的注册表加入名称 |
| 新增 backbone/decoder | 放到 `siclib/models/{encoders,decoders}/`，通过 `get_model("decoders.xxx")` 动态导入，无需改注册表 |
| 新增损失 | 在 `siclib/models/optimization/losses.py` 添加函数，配置里用 `loss_fn` 名字引用 |
| 新增初始化策略 | 在 `siclib/models/optimization/utils.py` 实现函数，并把 `init_conf.name` 扩展为对应分支 |
| 新增数据增强 | 在 `siclib/datasets/augmentations.py` 实现 `BaseAugmentation` 子类并注册到 `augmentations` 字典 |

---

## 12.7 症状 → 原因 → 解决

| 症状 | 可能原因 | 解决 |
| --- | --- | --- |
| 训练第一步报 `UnboundLocalError: init_vfov` | 12.2 的 P0 缺陷 | 改 `init_conf.name: heuristic`，或按建议修 `utils.py` |
| 配置改了但不生效 | `strict_conf=False` 静默丢键 | 对照字段表检查拼写；必要时开严格模式 |
| `KeyError: 'prior_dist'` 或先验不锁畸变 | `prior_k1` / `prior_dist` 混用 | 统一键名，见 P1 |
| 找不到 `weights/mscan_b.pth` | 相对 cwd 的路径 | 在仓库根目录运行，或改绝对路径/`TRAINING_PATH` |
| 评估复用旧结果 | `predictions.h5` 已存在 | 加 `--overwrite`（必要时 `--overwrite_eval`） |
| `--conf` 找不到配置 | 评估配置在 `siclib/eval/configs/` | 用模块名 `geocalib-pinhole` 等，不要带路径 |
| 图像尺寸报错/结果异常 | 宽高非 32 倍数、未走 preprocessor | 使用 `ImagePreprocessor` 或自行 pad 到 32 的倍数 |
| 主点偏移导致精度差 | 主点被固定为中心 | 裁剪后调用 `crop()`；或改用支持主点的流程 |
| DDP 启动卡住 | 各进程看不到同一输出目录 | 使用共享文件系统路径 |
| 读取 HDR 全景失败 | 未开启 OpenEXR | `export OPENCV_IO_ENABLE_OPENEXR=1` |
| `albumentations` 报 API 错误 | 安装了 2.x | 降级到 `<2` |
| 多 worker 报 “Too many open files” | 文件句柄上限 | `ulimit -n 4096` |

---

## 12.8 上手前自查清单

- [ ] 在仓库根目录运行，`weights/` 与 `data/` 路径正确
- [ ] `geocalib` 可 `import`，权重能下载或已本地放置
- [ ] 训练前确认 `init_conf.name` 已避开 P0 缺陷
- [ ] 训练 / 评估 / 推理三处的 `camera_model` 一致
- [ ] 评估前确认 benchmark 数据集已就位，且知道 `--overwrite` 的作用
- [ ] 输入图像的短边缩放与 32 倍数约束由 `ImagePreprocessor` 保证
- [ ] 修改几何/优化代码时，同步检查 `geocalib/` 与 `siclib/` 两份实现

---

**相关文档**：[01-architecture.md](./01-architecture.md) · [04-geometry-and-optimization.md](./04-geometry-and-optimization.md) · [09-configuration.md](./09-configuration.md) · [11-api-reference.md](./11-api-reference.md)
