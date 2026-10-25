# 09 · 配置系统与路径约定

> 对应文件：`siclib/configs/`、`siclib/eval/configs/`、`siclib/datasets/configs/`、`siclib/settings.py`
> 相关文档：[06-siclib-datasets.md](./06-siclib-datasets.md)、[07-siclib-evaluation.md](./07-siclib-evaluation.md)、[08-siclib-training.md](./08-siclib-training.md)

仓库中有**两套彼此独立的配置机制**：

| 用途 | 机制 | 位置 |
| --- | --- | --- |
| `geocalib` 推理包 | Python 类内 `default_conf` 字典 + 运行时参数 | `geocalib/*.py` |
| `siclib` 训练/评估 | Hydra + OmegaConf 组合 YAML | `siclib/configs/`、`siclib/eval/configs/`、`siclib/datasets/configs/` |

本章以 Hydra 配置为主，并说明路径约定与推理包的参数。

---

## 1. 配置目录总览

```
siclib/
├── configs/                       # 训练入口配置
│   ├── geocalib.yaml              #   针孔训练（入口）
│   ├── geocalib-radial.yaml       #   畸变微调（入口）
│   ├── deepcalib.yaml             #   DeepCalib 基线（入口）
│   ├── data/
│   │   ├── openpano.yaml
│   │   └── openpano-radial.yaml
│   ├── model/
│   │   ├── geocalib.yaml
│   │   └── deepcalib.yaml
│   └── train/
│       ├── geocalib.yaml
│       └── deepcalib.yaml
├── eval/configs/                  # 评估模型配置（--conf 搜索路径）
│   ├── geocalib-pinhole.yaml
│   ├── geocalib-simple_radial.yaml
│   ├── deepcalib.yaml
│   └── uvp.yaml
└── datasets/configs/              # 数据生成配置（create_dataset_from_pano）
    ├── openpano.yaml
    └── openpano-radial.yaml
```

---

## 2. Hydra 组合规则（训练）

### 2.1 入口配置的 `defaults`

```yaml
# siclib/configs/geocalib.yaml
defaults:
  - data: openpano        # → siclib/configs/data/openpano.yaml
  - train: geocalib       # → siclib/configs/train/geocalib.yaml
  - model: geocalib       # → siclib/configs/model/geocalib.yaml
  - _self_                # 最后应用本文件中的字段（可覆盖上面的默认值）

data:
  train_batch_size: 24
  val_batch_size: 24
  test_batch_size: 24
```

加载后形成一棵完整配置树 `{data, train, model}`。`_self_` 的位置决定本文件字段相对 defaults 的覆盖顺序。

### 2.2 优先级

```
Hydra defaults（data/train/model）  <  入口文件 _self_ 段  <  命令行 dotlist 覆盖
                                   <  train.py 中的 default_train_conf（仅补缺省项）
```

注意 `default_train_conf` 是在 `training()` 内 `OmegaConf.merge(default_train_conf, conf.train)`，因此它**只填充缺失字段**，不会覆盖 YAML 中已有的值。

`BaseModel` / `BaseDataset` / `BaseAugmentation` 内部还有一层类默认值合并：

```
base_default_conf  <  子类 default_conf  <  YAML 传入 conf
```

### 2.3 命令行覆盖示例

```bash
python -m siclib.train my-exp --conf geocalib \
    data.dataset_dir=data/openpano/openpano \
    data.train_batch_size=16 \
    train.lr=5e-5 \
    train.num_steps=50000 \
    model.optimizer.camera_model=simple_radial
```

点号路径与 YAML 结构一一对应；列表/字典可直接写 `{...}`。

---

## 3. 训练配置详解

### 3.1 `configs/train/geocalib.yaml`

| 字段 | 值 | 说明 |
| --- | --- | --- |
| `seed` | `0` | 随机种子 |
| `num_steps` | `150_000` | 总步数 |
| `writer` | `null` | 关闭 tensorboard；可改 `tensorboard`/`wandb` |
| `log_every_iter` | `500` | 日志间隔 |
| `eval_every_iter` | `1000` | 验证间隔 |
| `lr` | `1e-4` | 基础学习率 |
| `optimizer` | `adamw` | 优化器 |
| `clip_grad` | `1.0` | 梯度裁剪 |
| `best_key` | `loss/param_total` | 最佳 checkpoint 指标 |
| `lr_schedule.type` | `SequentialLR` | 组合调度 |
| `lr_schedule.options.milestones` | `[4000]` | 第一个调度切换点 |
| `lr_schedule.options.schedulers` | `LinearLR`(4k warmup) + `MultiStepLR`(`[80k,130k]`, `gamma=0.1`) | |
| `median_metrics` | roll/pitch/gravity/vfov/up/latitude 误差及 recall | 验证时额外记录中位数 |
| `recall_metrics` | `{roll_error: [1,3,5,10], …}` | 多阈值 recall |
| `plot` | `[3, "siclib.visualization.visualize_batch.make_perspective_figures"]` | 验证时抽 3 张可视化 |

### 3.2 `configs/train/deepcalib.yaml`（基线）

`num_steps=20_000`、`lr=1e-4`、`optimizer=adam`、`lr_schedule.type=null`、`eval_every_iter=3000`，指标仅 roll/pitch/vfov。

### 3.3 `configs/model/geocalib.yaml`

```yaml
name: networks.geocalib

ll_enc:
  name: encoders.low_level_encoder

backbone:
  name: encoders.mscan
  weights: weights/mscan_b.pth        # 相对仓库根目录

perspective_decoder:
  name: decoders.perspective_decoder
  up_decoder:
    name: decoders.up_decoder
    loss_type: l1
    use_uncertainty_loss: true
    decoder:
      name: decoders.light_hamburger
      predict_uncertainty: true
  latitude_decoder:
    name: decoders.latitude_decoder
    loss_type: l1
    use_uncertainty_loss: true
    decoder:
      name: decoders.light_hamburger
      predict_uncertainty: true

optimizer:
  name: optimization.lm_optimizer
  camera_model: pinhole
```

`weights` 由 `BaseModel.__init__` 直接 `torch.load(str(conf.weights))`，**按当前工作目录解析**，因此必须在仓库根目录运行训练，或把它改成绝对路径。

### 3.4 `configs/model/deepcalib.yaml`

```yaml
name: networks.deepcalib
bounds:
  roll: [-45, 45]
  rho:  [-2.836, 2.836]     # rho = tan(pitch)/tan(vfov/2)/2
  vfov: [20, 105]
  k1_hat: [-0.7, 0.7]
```

### 3.5 两个入口的差异

| 字段 | `geocalib.yaml` | `geocalib-radial.yaml` |
| --- | --- | --- |
| data | `openpano` | `openpano-radial` |
| batch size | 24 | 18（LM 更耗显存） |
| `model.optimizer.camera_model` | `pinhole` | `simple_radial` |
| `model.weights` | —（仅骨干 `mscan_b.pth`） | `weights/geocalib.tar` |
| `train.lr` | `1e-4` | `1e-5`（微调） |
| `train.num_steps` | 150k | 200k |
| 调度 milestones | `[4000] / [80k,130k]` | `[5000] / [110k,170k]` |

---

## 4. 数据配置

### 4.1 `configs/data/openpano.yaml`

```yaml
name: simple_dataset
dataset_dir: data/openpano/openpano     # 注意双层目录
preprocessing:
  resize: 320
  side: short
augmentations:
  name: geocalib
grayscale: false
use_up: true
use_latitude: true
train_batch_size: 64
val_batch_size: 64
test_batch_size: 64
num_workers: 8
prefetch_factor: 2
```

`openpano-radial.yaml` 除 `dataset_dir: data/openpano/openpano_radial` 外完全相同。

### 4.2 数据生成配置（`datasets/configs/`）

```yaml
name: openpano
base_dir: data/openpano
pano_dir: ${.base_dir}/panoramas
images_per_pano: 16
resize_factor: null
n_workers: 1
device: cpu
overwrite: true
parameter_dists:
  roll:  {type: uniform, options: {loc: -0.785, scale: 1.571}}
  pitch: {type: uniform, options: {loc: -0.785, scale: 1.571}}
  vfov:  {type: uniform, options: {loc: 0.349, scale: 1.484}}
  resize_factor: {type: uniform, options: {loc: 1.2, scale: 0.5}}
  shape: {type: fix, value: [640, 640]}
```

`openpano-radial.yaml` 额外定义 `k1_hat` 的 `truncnorm` 分布（见 06 章第 5.2 节）。

---

## 5. 评估配置

评估配置位于 `siclib/eval/configs/`，由 `--conf` 指定（不是 `siclib/configs/eval/`）：

| 文件 | 内容 |
| --- | --- |
| `geocalib-pinhole.yaml` | `model: {name: networks.geocalib_pretrained}` |
| `geocalib-simple_radial.yaml` | `model: {name: networks.geocalib_pretrained, camera_model: simple_radial, model_weights: distorted}` |
| `deepcalib.yaml` | `model: {name: networks.deepcalib, weights: weights/deepcalib.tar}` |
| `uvp.yaml` | `model: {name: optimization.vp_from_prior, ...}`（消失点 + 重力先验基线） |

合并顺序（见 07 章 `parse_eval_args`）：

```
benchmark default_conf  <  --conf 配置  <  CLI dotlist  <  checkpoint 中的配置（若有）
```

`checkpoint` 可以是实验名（读 `outputs/training/<name>/config.yaml`）或 `.tar` 文件。

---

## 6. 路径约定（`siclib/settings.py`）

```python
try:
    from settings import DATA_PATH, EVAL_PATH, TRAINING_PATH
except ModuleNotFoundError:
    root = Path(__file__).parent.parent
    DATA_PATH     = root / "data/"
    TRAINING_PATH = root / "outputs/training/"
    EVAL_PATH     = root / "outputs/results/"
```

- 若仓库根目录存在 `settings.py`（顶层），其路径优先生效，可自定义数据盘位置。
- 默认路径均相对仓库根：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DATA_PATH` | `<root>/data/` | 数据集与预训练权重 |
| `TRAINING_PATH` | `<root>/outputs/training/` | checkpoint 与实验日志 |
| `EVAL_PATH` | `<root>/outputs/results/` | 评估结果 |

需要注意的**相对路径**（均相对当前工作目录，而非 `DATA_PATH`）：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `model.backbone.weights` | `weights/mscan_b.pth` | MSCAN-B 骨干权重 |
| `model.weights` | `weights/geocalib.tar` | 径向微调的初始化权重 |
| `eval/configs/deepcalib.yaml` 的 `weights` | `weights/deepcalib.tar` | DeepCalib 权重 |
| `data.dataset_dir` | `data/openpano/openpano` 等 | 数据集根目录 |

因此推荐始终在仓库根目录执行训练/评估命令，或显式改写为绝对路径。

---

## 7. `geocalib` 推理包的参数

推理包不使用 Hydra，参数直接写在类默认值与函数签名中：

| 组件 | 参数 | 默认值 |
| --- | --- | --- |
| `GeoCalib.__init__` | `weights` | `"pinhole"`（或 `"distorted"`、权重路径） |
| `GeoCalib.calibrate` | `camera_model` | `"pinhole"` |
| | `priors` | `None` |
| | `shared_intrinsics` | `False` |
| | `camera_R_rig` | `None` |
| `ImagePreprocessor` | `resize` / `edge_divisible_by` | `320` / `32` |
| `LMOptimizer.default_conf` | `num_steps` | `30` |
| | `lambda_` | `0.1` |
| | `use_spherical_manifold` | `True` |
| | `use_log_focal` | `True` |
| | `loss_fn` | `"huber_loss"` |
| | `up_loss_fn_scale` / `lat_loss_fn_scale` | `1e-2` |
| 设备选择 | `get_device()` | 自动（`cuda` > `mps` > `cpu`） |

详见 [03-geocalib-inference-package.md](./03-geocalib-inference-package.md)。

---

## 8. 常用配置配方

```bash
# 小显存调试
python -m siclib.train debug --conf geocalib data.train_batch_size=2 train.num_steps=100

# 从已有实验微调
python -m siclib.train finetune --conf geocalib train.load_experiment=geocalib-pinhole-openpano train.lr=1e-5

# 开启 wandb
python -m siclib.train exp --conf geocalib train.writer=wandb

# 只评估、不重新推理
python -m siclib.eval.lamar2k --conf geocalib-pinhole --overwrite_eval

# 用绝对路径权重
python -m siclib.eval.megadepth2k --checkpoint /abs/path/checkpoint_best.tar
```

---

## 9. 常见配置陷阱

| 陷阱 | 说明 |
| --- | --- |
| `--conf` 路径混淆 | 训练配置在 `siclib/configs/`，评估配置在 `siclib/eval/configs/` |
| 相对权重路径 | `weights/*.pth` 相对 cwd；不在根目录运行会 `FileNotFoundError` |
| 双击目录 | `data/openpano/openpano`（第一层是下载目录，第二层是生成目录） |
| `strict_conf` | `BaseModel.strict_conf=False` 时未知字段不报错，容易静默忽略拼写错误 |
| 只读配置 | `BaseModel`/`BaseDataset` 合并后 `set_readonly`，运行期不可修改 |
| `num_steps` 与 `epochs` | `num_steps` 非空时由脚本换算 epochs；两者同时设置以 `num_steps` 为准 |
| 结构模式 | `train.py` 中 `OmegaConf.set_struct(conf, True)`，访问未定义键会抛错 |
