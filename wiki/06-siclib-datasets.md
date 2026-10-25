# 06 · 数据集体系（`siclib/datasets`）

> 对应目录：`siclib/datasets/`
> 相关文档：[09-configuration.md](./09-configuration.md)、[08-siclib-training.md](./08-siclib-training.md)

本章说明训练/评估所用的数据层：数据集如何注册、样本如何构造、透视场监督信号如何生成、图像增强如何组织，以及 OpenPano 数据集的下载与从全景图生成透视图像的完整流程。

---

## 1. 数据集注册机制

`siclib/datasets/__init__.py` 提供 `get_dataset(name)`，通过动态导入定位数据集类：

```python
def get_dataset(name):
    import_paths = [name, f"{__name__}.{name}"]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseDataset)          # 查找 BaseDataset 子类
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_dataset__              # 或模块级变量
                except AttributeError:
                    continue
    raise RuntimeError(...)
```

要点：

- `name` 可以是**完整模块路径**（如 `siclib.datasets.simple_dataset`），也可以是相对 `siclib.datasets.` 的短名（如 `simple_dataset`）。
- 优先寻找模块中**唯一的 `BaseDataset` 子类**；若有多个子类，则回退到模块级 `__main_dataset__`。
- Hydra 配置中的 `data.name` 直接传给它，例如 `siclib/configs/data/openpano.yaml` 中的 `name: simple_dataset`。

`siclib/models/__init__.py`、`siclib/eval/__init__.py` 使用完全相同的动态导入模式（见 05、07 章）。

---

## 2. `BaseDataset`：数据集的公共骨架

文件：[base_dataset.py](file:///workspace/siclib/datasets/base_dataset.py)

### 2.1 类契约

子类必须实现：

| 方法 | 说明 |
| --- | --- |
| `default_conf`（类属性） | 数据集默认配置，与 `base_default_conf` 合并 |
| `_init(self, conf)` | 初始化（`SimpleDataset` 中为空实现） |
| `get_dataset(self, split)` | 返回某个 split 的 `torch.utils.data.Dataset` |

构造时的配置优先级为：

```
base_default_conf  <  default_conf(子类)  <  用户传入 conf
```

合并后 `OmegaConf.set_readonly(self.conf, True)`，运行期不可修改。

`base_default_conf` 字段：

```yaml
name: "???"
num_workers: "???"
train_batch_size: "???"   # 三个 split 各自 batch size
val_batch_size: "???"
test_batch_size: "???"
shuffle_training: true
batch_size: 1             # 显式覆盖 / overfit 时使用
num_threads: 1
seed: 0
prefetch_factor: 2
```

### 2.2 `get_data_loader(split, shuffle=None, pinned=False, distributed=False)`

- 断言 `split in ["train", "val", "test"]`。
- 优先读取 `conf[f"{split}_batch_size"]`；缺失（`MissingMandatoryValue`）时回退到 `conf.batch_size`。
- `distributed=True`：使用 `DistributedSampler` 并强制 `shuffle=False`（供 DDP 训练）。
- 否则 `shuffle` 默认取 `split == "train" and conf.shuffle_training`。
- 统一使用自定义 `collate` 与 `worker_init_fn`。

### 2.3 `get_overfit_loader(split)`

调试用：基于 `train` 数据集与 `LoopSampler(batch_size, …)` 反复取同一小批数据。
`split=="train"` 时循环长度等于数据集大小，否则只取一个 batch。用于验证「loss 能否下降、指标是否与 loss 一致」。

### 2.4 辅助函数

| 名称 | 作用 |
| --- | --- |
| `LoopSampler(loop_size, total_size)` | 无限循环采样器，索引按 `i % loop_size` 重复 |
| `worker_init_fn(i)` | 为每个 worker 设置不同随机种子与线程数（读 `dataset.conf`） |
| `collate(batch)` | 扩展版 `default_collate`：支持 `TensorWrapper` / 自定义对象 `torch.stack`、`None` 透传、字符串保持 list、numpy 标量与数组、worker 内共享内存优化 |

`collate` 的关键差异在于最后的分支：

```python
else:
    # 尝试直接 stack，兼容 Gravity / BaseCamera 等 TensorWrapper
    return torch.stack(batch, 0)
```

因此 `camera`、`gravity` 这类包装张量可以被正常批处理。

---

## 3. `SimpleDataset`：从 CSV 读取透视图像样本

文件：[simple_dataset.py](file:///workspace/siclib/datasets/simple_dataset.py)

### 3.1 默认配置

```yaml
dataset_dir: "???"
train_img_dir: ${.dataset_dir}/train
val_img_dir:   ${.dataset_dir}/val
test_img_dir:  ${.dataset_dir}/test
train_csv:     ${.dataset_dir}/train.csv
val_csv:       ${.dataset_dir}/val.csv
test_csv:      ${.dataset_dir}/test.csv
use_up: true            # 是否监督上方向场
use_latitude: true      # 是否监督纬度场
use_prior_focal: false  # 是否把真值焦距作为先验输入
use_prior_gravity: false
use_prior_k1: false
grayscale: false
preprocessing: ImagePreprocessor.default_conf
augmentations: {name: geocalib, verbose: false}
p_rotate: 0.0
reseed: false
seed: 0
num_workers: 8
prefetch_factor: 2
train_batch_size: 32
val_batch_size: 32
test_batch_size: 32
```

`get_dataset(split)` 直接返回 `_SimpleDataset(self.conf, split)`。

### 3.2 CSV 格式与 `load_csv`

CSV 每行对应一张透视图像，由 `create_dataset_from_pano.py` 生成：

| 列 | 含义 | 必需 |
| --- | --- | --- |
| `fname` | 图像文件名（相对 `*_img_dir`） | 是 |
| `width` / `height` | 图像尺寸（像素） | 是 |
| `vfov` | 垂直视场角（**弧度**） | 是 |
| `roll` / `pitch` | 重力方向（**弧度**） | 是 |
| `px` / `py` | 主点，缺省为图像中心 | 否 |
| `k1` / `k2` | 径向畸变系数，缺省 0 | 否 |

`load_csv` 返回 `(infos, params, gravity)`：

- `params`：`[w, h, f, f, px, py, k1, k2]`，其中 `f = fov2focal(vfov, h)`；
- `gravity`：`[roll, pitch]`；
- `infos`：`{"name": fname, "file_name": 绝对路径}`。

### 3.3 `_SimpleDataset.__getitem__` 流程

```
__getitem__(idx)
  └─ (可选) fork_rng(seed+idx)        # conf.reseed=True 时保证可复现
      └─ getitem(idx)
          ├─ _read_image(infos, params, gravity)
          │    ├─ load_image(path, grayscale, return_tensor=False)  # HWC uint8
          │    ├─ augmentation(image, return_tensor=True)          # HWC float tensor
          │    ├─ SimpleRadial(params[None])                        # k1=0 时等价 pinhole
          │    ├─ Gravity.from_rp(roll, pitch)
          │    ├─ preprocessor(image)                               # 缩放 + 可选裁剪
          │    ├─ camera.scale(scales); camera.crop(crop_pad)
          │    └─ 组装可选先验 prior_gravity / prior_focal / prior_k1
          └─ (可选) _get_perspective(data)
               └─ get_perspective_field(camera, gravity,
                                        use_up, use_latitude)
```

注意：数据集内部**统一使用 `SimpleRadial`** 表示相机，当 `k1=0` 时与 `Pinhole` 等价。这样训练数据管线同时兼容针孔与畸变数据。

### 3.4 样本字典的键

| 键 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `name` | `str` | 样本名（文件名） |
| `path` | `str` | 原图路径 |
| `camera` | `BaseCamera`（8 维） | `w,h,fx,fy,cx,cy,k1,k2` |
| `gravity` | `Gravity`（3 维） | 相机坐标系下的重力单位向量 |
| `up_field` | `3×H×W` | 上方向场（x, y, valid 或 2 通道） |
| `latitude_field` | `1×H×W` | 纬度角（弧度） |
| `prior_gravity` / `prior_focal` / `prior_k1` | 同上 | 仅在对应 `use_prior_*` 为真时出现 |
| `image` | `3×H×W` float | 预处理后的图像 |
| `scales` | `[sx, sy]` | 缩放比例（网络输出需用它还原） |
| `image_size` | `np.array([w,h])` | 预处理后尺寸 |
| `original_image_size` | `np.array([w,h])` | 原图尺寸 |
| `transform` | `3×3` | 缩放对应的变换矩阵 |
| `crop_pad` | `[dx, dy]` | 当 `edge_divisible_by` 非空时的裁剪量 |

### 3.5 调试入口

`python -m siclib.datasets.simple_dataset --name simple_dataset --data_dir data/openpano --split train` 会取一个 batch 并调用 `make_perspective_figures` 可视化透视场（见 [visualization](./00-README.md)）。

---

## 4. 图像增强（`augmentations.py`）

文件：[augmentations.py](file:///workspace/siclib/datasets/augmentations.py)

### 4.1 `BaseAugmentation`

- 配置合并方式与 `BaseDataset` 一致；`base_default_conf`：`name / shuffle / p / verbose / dtype`。
- `dtype="uint8"`：先用 `A.FromFloat` 转 uint8，处理后用 `A.ToFloat` 回到 float；`dtype="float32"`：不做量化往返。
- `__call__(image, return_tensor=False)`：
  1. Tensor 自动转 numpy；
  2. 若 dtype 不符则 `preprocess`；
  3. 可选 `shuffle` 打乱 transform 顺序；
  4. `A.Compose(transforms, p=self.conf.p)`（`verbose=True` 时用 `A.ReplayCompose` 并打印 replay 串）；
  5. `postprocess`，可选 `ToTensorV2`。
- 数据集仅在 `split == "train"` 时使用配置的增强，其余 split 一律 `IdentityAugmentation`。

### 4.2 可用增强类

`augmentations` 注册表：

| 键 | 类 | 特点 |
| --- | --- | --- |
| `default` | `DefaultAugmentation` | 亮度/对比度、HSV、灰度、JPEG 压缩、轻微模糊 |
| `dark` | `DarkAugmentation` | 偏暗场景：负向亮度/对比度、雨、模糊、噪声、伽马、均衡化 |
| `perspective` | `PerspectiveAugmentation` | 与 `default` 基本一致（透视场任务用） |
| `deepcalib` | `DeepCalibAugmentations` | 强噪声/降采样/JPEG/颜色抖动，模拟 DeepCalib 训练分布 |
| `geocalib` | `GeoCalibAugmentations` | **GeoCalib 默认**：颜色 + 噪声（高斯/JPEG/ISO/模糊锐化）+ 多插值降采样 |
| `identity` | `IdentityAugmentation` | 仅 `IdentityTransform` |

`GeoCalibAugmentations` 分三组：

```python
self.color_transforms = [RandomGamma, RandomToneCurve, RandomBrightnessContrast,
                         ColorJitter, OneOf(ToGray, ToSepia, Identity)]
self.noise_transforms = [GaussNoise, ImageCompression, ISONoise,
                         OneOrOther(AdvancedBlur+Sharpen, Sharpen+AdvancedBlur)]
self.image_transforms = [OneOf(Downscale(多种插值组合))]
self.transforms = color + noise + image
```

此外定义了自定义 `RandomAdditiveShade`（椭圆遮罩 + 高斯模糊模拟光照不均），可在配置中显式引用。

---

## 5. OpenPano 数据集构建

### 5.1 下载与准备全景图

文件：[download_openpano.py](file:///workspace/siclib/datasets/utils/download_openpano.py)

```bash
python -m siclib.datasets.utils.download_openpano \
    --name openpano --laval_dir data/laval-tonemap
```

流程：

1. `PANO_URL`（polybox）下载 `openpano.zip` 到 `data/`，解压后删除 zip；
2. 读取 `panoramas/{split}_panos.txt`，把 Laval 目录中缺失的全景图复制到 `panoramas/{split}/`；
3. 统计 `train/test/val` 的 jpg 数量并打印。

`download_and_extract_dataset(name, url, output)` 做了幂等处理：目录已存在则跳过下载。

### 5.2 从全景图生成透视图像

文件：[create_dataset_from_pano.py](file:///workspace/siclib/datasets/create_dataset_from_pano.py)

入口（Hydra，默认 `config_name="SUN360"`，OpenPano 用 `openpano`）：

```bash
python -m siclib.datasets.create_dataset_from_pano --config-name openpano
```

#### `DatasetGenerator.default_conf`

```yaml
name: "???"
base_dir: "???"
pano_dir:       ${base_dir}/panoramas
pano_train/val/test: ${pano_dir}/{split}
perspective_dir:     ${base_dir}/${name}
perspective_train/val/test: ${perspective_dir}/{split}
train_csv/val_csv/test_csv: ${perspective_dir}/{split}.csv
camera_model: pinhole           # camera_models 中的任意模型
parameter_dists:
  roll:  {type: uniform, options: {loc: -45°, scale: 90°}}
  pitch: {type: uniform, options: {loc: -45°, scale: 90°}}
  vfov:  {type: uniform, options: {loc: 20°,  scale: 85°}}
  resize_factor: {type: uniform, options: {loc: 1.0, scale: 1.0}}
  shape: {type: fix, value: [640, 640]}
images_per_pano: 16
n_workers: 10
device: cpu
overwrite: false
```

支持两类 `type`：
- `fix`：直接返回常量；
- 其他值视作 `scipy.stats` 分布名（如 `uniform`、`truncnorm`），用 `sampler.rvs(**options)` 采样。

#### 关键方法

| 方法 | 说明 |
| --- | --- |
| `sample_value(param, seed)` | 按分布采样；`seed` 可为字符串（SHA-256 哈希后作 RNG seed），保证同一全景图可复现 |
| `generate_images_from_pano(pano_path, out_dir)` | 对单张全景图沿 0~2π 均匀取 `images_per_pano` 个 yaw，采样 roll/pitch/vfov/resize_factor/shape，构造相机与重力，调用 `camera.get_img_from_pano(...)` 渲染透视图像；过滤黑像素占比 ≥1% 的失败样本；写 jpg 并返回 CSV 行 |
| `generate_split(split, processor)` | 遍历 `pano_{split}` 下所有文件，`ProcessPoolExecutor` 并行渲染，汇总 `infos` 并写 CSV |
| `generate_dataset()` | 保存 `config.yaml`、依次生成 train/val/test、打印数量、绘制参数分布图（`distributions.pdf`、`distributions_scatter.pdf`） |
| `plot_distributions()` | 直方图 + 两两散点，检查 roll/pitch/vfov 覆盖是否均匀 |

#### `k1_hat` 特殊处理（畸变数据）

当 `parameter_dists` 包含 `k1_hat`（见 `openpano-radial.yaml`）时：

```python
focal     = fov2focal(vfov, height)
rel_focal = focal / height
k1        = k1_hat * rel_focal          # 归一化畸变系数 → 像素尺度

# 保证畸变模型在图像角落仍单调可逆（Brown 模型最大半径）
min_permissible_rmax = sqrt((h/2)^2 + (w/2)^2)
r_max = brown_max_radius(k1, 0)
lowest_possible_f_px = min_permissible_rmax / (r_max * (1 + k1 * r_max²))
f = max(focal, lowest_possible_f_px)    # 不足时抬高焦距
vfov = focal2fov(f, height)
```

`max_radius` / `brown_max_radius` 通过解导数判别式得到 Brown 畸变的可逆半径。这段逻辑保证生成的畸变图像不会出现像素折叠。

#### `ParallelProcessor`

封装 `ProcessPoolExecutor` + `tqdm` 进度条：`submit_tasks(task_func, task_args, pbar_key)` 提交并注册回调，`wait_for_completion(futures)` 收集结果，`shutdown()` 关闭进程池。

### 5.3 配置文件

| 文件 | 用途 |
| --- | --- |
| `siclib/datasets/configs/openpano.yaml` | 针孔数据：roll/pitch U[-45°,45°]，vfov U[20°,105°]，640×640，每全景 16 张 |
| `siclib/datasets/configs/openpano-radial.yaml` | 径向畸变数据，额外采样 `k1_hat`（truncnorm） |
| 其他（`SUN360.yaml` 等） | Hydra 默认配置，结构相同 |

生成后的目录结构：

```
data/openpano/
├── panoramas/{train,val,test}/*.jpg
└── openpano/
    ├── train/*.jpg  val/*.jpg  test/*.jpg
    ├── train.csv    val.csv    test.csv
    ├── config.yaml
    └── distributions*.pdf
```

---

## 6. 图像预处理（`siclib/utils/image.py`）

`ImagePreprocessor` 是数据层与评估层共用的预处理组件（`geocalib/extractor.py` 中有对应实现）：

```yaml
resize: 320            # 目标边长；null 表示不缩放
edge_divisible_by: null  # 裁剪到该值的整数倍
side: short            # 按短边 / 长边缩放
interpolation: bilinear
antialias: true
square_crop: false
resize_backend: kornia # kornia / torchvision / PIL
```

输出 `{"image", "scales", "image_size", "original_image_size", "transform"}`，若 `edge_divisible_by` 非空再追加 `crop_pad`。相机参数通过 `camera.scale(scales)` 与可选 `camera.crop(crop_pad)` 同步变换。

GeoCalib 推理时实际配置为 `resize=320, edge_divisible_by=32`（见 [03-geocalib-inference-package.md](./03-geocalib-inference-package.md)）。

---

## 7. 扩展新数据集

1. 新建 `siclib/datasets/my_dataset.py`，定义 `class MyDataset(BaseDataset)`；
2. 声明 `default_conf`，实现 `_init` 与 `get_dataset(split)`；
3. 返回的样本至少要包含 `image`、`camera`、`gravity`（若训练透视场还需 `up_field` / `latitude_field`），字段命名与本章第 3.4 节保持一致；
4. 在 Hydra 配置中设置 `data.name: siclib.datasets.my_dataset`（或短名 `my_dataset`）；
5. 若需要新增强，在 `augmentations` 字典中注册。

---

## 8. 文件索引

| 文件 | 职责 |
| --- | --- |
| `datasets/__init__.py` | `get_dataset` 动态导入 |
| `datasets/base_dataset.py` | `BaseDataset`、`LoopSampler`、`collate`、`worker_init_fn` |
| `datasets/simple_dataset.py` | `SimpleDataset`、`_SimpleDataset`、`load_csv` |
| `datasets/augmentations.py` | 增强基类与 6 种增强 |
| `datasets/create_dataset_from_pano.py` | OpenPano 生成器 |
| `datasets/utils/download_openpano.py` | 数据下载 |
| `datasets/utils/tonemapping.py` | HDR→LDR 色调映射（生成/处理全景图用） |
| `datasets/utils/align_megadepth.py` | MegaDepth 对齐数据准备 |
| `datasets/configs/*.yaml` | 数据生成/加载配置 |
| `utils/image.py` | `ImagePreprocessor`、`load_image`、`write_image` |
