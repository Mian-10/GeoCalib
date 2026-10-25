# 07 · 评估体系（`siclib/eval`）

> 对应目录：`siclib/eval/`
> 相关文档：[06-siclib-datasets.md](./06-siclib-datasets.md)、[08-siclib-training.md](./08-siclib-training.md)、[09-configuration.md](./09-configuration.md)

本章说明 GeoCalib 的 benchmark 评估流程：模型预测如何缓存、指标如何计算、结果如何落盘，以及每个 benchmark 的数据与配置。

---

## 1. 总体流程

每个 benchmark 是 `siclib/eval/` 下的一个可执行模块（如 `lamar2k.py`），既可命令行运行，也可被训练脚本调用。整体分为三步：

```
get_predictions()  →  predictions.h5
        ↓
run_eval()         →  results.h5 / summaries.json / figures/
        ↓
load_eval()        →  summaries（供打印或训练日志使用）
```

设计要点：

- **预测与评估解耦**：先对所有测试样本导出预测到 `predictions.h5`，再离线用 `CacheLoader` 读取，避免重复推理。
- **结果幂等**：已有 `results.h5` + `summaries.json` 时默认跳过；`--overwrite` / `--overwrite_eval` 控制是否重跑。
- **配置一致性校验**：`save_conf` 会比较新旧 `conf.yaml` 的 `data` / `model` / `eval` 段，不一致时要求显式覆盖。

---

## 2. `EvalPipeline`：评估流水线基类

文件：[eval_pipeline.py](file:///workspace/siclib/eval/eval_pipeline.py)

### 2.1 抽象接口

| 方法 | 说明 |
| --- | --- |
| `get_dataloader(data_conf)` | 返回测试 dataloader（类方法） |
| `get_predictions(experiment_dir, model, overwrite)` | 导出预测文件 |
| `run_eval(loader, pred_file)` | 在缓存预测上计算指标，返回 `(summaries, figures, results)` |
| `run(experiment_dir, model, overwrite, overwrite_eval)` | 编排上述步骤并保存结果 |
| `save_conf(experiment_dir, …)` | 保存/校验 `conf.yaml` |

类属性 `export_keys` / `optional_export_keys` 定义需要导出的预测字段。

### 2.2 `run()` 的完整逻辑

```python
def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
    self.save_conf(experiment_dir, ...)                     # conf.yaml
    pred_file = self.get_predictions(experiment_dir, ...)   # predictions.h5
    f = {}
    if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
        s, f, r = self.run_eval(self.get_dataloader(self.conf.data, 1), pred_file)
        save_eval(experiment_dir, s, f, r)                  # results.h5 / summaries.json / figures
    s, r = load_eval(experiment_dir)
    return s, f, r
```

### 2.3 I/O 函数

| 函数 | 行为 |
| --- | --- |
| `save_eval(dir, summaries, figures, results)` | `results` 逐 key 写入 `results.h5`；`summaries` 中非 list 的有限值写 `summaries.json`（非有限值 → `null`），list 原样保留；`figures` 保存为 `{name}.png` |
| `load_eval(dir)` | 读 `results.h5` 中维度 <3 的数组作为 `results`，读 `summaries.json`，`null` 转 `np.nan` |
| `exists_eval(dir)` | 同时存在 `results.h5` 与 `summaries.json` 时为真 |

---

## 3. 命令行与配置解析

文件：[io.py](file:///workspace/siclib/eval/io.py)

### 3.1 参数

```python
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str)              # 实验名（输出目录）
parser.add_argument("--checkpoint", type=str)       # 权重路径或训练实验名
parser.add_argument("--conf", type=str)             # eval 配置名 / 路径
parser.add_argument("--overwrite", action="store_true")       # 重跑预测与评估
parser.add_argument("--overwrite_eval", action="store_true")  # 只重跑评估
parser.add_argument("--plot", action="store_true")
parser.add_argument("dotlist", nargs="*")           # OmegaConf 点号覆盖，如 model.camera_model=pinhole
```

### 3.2 `parse_eval_args(benchmark, args, configs_path, default)`

1. 若给定 `--conf`：
   - `parse_config_path` 先在 `siclib.eval/configs/` 中按名称查找 yaml，否则按文件路径解析；
   - 用 Hydra `compose` 加载；
   - `extract_benchmark_conf` 把 `model` 与 `benchmarks.<benchmark>` 合并抽出。
2. 合并命令行点号覆盖 `OmegaConf.from_cli(args.dotlist)`。
3. `checkpoint` 解析：
   - `xxx.tar` 视为权重文件；
   - 否则视为 `outputs/training/` 下的实验名，读取其 `config.yaml` 继承模型配置。
4. 合并 benchmark 的 `default_conf`。
5. 生成实验名 `name`：优先级 `--tag` > `{conf}_{checkpoint}` > `conf` > `checkpoint`；若带点号覆盖且无 tag，则追加 `_{overrides}`。

### 3.3 `load_model(model_conf, checkpoint, get_last)`

```python
if checkpoint:
    model = load_experiment(checkpoint, conf=model_conf, get_last=get_last).eval()
else:
    model = get_model(model_conf.name)(model_conf).eval()
```

### 3.4 `get_benchmark` / `run_benchmark`

[eval/\_\_init\_\_.py](file:///workspace/siclib/eval/__init__.py)：

```python
def get_benchmark(benchmark):
    return get_class(f"{__name__}.{benchmark}", EvalPipeline)

@torch.no_grad()
def run_benchmark(benchmark, eval_conf, experiment_dir, model=None):
    experiment_dir.mkdir(exist_ok=True, parents=True)
    pipeline = get_benchmark(benchmark)(eval_conf)
    return pipeline.run(experiment_dir, model=model, overwrite=True, overwrite_eval=True)
```

`run_benchmark` 供训练脚本在 `do_evaluation` 中调用（见 08 章），结果写入 `outputs/results/<benchmark>/<experiment>/`。

---

## 4. `SimplePipeline`：通用单图评估实现

文件：[simple_pipeline.py](file:///workspace/siclib/eval/simple_pipeline.py)

绝大多数 benchmark（LaMAR、MegaDepth、TartanAir、Stanford2D3D、OpenPano）只是更换数据目录与 URL，均继承它。

### 4.1 默认配置

```yaml
data: {}          # 由具体 benchmark 覆盖
model: {}
eval:
  thresholds: [1, 5, 10]          # AUC 的误差阈值（度）
  pixel_thresholds: [0.5, 1, 3, 5] # 像素投影误差阈值
  num_vis: 10                     # 可视化样本数，0 关闭
  verbose: true
url: null         # benchmark.zip 下载地址
```

### 4.2 导出字段

```python
export_keys = ["camera", "gravity"]
optional_export_keys = [
    "focal_uncertainty", "vfov_uncertainty",
    "roll_uncertainty", "pitch_uncertainty", "gravity_uncertainty",
    "up_field", "up_confidence", "latitude_field", "latitude_confidence",
]
```

`camera` / `gravity` 必然导出；不确定度与透视场字段仅在模型输出中存在时才缓存。

### 4.3 `get_dataloader`

```python
data_conf["test_batch_size"] = batch_size
do_shuffle = data_conf["test_batch_size"] > 1
dataset = get_dataset(data_conf["name"])(data_conf)
return dataset.get_data_loader("test", shuffle=do_shuffle)
```

### 4.4 `get_predictions`

```python
pred_file = experiment_dir / "predictions.h5"
if not pred_file.exists() or overwrite:
    model = model or load_model(self.conf.model, self.conf.checkpoint)
    export_predictions(dataloader, model, pred_file,
                       keys=self.export_keys,
                       optional_keys=self.optional_export_keys,
                       verbose=self.verbose)
```

`export_predictions`（`siclib/utils/export_predictions.py`）逐 batch 推理并按样本名写入 h5，支持 `as_half` 压缩与自定义 `callback_fn`。

### 4.5 `run_eval`：指标计算

```python
cache_loader = CacheLoader({"path": pred_file, "collate": None}).eval()
for data in loader:
    pred = cache_loader(data)                 # 按 data["name"] 取缓存预测
    results["names"].append(...)

    # 真值
    results["roll_gt"], results["pitch_gt"], results["vfov_gt"], results["focal_gt"], results["k1_gt"]

    # 相机
    pred_camera = CameraModel(pred["camera"])[None].undo_scale_crop(data)[0]
    gt_camera   = gt_cam[None].undo_scale_crop(data)[0]
    compute_camera_metrics(pred_camera, gt_camera, conf.pixel_thresholds)

    # 重力
    compute_gravity_metrics(Gravity(pred["gravity"]), gt_gravity)

    # 不确定度（可选）
    focal_uncertainty / vfov_uncertainty / roll / pitch / gravity

    # 透视场（可选）
    up_error / up_med_error / up_weighted_error
    latitude_error / latitude_med_error / latitude_weighted_error
```

关键细节：

- `undo_scale_crop(data)` 把预测/真值相机还原到**原始图像尺寸**，保证误差在原始像素尺度上可比。
- `CacheLoader` 负责从 h5 按 `name` 取预测，并自动反缩放 field（见 05 章）。
- 每类误差同时记录**均值**与**中位数**；带置信度时计算置信度加权误差。

### 4.6 相机指标 `compute_camera_metrics`

| 指标 | 说明 |
| --- | --- |
| `vfov` / `vfov_error` | 预测 vFoV（度）与误差 |
| `focal` / `focal_error` / `rel_focal_error` | 焦距（像素）、绝对误差、相对误差 |
| `k1` / `k1_error` | 径向畸变系数及误差（仅含 `k1` 的模型） |
| `pixel_projection_error@{th}` | 用**预测内参+预测畸变**重投影网格点，误差小于 `th` 像素的比例 |
| `pixel_distortion_error@{th}` | 固定真值内参、仅替换畸变系数后的同口径比例 |

`calculate_pixel_projection_error(camera_pred, camera_gt, N=500, distortion_only=True)`：

1. 在 GT 图像上构造 `N×N` 像素网格；
2. 用 GT 的 pinhole 模型 `image2world` 得到归一化坐标（去掉畸变影响）；
3. 分别用 GT 与预测相机 `world2image` 投影；
4. 取两者都有效的点，计算像素距离。

`distortion_only=True` 时把预测的 `k1,k2` 拼到 GT 参数上，只衡量畸变估计质量。

### 4.7 重力指标 `compute_gravity_metrics`

| 指标 | 说明 |
| --- | --- |
| `roll` / `pitch` | 预测角（度） |
| `roll_error` / `pitch_error` | 角度误差（度） |
| `gravity_error` | 3D 重力向量夹角误差（度） |

对应 `siclib/models/utils/metrics.py` 中的 `roll_error / pitch_error / gravity_error / vfov_error / up_error / latitude_error`。

### 4.8 汇总与 AUC

```python
for k, v in results.items():
    if k.endswith("_error") or "recall" in k or "pixel" in k:
        summaries[f"mean_{k}"]  = round(nanmean(v), 3)
        summaries[f"median_{k}"] = round(nanmedian(v), 3)
        if any(kw in k for kw in ["roll", "pitch", "vfov", "gravity"]):
            auc = AUCMetric(elements=v, thresholds=conf.thresholds, min_error=1).compute()
            summaries[f"auc_{k}@{t}"] = round(auc[i], 3)
```

`AUCMetric` 见 `siclib/utils/tools.py`：对给定阈值序列计算累积误差曲线面积，`min_error=1` 表示误差先减去 1° 再归一化（论文口径）。

### 4.9 可视化 `get_figures`

`num_vis > 0` 时：

- `rpf_gt_error`：roll/pitch/vfov 真值 vs 误差散点；
- `rpf_gt_pred`：真值 vs 预测（带对角线）；
- `gl_gt_error`：真值 vs 透视场误差；
- 每个可视化样本单独保存 `figures/up-{i}-{err}.jpg`、`figures/latitude-{i}-{err}.jpg`。

---

## 5. Benchmark 一览

所有 benchmark 都位于 `siclib/eval/`，通过 `python -m siclib.eval.<name> --conf <config>` 运行。

| 模块 | 类 | 数据目录 | 下载 | 相机模型 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `lamar2k.py` | `Lamar2k` | `data/lamar2k` | 自动（cvg-data zip） | pinhole | LaMAR 室内/室外，真实相机 |
| `megadepth2k.py` | `Megadepth2k` | `data/megadepth2k` | 自动 | pinhole | MegaDepth 互联网图像 |
| `megadepth2k_radial.py` | `Megadepth2kRadial` | `data/megadepth2k-radial` | 自动 | **simple_radial** | 带径向畸变的 MegaDepth |
| `tartanair.py` | `TartanAir` | `data/tartanair` | 自动 | pinhole | 合成环境 |
| `stanford2d3d.py` | `Stanford2d3d` | `data/stanford2d3d` | 自动（需确认使用条款） | pinhole | 室内全景渲染 |
| `openpano.py` | `OpenPano` | `data/poly+maps+laval/poly+maps+laval` | 无（本地生成） | pinhole | 训练分布内测试集 |
| `openpano_radial.py` | `OpenPanoRadial` | `data/poly+maps+laval/pano_dataset_distorted` | 无 | simple_radial | 畸变测试集 |

通用数据配置：

```yaml
data:
  name: simple_dataset
  dataset_dir: data/<benchmark>
  test_img_dir: ${.dataset_dir}/images
  test_csv: ${.dataset_dir}/images.csv
  augmentations: {name: identity}
  preprocessing: {resize: 320, edge_divisible_by: 32}
  test_batch_size: 1
eval:
  thresholds: [1, 5, 10]
  pixel_thresholds: [0.5, 1, 3, 5]
  num_vis: 10
```

`download_and_extract_benchmark(name, url, output)`（`eval/utils.py`）：

- 目录已存在则跳过；
- `stanford2d3d` 需交互确认数据使用条款；
- 下载 zip → 解压 → 删除 zip。

### 5.1 评估配置（`siclib/eval/configs/`）

| 配置 | 内容 | 适用 |
| --- | --- | --- |
| `geocalib-pinhole.yaml` | `model: {name: networks.geocalib_pretrained}` | 预训练针孔权重，全部针孔 benchmark |
| `geocalib-simple_radial.yaml` | `model: {name: networks.geocalib_pretrained, camera_model: simple_radial, model_weights: distorted}` | 畸变 benchmark |
| `deepcalib.yaml` | `model: {name: networks.deepcalib, weights: weights/deepcalib.tar}` | DeepCalib 对比 |
| `uvp.yaml` | `model: {name: optimization.vp_from_prior, ...}` | 基于消失点 + 先验重力的基线 |

> 注意：`--conf` 的搜索路径是 `siclib/eval/configs/`，而不是 `siclib/configs/eval/`。

---

## 6. 常用命令

```bash
# 使用预训练权重评估 LaMAR（自动下载数据）
python -m siclib.eval.lamar2k --conf geocalib-pinhole

# 评估训练得到的 checkpoint
python -m siclib.eval.megadepth2k --checkpoint geocalib-pinhole-openpano

# 指定权重文件 + 覆盖模型配置
python -m siclib.eval.megadepth2k_radial --conf geocalib-simple_radial \
    --checkpoint weights/geocalib-distorted.tar

# 只重跑评估（复用 predictions.h5）
python -m siclib.eval.tartanair --conf geocalib-pinhole --overwrite_eval

# 命令行覆盖
python -m siclib.eval.stanford2d3d --conf geocalib-pinhole eval.num_vis=0
```

输出结构（`siclib/settings.py` 的 `EVAL_PATH = <root>/outputs/results`）：

```
outputs/results/<benchmark>/<name>/
├── conf.yaml
├── predictions.h5
├── results.h5
├── summaries.json
└── figures/
    ├── rpf_gt_error.png  rpf_gt_pred.png  gl_gt_error.png
    └── up-*.jpg  latitude-*.jpg
```

`<name>` 由 `--tag` / `--conf` / `--checkpoint` 组合决定。

---

## 7. 新增 benchmark

1. 在 `siclib/eval/` 新建 `<name>.py`；
2. 继承 `SimplePipeline`，覆盖 `default_conf.data`（`dataset_dir`、`test_csv` 等）与 `url`；
3. 若相机含畸变，在 `_init` 中设置 `self.CameraModel = SimpleRadial`；
4. 添加 `__main__` 块（复制现有 benchmark 的模板：`get_eval_parser` → `parse_eval_args` → `pipeline.run`）；
5. 可选：在 `siclib/eval/configs/` 增加模型配置；
6. 在训练配置的 `benchmarks` 列表中登记，训练期间即可自动评估。

---

## 8. 文件索引

| 文件 | 职责 |
| --- | --- |
| `eval/__init__.py` | `get_benchmark`、`run_benchmark` |
| `eval/eval_pipeline.py` | `EvalPipeline`、`load/save/exists_eval` |
| `eval/simple_pipeline.py` | `SimplePipeline`、相机/重力指标、像素投影误差 |
| `eval/io.py` | 命令行解析、配置合并、`load_model` |
| `eval/utils.py` | benchmark 下载、`plot_scatter_grid`、`check_keys_recursive` |
| `eval/lamar2k.py` / `megadepth2k.py` / `megadepth2k_radial.py` | 真实/互联网/畸变 benchmark |
| `eval/tartanair.py` / `stanford2d3d.py` | 合成/室内 benchmark |
| `eval/openpano.py` / `openpano_radial.py` | OpenPano 测试集 |
| `eval/configs/*.yaml` | 模型配置（geocalib / deepcalib / uvp） |
| `models/cache_loader.py` | 预测缓存读取 |
| `utils/export_predictions.py` | 预测导出 |
| `utils/tools.py` | `AUCMetric`、`RecallMetric`、`compute_auc` 等 |
