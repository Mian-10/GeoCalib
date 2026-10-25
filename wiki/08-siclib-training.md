# 08 · 训练流程（`siclib.train`）

> 对应文件：`siclib/train.py`、`siclib/configs/`、`siclib/utils/experiments.py`、`siclib/utils/summary_writer.py`
> 相关文档：[09-configuration.md](./09-configuration.md)、[07-siclib-evaluation.md](./07-siclib-evaluation.md)

`siclib.train` 是一个与具体模型/数据集无关的通用训练脚本，通过 Hydra 组合 `data / train / model` 三组配置。GeoCalib 的训练命令为：

```bash
python -m siclib.train geocalib-pinhole-openpano --conf geocalib --distributed
```

第一个位置参数是**实验名**，决定输出目录 `outputs/training/<experiment>/`。

---

## 1. 命令行参数

| 参数 | 说明 |
| --- | --- |
| `experiment`（位置参数） | 实验名 / 输出目录名 |
| `--conf` | Hydra 配置名（`siclib/configs/<conf>.yaml`） |
| `--mixed_precision {float16,bfloat16}` / `--mp` | AMP 精度 |
| `--compile [default\|reduce-overhead\|max-autotune]` | `torch.compile` 模式 |
| `--overfit` | 用同一 batch 训练+验证，调试用 |
| `--restore` | 从该实验最近 checkpoint 恢复（含优化器状态） |
| `--distributed` | 多 GPU DDP |
| `--profile` | `torch.profiler` |
| `--print_arch` / `--pa` | 打印模型结构 |
| `--detect_anomaly` / `--da` | 开启 autograd anomaly detection |
| `--log_it` | tensorboard x 轴用迭代数而非样本数 |
| `--no_eval_0` | 跳过第 0 次迭代的验证 |
| `--no_test_0` | 跳过第 0 个 epoch 的 benchmark 测试 |
| `dotlist` | OmegaConf 点号覆盖，如 `train.lr=1e-5 data.num_workers=4` |

---

## 2. 配置组合与启动流程

### 2.1 主流程（`__main__`）

```python
output_dir = TRAINING_PATH / args.experiment          # outputs/training/<experiment>
conf = OmegaConf.from_cli(args.dotlist)
if args.conf:
    initialize(config_path="configs"); conf = compose(args.conf, overrides=args.dotlist)
elif args.restore:
    conf = merge(load(output_dir/"config.yaml"), conf)
if not args.restore:
    conf.train.seed = conf.train.seed or 随机种子
    OmegaConf.save(conf, output_dir/"config.yaml")
for module in conf.train.submodules + [siclib]:
    copytree(模块目录, output_dir/module)              # 代码快照，便于复现
if args.distributed:
    args.n_gpus = cuda.device_count()
    spawn(main_worker, nprocs=args.n_gpus, ...)
else:
    main_worker(0, conf, output_dir, args)
```

要点：

- **代码快照**：训练开始会把 `siclib`（以及 `train.submodules` 指定的包）复制到输出目录，保证实验可复现。
- rank 0 的 stdout/stderr 通过 `capture_outputs` 写入 `output_dir/log.txt`。

### 2.2 DDP

- `torch.multiprocessing.spawn` 启动 `n_gpus` 个进程；
- 每个进程 `init_process_group(backend="nccl", init_method="file://<output>/distributed_lock")`；
- `batch_size / train_batch_size / num_workers` 按 GPU 数均分（`num_workers` 向上取整）；
- `SyncBatchNorm.convert_sync_batchnorm` + `DistributedDataParallel`；
- 训练时 `train_loader.sampler.set_epoch(epoch)`。

---

## 3. `training(rank, conf, output_dir, args)` 详解

### 3.1 初始化

| 步骤 | 行为 |
| --- | --- |
| 恢复/微调 | `--restore` 读最近（或最佳）checkpoint，合并旧 conf，`epoch = ckpt.epoch+1`；`train.load_experiment` 只加载权重并合并旧模型配置 |
| 合并默认训练配置 | `conf.train = merge(default_train_conf, conf.train)`，随后 `OmegaConf.set_struct(conf, True)` |
| 随机性 | `set_seed(conf.train.seed)` |
| 日志 | rank 0 创建 `SummaryWriter`（tensorboard / wandb / null） |
| 数据 | `dataset = get_dataset(data_conf.name)(data_conf)`；若存在 `conf.data_val` 则单独构造验证集 |
| loader | 训练集 `get_data_loader("train", distributed=...)`，验证集 `get_data_loader("val")`；`--overfit` 时两者都用 `get_overfit_loader` |
| 模型 | `get_model(conf.model.name)(conf.model).to(device)`；可选 `torch.compile`；`loss_fn = model.loss` |
| 优化器 | 按 `train.optimizer` 取 `SGD/Adam/AdamW/RMSprop`；`opt_regexp` 过滤参数；`lr_scaling` 分组缩放学习率 |
| AMP | `GradScaler(enabled=mp is not None)`，`bfloat16/float16` |
| 调度器 | `get_lr_scheduler(optimizer, conf.train.lr_schedule)` |

### 3.2 `default_train_conf`

```yaml
seed: "???"
epochs: 1
num_steps: null              # 非空时覆盖 epochs
optimizer: adam
opt_regexp: null
optimizer_options: {}
lr: 0.001
lr_schedule: {type: null, start: 0, exp_div_10: 0, on_epoch: false, factor: 1.0}
lr_scaling: [(100, ["dampingnet.const"])]
eval_every_iter: 1000        # 验证集评估间隔
save_every_iter: 5000        # checkpoint 保存间隔
log_every_iter: 200          # 控制台/tensorboard 记录间隔
log_grad_every_iter: null
writer: tensorboard          # tensorboard / wandb / null
test_every_epoch: 1          # benchmark 测试间隔
keep_last_checkpoints: 10
load_experiment: null
median_metrics: []
recall_metrics: {}
pr_metrics: {}
best_key: loss/total         # 选择最佳 checkpoint 的指标
dataset_callback_fn: null
dataset_callback_on_val: false
clip_grad: null
pr_curves: {}
plot: null
submodules: []
```

### 3.3 学习率调度

`get_lr_scheduler` 支持三种形态：

1. **兼容模式**：`type in ["factor", "exp", None]` → `MultiplicativeLR`；
2. **标准 PyTorch 调度器**：`type` 为任意 `torch.optim.lr_scheduler` 类名，`options` 直接透传；
3. **组合调度**：`options.schedulers` 存在时递归构建，再用 `SequentialLR` 串联（GeoCalib 用此模式）。

GeoCalib 的调度（`configs/train/geocalib.yaml`）：

```yaml
lr_schedule:
  type: SequentialLR
  on_epoch: false
  options:
    milestones: [4000]
    schedulers:
      - {type: LinearLR,    options: {start_factor: 1e-3, total_iters: 4000}}
      - {type: MultiStepLR, options: {gamma: 0.1, milestones: [80000, 130000]}}
```

即：前 4k 步 warmup，80k 与 130k 步各衰减 10 倍。

若 `on_epoch: false`，每个训练迭代调用 `lr_scheduler.step()`（DDP 下调用 `n_gpus` 次）；否则每 epoch 调用一次。

### 3.4 单次迭代

```python
model.train(); optimizer.zero_grad()
with autocast(enabled=mp is not None, dtype=mp_dtype):
    data   = batch_to_device(data, device)
    pred   = model(data)
    losses, metrics = loss_fn(pred, data)
    loss   = torch.mean(losses["total"])

if loss_has_nan(loss, distributed):      # 任一 rank NaN → 全组跳过
    continue

if do_backward:                          # DDP 下 all_reduce(PRODUCT) 一致性判断
    scaler.scale(loss).backward()
    if clip_grad:
        scaler.unscale_(optimizer)
        clip_grad_norm_(params, max_norm=clip_grad, error_if_nonfinite=True)
    scaler.step(optimizer); scaler.update()
    if not on_epoch: lr_scheduler.step()
```

- `losses["total"]` 是所有 loss 项的加权和，由模型的 `loss()` 返回。
- 梯度裁剪失败（非有限梯度）时打印警告并跳过该迭代。

### 3.5 记录与保存

| 触发条件 | 行为 |
| --- | --- |
| `it % log_every_iter == 0` | 打印 `loss {...}`，写 `training/<metric>`、`training/lr`、`training/epoch` |
| `it % eval_every_iter == 0` 或 epoch 末或 `stop` | `do_evaluation` → 写 `val/<metric>`；若 `results[best_key]` 改进则保存 `checkpoint_best.tar` |
| `tot_it % save_every_iter == 0` | `save_experiment` 保存 `checkpoint_{epoch}_{it}.tar` |
| 每个 epoch 结束 | 再保存一次；rank 0 关闭 writer |

`tot_it` 在 DDP 下乘以 GPU 数；除非 `--log_it`，tensorboard 的 step 再乘以 batch size（即按样本数计）。

### 3.6 `do_evaluation`

对验证集逐 batch 前向（`model.eval()`、`no_grad`），用同一个 `loss_fn` 得到 `losses/metrics`：

- `metrics` 与 `loss/xxx` 全部进入 `AverageMetric`；
- `conf.median_metrics` 中的键额外记录中位数；
- `conf.recall_metrics` 中的键按阈值记录 recall；
- `conf.pr_curves` 记录 PR 曲线；
- `conf.plot = [n, "module.fn"]` 时随机抽 n 个 batch 生成可视化图。

返回 `(results, pr_metrics, figures)`；用 `fork_rng(seed)` 保证评估可复现。

### 3.7 Benchmark 测试

每个 epoch 开始时（rank 0），若 `epoch % test_every_epoch == 0`：

```python
for bname, eval_conf in conf.get("benchmarks", {}).items():
    s, f, r = run_benchmark(bname, eval_conf,
                            EVAL_PATH / bname / args.experiment / str(epoch),
                            model.eval())
    writer.add_scalar(f"test/{bname}/{metric}", value, step=tot_n_samples)
```

`benchmarks` 段来自顶层配置（GeoCalib 默认配置中为空，可按需添加，如 `benchmarks.lamar2k.model.name=...`）。

---

## 4. Checkpoint 管理（`utils/experiments.py`）

### 4.1 保存内容

```python
checkpoint = {
    "model": state_dict,
    "optimizer": optimizer.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
    "conf": OmegaConf.to_container(conf, resolve=True),
    "epoch": epoch,
    "losses": losses,
    "eval": results,
}
```

### 4.2 函数

| 函数 | 说明 |
| --- | --- |
| `list_checkpoints(dir)` | 解析 `checkpoint_*.tar` 中的数字，返回 `[(step, path)]` |
| `get_last_checkpoint(exper, allow_interrupted)` | 最近 checkpoint（可排除 `_interrupted`） |
| `get_best_checkpoint(exper)` | 固定返回 `checkpoint_best.tar` |
| `delete_old_checkpoints(dir, num_keep)` | 只保留最近 `num_keep` 个；中断 checkpoint 优先删除 |
| `save_experiment(...)` | 保存当前状态；若指标优于 `best_eval` 则复制为 `checkpoint_best.tar`；随后清理旧 checkpoint |
| `load_experiment(exper, conf, get_last, ckpt)` | 加载 `.tar` 或实验目录；合并旧模型配置与当前配置；`strict=False` 加载权重并提示缺失参数 |

`load_experiment` 是评估与微调的共同入口（见 07 章 `load_model`）。

### 4.3 输出目录结构

```
outputs/training/<experiment>/
├── config.yaml
├── log.txt
├── checkpoint_0_999.tar
├── checkpoint_0_4999.tar
├── checkpoint_best.tar
├── siclib/                  # 代码快照
└── events.out.tfevents.*    # tensorboard（若启用）
```

---

## 5. 日志（`utils/summary_writer.py`）

`SummaryWriter` 根据 `train.writer` 选择后端：

- 包含 `"tensorboard"`：`torch.utils.tensorboard.SummaryWriter(log_dir=output_dir)`；
- 包含 `"wandb"`：`wandb.init(project=siclib, name=experiment, config=dot_conf(conf))`（未安装则报错）；
- `null`：不记录。

统一 API：`add_scalar / add_scalars / add_figure / add_histogram / add_text / add_pr_curve / watch / close`，内部按后端分发。

---

## 6. 训练 GeoCalib 的完整步骤

```bash
# 1) 准备数据（见 06 章）
python -m siclib.datasets.utils.download_openpano --name openpano --laval_dir data/laval-tonemap
python -m siclib.datasets.create_dataset_from_pano --config-name openpano

# 2) 训练针孔模型
python -m siclib.train geocalib-pinhole-openpano --conf geocalib --distributed --mp bfloat16

# 3) 微调径向畸变模型（从针孔权重初始化）
python -m siclib.train geocalib-radial-openpano --conf geocalib-radial --distributed

# 4) 评估
python -m siclib.eval.lamar2k --checkpoint geocalib-pinhole-openpano
```

关键差异：

| 配置 | 数据 | 相机模型 | lr / steps | 权重初始化 |
| --- | --- | --- | --- | --- |
| `geocalib.yaml` | `openpano` | `pinhole` | 1e-4 / 150k | `weights/mscan_b.pth`（仅骨干） |
| `geocalib-radial.yaml` | `openpano-radial` | `simple_radial` | 1e-5 / 200k | `weights/geocalib.tar`（完整模型） |

> 注意：`geocalib-radial.yaml` 通过 `model.weights` 加载预训练权重，由 `BaseModel` 的权重加载逻辑处理（见 05 章）。

---

## 7. 常见问题与调试

| 现象 | 处理 |
| --- | --- |
| 显存不足（尤其 radial） | 降低 `data.train_batch_size`（radial 默认 18 vs 针孔 24） |
| 训练 NaN | 脚本会自动跳过该迭代；可开 `--detect_anomaly` 定位 |
| 想验证代码正确性 | `--overfit`，loss 应快速下降 |
| 想从上次中断继续 | `--restore`（含优化器/调度器状态） |
| 只想微调 | 配置 `train.load_experiment` 或 `model.weights` |
| 无 tensorboard | 设 `train.writer=null`（GeoCalib 默认即为 null） |
| 多机多卡 | `--distributed` 使用 `file://` 初始化，需共享文件系统 |

---

## 8. 文件索引

| 文件 | 职责 |
| --- | --- |
| `train.py` | 训练主脚本、DDP/AMP、评估循环、调度器 |
| `configs/geocalib.yaml` | 针孔训练入口配置 |
| `configs/geocalib-radial.yaml` | 畸变微调配置 |
| `configs/train/geocalib.yaml` | 训练超参 |
| `configs/model/geocalib.yaml` | 模型结构 |
| `configs/data/openpano*.yaml` | 数据配置 |
| `utils/experiments.py` | checkpoint 保存/加载/清理 |
| `utils/summary_writer.py` | tensorboard / wandb 封装 |
| `utils/stdout_capturing.py` | rank 0 日志捕获 |
| `settings.py` | `DATA_PATH / TRAINING_PATH / EVAL_PATH` |
