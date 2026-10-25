# GeoCalib Code Wiki

> 项目：**GeoCalib — Single-image Calibration with Geometric Optimization**（ECCV 2024）
> 仓库：`/workspace`
> 文档语言：中文
> 代码许可：Apache-2.0；预训练权重许可：CC BY 4.0

GeoCalib 从**单张图像**估计相机内参（焦距 / 垂直视场角）与重力方向（roll / pitch）。核心思路是：

1. 用深度网络预测两张**透视场（perspective fields）**——逐像素的上方向向量场与纬度场；
2. 以透视场为观测，用**可微 Levenberg–Marquardt（LM）** 几何优化拟合相机参数与重力；
3. 由 LM 的协方差给出 roll / pitch / focal / vFoV 的**不确定度**。

---

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [01-architecture.md](./01-architecture.md) | 项目整体架构、双包结构、核心数据流、目录树 |
| [02-quickstart.md](./02-quickstart.md) | 安装、推理、Demo、评估、训练、数据集构建等全部运行方式 |
| [03-geocalib-inference-package.md](./03-geocalib-inference-package.md) | `geocalib` 轻量推理包逐模块说明 |
| [04-geometry-and-optimization.md](./04-geometry-and-optimization.md) | 相机模型、重力流形、透视场、LM 优化与不确定度的数学与实现 |
| [05-siclib-model-zoo.md](./05-siclib-model-zoo.md) | `siclib` 模型框架：BaseModel、网络、编码器、解码器、优化器 |
| [06-siclib-datasets.md](./06-siclib-datasets.md) | 数据集基类、SimpleDataset、增强、OpenPano 生成 |
| [07-siclib-evaluation.md](./07-siclib-evaluation.md) | 评估流水线、预测缓存、指标与 AUC、各 benchmark |
| [08-siclib-training.md](./08-siclib-training.md) | 训练脚本、DDP/AMP、调度器、checkpoint 与日志 |
| [09-configuration.md](./09-configuration.md) | Hydra 配置树、各 YAML 字段、命令行覆盖、路径设置 |
| [10-dependencies.md](./10-dependencies.md) | 外部依赖、包内模块依赖图、权重/数据依赖 |
| [11-api-reference.md](./11-api-reference.md) | 关键类与函数 API 速查表 |
| [12-notes-and-pitfalls.md](./12-notes-and-pitfalls.md) | 使用注意事项、重复实现、已知问题 |

---

## 核心概念速览

| 概念 | 说明 | 主要实现 |
| --- | --- | --- |
| 透视场 | 每像素 `up` 单位向量（2 通道）+ 纬度角 `latitude`（1 通道） | `geocalib/perspective_fields.py` |
| 相机模型 | `pinhole` / `simple_radial` / `radial` / `simple_divisional` | `geocalib/camera.py` |
| 重力 | 相机坐标系下的 3D 单位向量，支持 roll/pitch 参数化与球面流形更新 | `geocalib/gravity.py` |
| LM 优化 | 以透视场残差构建正规方程，Cholesky 求解，阻尼自适应 | `geocalib/lm_optimizer.py` |
| 不确定度 | 由 LM 近似 Hessian 的逆得到参数协方差 | `LMOptimizer.estimate_uncertainty` |
| 网络骨干 | MSCAN-B + LowLevelEncoder + LightHamHead 解码器 | `geocalib/modules.py` |
| 训练/评估库 | `siclib`：Hydra + OmegaConf 驱动的模型/数据/评估体系 | `siclib/` |

## 两套代码的关系

仓库包含两个可独立安装的 Python 包：

- **`geocalib`**：面向用户的轻量推理包，只依赖 `torch / torchvision / opencv-python / kornia / matplotlib`。
- **`siclib`**：训练与评估库，额外依赖 `hydra-core / omegaconf / albumentations / h5py / pandas / tensorboard / wandb` 等。

两者在 `camera`、`gravity`、`perspective_fields`、`lm_optimizer`、`modules` 上存在**并行实现**：`siclib` 版本面向训练（可配置损失、initializer、输出 loss/metrics），`geocalib` 版本面向推理（接口更简、支持多图共享内参与多相机 rig）。详见 [12-notes-and-pitfalls.md](./12-notes-and-pitfalls.md)。
