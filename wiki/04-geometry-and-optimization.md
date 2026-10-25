# 04 · 几何模型与优化原理

本文档说明 GeoCalib 的数学核心：相机模型、重力表示、透视场正演、LM 优化与不确定度。

## 1. 坐标系约定

- 相机坐标系：`x` 向右、`y` 向下、`z` 向前（与 OpenCV 一致）。
- 世界/重力坐标系：重力方向为 `+Z`（即“上”）。
- 图像坐标：像素坐标 `(u, v)`，原点左上角。
- 归一化坐标：`(x, y) = ((u - cx)/fx, (v - cy)/fy)`。

相机参数张量：

```text
[w, h, fx, fy, cx, cy, k1, k2]
```

其中主点固定为 `cx = w/2, cy = h/2`。

## 2. 相机模型与投影

### 2.1 透视投影

```text
p2d = (X/Z, Y/Z),  valid = Z > eps
```

雅可比：

```text
J_project = [[1/Z, 0, -X/Z²],
             [0, 1/Z, -Y/Z²]]
```

### 2.2 畸变模型

设 `r² = x² + y²`，归一化点 `(x, y)`。

| 模型 | 畸变（distort） | 去畸变（undistort） |
| --- | --- | --- |
| Pinhole | `1` | `1` |
| SimpleRadial | `1 + k1 r²` | `1 - k1 r²` |
| Radial | `1 + k1 r² + k2 r⁴` | `1 - k1 r² + (3k1² - k2) r⁴` |
| SimpleDivisional | `(1 - sqrt(1 - 4k1r²)) / (2k1r²)` | `1 / (1 + k1 r²)` |

畸变后的点再乘以焦距并加主点：

```text
(u, v) = (fx·x_d + cx, fy·y_d + cy)
```

### 2.3 完整变换链

```text
world2image: p3d --project--> p2d --distort--> p2d_d --denormalize--> uv
image2world: uv --normalize--> p2d --undistort--> p2d_u --(x,y,1)--> bearing
```

雅可比通过链式乘法组合：

```text
J_world2image = J_denormalize · J_distort · J_project
```

### 2.4 焦距与 vFoV

```text
focal = size / (2·tan(fov/2))
fov   = 2·atan(size / (2·focal))
```

`update_focal` 会把 vFoV 限制在 `[5°, 150°]`，避免优化发散。

## 3. 重力表示

重力是相机坐标系下的 3D 单位向量 `g = (gx, gy, gz)`。

由 roll / pitch 构造：

```text
gx = -sin(roll)·cos(pitch)
gy = -cos(roll)·cos(pitch)
gz =  sin(pitch)
```

反解：

```text
pitch = asin(gz)
roll  = 由 (gx, gy) 分支计算，范围 (-π, π]
```

### 3.1 两种更新方式

| 方式 | 更新 | 维度 |
| --- | --- | --- |
| 欧氏（roll/pitch） | `(r, p) += Δ` | 2 |
| 球面流形 | `g = SphericalManifold.plus(g, Δ)` | 2（切空间） |

球面流形使用 Householder 变换：

```text
H = I - β·v·vᵀ,  H·x = ||x||·e_n
plus(x, Δ) = ||x|| · H · [sinc(||Δ||)·Δ, cos(||Δ||)]
```

优点：更新后自动保持单位范数，避免长时间迭代后的数值漂移。

## 4. 透视场（Perspective Fields）

### 4.1 up 向量场

对每个像素，先取归一化坐标 `(u, v)`（注意此处是去畸变前的归一化坐标）：

```text
up_2d = (gx, gy) - gz · (u, v)
up_2d = normalize(up_2d)
```

直觉：3D 重力方向投影到图像平面。对于畸变相机，还要乘以畸变雅可比与偏移项：

```text
up_2d ← (diag(d_u) + offset(u,v)) · up_2d
```

输出形状 `(B, 2, H, W)`。

### 4.2 纬度场

对每个像素求 bearing `r = normalize(x, y, 1)`，纬度定义为：

```text
latitude = asin(r · g)
```

输出形状 `(B, 1, H, W)`，范围 `(-π/2, π/2)`。

### 4.3 地平线

```text
horizon_midpoint = K · R · (0,0,1)
```

再根据 roll 求左右边界交点，返回归一化到图像高度的 `(left, right)`。

### 4.4 雅可比

`J_perspective_field` 返回：

- `J_up`: `(B, H, W, 2, K)`
- `J_lat`: `(B, H, W, 1, K)`

`K` 的组成（按开关）：

```text
[gravity Δ (2) , focal (1) , dist (num_dist_params)]
```

各分量通过链式法则组合：

```text
J_up  = J_vecnorm · J_up_projection · J_update
J_lat = J_vecnorm · J_image2world · J_update
```

当 `log_focal=True` 时，焦距雅可比乘以 `f`（即对 `log f` 求导）。

当 `camera_R_rig` 存在时，先把重力转到 rig 坐标系 `g_rig = R_rigᵀ g`，只对共享的 rig 重力求导。

## 5. LM 优化

### 5.1 残差

```text
r_up  = up_field_target - up_field(camera, gravity)          # 2 通道
r_lat = sin(lat_target) - sin(lat_field(camera, gravity))    # 1 通道
```

纬度在 `sin` 空间做差，避免 `asin` 在边界处的数值问题。

### 5.2 代价与权重

```text
cost_up  = Σ_channel r_up²
cost_lat = r_lat²
```

对 `cost` 应用 **scaled Huber**：

```text
scaled_loss(x, huber, a) = a² · huber(x / a²)
```

再乘以网络输出的逐像素置信度：

```text
weight = huber'(x/a²) · confidence
cost   = huber(x/a²) · confidence
```

`up_loss_fn_scale` 与 `lat_loss_fn_scale` 默认均为 `1e-2`。

### 5.3 正规方程

```text
g = Σ_i w_i · J_iᵀ r_i
H = Σ_i w_i · J_iᵀ J_i
```

求解：

```text
(H + λ·I) Δ = -g
```

使用 Cholesky 分解；若失败则增大 `λ` 并重试。

### 5.4 阻尼自适应

```text
if new_cost < prev_cost:
    accept;  λ = max(λ / ν, λ_min)
else:
    reject;  λ = min(λ · ν, λ_max)
```

默认 `λ_min = 1e-6`、`λ_max = 1e2`。当 `fix_lambda=True` 时保持恒定。

### 5.5 早停

```python
torch.allclose(new_cost, prev_cost, atol=1e-8, rtol=1e-8)
```

可通过 `early_stop=True` 开启。

### 5.6 参数更新

- 重力：球面流形或 `(roll, pitch)` 欧氏更新；
- 焦距：`log f` 参数化（`f_new = exp(log f + Δ)`）或线性更新；
- 畸变：`k += Δ`，并 clamp 到 `(-0.7, 0.7)`（`Radial` 的 `k2` 单独范围）；
- 主点不更新。

## 6. 多图与多相机 rig

### 6.1 共享内参（shared intrinsics）

```text
所有图像共享同一组 (fx, fy) 与畸变参数；
重力逐图独立。
```

实现：`setup_system_shared_intrinsics` 把各图的梯度/海森按参数块聚合，焦距块求和，重力块保持块对角。

### 6.2 刚性 rig（camera_R_rig）

```text
g_rig  = R_rig,iᵀ · g_cam,i     （对每台相机）
```

所有相机共享同一个 `g_rig`，因此：

- 重力参数只有 2 维（rig 坐标系下的 roll/pitch 或球面切空间）；
- 内参可以共享（`shared_intrinsics=True`）或各自独立；
- 雅可比通过 `camera_R_rig` 旋转到 rig 坐标系后再对共享重力求导。

`setup_system_rig_unshared_intrinsics` 负责把“共享重力 + 各相机内参”拼成联合系统。

## 7. 不确定度估计

在最终解处，用 **roll/pitch/焦距** 参数化重新组装 Hessian（`as_rpf=True`，关闭球面与 log-focal 参数化），然后：

```text
Cov = H⁻¹
```

返回：

| 量 | 计算 |
| --- | --- |
| `covariance` | 完整参数协方差 |
| `roll_uncertainty` | `sqrt(Cov[roll, roll])` |
| `pitch_uncertainty` | `sqrt(Cov[pitch, pitch])` |
| `gravity_uncertainty` | 由 2×2 重力协方差块转换 |
| `focal_uncertainty` | `sqrt(Cov[f, f]) / 2` |
| `vfov_uncertainty` | `sqrt(J_focal2fov² · Cov[f,f] / 2)` |

这些不确定度是**近似值**，依赖：

- 透视场预测的置信度权重；
- Huber 损失的鲁棒性；
- 局部二次近似（Hessian）是否成立。

## 8. 初始化

| 初始化 | 规则 |
| --- | --- |
| `trivial` | `roll=0, pitch=0, vfov = focal2fov(0.7·max(h,w))`（或使用 `prior_focal`） |
| `heuristic` | roll 取图像中心 up 向量角度；pitch 取中心纬度；vFoV 取中心列上下纬度差；畸变为 0 |
| 先验 | 若提供 `prior_gravity` / `prior_focal` / `prior_dist`，则对应参数固定不优化 |

`siclib` 版本还支持把其他网络（如 DeepCalib）作为 `initializer`。

## 9. 为什么优化是可微的

- 所有正演函数（投影、畸变、透视场、LM 更新）均用 PyTorch 实现；
- LM 的迭代过程本身在计算图中（`num_steps` 有限）；
- 训练时 `param_total` 损失对最终 `camera / gravity` 求导，梯度可回传到透视场预测网络。

因此网络不仅学习“像透视场”，还学习“**易于被几何优化拟合**的透视场”。

## 10. 数值稳定性细节

| 位置 | 处理 |
| --- | --- |
| `project` | `z.clamp(min=eps)`，`eps=1e-3` |
| `J_vecnorm` | 范数为 0 时加 `1e-6` |
| `latitude asin` | `clamp(-1+eps, 1-eps)` |
| `SphericalManifold.plus` | `||Δ|| < eps` 时用 `sinc ≈ 1` |
| `huber_loss` | `sqrt(x + 1e-8)` 避免 backward NaN |
| `update_focal` | clamp vFoV 到 5°~150° |
| `update_dist` | clamp 畸变到 `(-0.7, 0.7)` |
