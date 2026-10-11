# 服务器正式实验 · 冻结协议（Server Protocol）

> 状态：已冻结（2026-09-19；正式训练仍须通过环境、数据与全链路 smoke 门禁）
> 项目：SQA-YOLO26-DCL — 跨传感器、质量条件与退化一致性学习的事件级洪水语义分割
> 日期：2026-09-17
> 本文件是 `implementation_status.md` 服务器阶段结论的落地协议，只写"要做什么、怎么做、怎么判定"，不展开历史讨论。

---

## 0. 目的与范围

在事件级、固定退化验证、≥3 随机种子的条件下，对四个冻结模型做完整训练/验证/测试，
产出可复现、可辩护的对比结论，并回答三个问题：

1. decoder（P2/P1/input 高分辨率）相比 native head 是否有稳定增益；
2. DCL 在 3 个种子上是否稳定改善 degraded F1 或缩小 clean/degraded 差距；
3. 是否具备服务器阶段的论文创新证据（或明确哪些只能算失败探索）。

**边界**：不继续扩大 YOLO 家族搜索；不把 M2 动态质量路由当主方法；不调参到"好看"为止。

---

## 1. 环境审计清单（训练前必须逐项通过）

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | 目标机器确认 | 明确机器名/IP/GPU 型号，确认就是"服务器"，不是本机 |
| 2 | Python | 与 `pytorch_env` 对齐：Python 3.10.x（本机为 3.10.20） |
| 3 | PyTorch/CUDA | `torch.cuda.is_available()==True`，记录 `torch.__version__` 与 `nvcc`/驱动版本 |
| 4 | 依赖 | `rasterio`、`ultralytics`、`opencv-python`、`scipy`、`sklearn` 可 import，`pip check` 干净 |
| 5 | GeoTIFF 读取 | 随机抽取 S1/S2 各 5 张，`rasterio.open` 成功，波段数 S1=2、S2=9 |
| 6 | 模型加载 | `models/pretrained/yolo26n-sem.pt` 存在且可被 `load_matching_yolo26_weights` 加载 |
| 7 | 前向/反向 | 128×128 输入，S1/S2 各 1 批，forward+loss+backward 全有限，无 NaN/Inf |
| 8 | 标签健康 | `manifest.csv` 中 mask 值在协议范围内，无越界，无全空掩膜进入训练 |

**禁止**：修改系统级环境、安装系统级 GDAL、源码编译 Rasterio、任何破坏性清理命令。
依赖只装到 conda/项目级环境，优先二进制 wheel。

---

## 2. 数据与协议确认

- **事件划分（冻结，不再变动）**：训练 36 / 验证 12 / 测试 12（`configs/project.yaml` 的 `event_counts`）。
- **划分文件**：`data/splits/event_split_v1.csv`（主划分）。
- **无泄漏**：同事件的所有瓦片只能出现在同一 split，禁止 tile-level 泄漏。
- **标签协议**：raw mask `0 -> 0`，`>0 -> 1`（`positive_rule: raw_mask_gt_zero`）；非有限/声明 nodata 区域 `ignore_index=255`，用显式 `valid_mask` 排除。
- **输入**：S1 `[B,2,128,128]`，S2 `[B,9,128,128]`；S2 已知 NaN 路径必须走有限值替换 + valid_mask。
- **测试集纪律**：测试集不参与 checkpoint 选择、不参与任何调参，只做最终一次性评估。

---

## 3. 冻结模型矩阵

### 主模型（5 个，全部 clean+degraded 训练）

| ID | 模型 | 构造要点 | 定位 |
|---|---|---|---|
| A | 共享 YOLO26 native baseline | `SharedYolo26(scale="n", common_channels=3, num_classes=2)`，native head | 架构对照 / 历史基线 |
| B | 共享 YOLO26 + common=16 + P2/P1/input decoder | `Yolo26UNetDecoder(scale="n", common_channels=16, num_classes=2, use_input_skip=True)` | **主候选** |
| C | B + DCL | B 的 decoder 结构 + DCL 一致性项（ramp 到 0.10） | 注册消融 |
| D | YOLO11n + ECA | `SharedYolo11(scale="n", use_attention=True)` | 主要轻量架构对照 |
| E | 原论文 U-Net 拓扑复现 | `SturmPaperUNet(base_channels=64, native_input=True)`，S1/S2 独立各一 | 原论文基线 |

### 次级控制（预算允许时补跑，不得影响主实验进度）

| ID | 模型 | 用途 |
|---|---|---|
| E1 | `Yolo26UNetDecoder(common_channels=16, use_input_skip=False)` | 分离 P2/P1 skip 与 input skip 的贡献 |
| E2 | c3 + per-band normalization decoder | 分离"输入归一化修复 c3 塌缩" |
| E3 | c16 decoder + class-weighted focal+Dice | 分离损失函数贡献 |

> E2 的"per-band normalization"确切实现需从本地
> `scripts/pilot_yolo26_decoder_normalization_ablation.py` 提取，服务器脚本里必须复现同一实现。

---

## 4. 训练协议（冻结超参数）

- **优化器**：AdamW(lr=3e-4, weight_decay=1e-4)。
- **batch_size**：16。
- **BatchNorm 运行统计**：冻结（本地 pilot 结论，小 batch 下冻结更稳）。
- **训练视图**：clean + 固定合成 degraded（S1 乘性噪声 `0.12*severity`，S2 加性噪声 `0.20*severity*scale`，dropout `0.05+0.10*severity`）。
- **损失**：`MaskedSemanticLoss`（有效像元加权 CE + soft Dice，`ignore_index=255`）。
- **种子**：{2026, 2027, 2028}，每个模型 ≥3 种子。
- **预训练初始化**：`load_matching_yolo26_weights(model.core, "models/pretrained/yolo26n-sem.pt")`；
  common_channels=16 时首个 YOLO stem 因输入宽度不同必须重新初始化（代码已如此处理）。
- **epoch 上限（已冻结）**：YOLO26/YOLO11 为 50 epoch；U-Net 为 30 epoch。
- **公平对照**：除被消融项（DCL 有无 / decoder 结构 / 损失）外，模型 A/B/C/D 的瓦片、事件、
  seed、优化器、epoch、batch、退化视图、评估协议必须完全一致。

---

## 5. 固定退化验证（必须复现本地机制）

"固定"的准确含义（已从本地代码确认）：评估时用 `set_seed(seed + 10000 + tag)` 固定 RNG，
且 **tag 固定为常数**，使同一 seed 的 initial / 每 epoch / final 评估复用**完全相同的退化视图**。
跨种子退化视图不同，但每个种子内部可复现。

服务器脚本必须满足：

- 每个模型/种子在评估 degraded 指标前，进入固定的 `(seed + 10000 + tag)` RNG 状态；
- 同一 seed 的学习曲线里，所有 degraded 评估用同一个固定 tag（本地用 `1`）；
- 评估结束恢复原 RNG 状态（本地 `_fixed_evaluate` 的 save/restore 写法）；
- clean 评估与 degraded 评估分开记录，不可混用同一 RNG 视图。

> 若服务器脚本改用显式 `severity`（例如 `severity=torch.ones(B)`）也可接受，但必须在
> 脚本内**写死数值**，并保证与本地结果可比；默认禁止沿用 `degrade_sensor_batch` 的随机 severity。

---

## 6. checkpoint 与 early-stopping（预注册规则）

> 项目当前 `engine.py` **没有** checkpoint 保存和 early-stopping，服务器脚本必须新增以下逻辑。

- **选择指标**：固定退化验证集上的 **overall degraded flood F1**（核心鲁棒性指标）。
- **best checkpoint 判定（依次）**：
  1. degraded F1 更高者胜；
  2. 若并列，clean F1 更高者胜；
  3. 仍并列，取更早 epoch。
- **early-stopping**：监控固定退化验证 degraded F1，**耐心 10 epoch**（连续 10 epoch 无新高即停），
  且不超过第 4 节的 epoch 上限。
- **每 epoch 记录**：train loss、clean/degraded F1/IoU/Precision/Recall（overall + 分 S1/S2）、
  有效像元数、NaN/无效像元比例，逐行写入该实验的 `*.csv` 日志。
- **checkpoint 保存**：覆盖式保存当前 best（含 `model.state_dict()`、epoch、seed、optimizer 状态、
  指标快照）；最终另存 last epoch 供诊断。
- **测试集**：只在 best checkpoint 选定后，对测试集做**一次**评估。

---

## 7. 评价指标与报告字段

每个模型 × 种子产出以下字段（JSON + 汇总 CSV）：

- overall：flood F1、IoU、Precision、Recall；
- 分传感器：S1 F1、S2 F1（clean 与 degraded 各一份）；
- clean F1 与 degraded F1 及 **clean−degraded 差值**；
- 事件级均值 ± 标准差（跨事件口径）；
- 跨种子均值 ± 样本标准差（跨种子口径，与事件级口径分开呈现）；
- 有效像元数量、NaN/无效像元比例；
- 参数量、FLOPs、峰值显存、推理速度（同硬件、同 batch=16、同 128×128 下测）。

---

## 8. 复现性要求

训练前必须留存并归档：

1. **git 快照**：项目顶层已初始化 Git；服务器实验前提交代码、配置、协议与关键文档的
   训练前基线，并由每次运行的 `provenance.json` 记录 commit hash 和权威文件 SHA-256。
2. 完整启动命令（含所有参数）；
3. 环境快照（`pip freeze` / `conda env export`）；
4. 随机种子、事件划分文件路径；
5. 每实验的日志、checkpoint、结果 JSON/CSV；
6. 汇总对比表 + 结论报告。

---

## 9. 执行顺序（门禁式）

1. 环境审计（第 1 节）——未通过则停止并报告阻断原因；
2. 数据/协议确认（第 2 节）——路径、划分、标签协议逐项核对；
3. 服务器 smoke test（1 个小规模固定子集，验证前向/反向/固定退化/checkpoint 全链路）——通过才进入正式训练；
4. 正式训练 A → B → C → D → E（先主模型，再注册消融与基线）；
5. 主矩阵完成且资源允许时，运行 YOLO26s common=16 full decoder 三种子尺度敏感性补充；
6. 汇总报告（第 7 节字段）。

### 9.1 标准入口

环境与数据门禁：

```powershell
C:\Users\hhby\anaconda3\envs\pytorch_env\python.exe scripts\audit_server_environment.py --assert-server --output results\formal\environment_audit.json
C:\Users\hhby\anaconda3\envs\pytorch_env\python.exe src\data_audit\validate_outputs.py
```

全链路 smoke 只使用训练/验证子集，并以验证子集模拟 test，不读取真实 test：

```powershell
C:\Users\hhby\anaconda3\envs\pytorch_env\python.exe scripts\server_main.py --model yolo26n_decoder --seed 2026 --smoke
```

正式训练每次只运行一个模型和一个种子，例如：

```powershell
C:\Users\hhby\anaconda3\envs\pytorch_env\python.exe scripts\server_main.py --model yolo26n_decoder --seed 2026
```

主矩阵 ID：`yolo26n_native`、`yolo26n_decoder`、
`yolo26n_decoder_dcl`、`yolo11n_eca`、`sturm_unet`。补充尺度 ID：
`yolo26s_decoder`，且必须完成三种子，不得以单种子替代。

---

## 10. 风险与回退策略

| 风险 | 回退 |
|---|---|
| `shared_dcl` 训练分支与 `Yolo26UNetDecoder` 结构不兼容 | 停止 C 并修复同一 decoder+DCL 接口；不得换成 SharedYolo26+DCL 冒充公平消融 |
| E 模型（约 69M U-Net、30 epoch）算力/时间超预算 | 可延后运行并报告未完成；不得用单种子替代正式三种子结果 |
| degraded F1 提升不明显 | 先诊断数据、归一化、decoder、损失，不直接改主方法 |
| 主模型明显不稳定 | 才考虑轻量化注意力（ECA）、蒸馏或 decoder 宽度调整，且必须预注册目标 |

---

## 11. 最终交付

1. 环境审计结果；
2. 数据与协议确认结果；
3. 训练完成情况；
4. 各模型完整对比表；
5. 均值 ± 标准差（事件级与跨种子分开）；
6. 主方法推荐；
7. DCL 是否保留的判定；
8. M2 放入正文还是仅作为失败探索；
9. 是否具备服务器阶段论文创新证据。
