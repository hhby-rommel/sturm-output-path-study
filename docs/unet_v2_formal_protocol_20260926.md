# U-Net-v2 正式实验协议（2026-09-26）

## 身份与冻结点

版本 ID：`unet_no_bn_optimizer_v2_v1`。这是在本项目事件级、有效像元、clean+degraded 双视图协议下的独立 S1/S2 U-Net 基线；它不宣称严格复现 STURM-Flood 原论文最终训练。旧 `optimizer_v1` 正式结果与既有 optimizer-v2 诊断保持原位，不回填。

本版本只因已量化的 S1 BN/dropout 统计失配而关闭 U-Net 块入口 BN。短程证据见 `docs/unet_bn_failure_mechanism_20260926.md`：固定 epoch 2 权重，训练集重估 BN 且关闭 dropout 将 S1 固定验证面板 F1 从 0.3664 提升到 0.7701；保留 dropout 的重估为 0.3591。独立无 BN seed 2026 的 6 轮全量验证 degraded F1 为 `0.7836, 0.7372, 0.7940, 0.8544, 0.8032, 0.8376`，峰值为第 4 轮；对应 S1/S2 为 `0.8753/0.5963`，S2 仍是明确弱项。**这不是跨种子稳定性的既成证明**。用户授权进入正式三种子实验，因此从本版本开始不再根据正式测试结果调整配置。若后续种子不稳，应如实报告，不在同一版本内挑选有利种子或重新调参。

## 固定配置

唯一机器配置为 `configs/project.yaml::formal_unet_v2`，由 `scripts/server_main.py` 精确校验：

- 模型：`sturm_unet`，S1 两通道和 S2 九通道各自独立五层 U-Net；base channels 64，dropout 0.30，**无 BN**；原始输入 `none_v1`，不启用 z-score。
- 数据：`event_split_v1.csv`，train/val/test 事件分离；S2 无效像元不参与 loss/metric。两传感器仍采用原 50:50 replacement sampler。
- 损失和退化：原 `m1_degraded`，有效像元 masked CE+Dice，传感器感知退化；不改变标签、强度、质量特征或批量聚合。
- 优化：`unet_optimizer_v2_lr1e4` AdamW、peak LR 1e-4、5% warmup、cosine 衰减、weight decay 1e-4、norm/bias 免衰减、gradient clip 5.0；固定 batch size 16、max epochs 30、early stopping patience 10。
- 选权：固定 tag=1 的 degraded **pixel-pooled overall flood F1**；精确平局时比较 clean F1，再选更早 epoch。event-macro 用于报告，不参与选权。
- 种子：2026、2027、2028。每个模型/种子独立目录，smoke 只用 train/val 子集模拟测试，正式训练结束后每个 seed 只对真实 test 评估一次。

## 执行与审计

结果根目录为 `results/formal_unet_v2_20260926/`，子目录含版本 ID、模型 ID、seed 与 `formal`；不得指向 `results/formal`。先运行隔离的 smoke，再串行运行三个正式种子。`provenance.json` 记录命令、环境、manifest/split、配置、runner、模型、优化与本协议哈希。正式运行不保存每轮模型快照，以免磁盘不足；保留 `best.pt`、`last.pt`、每轮 history、验证详情、一次性测试评估和结果 JSON。每个 seed 的结果必须显示 `FORMAL_COMPLETED`、`real_test_split_used=true` 和 `formal_protocol_version=unet_no_bn_optimizer_v2_v1` 方可进入主表。

当前服务器空间与算力均须在启动前复核；任何一个 seed 中断时只允许从完整 `last.pt` 显式 `--resume`，不得覆盖已有结果。三个种子结束后汇总 S1/S2、clean/degraded、pooled/event-macro、逐事件和复杂度；如果无 BN U-Net 接近或优于 YOLO decoder，论文应收缩架构优越性主张。如果较弱，也只能在身份、训练健康和完整三种子证据成立时称为公平对照。

## 2027/2028 日志增补

seed 2026 启动后的第 10 轮出现 S1 假阳性陡增，因此在它自然完成、下一种子启动前，为后续种子追加只读训练诊断。每个训练步在原有全模型裁剪之前，分别测量独立 S1/S2 子模型的梯度范数；逐轮记录均值、最大值、各分支超过原全局裁剪阈值 5.0 的步骤占比，以及每个子模型相对于本轮起点的参数更新 L2 和相对更新 L2。分支“超过阈值占比”是诊断量，**不是实际分支裁剪比例**：优化仍对全模型执行原来的一次全局裁剪。模型、数据、损失、优化器、学习率、采样、早停、选权和测试访问均不改变。

seed 2026 的原始训练日志不回填这些字段；2027/2028 的 provenance 记录新代码哈希。启用前须通过隔离 smoke，确认新增数值有限且不会改变旧全局记录与训练更新。三种子性能可以按相同训练协议汇总，但新增诊断字段仅能跨 2027/2028 比较。

## 2028 中断与恢复记录

2026-09-27 服务器重启使 seed 2028 在第 2 轮完整 checkpoint 保存后中断；当时没有 `result.json` 或真实测试评估。首次显式 `--resume` 在恢复随机状态之前失败：`torch.load(map_location=cuda)` 将原本应留在 CPU 的全局和采样器 RNG ByteTensor 一并映射到 GPU，`torch.random.set_rng_state` 因此拒绝该状态。修复仅在恢复 RNG 时把这些状态转回 CPU；不改变模型、数据、损失、优化器、调度器、裁剪、选权或评估。续跑须沿用同一 `last.pt`（第 2 轮）、30 轮总预算及原目录，保留启动和失败日志，并由新 provenance 记录修复后的代码哈希。
