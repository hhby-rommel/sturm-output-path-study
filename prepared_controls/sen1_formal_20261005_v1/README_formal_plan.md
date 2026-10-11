# Sen1Floods11 S1正式方案：三模型、三种子

配置、评价范围、样本序列与存储预算已冻结；没有启动训练、下载test或执行test。本文件是正式计划，不是完成结果。执行入口尚待实现、来源锁定与准入检查。

## 模型与日程

| 模型 | 初始化 | 优化与BN | 完整上限 |
|---|---|---|---|
| 完整decoder | 原Cityscapes权重按名称/形状加载；新增路径原随机初始化 | AdamW LR3e-4/wd1e-4，无调度；主体BN冻结，仅六层新增BN适配 | max50/patience10，最多12600更新 |
| 主输出控制 | 同种子共同主体初始化；原主输出监督 | AdamW LR3e-4/wd1e-4，无调度；BN统计冻结 | max50/patience10，最多12600更新 |
| U-Net | 原随机Xavier、base64/dropout0.30、无BN | AdamW LR1e-4/wd1e-4，原warmup+cosine；norm/bias衰减排除规则保留 | max30/patience10，最多7560更新，378步warmup |

每轮252次更新，batch16、clip5、clean与degraded CE+Dice原监督相加。三个模型各2026/2027/2028，从原初始化开始，不继承CPU/GPU诊断权重。顺序为2026 decoder→主输出→U-Net，再2027、2028同序，共九任务。最大98280更新、1572480次样本暴露，早停可降低实际成本，不能因模型未领先延长。

新增BN适配仅属于Sen1Floods11配方。STURM原BN策略与历史实验均保持。Decoder与主输出比较包括输出路径及BN配方差异，不写成仅改变结构的严格消融。

## 数据与评价

沿用官方252train/89val/90test S1 HandLabeled，属于非事件隔离划分。本轮明确不含15个Bolivia chip，不含S2。原VV/VH dB输入，不新增归一化、缩放或test拟合统计；标签-1忽略、0背景、1水体。512×512 chip按4×4不重叠128窗口评价，标签不插值；主输出logits按双线性align_corners=False对齐128。预测恢复原chip位置，不重复计像素。

每轮使用完整val，degraded pooled前景F1优先，精确并列比较clean，再保留较早epoch。初始化不参与best。三个正式种子共享固定评价随机流12026/stream1及文件顺序/batch16；训练退化由种子与全局更新号独立生成，不与dropout耦合。采样.npy已经事先生成并SHA锁定，同种子三模型使用同一序列前缀。

每任务训练结束先冻结best来源、轮次及SHA，再创建排他一次性test claim；90chip clean与固定degraded各评价一遍，同属一次claim。test成功才进入下一任务。claim存在但无有效结果停止审查，不能自动重测；test不参与选择模型、BN配方或修改剩余队列。当前test资产未下载，SHA与网格检查仍需在访问评价前补齐，不能称已核验test内容。

报告三种子均值与样本SD，同种子配对差、precision/recall、TP/FP/FN/TN及文件分组宏平均。文件组并不自动构成独立事件。零分母指标返回0并保留原计数，空有效域瓦片保留、不替换。

## 存储核算

依据本次远端真实文件：U-Net模型best138069340字节，含AdamW等状态last414365486字节；其余模型按对应实测文件核算。九任务best+last合计约1.82GB（十进制），每任务保留一个滚动last，不保留逐epoch大文件。

额外计入：最大best/last原子保存重叠、checkpoint15%余量、test90chip缓存、以最大已有train/val TIFF对估算的test影像容量并加25%余量、1GiB日志/元数据/预测余量，以及2GiB不使用的安全空间。总新增空闲需求约5.95GiB；服务器快照约7.17GiB，当前计算通过，预计仍留约1.21GiB额外余量（另已有2GiB保留空间）。这不是未来磁盘使用保证。

不删除或覆盖历史资产，无需目前扩容。启动前、每任务前及原子保存前重新核算，实际test文件大小取得后更新容量收据；不足即停止，不自动清理历史结果。详情见storage_budget.json。

## 成本与准入

按GPU诊断实测平均更新耗时，九任务达到全部上限的纯更新约2.73小时；每轮完整val、保存、test、下载与恢复成本另计，不将其称为总运行时间。

启动前仍需：实现正式train/test串行入口、锁定来源与依赖、确认完整RNG/采样状态保存、检查非训练接口、防覆盖/防重复test/首错停止保护，以及取得实时存储与test资产收据。不能由GPU预验证控制器自动接入正式训练。用户本次批准的是正式方案冻结，当前没有激活正式训练/test执行凭据。

机器可读方案：protocol.json；test清单：test_manifest.json；存储：storage_budget.json；冻结收据：freeze_receipt.json。冻结生成入口为scripts/freeze_sen1_formal_plan_20261005.py，仅做元数据与采样序列准备，不训练、不推理。
