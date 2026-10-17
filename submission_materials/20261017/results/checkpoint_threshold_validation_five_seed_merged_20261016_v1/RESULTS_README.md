# 五 seed validation 与 threshold-transfer 本地汇总

状态：服务器补评价已完成；44 个交接文件验收通过。后续无需服务器。

范围：Native、Decoder、FPN × seeds 2026–2030；fixed12026；新增六个 checkpoint 无重训。
新增 validation 4853 tiles；test 4863 tiles；各 12 events。19 点阈值网格、按传感器选择规则与历史三 seed 一致。
先在事件内合并 S1/S2 计数，再计算事件指标；先在事件内平均 seed 配对差，再事件等权汇总。

## Validation 基线（阈值 0.5）

| 条件 | 路径 | pooled F1 | event-macro F1 | event-equal extent error (%) |
|---|---|---:|---:|---:|
| clean | main_only | 0.866563 | 0.582224 | 46.026 |
| clean | decoder | 0.846151 | 0.570584 | 48.984 |
| clean | fpn | 0.865870 | 0.585674 | 48.038 |
| degraded | main_only | 0.866979 | 0.585330 | 45.645 |
| degraded | decoder | 0.858556 | 0.584802 | 45.858 |
| degraded | fpn | 0.861894 | 0.590669 | 47.525 |

## Validation 事件配对差

| 条件 | 比较 | mean ΔF1 (pp) | 正/负事件 | ≥0.5 pp 事件 |
|---|---|---:|---:|---:|
| clean | decoder-main_only | -1.164 | 3/9 | 2 |
| clean | fpn-main_only | +0.345 | 7/5 | 4 |
| clean | fpn-decoder | +1.509 | 9/3 | 9 |
| degraded | decoder-main_only | -0.053 | 6/6 | 3 |
| degraded | fpn-main_only | +0.534 | 8/4 | 5 |
| degraded | fpn-decoder | +0.587 | 8/4 | 7 |

## Extent-selected 阈值迁移

变化为 selected − baseline；负值表示 extent error 降低。单位为 pp。

| 条件 | 路径 | val baseline → selected (%) | test baseline → selected (%) | test 配对变化 mean ± SD (pp) | 改善 seeds |
|---|---|---:|---:|---:|---:|
| clean | main_only | 46.026 → 27.869 | 35.322 → 35.050 | -0.272 ± 5.153 | 3/5 |
| clean | decoder | 48.984 → 30.928 | 38.466 → 36.068 | -2.399 ± 4.254 | 3/5 |
| clean | fpn | 48.038 → 36.761 | 34.709 → 35.033 | +0.324 ± 4.460 | 3/5 |
| degraded | main_only | 45.645 → 27.825 | 34.512 → 35.179 | +0.667 ± 3.672 | 3/5 |
| degraded | decoder | 45.858 → 31.106 | 31.442 → 34.912 | +3.470 ± 2.267 | 0/5 |
| degraded | fpn | 47.525 → 34.422 | 33.326 → 36.565 | +3.239 ± 7.038 | 2/5 |

## 核验范围与解释

- 新旧 validation/test 输入流一致；数值环境设置一致；阈值选择协议一致。
- 新增 test 的阈值 0.5 基线计数与既有 fixed12026 结果完全一致；一次 test 完成，无 checkpoint 重选。
- 15 个模型、60 个传感器/条件阈值选择单位齐全；所有事件均保留，当前相对误差分母均非零。
- validation 使用参与 checkpoint selection 的事件，属于描述性诊断，不是独立泛化证据。
- 五 seed 结果应直接读取本批输出；不能继续沿用旧三 seed 的阈值迁移数字或笼统的全路径恶化结论。
- 本轮未修改论文、未发布 release、未制作 ZIP、未启动其他实验。

全部逐 seed 指标和 F1-selected 敏感性结果见同目录 CSV。
