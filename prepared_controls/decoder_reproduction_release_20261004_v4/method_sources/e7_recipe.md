# DeepLabV3+式适配与SegFormer-B0 配置来源卡片

协议／记录日期：E7冻结2026-09-27；SegFormer诊断2026-09-29。

e7_pretrained_modern_baselines_v1。DeepLab LR1×10^-4，SegFormer LR6×10^-5；AdamW wd10^-4，5% warmup+cosine至1%，norm/bias免衰减，batch16，max50/patience10。来源为预设项目配方；健康与短程学习检查不证明学习率最优。

原始记录身份（完整内部档案保留，公开卡片不冒充原文件）：

- research_comparators_e7.yaml: `8227996224c462be42d845c52ebd4d9b20702c6cfbc89ef0e8eefbfd55f9adbf`
- e7_preformal_server_readiness_20260927.md: `a1e9a883c1359e756a9a55dc03688c62625c1603b95df5f41f1984994b1c0a85`
- e7_segformer_formal_20260929.md: `4a45955f7470a47a5e7bbc065d1f3e40e0b9e909bb9518c47367099d656661aa`
