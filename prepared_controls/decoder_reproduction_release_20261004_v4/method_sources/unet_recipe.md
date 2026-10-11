# U-Net-v2 配置来源卡片

协议／记录日期：2026-09-26。

unet_no_bn_optimizer_v2_v1；unet_optimizer_v2_lr1e4。AdamW peak1×10^-4、wd10^-4，5% warmup+cosine至1%，norm/bias免衰减，无BN，dropout0.30，max30/patience10。来源为项目优化诊断后冻结配方，不宣称与其他模型等量调参。

原始记录身份（完整内部档案保留，公开卡片不冒充原文件）：

- unet_v2_formal_protocol_20260926.md: `5ee62f8f7f72ad332146fd9257c1ed449c5ad600a6d9944fe449599a64a2f036`
- optimizer_v2_diagnostic_protocol.md: `11c2535b24db047b3ef3792652bdcb85d950a6d564ba008b17e92be8e0ccb989`
- unet_bn_failure_mechanism_20260926.md: `bed85d74bc02718741b79b88d0bb140ee88626a77c755ffea8e3e763394860fd`
- project.yaml: `2cd2b12f4765fb8bc478cca6779ba59f9cd0ddad790ee096833879a642d283ad`
