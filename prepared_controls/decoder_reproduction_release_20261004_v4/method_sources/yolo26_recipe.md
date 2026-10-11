# YOLO26主输出与decoder组件 配置来源卡片

协议／记录日期：2026-10-03组件入口冻结。

AdamW3×10^-4、wd10^-4、batch16、无调度、原CE+Dice、BN运行冻结、clip5、degraded pooled→clean精确并列→更早轮次、max50/patience10。

原始记录身份（完整内部档案保留，公开卡片不冒充原文件）：

- entry_lock.json: `27092dd1c2a846b93600fb9c4481db2e9b27d62bb415b2bc4fc79c803ebd83c6`
- PROJECT_CONTEXT.md: `73faefa08befab19778e4d84b5a0b2159d875020c762e7bd11e466c199c5d98f`
- decoder_attribution_train_val_20261003_v1.py: `4ab570bd5ccfe9a3406dc7b565b1c0dc64c591c0f131d4ac3c9171c71b4f962f`
