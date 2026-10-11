# YOLOv12轻语义头 配置来源卡片

协议／记录日期：2026-10-01。

官方检测主体加项目P3轻头。AdamW3×10^-5、wd10^-4、无调度、batch16、BN运行冻结、clip5、max50/patience10。轻头LR来源于正式前本地学习恢复；不混用历史高分辨率decoder的3×10^-4。

原始记录身份（完整内部档案保留，公开卡片不冒充原文件）：

- yolov12_light_server_registration_20261001_v1.md: `d03fb7d6e57bae4cd87cbe7523f2dd5c666d78d9529e15dcfb7316791b6edff3`
- server_yolov12_light_formal_v1.py: `7315eb79f5fecee8a2ed6a42bac1860309632887e926504149b865b0e97b9846`
