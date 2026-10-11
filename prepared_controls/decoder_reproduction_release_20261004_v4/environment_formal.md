# 原正式执行环境与依赖边界

正式GPU环境：PyTorch 2.1.0+cu121、CUDA 12.1；Ultralytics 8.4.152。requirements-observed-local.txt是本地环境记录，不是已验证的服务器锁文件。原训练checkpoint包含模型、优化器和随机状态；新版本PyTorch的默认加载行为可能不同，应先核对受信任资产与加载兼容性。不得以兼容性修改为由改变监督、初始化或数值配方。完整外部依赖及其许可证仍需发布前审核。
