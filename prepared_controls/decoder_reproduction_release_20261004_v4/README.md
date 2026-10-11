# Decoder组件复现候选包v3（未公开发布）

本包包含已完成九项结构删除对照的49份冻结源文件、原配置和划分、受保护的运行入口及参考结果。源码SHA与entry_lock.json完全一致，公开许可尚未确定。project.yaml只将本机路径改为相对路径，数值字段未变；公共源码锁中的配置SHA据此更新，原锁与配置SHA在public_source_lineage.json保留，不能将公共配置声称为历史字节副本。它不承诺七系统全部重训，也不把短程检查计入正式精度表。

## 1. 环境与外部资产

Python 3.10。独立环境验证使用PyTorch 2.1.0+cpu/Torchvision 0.16.0+cpu；原正式环境使用2.1.0+cu121/0.16.0+cu121。CPU验证不能替代新GPU环境准入。依赖安装后保存pip freeze并运行pip check。

```bash
python -m venv .venv
# 激活环境后，使用本次验证的pip版本
python -m pip install pip==26.2.1 --index-url https://pypi.org/simple
# 先取得CPU二进制，再由锁文件安装其余依赖
python -m pip install torch==2.1.0 torchvision==0.16.0 --no-deps --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-local-validation-lock.txt --index-url https://pypi.org/simple
python -m pip check
```
requirements-local-validation-lock.txt记录已实际验证的Windows CPU环境；GPU环境应安装cu121构建并重新检查，而非照搬CPU标识。

从原公开数据来源取得STURM-Flood，置于data/raw/Dataset，保持manifest中的相对路径。取得models/pretrained/yolo26n-sem.pt，核对entry_lock.json中的SHA。原影像、初始化和正式checkpoint不包含在发布包中。raw_train_val_identity.json列出全部train/val身份；health_fixture.json为已有16 train/16 val样本索引。不能用随机初始化或改写锁文件绕过缺失资产。

## 2. 先执行train/val健康检查

```bash
python scripts/decoder_attribution_health_20261003_v1.py --project-root . --lock entry_lock.json --output-root runs/cpu_health --device cpu
```
该入口固定诊断种子10501，对full/no_skip_capacity/no_input/no_p1分别做一个16样本训练epoch、clean/degraded 16样本val、保存和两个一致恢复探针。采用原LR、损失、采样、BN、裁剪；不读取test、不启动正式训练。输出目录必须不存在，首错停止，不自动重试。CPU收据不能作为GPU正式准入。

## 3. GPU收据与独立授权

在已准备完整train/val及兼容GPU环境时，实际生成收据：
```bash
python scripts/decoder_attribution_gpu_preflight_20261003_v1.py --project-root . --lock entry_lock.json --output-root runs/gpu_preflight
python scripts/prepare_reproduction_scope.py --output-dir scopes/run_v1
```
第一条会检查全部train/val数据字节、四组GPU最小运行和完整4853瓦片val，生成runs/gpu_preflight/server_gpu_admission_receipt.json，不能手写成功收据。第二条只生成approved=false的模板，不运行模型或授予许可。

审查准确九任务与源码锁后，由实验负责人明确批准train_approval.json；test_authorization.json单独批准，不能因为训练批准就访问test。个人批准文件不得复制历史作者授权。正式单任务示例：
```bash
python scripts/decoder_attribution_train_val_20261003_v1.py --phase formal --arm no_input --seed 2026 --project-root . --output-dir runs/formal/no_input/2026 --lock entry_lock.json --approval scopes/run_v1/train_approval.json --health-receipt runs/gpu_preflight/server_gpu_admission_receipt.json --device cuda
```
默认只训练。原串行train/val入口decoder_attribution_serial_20261003_v1.py接收相同--project-root、--output-root、--lock、--approval、--health-receipt，按固定九任务执行；首错停止，结果标记test pending。

如准确测试范围另获批准，完成训练并冻结best后才能调用：
```bash
python scripts/decoder_attribution_test_once_v1.py --project-root . --task-dir runs/formal/no_input/2026 --lock entry_lock.json --authorization scopes/run_v1/test_authorization.json --arm no_input --seed 2026
```
test入口会先创建一次性claim；已完成测试直接复用，claim存在但无有效结果时停下审查，不能重新调用补跑。CPU诊断checkpoint不允许进入此入口。历史不中断交接控制器仅保留在historical_archive，不能用于新实验启动。

## 4. 验证范围与发布状态

独立验证记录见independent_validation.json：它报告实际CPU环境、加载、四组短程train/val、保存恢复和数据范围；不代表完整训练复现、收敛或正式精度复现。许可证和公开范围待作者确定；本包仍未公开发布。


## 论文与复现范围

配套通用稿v17将贡献集中于共同骨干／neck／decoder的训练系统比较，以及整体高分辨率输出路径的结构删除对照。参考结果中的F1为水体前景F1，pooled由合并混淆计数计算；三种子均值为主要结果，样本SD及同种子差描述波动。跨架构为模型特定配方下的系统比较，内部结构删除保持YOLO26数值协议。

CPU检查验证四组短程train/val、保存恢复与严格权重加载，不独立复现正式精度或GPU效率。v4只同步文档及来源关系，未改变v3源码、数值配置、权重／数据身份和授权保护；尚未公开发布。
