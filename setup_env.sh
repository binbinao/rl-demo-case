#!/usr/bin/env bash
# P0 环境搭建（幂等，可重复执行）— 见 PLAN.md §4 P0
set -euo pipefail
source /data/miniconda3/etc/profile.d/conda.sh

# 磁盘约束：根分区小（曾 2.4GB 可用）。pip 缓存与临时目录一律放 /data，
# 否则 cudnn/cublas 等 >500MB wheel 会把 / 撑爆（Errno 28）。
export PIP_CACHE_DIR=/data/pipcache
export TMPDIR=/data/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
# 1. Python 3.12 环境（幂等）
conda env list | grep -q "^rldemo " || conda create -y -n rldemo python=3.12
conda activate rldemo

# 2. 先装 cu128 的 torch（精确 pin 带本地版本号：防止 pip 因镜像 extra-index 选到 cu130
#    wheel —— 本机驱动 570 仅支持 cu128，cu130 会报 "driver too old"）。
#    torch 2.10 满足 lerobot 的约束 >=2.7,<2.12。
PIP_CONFIG_FILE=/dev/null pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"
# （nccl-cu12 已随 torch 依赖自动安装；若 pip 源断流，可用 curl -C - 断点续传下载
#  该 wheel 到本地后以标准文件名安装 —— 实测 pip 缓存恢复路径可行，无需此步骤。）
# 3. 再装 lerobot（torch 已满足 >=2.7,<2.12，不会被重装）
pip install "lerobot[training,smolvla]"

# 4. 验证（每条都必须通过）
python - <<'EOF'
import torch, torchvision, lerobot, transformers
print("torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA 不可用"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))  # 应为 (7, 5)
print("lerobot", lerobot.__version__)
EOF
ffmpeg -version | head -1
python -c "import torchcodec" 2>/dev/null && echo "torchcodec OK" || echo "torchcodec 缺失→按 P1 降级路径处理(pyav)"

echo "P0 完成。磁盘占用："
du -sh /data/miniconda3/envs/rldemo
df -h /data | tail -1
