#!/usr/bin/env bash
# P2a 冒烟训练 2000 步 — 见 PLAN.md §4 P2a
# 目的：实测显存峰值、验证 loss 下降、测步速以校准全量训练时长。
set -euo pipefail
source /data/miniconda3/etc/profile.d/conda.sh
conda activate rldemo
cd /data/robinji/rl-demo-case

# 磁盘约束（根分区小）：缓存/临时目录放 /data
export PIP_CACHE_DIR=/data/pipcache
export TMPDIR=/data/tmp
export HF_LEROBOT_HOME=/data/robinji/rl-demo-case/data

# 相机名映射：数据集(top,wrist) → smolvla_base 预训练命名(camera1,camera2)；
# camera3 由 --policy.empty_cameras=1 以全零图像+mask=0 填充（模型原生支持缺失相机）。
# rename_map 方向：dataset key → policy key（见 lerobot/processor/rename_processor.py）。
lerobot-train \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --dataset.video_backend=pyav \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --batch_size=4 \
  --num_workers=4 \
  --steps=2000 \
  --save_freq=1000 \
  --log_freq=50 \
  --seed=1000 \
  --output_dir=outputs/train/smoke \
  --job_name=smoke \
  --wandb.enable=false
