#!/usr/bin/env bash
# P2b 正式训练 20000 步（≈4 epoch）— 见 PLAN.md §4 P2b
# 冒烟实测：1.60s/step、显存 8.6GB/15.4GB、loss 0.46→0.13（2000 步）。
# 预计时长：20000 × 1.60s ≈ 8.9 小时。
set -euo pipefail
source /data/miniconda3/etc/profile.d/conda.sh
conda activate rldemo
cd /data/robinji/rl-demo-case

# 磁盘约束（根分区小）：缓存/临时目录放 /data
export PIP_CACHE_DIR=/data/pipcache
export TMPDIR=/data/tmp
export HF_LEROBOT_HOME=/data/robinji/rl-demo-case/data

# RESUME=true 时续训（崩了/断了之后用）
if [[ "${RESUME:-false}" == "true" ]]; then
  lerobot-train \
    --config_path=outputs/train/smolvla_pickplace/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
  exit 0
fi

# 相机名映射：数据集(top,wrist) → smolvla_base 预训练命名(camera1,camera2)；
# camera3 由 --policy.empty_cameras=1 以全零图像+mask=0 填充（模型原生支持缺失相机）。
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
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --seed=1000 \
  --output_dir=outputs/train/smolvla_pickplace \
  --job_name=smolvla_pickplace \
  --wandb.enable=false
