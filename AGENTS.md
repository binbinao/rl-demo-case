# Repository Guidelines

具身智能案例仓库：在 SO-100 机械臂抓放数据集上微调 Hugging Face `lerobot/smolvla_base`（450M VLA 模型），全程在单张 Tesla T4（16GB）上完成。范式为流匹配模仿学习（非奖励驱动 RL——用户知情选择）。

## Project Overview

- **做什么**：`lerobot/svla_so100_pickplace`（50 集 / 19631 帧 / 双相机 480×640 AV1 / 6-DoF 动作）→ 微调 `lerobot/smolvla_base` → 离线动作预测对比（base vs finetuned 的逐关节 MSE）作为交付证明。
- **产出**：`outputs/train/smolvla_pickplace/checkpoints/last/pretrained_model/`（可加载推理）、`outputs/eval/report.md` + `traj_ep48.png`。
- **完整设计与执行实录**见 `PLAN.md`（含 7 个执行期踩坑记录 §8，排障先查它）。

## Directory Layout

```
PLAN.md               # 施工图 + 执行实录（排障首选文档）
setup_env.sh          # 幂等环境搭建（conda rldemo + cu128 torch + lerobot）
scripts/download_data.py   # 数据集下载 + 冒烟校验
scripts/smoke_train.sh     # 2000 步冒烟训练（改参数前先跑这个）
scripts/train_full.sh      # 20000 步正式训练（RESUME=true 续训）
scripts/offline_eval.py    # base vs finetuned 离线动作 MSE 对比 + 轨迹图
data/                 # HF_LEROBOT_HOME（数据集缓存 ~1GB）
outputs/              # 训练 checkpoint 与评估产物
```

## Development Commands

```bash
# 环境重建（幂等，可重复执行）
bash setup_env.sh
conda activate rldemo          # Python 3.12；所有脚本内部自带 activate

# 数据集校验（应打印 50/19631 且首中末帧解码 OK）
python scripts/download_data.py

# 训练
bash scripts/smoke_train.sh                                  # 冒烟：2000 步 ≈ 53 min
nohup bash scripts/train_full.sh > outputs/train_full.log 2>&1 &   # 全量：20000 步 ≈ 9 h
RESUME=true bash scripts/train_full.sh                       # 崩溃后续训

# 评估（退出码 0 = finetuned MSE 优于 base）
python scripts/offline_eval.py
```

监控：`grep -oE "step:[0-9]+ .* loss:[0-9.]+" outputs/train_full.log | tail`；`nvidia-smi`。

## Runtime / Tooling Constraints（重要，违反即失败）

| 约束 | 原因 |
|---|---|
| torch 必须 `2.10.0+cu128`（精确 pin，从 `download.pytorch.org/whl/cu128` 安装，且 `PIP_CONFIG_FILE=/dev/null`） | 驱动 570.86 仅支持 cu128；PyPI 默认 cu130 wheel 报 "driver too old"；腾讯镜像 extra-index 会让 pip 选错版本 |
| pip 缓存/临时目录必须重定向 `/data`（`PIP_CACHE_DIR` / `TMPDIR`） | 根分区极小（~2.4GB 可用），大 wheel 下载会 `Errno 28` |
| 视频解码一律 `video_backend="pyav"` | 系统 `libldap.so.2` 与 OpenSSL 符号冲突（`EVP_md2`）导致 torchcodec 永远加载失败；av 15.1.0 wheel 自带 libdav1d 可解 AV1 |
| 训练保持默认 float32，禁 `--policy.dtype=bfloat16` | T4 是 sm_75 Turing，无 bf16 |
| 显存预算 ~9.5GB（另有 5.9GB 被无关进程占用） | batch 4 + 默认 `freeze_vision_encoder=true` 实测 8.6GB；OOM 时依次降 batch 2 → 开 gradient checkpointing |
| 不推 HF Hub、不开 wandb | `--policy.push_to_hub=false --wandb.enable=false` 必带；否则 config 校验要求 repo_id |

## Key Training Flags（相机名适配，漏掉即报 feature mismatch）

```bash
--rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
--policy.empty_cameras=1   # camera3 以全零图 + mask=0 填充（模型原生支持缺失相机）
```

## Evaluation Caveat（评估必须用数据集 stats 构造 processor）

不要用 checkpoint 自带 processor 评估 base 模型：其 normalizer stats 来自预训练数据集命名空间（`so100.buffer.action.mean` 等），输出动作量级完全错误。`scripts/offline_eval.py` 对两个模型统一用 `make_smolvla_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)` 现场构造（与训练时对 finetuned 的做法一致）。改动评估逻辑时保持这一点。

## Measured Numbers（T4 实测，供容量规划）

- 步速：1.60 s/step（batch 4，fp32，冻结视觉编码器，512×512 双相机）
- 显存：8597 MiB 稳定；磁盘：conda env 9.8GB + 每 checkpoint ~800MB×2 文件
- 冒烟收敛：loss 0.460 → 0.128（2000 步 / 0.41 epoch）
- 磁盘红线：`df -h /data` 可用 <5GB 时删 `checkpoints/` 下非 `last` 的旧步数目录
