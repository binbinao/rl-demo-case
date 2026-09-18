# 具身智能案例：SmolVLA 在 SO-100 抓放数据集上的微调（执行计划）

> 本文档是给执行编码的模型（或人）的**完整施工图**。所有事实均已在线验证（2026-09-17）。
> 规则：**严格按本文档执行，不要自由发挥、不要扩大范围、不要引入本文档之外的依赖或抽象。**
> 遇到与本文档不符的现实情况，按「风险与对策」表的降级路径处理；再不行，停下来向用户提问。

---

## 1. 目标与范围

**目标**：在本机（Tesla T4 16GB）上，用 Hugging Face 生态完成一个具身智能案例：
从 `lerobot/smolvla_base`（450M 参数 VLA 模型）出发，在 `lerobot/svla_so100_pickplace`
数据集（SO-100 机械臂抓放，50 集）上微调，产出可加载、可推理的 checkpoint，并用
**离线动作预测对比**证明微调有效（预测动作 vs 数据集真值动作的误差下降）。

**明确不做**（用户已拍板，勿自行加回）：
- 不做真 RL（SAC/PPO/HIL-SERL）——本案例采用 SmolVLA 的流匹配（flow matching）模仿学习微调。
- 不做 Mac/MPS 适配——所有脚本仅面向本 Linux + CUDA 机器。
- 不接真机、不做仿真 rollout（无 SO-100 硬件；LIBERO 仿真与 SO-100 数据不匹配）。
- 不推 Hugging Face Hub（本地私有产出即可），不启用 wandb（离线环境）。

**范式说明**（给执行者的背景）：SmolVLA 是 VLA（视觉-语言-动作）模型，训练目标是
模仿数据集中的专家动作（flow matching 损失）。它在 LeRobot 生态中属于 VLA/模仿学习，
不是奖励驱动的 RL；这是用户知情后的选择。

---

## 2. 已验证事实（全部带证据，执行时直接引用）

### 2.1 本机环境
| 事实 | 值 | 证据 |
|---|---|---|
| GPU | Tesla T4 16GB，sm_75（Turing，**无 bf16**），compute capability 7.5 | `nvidia-smi --query-gpu=name,memory.total,compute_cap` |
| 驱动 / CUDA | 570.86.10 / 12.8 | `nvidia-smi` |
| 显存占用 | 另一进程 `/data/venv-lm/bin/python` 占 ~5.9GB，**空闲 ~9.5GB** | `nvidia-smi` |
| 磁盘 | `/data` 剩 **37GB**（已用 88%） | `df -h /data` |
| 内存 | 78GB，可用 ~53GB | `free -g` |
| ffmpeg | 系统 ffmpeg **7.0.2**（`/usr/bin/ffmpeg`） | `ffmpeg -version` |
| Python | 系统仅 3.11；lerobot 0.6.1 要求 **Python ≥3.12**；`/data/miniconda3` 可用 | PyPI 元数据 classifiers |
| pip 源 | 默认腾讯镜像（`mirrors.tencent.com`），可达；HF 与 hf-mirror 均可达 | `pip config list`、curl |

### 2.2 关键版本约束（⚠️ 最大的坑，务必照做）
- **PyPI 默认 torch wheel 是 cu130（要求驱动 ≥580.65），本机驱动 570.86 只能用 cu128 wheel。**
  → 必须先从 `https://download.pytorch.org/whl/cu128` 显式安装 torch+torchvision，
  再装 lerobot（pip 检测到 torch 已满足 `>=2.7,<2.12` 就不会重装）。
  cu128 index 实测下载速度 ~9.8MB/s，2GB wheel 约 4 分钟。
- lerobot 0.6.1 依赖：`torch>=2.7,<2.12`、`torchvision>=0.22,<0.27`、
  `transformers>=5.4,<5.6`（smolvla extra）、`datasets>=4.8,<5`、`torchcodec>=0.3,<0.12`（linux x86_64）。
- **TorchCodec ≥0.10 + PyTorch ≥2.10 支持动态链接系统 ffmpeg**；本机 ffmpeg 7.0.2 满足。
  若 torchcodec 运行时报 ffmpeg 版本不兼容 → 降级路径见 §4 P1。
- T4（sm_75）**不支持 bfloat16**；训练保持默认 **float32**，不要传 `--policy.dtype=bfloat16`。

### 2.3 模型：`lerobot/smolvla_base`
- 450M 参数，Apache-2.0，VLA；输入多视角图像 + 本体状态 + 语言指令，输出连续动作块（action chunk，chunk_size=50）。
- 训练目标：flow matching；骨干 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。
- 微调默认值（来自 `src/lerobot/policies/smolvla/configuration_smolvla.py`）：
  `freeze_vision_encoder=True`（冻结视觉编码器，**大幅省显存**）、`train_expert_only=True`、
  `optimizer_lr=1e-4`、warmup 1000 步、cosine decay 至 30000 步、图像 resize 到 512×512。
- 显存参考（官方硬件指南）：Small VLA 组 BS 8 峰值 ~10–16GB；BS 4 + 冻结视觉编码器可入 9.5GB。
- 首次微调会下载 SmolVLM2-500M 骨干权重（`load_vlm_weights=False` 时仅取结构；
  以运行时实际行为为准，磁盘预留 3GB）。

### 2.4 数据集：`lerobot/svla_so100_pickplace`
- **`main` 分支已是 v3.0 格式**（`meta/info.json`: `"codebase_version": "v3.0"`），无需 v2.1→v3.0 转换。
- 内容：50 集 / 19631 帧 / 1 个任务 / fps 30；SO-100 六自由度
  （shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper）。
- 相机 2 路：`observation.images.top`、`observation.images.wrist`，480×640，**视频编码 AV1**（系统 ffmpeg 7.0.2 带 libdav1d，可解）。
- 体量 ~941MB（data ~100MB + videos ~500MB，实际占用以下载为准，磁盘预留 2GB）。

### 2.5 训练入口
- CLI：`lerobot-train`（lerobot 0.6.1，extras：`training` 含 datasets/accelerate/wandb；`smolvla` 含 transformers）。
- 关键 flag（已从 `src/lerobot/configs/train.py` 的 `TrainPipelineConfig` 核实）：
  `--dataset.repo_id`、`--policy.path`、`--policy.device`、`--batch_size`、`--num_workers`、
  `--steps`、`--save_freq`（≤0 表示只存最终 checkpoint）、`--log_freq`、`--output_dir`、
  `--job_name`、`--resume`、`--seed`、`--wandb.enable`。
- 恢复训练：`lerobot-train --config_path=outputs/train/<RUN>/checkpoints/last/pretrained_model/train_config.json --resume=true`。

---

## 3. 项目目录结构（施工完成后应如此）

```
/data/robinji/rl-demo-case/
├── PLAN.md                  # 本文档
├── AGENTS.md                # 施工完成后补写的仓库指南（见 P4）
├── setup_env.sh             # P0：环境搭建（幂等，可重复执行）
├── scripts/
│   ├── download_data.py     # P1：下载数据集 + 冒烟校验（帧解码、统计、打印样例）
│   ├── smoke_train.sh       # P2a：2000 步冒烟训练（测显存、测步速）
│   ├── train_full.sh        # P2b：正式训练（支持 RESUME=true 续训）
│   └── offline_eval.py      # P3：离线评估 + 动作轨迹可视化
├── data/                    # HF_LEROBOT_HOME 指向这里（数据集缓存，~1GB）
└── outputs/train/           # 训练输出（checkpoint 在 outputs/train/<RUN>/checkpoints/）
```

所有脚本必须用**相对路径引用项目内文件**，环境变量显式导出；
`bash` 脚本开头 `set -euo pipefail`。

---

## 4. 分阶段执行

### P0 环境搭建（脚本：`setup_env.sh`）

```bash
#!/usr/bin/env bash
set -euo pipefail
source /data/miniconda3/etc/profile.d/conda.sh

# 1. Python 3.12 环境（幂等）
conda env list | grep -q "^rldemo " || conda create -y -n rldemo python=3.12
conda activate rldemo

# 2. 先装 cu128 的 torch（关键：驱动 570 只支持 cu128，PyPI 默认 cu130 会装不上/跑不了）
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 3. 再装 lerobot（torch 已满足，不会重装）
pip install "lerobot[training,smolvla]"

# 4. 验证（每条都必须通过，失败则停止并向用户报告）
python - <<'EOF'
import torch, torchvision, lerobot, transformers
print("torch", torch.__version__)
assert torch.cuda.is_available(), "CUDA 不可用"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))  # 应为 (7, 5)
print("lerobot", lerobot.__version__)
EOF
ffmpeg -version | head -1
python -c "import torchcodec" 2>/dev/null && echo "torchcodec OK" || echo "torchcodec 缺失→按 P1 降级路径处理"
```

**验收**：上述验证全部通过；`conda run -n rldemo python -c "import lerobot"` 成功。
磁盘预算：venv 约 8–10GB（装完后 `du -sh /data/miniconda3/envs/rldemo` 确认 <12GB）。

### P1 数据集下载与冒烟校验（脚本：`scripts/download_data.py`）

用 Python API（不要手写 wget，LeRobotDataset 负责校验与布局）：

```python
import os
os.environ.setdefault("HF_LEROBOT_HOME", "/data/robinji/rl-demo-case/data")
# 如遇 HF 限速，可加： os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/svla_so100_pickplace")
print("episodes:", ds.meta.total_episodes, "frames:", len(ds))
frame = ds[0]                       # 必须能解出一帧（验证 AV1 视频解码链路）
print({k: (v.shape if torch.is_tensor(v) else type(v).__name__) for k, v in frame.items()})
assert frame["observation.images.top"].shape[-2:] == (480, 640)
assert frame["action"].shape == (6,)
print("task:", ds.meta.tasks)       # 应含 1 个抓放任务描述
```

**验收**：打印 50/19631，帧解码成功，action 形状 (6,)，图像 480×640。
**降级路径**：
- torchcodec 报 ffmpeg 不兼容 → 数据集与训练命令均加 `--dataset.video_backend=pyav`（`av` wheel 自带 ffmpeg，已在 `dataset` extra 里）。
- HF 下载限速 → 设 `HF_ENDPOINT=https://hf-mirror.com` 重试（实测可达且更快）。

### P2 训练

**P2a 冒烟（`scripts/smoke_train.sh`，2000 步）——先跑这个，不要直接长跑：**

```bash
#!/usr/bin/env bash
set -euo pipefail
source /data/miniconda3/etc/profile.d/conda.sh && conda activate rldemo
cd /data/robinji/rl-demo-case
export HF_LEROBOT_HOME=/data/robinji/rl-demo-case/data

lerobot-train \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --batch_size=4 \
  --num_workers=4 \
  --steps=2000 \
  --save_freq=1000 \
  --log_freq=50 \
  --seed=1000 \
  --output_dir=outputs/train/smoke \
  --job_name=smoke \
  --wandb.enable=false
```

冒烟运行时另开终端记录：`nvidia-smi --query-gpu=memory.used --format=csv -l 5`。
**冒烟要看的三个数**：
1. 显存峰值（若 >9GB 或 OOM → 见降级路径）；
2. 每 50 步打印的 loss 是否从 ~1.0 量级开始**下降**（flow matching loss 下降即学习发生）；
3. 步速（日志时间戳）：估算全量时长，填入下表决策。

**P2b 正式训练（`scripts/train_full.sh`）**：

```bash
#!/usr/bin/env bash
set -euo pipefail
source /data/miniconda3/etc/profile.d/conda.sh && conda activate rldemo
cd /data/robinji/rl-demo-case
export HF_LEROBOT_HOME=/data/robinji/rl-demo-case/data

# RESUME=true 时续训（崩了/断了之后用）
if [[ "${RESUME:-false}" == "true" ]]; then
  lerobot-train \
    --config_path=outputs/train/smolvla_pickplace/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
  exit 0
fi

lerobot-train \
  --dataset.repo_id=lerobot/svla_so100_pickplace \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --batch_size=4 \
  --num_workers=4 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --seed=1000 \
  --output_dir=outputs/train/smolvla_pickplace \
  --job_name=smolvla_pickplace \
  --wandb.enable=false
```

**参数依据**：19631 帧 ÷ batch 4 ≈ 4908 步/epoch；20000 步 ≈ 4 epoch，
官方指南称模仿学习 5–10 epoch 收敛 → 20000 步是合理起点，必要时续训追加
（LeRobot 支持 `--resume`，不浪费已训进度）。
**时长预期**：T4 上 8–16 小时（fp32、冻结视觉编码器、512×512 双相机）。
用冒烟实测步速校准；如果外推 >24h，向用户报告并给出选项（减 steps 至 12000≈2.5 epoch / 换云端），
不要擅自跑通宵后才发现。

**训练监控**：`watch -n 5 nvidia-smi`；loss 在
`outputs/train/smolvla_pickplace/train.log`（或终端输出）。
**磁盘红线**：正式训练 checkpoints 450M×fp32 ≈ 1.8GB/个 × 5（5000/10000/15000/20000/last）≈ 9GB；
训练中若 `df -h /data` 可用 <5GB，删除非 last 的旧 checkpoint（`checkpoints/` 下按步数命名的目录）。

**降级路径（显存）**——按顺序尝试，每次只改一项：
1. `--batch_size=2`（步数不变；等效 batch 变小，loss 仍会降）；
2. `--policy.gradient_checkpointing=true`（速度换显存）；
3. 若 `nvidia-smi` 显示 `venv-lm` 进程已退出、空闲显存 >14GB → 可回到 batch 8 加速。
**禁止**：传 `--policy.dtype=bfloat16`（T4 不支持，会直接报错或回退 CPU）；
`--policy.dtype=float16` 未经冒烟验证前也不要用（flow matching 数值稳定性风险）。

### P3 离线评估与可视化（脚本：`scripts/offline_eval.py`）

无真机/无匹配仿真，评估方式 = **离线动作预测对比**：把数据集留出集的观测喂给模型，
比较预测动作块与真值。这是本案例的「交付证明」。

脚本规格：
1. 加载两个策略对比：
   - base：`lerobot/smolvla_base`
   - finetuned：`outputs/train/smolvla_pickplace/checkpoints/last/pretrained_model`
   用 `SmolVLAPolicy.from_pretrained(...) + make_pre_post_processors(...)`（模式见
   smolvla_base 模型卡的 Quick start 代码，逐行照抄，勿改 API）。
2. 留出集：episode 45–49（训练集的最后 5 集为固定评估集；训练并未做集划分，说明这一点即可）。
3. 对每个评估帧：`policy.select_action(frame)` → `postprocess` 得到动作，
   与 `frame["action"]`（真值）算逐关节 MSE；两端各汇总。
4. 输出：
   - `outputs/eval/report.md`：两端的 overall MSE 与逐关节 MSE 表格；
   - `outputs/eval/traj_ep{48}.png`：episode 48 的 6 个关节「真值 vs base 预测 vs finetuned 预测」
     曲线（matplotlib，3 种颜色，标题含 MSE）。
5. 退出码：finetuned MSE < base MSE → 0（成功）；否则非 0 并在 report 里如实写出。

**验收**：`report.md` 存在且数字完整；PNG 可打开；finetuned 的动作 MSE 低于 base。
（若未低于：检查 loss 曲线是否还在下降 → 续训更多步，而不是调参重跑。）

### P4 收尾

1. 删除冒烟输出 `outputs/train/smoke/`（磁盘红线）。
2. 写 `AGENTS.md`（仓库指南，放项目根）：概述、目录、如何重建环境（`setup_env.sh`）、
   如何下载数据（P1）、如何训练/续训（P2）、如何评估（P3）、已知约束（cu128-only、无 bf16、显存红线）。
   简明扼要，面向后续接手的 AI 助手。
3. 最终交付检查单（向用户报告时逐项确认）：
   - [ ] `setup_env.sh` 幂等重跑通过
   - [ ] 数据集 50 集全部可解码
   - [ ] 训练 loss 曲线下降（附最终 loss 值）
   - [ ] `checkpoints/last/pretrained_model/model.safetensors` 存在且 <3GB
   - [ ] `offline_eval.py` 退出码 0，report + PNG 齐全
   - [ ] `df -h /data` 可用空间 >5GB

---

## 5. 风险与对策总表

| # | 风险 | 概率 | 对策（按序） |
|---|---|---|---|
| 1 | torch 误装 cu130 → CUDA 不可用 | 高（若跳过 P0 顺序） | 严格先装 cu128 torch 再装 lerobot；验证 `torch.cuda.is_available()` |
| 2 | OOM（venv-lm 占 5.9GB） | 中 | batch 4→2 → gradient_checkpointing → 等 venv-lm 退出后 batch 8 |
| 3 | torchcodec/ffmpeg 链路失败 | 低 | `--dataset.video_backend=pyav` |
| 4 | HF 下载限速 | 低 | `HF_ENDPOINT=https://hf-mirror.com` |
| 5 | 磁盘 <5GB | 中（37GB 起步，env 10 + ckpt 9 + data 2） | 删旧 checkpoint；删 smoke；不推 Hub |
| 6 | 全量训练 >24h | 中 | 冒烟实测步速外推；报告用户，给减步数/云端选项 |
| 7 | finetuned 不优于 base | 低 | 续训（`--resume`）；报告真实数字，不粉饰 |

## 6. 云端备选（仅当用户主动要求时才提供，不在本项目内执行）

`lerobot-train --job.target=a10g-small`（HF Jobs，A10G 24GB，官方指南标注 smolvla batch 4 约 3–6h）。
需要 `hf auth login`；会产生费用。T4 跑通前不要主动提议。

---

## 7. 给执行编码模型的硬性规则

1. 只创建 §3 列出的文件；不加配置框架、不抽象「可扩展训练器」、不写测试套件。
2. 所有命令逐字使用本文档的版本；不要 `pip install lerobot[all]`（会拉进硬件/机器人依赖，浪费磁盘且可能编译失败）。
3. 脚本失败时：读错误信息 → 查 §5 对策 → 应用对策重试一次 → 仍失败则停止并完整粘贴错误给用户。
4. 不要为了「让训练跑起来」修改 lerobot 源码或 monkey-patch。
5. 每个阶段的验收不通过，不进入下一阶段。
6. 所有对用户的汇报引用真实数字（loss、显存峰值、步速、MSE），不编造。

---

## 8. 执行实录（2026-09-17 实际落地时发现，已全部处理）

以下偏差在执行中出现，均已按降级路径解决，供复盘：

| # | 发现 | 处理 |
|---|---|---|
| E1 | pip 腾讯镜像的 `extra-index-url` 会与 pytorch 官方 index 合并，lerobot 的 `torch<2.12` 约束让 pip 选了 `2.11.0+cu130`（本地版本比较 cu130>cu128），驱动 570 报 "driver too old" | torch 安装改用**精确 pin** `torch==2.10.0+cu128`（本地版本号让 cu130 候选不匹配），并用 `PIP_CONFIG_FILE=/dev/null` 绕开镜像 |
| E2 | `nvidia-nccl-cu12` 大 wheel（322MB）在腾讯镜像/pytorch.org/pypi.nvidia.com 全部在 ~144MB 处确定性断流 | 先用 `curl -C -` 断点续传从腾讯镜像拿全文件（sha256 校验一致）；后续重跑时 pip 从缓存恢复成功，未再需要 |
| E3 | pip 缓存/临时目录默认在根分区 `/`（仅 2.4GB 可用），cudnn 706MB + cublas 594MB wheel 下载直接 `Errno 28` | `export PIP_CACHE_DIR=/data/pipcache TMPDIR=/data/tmp`；并清理了历史 12GB `/root/.cache/pip` |
| E4 | torchcodec 0.11.1 因系统 `/lib64/libldap.so.2` 与 OpenSSL 符号冲突（`undefined symbol: EVP_md2`）无法加载（torchcodec 的报错只显示最后尝试的 core4 版本，误导性；真实错误在 core7） | 卸载 torchcodec，全程 `video_backend=pyav`（av 15.1.0 wheel 自带 libdav1d，AV1 解码验证通过） |
| E5 | `lerobot-train` 默认 `push_to_hub=true`，config 校验要求 `policy.repo_id` | 加 `--policy.push_to_hub=false` |
| E6 | 数据集相机名 `top/wrist` 与 smolvla_base 期望的 `camera1/2/3` 不相交，触发 feature mismatch | `--rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'` + `--policy.empty_cameras=1`（模型原生把缺失相机填全零图+mask=0） |
| E7 | 评估时若加载 checkpoint 自带 processor：base 的 normalizer stats 是其预训练数据集命名空间（`so100.buffer.action.mean` 等），输出动作量级完全错误（~0.3 vs 真值 ~±90） | 评估脚本改用 `make_smolvla_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)` 对两个模型统一现场构造（与训练时对 finetuned 的做法一致），单帧验证 pred/gt 同量级 |

**冒烟实测（P2a）**：1.60s/step、显存稳定 8597MiB/15360MiB、loss 0.460→0.128（2000 步，0.41 epoch）、checkpoints 正常落盘（001000/002000/last）。
**全量训练（P2b）**：20000 步 ETA ~9 小时。

### 最终结果（2026-09-18，P2b/P3/P4 完成）

- **P2b 全量训练**：20000 步 / 8h54m（1.60s/step），loss 0.495 → **0.077**（约 4.08 epoch），
  显存稳定 8597MiB，checkpoints 005000/010000/015000/020000（`last -> 020000`，model.safetensors 865MB < 3GB）。
- **P3 离线评估**（留出集 ep45–49，1877 帧）：finetuned overall MSE **28.39** vs base **264.91**
  （**↓89.3%**），6 关节全部提升（+81.5% ~ +97.4%）；`offline_eval.py` 退出码 0；
  产物 `outputs/eval/report.md` + `traj_ep48.png`（1200×1920，已人工核验渲染）。
- **P4 清理**：`outputs/train/smoke/`（2.5GB）已删；`setup_env.sh` 幂等重跑通过（env 9.9GB）；
  50 集边界帧（首/中/末）解码全部通过；`df -h /data` 可用 20GB。
- 评估期额外坑：环境未装 matplotlib（补装 3.11.2）；子集加载的 LeRobotDataset 必须用
  相对索引（`meta.episodes` 的 from/to 是全局索引，直接用会 IndexError）；HF Hub 偶发
  网络不可达（权重与数据集均已本地缓存，重试即恢复）。
