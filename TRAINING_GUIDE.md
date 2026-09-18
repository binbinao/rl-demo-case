# 拾放智造 · 技术复现手册

> **PickTeach · 小算力 × 小数据，让机械臂学会「抓放」**
>
> 本文档面向**机械电气背景、能读懂但未实操过 VLA 的工程/算法团队**，目标是照着本文档
> 能完整复现「在单张 Tesla T4 上，用少量演示数据把一个通用 VLA 模型微调成专用抓放模型」，
> 并理解每一步**为什么这么做**。配套施工记录见 [PLAN.md](./PLAN.md)，排障先查本文档第 8 章。

---

## 目录

1. [项目概述](#1-项目概述)
2. [概念科普：先分清三种「教机器人」的方式](#2-概念科普先分清三种教机器人的方式)
3. [技术栈与硬件要求](#3-技术栈与硬件要求)
4. [环境搭建](#4-环境搭建)
5. [数据准备](#5-数据准备)
6. [训练流程](#6-训练流程)
7. [评估与验收](#7-评估与验收)
8. [排障手册](#8-排障手册)
9. [附录：实测指标](#9-附录实测指标)

---

## 1. 项目概述

### 1.1 做什么

从一个**通用**的视觉-语言-动作模型（VLA）`lerobot/smolvla_base`（4.5 亿参数）出发，
在 `lerobot/svla_so100_pickplace` 数据集（SO-100 六自由度机械臂「抓取-放置」演示，50 集）
上做**微调（fine-tuning）**，产出可加载、可推理的专用 checkpoint，并用**离线动作预测对比**
证明微调有效。

### 1.2 为什么

制造现场机械臂智能化改造的传统方式（示教编程）存在「编程门槛高、换品种重示教、依赖
现场工程师」三大痛点。本项目验证的路径是：**用少量演示数据 + 单卡算力**，让通用模型
快速适配到具体抓放任务，从而把「重新编程」变成「补数据再微调」。

### 1.3 能复现出什么结果

离线评估（留出集 ep45–49，共 1877 帧），指标为「预测动作 vs 真值动作」的逐关节 MSE：

| 关节 | base MSE | 微调后 MSE | 提升 |
|---|---|---|---|
| shoulder_pan | 604.11 | 15.82 | +97.4% |
| shoulder_lift | 409.03 | 75.52 | +81.5% |
| elbow_flex | 183.10 | 33.09 | +81.9% |
| wrist_flex | 202.49 | 23.49 | +88.4% |
| wrist_roll | 98.07 | 15.57 | +84.1% |
| gripper | 92.69 | 6.83 | +92.6% |
| **总体** | **264.91** | **28.39** | **+89.3%** |

> 一句话：微调后动作预测误差下降 **89.3%**，6 个关节全部提升。

---

## 2. 概念科普：先分清三种「教机器人」的方式

这一章不堆公式，目标是让机械电气背景的读者在动手前先建立正确的**心智模型**。

### 2.1 三种方式对比

| 维度 | 传统示教编程 | 模仿学习（本方案） | 强化学习 |
|---|---|---|---|
| 怎么教 | 工程师逐点示教 | 演示正确动作给模型看 | 让机器人试错 + 给奖励 |
| 数据需求 | 无需数据（人工编程） | **少量演示**（几十集） | 大量交互试错 |
| 算力需求 | 无 | **单卡可训** | 高（常需集群） |
| 换品种成本 | 重新示教 | 补数据再微调 | 重新训练/调奖励 |
| 依赖专家 | 高 | 中 | 中（奖励设计难） |

**结论**：在制造现场「小批量、多品种、算力有限」的约束下，**模仿学习**是落地阻力最小的路线。

### 2.2 什么是 VLA（视觉-语言-动作模型）

VLA = Vision-Language-Action。它的输入是三类东西：

1. **视觉**：相机拍到的图像（本项目是机械臂的 2 路相机：顶部视角 + 腕部视角）；
2. **语言**：任务指令（如「把物体从 A 放到 B」）；
3. **本体状态**：机械臂当前各关节的角度。

它的输出是**一串连续动作**（每个关节下一步要转多少度、夹爪开合多少）。

> 类比：可以把 VLA 理解为「**能看图、能听懂指令、并且会直接输出关节动作**」的模型，
> 而不是只输出文字或分类标签的普通模型。

### 2.3 什么是 flow matching（流匹配）

SmolVLA 的训练目标叫 flow matching。不需要记公式，只需要抓住两个直觉：

- 训练时：给模型「真值动作」，教它如何把**一团噪声**逐步「流动」成正确动作；
- 推理时：模型从噪声出发，逐步生成一条干净的动作轨迹。

> 类比：就像把一滴墨水滴进水里，观察它如何扩散；训练是学会「逆向」这个过程，
> 把扩散的墨水重新聚回一滴。`loss` 下降，意味着模型越来越会「逆向还原」。

### 2.4 为什么要「微调」而不是从零训练

通用 VLA（SmolVLA）已经见过海量机器人数据，具备基本的视觉理解和动作生成能力。
微调只是让它在这个基础上**再适配到你的具体任务**（本项目是 SO-100 的抓放），
所以几十集数据 + 单卡就够，而不需要从零开始训练（那需要海量数据和 GPU 集群）。

---

## 3. 技术栈与硬件要求

| 组件 | 版本 / 说明 | 作用 |
|---|---|---|
| Python | 3.12（lerobot 要求 ≥3.12） | 运行环境 |
| PyTorch | **2.10.0+cu128**（精确 pin） | 深度学习框架 |
| LeRobot | 0.6.1（`training` + `smolvla` extras） | Hugging Face 机器人训练框架 |
| SmolVLA | `lerobot/smolvla_base`，450M 参数 | 待微调的 VLA 模型 |
| 数据集 | `lerobot/svla_so100_pickplace`，50 集 / 19631 帧 | SO-100 机械臂抓放演示 |
| GPU | Tesla T4 16GB（sm_75，**无 bf16**） | 训练算力 |

### 硬件红线（违反即失败）

| 约束 | 原因 |
|---|---|
| 驱动 570.86 仅支持 **cu128** | PyPI 默认 cu130 wheel 会报 "driver too old" |
| T4 是 sm_75，**不支持 bfloat16** | 训练保持默认 float32，禁 `--policy.dtype=bfloat16` |
| 根分区极小（曾 2.4GB 可用） | pip 缓存/临时目录必须重定向到 `/data` |
| 显存预算 ~9.5GB（另有 5.9GB 被无关进程占用） | batch 4 实测 8.6GB，够用 |

---

## 4. 环境搭建

执行 [setup_env.sh](./setup_env.sh)（幂等，可重复执行）：

```bash
bash setup_env.sh
conda activate rldemo
```

脚本做了四件事，对应四个关键点：

1. **创建 Python 3.12 的 conda 环境 `rldemo`**（lerobot 要求 ≥3.12）。
2. **先装 cu128 的 torch**（关键顺序！）：
   ```bash
   PIP_CONFIG_FILE=/dev/null pip install --index-url https://download.pytorch.org/whl/cu128 \
     "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"
   ```
   必须**精确 pin 带本地版本号 `+cu128`**，并用 `PIP_CONFIG_FILE=/dev/null` 绕开腾讯镜像，
   否则 pip 会因镜像 extra-index 选到 cu130 wheel → 驱动报 "driver too old"（见第 8 章 E1）。
3. **再装 lerobot**（此时 torch 已满足 `>=2.7,<2.12`，不会被重装）：
   ```bash
   pip install "lerobot[training,smolvla]"
   ```
4. **验证**：脚本会自动检查 `torch.cuda.is_available()`、设备名、compute capability `(7,5)`、
   lerobot 版本、ffmpeg 版本、torchcodec 是否可用。**全部通过才算环境就绪**。

> 三个大坑提前预警（详见第 8 章）：**cu128**、**pip 缓存重定向 `/data`**、**torchcodec→pyav**。

---

## 5. 数据准备

### 5.1 下载 + 冒烟校验

执行 [scripts/download_data.py](./scripts/download_data.py)：

```bash
python scripts/download_data.py
```

脚本用 `LeRobotDataset` API 下载并校验（不手写 wget），并做三项冒烟校验：

1. 集数/帧数正确（应打印 `50 / 19631`）；
2. 首帧能解出（验证 AV1 视频解码链路），图像分辨率 `480×640`，动作维度 `(6,)`；
3. 中间帧、末尾帧也能解出（覆盖视频分块边界）。

### 5.2 数据集结构（记牢，后面训练 flag 依赖它）

| 属性 | 值 |
|---|---|
| 集数 / 帧数 | 50 集 / 19631 帧 |
| 任务 | 1 个抓放（pick & place）任务 |
| 自由度 | 6（shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper） |
| 相机 | 2 路：`observation.images.top`、`observation.images.wrist`，480×640，AV1 编码 |
| fps | 30 |

> 关键点：数据集的相机名是 **`top` / `wrist`**，但 smolvla_base 预训练时用的是
> **`camera1` / `camera2` / `camera3`** 这套命名。二者不相交，直接训练会触发
> feature mismatch（见第 8 章 E6），必须在训练/评估时做「改名映射」。

---

## 6. 训练流程

训练分两步：**先冒烟，再全量**。改任何参数前，先跑冒烟。

### 6.1 第一步：冒烟训练（2000 步，约 53 分钟）

执行 [scripts/smoke_train.sh](./scripts/smoke_train.sh)：

```bash
bash scripts/smoke_train.sh
```

冒烟的目的不是出好模型，而是拿到三个关键数：
1. **显存峰值**是否 < 9GB（是否 OOM）；
2. **loss 是否下降**（从 ~0.4 量级开始往下降，说明学习在发生）；
3. **步速**（时间戳估算全量时长）。

本项目冒烟实测：**1.60 s/step、显存 8597 MiB、loss 0.460 → 0.128**。

### 6.2 第二步：全量训练（20000 步，约 9 小时）

执行 [scripts/train_full.sh](./scripts/train_full.sh)：

```bash
nohup bash scripts/train_full.sh > outputs/train_full.log 2>&1 &
```

崩溃/断线后**续训**（不浪费已训进度）：

```bash
RESUME=true bash scripts/train_full.sh
```

监控：

```bash
grep -oE "step:[0-9]+ .* loss:[0-9.]+" outputs/train_full.log | tail
watch -n 5 nvidia-smi
```

### 6.3 训练 flag 逐项解释（重点）

| flag | 值 | 为什么 |
|---|---|---|
| `--dataset.repo_id` | `lerobot/svla_so100_pickplace` | 数据集 |
| `--dataset.video_backend` | `pyav` | 本机 torchcodec 加载失败，用自包含的 av 解 AV1（E4） |
| `--policy.path` | `lerobot/smolvla_base` | 待微调模型 |
| `--policy.device` | `cuda` | 用 GPU |
| `--policy.push_to_hub` | `false` | 不推 HF Hub；否则 config 校验要求 repo_id（E5） |
| `--rename_map` | `{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}` | 数据集相机名 → 模型期望名（E6） |
| `--policy.empty_cameras` | `1` | camera3 用全零图 + mask=0 填充（模型原生支持缺失相机） |
| `--batch_size` | `4` | 显存预算下能跑的最大 batch |
| `--num_workers` | `4` | 数据加载并行度 |
| `--steps` | `20000` | 约 4 epoch（19631 ÷ 4 ≈ 4908 步/epoch） |
| `--save_freq` | `5000` | 每 5000 步存一个 checkpoint |
| `--seed` | `1000` | 可复现 |
| `--wandb.enable` | `false` | 离线环境不开 wandb |

> 两个容易被忽略但**漏掉就报错**的 flag：`rename_map` 和 `empty_cameras`。
> 关于 `freeze_vision_encoder`：smolvla 微调默认**冻结视觉编码器**（省显存），无需手动传。

### 6.4 降级路径（遇到 OOM 时，按顺序每次只改一项）

1. `--batch_size=2`（loss 仍会降，只是等效 batch 变小）；
2. `--policy.gradient_checkpointing=true`（用速度换显存）；
3. 若 `venv-lm` 进程已退出、空闲显存 >14GB，可回 batch 8 加速。

**禁止**：`--policy.dtype=bfloat16`（T4 不支持）；`float16` 未经冒烟验证不要用。

---

## 7. 评估与验收

### 7.1 评估原理

无真机/无匹配仿真条件下，交付证明 = **离线动作预测对比**：把留出集的观测喂给
base 与 finetuned 两个模型，比较各自预测动作与真值动作的逐关节 MSE。执行：

```bash
python scripts/offline_eval.py
```

**退出码约定**：finetuned MSE < base MSE → 退出码 0（成功）；否则非 0。

### 7.2 一个关键的坑：为什么必须用数据集 stats 构造 processor

**不要**用 checkpoint 自带的 processor 去评估 base 模型。原因：

- base 模型自带的 normalizer stats 来自它**预训练数据集**的命名空间
  （键如 `so100.buffer.action.mean`），其动作量级与本数据集的原始电机角度空间不匹配，
  会导致输出动作量级完全错误（~0.3 vs 真值 ~±90）。

**正确做法**（`offline_eval.py` 已实现）：对两个模型统一用
`make_smolvla_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)` 现场构造 processor。
改动评估逻辑时保持这一点（对应 E7）。

### 7.3 指标怎么读

- **MSE 越低越好**：它衡量「预测动作离真值动作有多远」。
- **提升 = (base − finetuned) / base**：正值且越大越好。
- 本项目结果：总体 MSE 264.91 → 28.39（+89.3%），6 关节全部提升。

### 7.4 验收清单

- [ ] `outputs/eval/report.md` 存在且数字完整；
- [ ] `outputs/eval/traj_ep48.png` 可打开（6 关节「真值 vs base vs finetuned」曲线）；
- [ ] finetuned MSE 低于 base（若未低于：检查 loss 是否还在下降 → 续训，而非调参重跑）。

---

## 8. 排障手册

按「症状 → 原因 → 解法」整理，全部来自本项目执行实录（`PLAN.md` §8）。

| # | 症状 | 原因 | 解法 |
|---|---|---|---|
| E1 | torch 装完 CUDA 不可用 / "driver too old" | pip 因腾讯镜像 extra-index 选到 cu130 wheel，驱动 570 只支持 cu128 | 精确 pin `torch==2.10.0+cu128` 并用 `PIP_CONFIG_FILE=/dev/null` 绕开镜像 |
| E2 | `nvidia-nccl-cu12`（322MB）下载到 ~144MB 确定性断流 | 大 wheel 网络传输中断 | `curl -C -` 断点续传拿全文件后本地安装；后续 pip 从缓存恢复 |
| E3 | pip 下载报 `Errno 28`（磁盘满） | pip 缓存/临时目录默认在根分区 `/`（仅 2.4GB） | `export PIP_CACHE_DIR=/data/pipcache TMPDIR=/data/tmp` |
| E4 | torchcodec 加载失败 `undefined symbol: EVP_md2` | 系统 `/lib64/libldap.so.2` 与 OpenSSL 符号冲突 | 卸载 torchcodec，全程 `--dataset.video_backend=pyav`（av 自带 libdav1d） |
| E5 | `lerobot-train` 报 config 校验错误 | 默认 `push_to_hub=true`，要求 repo_id | 加 `--policy.push_to_hub=false` |
| E6 | 训练报 feature mismatch | 数据集相机名 `top/wrist` 与模型期望 `camera1/2/3` 不相交 | `--rename_map` 映射 + `--policy.empty_cameras=1` |
| E7 | 评估时动作量级错误（~0.3 vs ~±90） | 误用 checkpoint 自带 processor（stats 来自预训练命名空间） | 用 `dataset_stats=ds.meta.stats` 现场构造 processor |
| E8 | 评估报 IndexError / matplotlib 缺失 / HF 偶发网络不可达 | 子集索引需用相对索引；环境未装 matplotlib；HF 网络抖动 | 用 `meta.episodes` 的 length 算相对 offset；补装 matplotlib；重试（权重已本地缓存） |

> 遇到新问题，先读错误信息 → 查本表 → 应用对策重试一次 → 仍失败则完整粘贴错误给团队。

---

## 9. 附录：实测指标

以下均为本项目在本机 Tesla T4（16GB）上的实测值，供容量规划参考：

| 指标 | 值 |
|---|---|
| 步速 | 1.60 s/step（batch 4，fp32，冻结视觉编码器，512×512 双相机） |
| 显存 | 8597 MiB 稳定（共 15360 MiB） |
| 冒烟 loss | 0.460 → 0.128（2000 步 / 0.41 epoch） |
| 全量 loss | 0.495 → 0.077（20000 步 / 4.08 epoch / 8h54m） |
| 磁盘 | conda env 9.8GB + 每 checkpoint ~800MB×2 文件 |
| 评估帧数 | 1877 帧（ep45–49） |
| 最终 MSE | base 264.91 → finetuned 28.39（↓89.3%） |
| checkpoint 大小 | model.safetensors 865MB（< 3GB） |

**磁盘红线**：`df -h /data` 可用 <5GB 时，删除 `checkpoints/` 下非 `last` 的旧步数目录。

---

> 本手册数据来源：`AGENTS.md`、`PLAN.md`、`setup_env.sh`、`scripts/download_data.py`、
> `scripts/smoke_train.sh`、`scripts/train_full.sh`、`scripts/offline_eval.py`、
> `outputs/eval/report.md`，全部为项目真实执行结果。
