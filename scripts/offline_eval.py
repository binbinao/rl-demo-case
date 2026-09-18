#!/usr/bin/env python
"""P3: 离线动作预测对比评估 — 见 PLAN.md §4 P3。

无真机/无匹配仿真条件下的交付证明：把留出集（episode 45–49）的观测喂给
两个策略（base vs finetuned），比较预测动作与数据集真值动作的逐关节 MSE。

API 模式逐行对照 lerobot/smolvla_base 模型卡 Quick start 与 PLAN.md §2.5。
"""
import os
import sys

PROJECT_ROOT = "/data/robinji/rl-demo-case"
os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(PROJECT_ROOT, "data"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

BASE_MODEL = "lerobot/smolvla_base"
FINETUNED = os.path.join(
    PROJECT_ROOT, "outputs/train/smolvla_pickplace/checkpoints/last/pretrained_model"
)
EVAL_EPISODES = [45, 46, 47, 48, 49]  # 留出集：训练未做集划分，固定取最后 5 集
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

RENAME_MAP = {  # 数据集相机名 → 策略期望名（与训练时 --rename_map 一致）
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def load_policy(model_id: str, device: torch.device, dataset_stats):
    """加载策略 + 用「数据集 stats」现场构造 processor（两个模型同一套，公平对比）。

    不用 checkpoint 自带 processor：base 的 normalizer stats 来自其预训练数据集
    （键如 so100.buffer.action.mean），与本数据集的原始电机角度空间不匹配，
    会导致输出动作量级错误。训练脚本对 finetuned 正是传 dataset_stats 构造的。
    """
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    policy.config.device = str(device)
    preprocess, postprocess = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=dataset_stats
    )
    return policy, preprocess, postprocess




def evaluate(policy, preprocess, postprocess, ds, episode_index: int, device):
    """对一个 episode 逐帧推理，返回 (pred_actions[T,6], gt_actions[T,6])。"""
    # ds 以 episodes=[45..49] 子集加载（len=各集帧数和），meta 里的 from/to 是
    # 全局索引，不能直接用；子集帧顺序即 EVAL_EPISODES 顺序（已实测验证），
    # 按该顺序累计各集长度得到相对索引区间。
    ep_lens = [ds.meta.episodes["length"][e] for e in EVAL_EPISODES]
    offset = sum(ep_lens[: EVAL_EPISODES.index(episode_index)])
    n_frames = ds.meta.episodes["length"][episode_index]

    preds, gts = [], []
    with torch.inference_mode():
        for idx in range(offset, offset + n_frames):
            frame = dict(ds[idx])
            gt = frame["action"].clone()
            # 与训练一致：把数据集相机名映射到策略期望名
            for old, new in RENAME_MAP.items():
                if old in frame:
                    frame[new] = frame.pop(old)
            batch = preprocess(frame)
            action = policy.select_action(batch)
            action = postprocess(action)
            preds.append(action.detach().cpu().reshape(-1)[: gt.shape[0]])
            gts.append(gt)
    return torch.stack(preds), torch.stack(gts)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = LeRobotDataset(
        "lerobot/svla_so100_pickplace",
        episodes=EVAL_EPISODES,
        video_backend="pyav",
    )
    print(f"eval frames: {len(ds)} (episodes {EVAL_EPISODES})")

    results = {}
    traj_cache = {}
    for name, model_id in [("base", BASE_MODEL), ("finetuned", FINETUNED)]:
        print(f"\n=== evaluating {name}: {model_id} ===")
        policy, preprocess, postprocess = load_policy(model_id, device, ds.meta.stats)
        total_se = torch.zeros(len(JOINT_NAMES))
        total_frames = 0
        for ep in EVAL_EPISODES:
            preds, gts = evaluate(policy, preprocess, postprocess, ds, ep, device)
            se = ((preds - gts) ** 2).sum(dim=0)
            total_se += se
            total_frames += preds.shape[0]
            if name not in traj_cache:
                traj_cache[name] = {}
            traj_cache[name][ep] = (preds, gts)
            print(f"  episode {ep}: frames={preds.shape[0]} "
                  f"mse={se.sum().item() / (preds.shape[0] * len(JOINT_NAMES)):.4f}")
        results[name] = (total_se / total_frames).tolist()
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mse_base = sum(results["base"]) / len(JOINT_NAMES)
    mse_ft = sum(results["finetuned"]) / len(JOINT_NAMES)

    # 报告
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs/eval"), exist_ok=True)
    report = os.path.join(PROJECT_ROOT, "outputs/eval/report.md")
    with open(report, "w") as f:
        f.write("# 离线动作预测对比评估（P3）\n\n")
        f.write(f"- 留出集：episodes {EVAL_EPISODES}（训练未做集划分，固定取最后 5 集）\n")
        f.write(f"- 指标：逐关节 MSE（预测动作 vs 数据集真值动作），帧数加权\n\n")
        f.write("| 关节 | base MSE | finetuned MSE | 提升 |\n|---|---|---|---|\n")
        for i, jn in enumerate(JOINT_NAMES):
            b, t = results["base"][i], results["finetuned"][i]
            f.write(f"| {jn} | {b:.4f} | {t:.4f} | {(b - t) / b * 100:+.1f}% |\n")
        b, t = mse_base, mse_ft
        f.write(f"| **overall** | **{b:.4f}** | **{t:.4f}** | **{(b - t) / b * 100:+.1f}%** |\n")

    # 轨迹图：episode 48，6 关节 真值 vs base vs finetuned
    ep = 48
    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    gt = traj_cache["base"][ep][1].numpy()
    pb = traj_cache["base"][ep][0].numpy()
    pf = traj_cache["finetuned"][ep][0].numpy()
    for i, ax in enumerate(axes):
        ax.plot(gt[:, i], "k-", label="ground truth", linewidth=1.5)
        ax.plot(pb[:, i], "r--", label=f"base (MSE {results['base'][i]:.3f})", linewidth=1)
        ax.plot(pf[:, i], "g-", label=f"finetuned (MSE {results['finetuned'][i]:.3f})", linewidth=1)
        ax.set_ylabel(JOINT_NAMES[i])
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title(
        f"Episode {ep}: predicted vs ground-truth actions\n"
        f"overall MSE — base: {mse_base:.4f}, finetuned: {mse_ft:.4f}"
    )
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    png = os.path.join(PROJECT_ROOT, "outputs/eval/traj_ep48.png")
    fig.savefig(png, dpi=120)

    print(f"\nreport: {report}")
    print(f"trajectory plot: {png}")
    print(f"\noverall MSE — base: {mse_base:.4f}, finetuned: {mse_ft:.4f}")
    if mse_ft < mse_base:
        print("P3 PASS: finetuned 优于 base")
        return 0
    print("P3 FAIL: finetuned 未优于 base（检查 loss 是否仍在下降，考虑续训）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
