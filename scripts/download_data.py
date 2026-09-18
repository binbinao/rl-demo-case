#!/usr/bin/env python
"""P1: 下载 svla_so100_pickplace 数据集 + 冒烟校验（帧解码、统计、打印样例）— 见 PLAN.md §4 P1。

视频后端固定为 pyav：本机系统 libldap 与 OpenSSL 符号冲突导致 torchcodec 无法加载
（/lib64/libldap.so.2: undefined symbol: EVP_md2），而 av 15.1.0 wheel 自带
libdav1d（AV1 解码）与全部 ffmpeg 库，完全自包含。
"""
import os
import sys

PROJECT_ROOT = "/data/robinji/rl-demo-case"
os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(PROJECT_ROOT, "data"))
# 如遇 HF 限速，取消下行注释走镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> int:
    ds = LeRobotDataset("lerobot/svla_so100_pickplace", video_backend="pyav")
    total_episodes = ds.meta.total_episodes
    total_frames = len(ds)
    print(f"episodes: {total_episodes}")
    print(f"frames:   {total_frames}")
    assert total_episodes == 50, f"expected 50 episodes, got {total_episodes}"
    assert total_frames == 19631, f"expected 19631 frames, got {total_frames}"

    # 首帧必须能解出（验证 AV1 解码链路）
    frame = ds[0]
    summary = {
        k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)
        for k, v in frame.items()
    }
    print("frame[0]:", summary)
    assert frame["observation.images.top"].shape[-2:] == (480, 640), "top 图像分辨率不符"
    assert frame["observation.images.wrist"].shape[-2:] == (480, 640), "wrist 图像分辨率不符"
    assert frame["action"].shape == (6,), "action 维度不符"

    print("task:", ds.meta.tasks)

    # 中间与末尾各抽一帧再验（视频分块边界）
    for idx in (total_frames // 2, total_frames - 1):
        f = ds[idx]
        assert f["observation.images.top"].shape[-2:] == (480, 640), f"帧 {idx} 解码失败"
    print("sampled frames [0, mid, last] decode OK (pyav/dav1d)")

    print("P1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
