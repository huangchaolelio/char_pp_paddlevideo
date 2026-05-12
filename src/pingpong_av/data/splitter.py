"""按源视频 ID 分层划分数据 (章程 IV 不可妥协的核心).

核心约束:
- **同一 source_video_id 的所有片段必须落在同一 split** (训练/验证/测试).
  否则模型会在测试集上"看到"训练时已看过的源视频, 导致评估失真 (章程 IV).
- 划分种子来自配置, 同种子可复现 (FR-018, 章程 II).
- 划分类别比例尽量贴近期望 (例如 0.7/0.15/0.15), 但不强求精确;
  优先级是 "同源不跨 split" > "比例精确".

提供两个公开 API:
- :func:`split_by_video_id` — 按比例做随机划分 (受 ratios + seed 控制)
- :func:`verify_no_leakage` — 章程 IV 自动化闸门; 扫描三份 split, 发现重叠即抛错

不在本模块的范围:
- 实际产出 list 文件 (那是 ``data.list_writer`` 的职责).
- 类别表 / 来源数据集差异处理 (那是 ``data.public_datasets`` 的职责).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from pingpong_av.experiment.run_manifest import ConstitutionViolation
from pingpong_av.utils.logging import get_logger

__all__ = [
    "VideoClip",
    "Splits",
    "SplitName",
    "split_by_video_id",
    "verify_no_leakage",
]

_log = get_logger(__name__)

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class VideoClip:
    """一段被标注的乒乓球片段 (与 data-model.md 中 VideoClip 实体对齐).

    本类是**划分阶段**的内部数据载体, 含有比 list 行更多的信息 (clip_id /
    source_video_id / start/end). list_writer 在落盘时只把 path + label_id 写入
    PaddleVideo 兼容的 list.txt, 把完整对象写入 meta.jsonl 方便回溯.
    """

    clip_id: str
    source_video_id: str   # **章程 IV 关键字段**: 划分时的去重键
    path: str              # 相对 data/clips/ 的路径 (或绝对路径)
    label_id: int
    start_sec: float | None = None
    end_sec: float | None = None
    split: SplitName | None = None  # 由 split_by_video_id 设置


@dataclass(frozen=True)
class Splits:
    """三份划分结果, 字段名与 data-model.md DatasetSplit 对齐."""

    train: list[VideoClip]
    val: list[VideoClip]
    test: list[VideoClip]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def all_clips(self) -> list[VideoClip]:
        return [*self.train, *self.val, *self.test]


# --------------------------------------------------------------------------------------
# 划分: 按 source_video_id 分层
# --------------------------------------------------------------------------------------


def split_by_video_id(
    clips: Iterable[VideoClip],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> Splits:
    """按 ``source_video_id`` 把片段分配到 train/val/test 三份, **同源不跨 split**.

    算法:
        1. 把所有片段按 ``source_video_id`` 分组;
        2. 用 ``seed`` 打乱"源视频列表"的顺序;
        3. 按 ratios.train / ratios.val / ratios.test 顺序切分**源视频**到三个 bucket;
        4. 每个 source_video_id 对应的全部片段进入同一 bucket.

    参数:
        clips: 可迭代的 VideoClip 集合.
        ratios: 含 ``train`` / ``val`` / ``test`` 三键的浮点字典, 三者之和应 ≈ 1.0;
                精确 floor 切分, 余数全部归 ``train``.
        seed: 随机种子 (FR-018, 章程 II).

    返回:
        :class:`Splits`, 每个 VideoClip 的 ``split`` 字段已被填充.

    抛出:
        ValueError: ratios 缺键 / 负数 / 总和偏离 1.0 太多.
    """
    _validate_ratios(ratios)

    # 1) 按 source_video_id 分组 (保持原 clip 顺序在组内, 便于调试)
    groups: dict[str, list[VideoClip]] = defaultdict(list)
    for clip in clips:
        groups[clip.source_video_id].append(clip)

    if not groups:
        raise ValueError("split_by_video_id 收到空 clips 集合")

    # 2) 打乱源视频列表 (deterministic 给定 seed)
    video_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    n_videos = len(video_ids)

    # 3) 按 ratio 把源视频切到三个 bucket
    n_val = int(n_videos * ratios["val"])
    n_test = int(n_videos * ratios["test"])
    n_train = n_videos - n_val - n_test  # 余数归 train, 保证 train 至少有 1
    assert n_train >= 0, f"内部错误: ratios 导致 train 数量为负 (n={n_videos}, val={n_val}, test={n_test})"

    train_ids = set(video_ids[:n_train])
    val_ids = set(video_ids[n_train:n_train + n_val])
    test_ids = set(video_ids[n_train + n_val:])

    # 4) 把片段铺到三个列表; 通过 dataclasses.replace 写入 split 字段
    from dataclasses import replace

    train_clips: list[VideoClip] = []
    val_clips: list[VideoClip] = []
    test_clips: list[VideoClip] = []
    for vid in video_ids:
        if vid in train_ids:
            target, name = train_clips, "train"
        elif vid in val_ids:
            target, name = val_clips, "val"
        else:
            target, name = test_clips, "test"
        for c in groups[vid]:
            target.append(replace(c, split=name))  # type: ignore[arg-type]

    splits = Splits(train=train_clips, val=val_clips, test=test_clips)
    _log.info(
        "split_by_video_id done",
        extra={
            "n_videos": n_videos,
            "videos_per_split": {"train": n_train, "val": n_val, "test": len(test_ids)},
            "clips_per_split": splits.counts(),
            "seed": seed,
        },
    )
    return splits


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    required = ("train", "val", "test")
    missing = [k for k in required if k not in ratios]
    if missing:
        raise ValueError(f"ratios 缺少键: {missing}; 必须包含 {required}")
    for k in required:
        v = ratios[k]
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"ratios.{k} 必须是非负数, 实际为 {v!r}")
    total = sum(ratios[k] for k in required)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"ratios 之和应 ≈ 1.0, 实际为 {total:.4f}")


# --------------------------------------------------------------------------------------
# 章程 IV 自动化闸门
# --------------------------------------------------------------------------------------


def verify_no_leakage(splits: Splits) -> None:
    """扫描三份 split, 任意两份之间存在共同 source_video_id 即抛错.

    这是**章程 IV 不可协商**的硬约束闸门. 由 ``pp data-prepare`` (T039) 在写入 list 文件
    **之前**调用, 失败时 ``pp data-prepare`` 退出码 3, 数据划分**不**会落盘.

    抛出:
        :class:`ConstitutionViolation`: 检测到泄漏 (含具体重叠的 source_video_id).
    """
    train_ids = {c.source_video_id for c in splits.train}
    val_ids = {c.source_video_id for c in splits.val}
    test_ids = {c.source_video_id for c in splits.test}

    overlaps: dict[str, set[str]] = {}
    if (tv := train_ids & val_ids):
        overlaps["train ∩ val"] = tv
    if (tt := train_ids & test_ids):
        overlaps["train ∩ test"] = tt
    if (vt := val_ids & test_ids):
        overlaps["val ∩ test"] = vt

    if overlaps:
        # 取每对最多 5 个示例 ID, 避免错误信息过长
        details = []
        for pair, ids in overlaps.items():
            sample = sorted(ids)[:5]
            more = "" if len(ids) <= 5 else f" (... 共 {len(ids)} 个)"
            details.append(f"{pair}: {sample}{more}")
        raise ConstitutionViolation(
            "数据划分泄漏 (章程 IV 不可协商). 同一 source_video_id 出现在多个 split:\n  "
            + "\n  ".join(details)
            + "\n请检查标注或调整 split_strategy 后重试."
        )

    _log.info(
        "no-leakage check passed",
        extra={
            "videos_train": len(train_ids),
            "videos_val": len(val_ids),
            "videos_test": len(test_ids),
        },
    )
