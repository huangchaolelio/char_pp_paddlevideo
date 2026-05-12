"""splitter.py 单元测试 (章程 IV 自动化闸门).

覆盖范围:
- ``split_by_video_id``:
  * 同一 source_video_id 的所有片段必须落在同一 split (核心硬约束)
  * 划分受 seed 控制, 同种子可复现
  * ratios 校验 (缺键 / 负数 / 总和偏离)
- ``verify_no_leakage``:
  * 干净的划分通过
  * 故意构造跨 split 的 source_video_id 时抛 ConstitutionViolation
  * 错误信息中含具体重叠的 source_video_id
"""

from __future__ import annotations

import pytest

from pingpong_av.data.splitter import (
    Splits,
    VideoClip,
    split_by_video_id,
    verify_no_leakage,
)
from pingpong_av.experiment.run_manifest import ConstitutionViolation


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _clip(clip_id: str, src: str, label: int = 0) -> VideoClip:
    return VideoClip(
        clip_id=clip_id,
        source_video_id=src,
        path=f"/tmp/{clip_id}.mp4",
        label_id=label,
    )


@pytest.fixture()
def many_clips_with_groups() -> list[VideoClip]:
    """20 个源视频, 每个 3 个片段; 共 60 个片段."""
    out: list[VideoClip] = []
    for vid in range(20):
        src = f"video_{vid:03d}"
        for cid in range(3):
            out.append(_clip(f"{src}_clip{cid}", src, label=vid % 5))
    return out


# --------------------------------------------------------------------------------------
# split_by_video_id: 核心不变量
# --------------------------------------------------------------------------------------


def test_no_source_video_crosses_splits(many_clips_with_groups: list[VideoClip]) -> None:
    """章程 IV 不可妥协: 同一 source_video_id 必须只在一个 split."""
    splits = split_by_video_id(
        many_clips_with_groups,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        seed=2026,
    )
    train_ids = {c.source_video_id for c in splits.train}
    val_ids = {c.source_video_id for c in splits.val}
    test_ids = {c.source_video_id for c in splits.test}
    assert not (train_ids & val_ids), f"train ∩ val 有源视频: {train_ids & val_ids}"
    assert not (train_ids & test_ids), f"train ∩ test 有源视频: {train_ids & test_ids}"
    assert not (val_ids & test_ids), f"val ∩ test 有源视频: {val_ids & test_ids}"


def test_all_clips_assigned(many_clips_with_groups: list[VideoClip]) -> None:
    splits = split_by_video_id(many_clips_with_groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=2026)
    total = sum(splits.counts().values())
    assert total == len(many_clips_with_groups)


def test_split_field_set(many_clips_with_groups: list[VideoClip]) -> None:
    splits = split_by_video_id(many_clips_with_groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=2026)
    for c in splits.train: assert c.split == "train"
    for c in splits.val:   assert c.split == "val"
    for c in splits.test:  assert c.split == "test"


def test_same_seed_is_reproducible(many_clips_with_groups: list[VideoClip]) -> None:
    """同种子两次划分必须完全一致 (FR-018, 章程 II)."""
    s1 = split_by_video_id(many_clips_with_groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    s2 = split_by_video_id(many_clips_with_groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert [c.clip_id for c in s1.train] == [c.clip_id for c in s2.train]
    assert [c.clip_id for c in s1.val]   == [c.clip_id for c in s2.val]
    assert [c.clip_id for c in s1.test]  == [c.clip_id for c in s2.test]


def test_different_seeds_differ() -> None:
    clips = [_clip(f"v{v}_c{c}", f"v{v}") for v in range(50) for c in range(2)]
    s1 = split_by_video_id(clips, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    s2 = split_by_video_id(clips, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=2)
    # 不强求 100% 不同, 但两次划分的 train video set 不应完全相同
    t1 = {c.source_video_id for c in s1.train}
    t2 = {c.source_video_id for c in s2.train}
    assert t1 != t2, "不同 seed 划分结果应有差异"


def test_empty_clips_raises() -> None:
    with pytest.raises(ValueError, match="空"):
        split_by_video_id([], ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)


# --------------------------------------------------------------------------------------
# ratios 校验
# --------------------------------------------------------------------------------------


def test_ratios_missing_key_raises(many_clips_with_groups: list[VideoClip]) -> None:
    with pytest.raises(ValueError, match="缺少键"):
        split_by_video_id(many_clips_with_groups, ratios={"train": 0.8, "val": 0.2}, seed=0)


def test_ratios_negative_raises(many_clips_with_groups: list[VideoClip]) -> None:
    with pytest.raises(ValueError, match="非负"):
        split_by_video_id(
            many_clips_with_groups,
            ratios={"train": 0.7, "val": -0.1, "test": 0.4},
            seed=0,
        )


def test_ratios_sum_must_be_one(many_clips_with_groups: list[VideoClip]) -> None:
    with pytest.raises(ValueError, match="≈ 1.0"):
        split_by_video_id(
            many_clips_with_groups,
            ratios={"train": 0.5, "val": 0.2, "test": 0.2},   # 0.9
            seed=0,
        )


# --------------------------------------------------------------------------------------
# verify_no_leakage: 章程 IV 闸门
# --------------------------------------------------------------------------------------


def test_verify_no_leakage_passes_clean_split(many_clips_with_groups: list[VideoClip]) -> None:
    splits = split_by_video_id(many_clips_with_groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=2026)
    # 不应抛错
    verify_no_leakage(splits)


def test_verify_no_leakage_detects_train_test_overlap() -> None:
    """构造 train/test 共有的 source_video_id, 必须抛 ConstitutionViolation."""
    splits = Splits(
        train=[_clip("a1", "video_X"), _clip("a2", "video_A")],
        val=[_clip("b1", "video_B")],
        test=[_clip("c1", "video_X"), _clip("c2", "video_C")],  # video_X 出现在 train 也出现在 test
    )
    with pytest.raises(ConstitutionViolation, match="train ∩ test"):
        verify_no_leakage(splits)


def test_verify_no_leakage_detects_train_val_overlap() -> None:
    splits = Splits(
        train=[_clip("a1", "video_X")],
        val=[_clip("b1", "video_X")],   # X 在 train 也在 val
        test=[_clip("c1", "video_C")],
    )
    with pytest.raises(ConstitutionViolation, match="train ∩ val"):
        verify_no_leakage(splits)


def test_verify_no_leakage_detects_val_test_overlap() -> None:
    splits = Splits(
        train=[_clip("a1", "video_A")],
        val=[_clip("b1", "video_X")],
        test=[_clip("c1", "video_X")],   # X 同时出现在 val 和 test
    )
    with pytest.raises(ConstitutionViolation, match="val ∩ test"):
        verify_no_leakage(splits)


def test_verify_no_leakage_lists_overlapping_ids_in_message() -> None:
    """错误消息必须包含具体的重叠 source_video_id, 便于排查."""
    splits = Splits(
        train=[_clip("a1", "shared_vid_42")],
        val=[],
        test=[_clip("c1", "shared_vid_42")],
    )
    with pytest.raises(ConstitutionViolation) as excinfo:
        verify_no_leakage(splits)
    assert "shared_vid_42" in str(excinfo.value)


def test_verify_no_leakage_truncates_long_overlap_lists() -> None:
    """重叠 > 5 个时, 错误消息应只列前 5 + '共 N 个' 提示."""
    train = [_clip(f"t{i}", f"vid_{i:02d}") for i in range(10)]
    test = [_clip(f"e{i}", f"vid_{i:02d}") for i in range(10)]   # 全部重叠
    splits = Splits(train=train, val=[], test=test)
    with pytest.raises(ConstitutionViolation) as excinfo:
        verify_no_leakage(splits)
    msg = str(excinfo.value)
    assert "共 10 个" in msg
