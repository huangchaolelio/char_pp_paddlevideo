"""post_process.apply_threshold_and_merge 单元测试 (FR-014, R5).

测试场景 (覆盖 tasks.md T060 列出的三类):
  (a) 全部高置信度同类连续 → 合并为一段
  (b) 夹杂低置信窗口 → 中间出现 unknown 段
  (c) merge_gap_sec 边界情况

外加:
  - 不同 label 不合并
  - 失败窗口 (error 非空) 视为 unknown
  - 段间无空隙 / 无重叠 (data-model.md 硬约束)
  - video_duration_sec 末端补齐
  - 输入校验 (conf_threshold 越界)
"""

from __future__ import annotations

import numpy as np
import pytest

from pingpong_av.inference.post_process import (
    TimelineSegment,
    apply_threshold_and_merge,
)
from pingpong_av.inference.sliding_window import Window, WindowResult


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _wr(idx: int, start: float, end: float, label_id: int, label: str, conf: float,
        error: str | None = None, n_classes: int = 3) -> WindowResult:
    return WindowResult(
        window=Window(index=idx, start_sec=start, end_sec=end),
        label_id=label_id,
        label_name=label,
        confidence=conf,
        scores=np.zeros(n_classes, dtype=np.float32),
        error=error,
    )


# --------------------------------------------------------------------------------------
# (a) 全部高置信度同类连续 → 合并为一段
# --------------------------------------------------------------------------------------


def test_all_same_label_high_confidence_merges_to_one_segment() -> None:
    windows = [
        _wr(0, 0.0, 2.0, 1, "serve", 0.9),
        _wr(1, 1.0, 3.0, 1, "serve", 0.85),
        _wr(2, 2.0, 4.0, 1, "serve", 0.95),
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=1.0)
    assert len(segs) == 1
    s = segs[0]
    assert s.label == "serve"
    assert s.label_id == 1
    assert s.start == pytest.approx(0.0)
    assert s.end == pytest.approx(4.0)
    assert s.n_windows == 3
    assert s.confidence == pytest.approx((0.9 + 0.85 + 0.95) / 3)


# --------------------------------------------------------------------------------------
# (b) 夹杂低置信窗口 → 中间出现 unknown 段
# --------------------------------------------------------------------------------------


def test_low_confidence_middle_window_becomes_unknown_segment() -> None:
    windows = [
        _wr(0, 0.0, 2.0, 1, "serve", 0.9),
        _wr(1, 1.0, 3.0, 1, "serve", 0.3),     # below threshold → unknown
        _wr(2, 2.0, 4.0, 1, "serve", 0.85),
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=1.0)
    # 应得到 3 段: serve / unknown / serve
    labels = [s.label for s in segs]
    assert labels == ["serve", "unknown", "serve"]
    # unknown 段的边界由低置信窗口决定
    assert segs[1].label_id == -1
    # 段之间无空隙、无重叠
    for i in range(len(segs) - 1):
        assert segs[i].end == pytest.approx(segs[i + 1].start)


def test_failed_window_treated_as_unknown() -> None:
    windows = [
        _wr(0, 0.0, 2.0, 1, "serve", 0.9),
        _wr(1, 1.0, 3.0, -1, "unknown", 0.0, error="decode failure"),
        _wr(2, 2.0, 4.0, 1, "serve", 0.85),
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=1.0)
    assert [s.label for s in segs] == ["serve", "unknown", "serve"]


# --------------------------------------------------------------------------------------
# (c) merge_gap_sec 边界
# --------------------------------------------------------------------------------------


def test_merge_gap_within_threshold_merges() -> None:
    """两个同类窗口之间有 0.5s 空隙, merge_gap_sec=1.0 → 合并."""
    windows = [
        _wr(0, 0.0, 1.0, 1, "serve", 0.9),
        _wr(1, 1.5, 2.5, 1, "serve", 0.9),  # gap=0.5
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=1.0)
    # 应合并为一段; 但在它们之间会插入空隙填充段? 看实现:
    # 当前逻辑: 同 label + gap ≤ merge_gap_sec → 直接合并 (cur_end 推进到 nxt.end), 不留空隙
    # 因此应得到 1 段, 从 0 到 2.5
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(0.0)
    assert segs[0].end == pytest.approx(2.5)
    assert segs[0].label == "serve"


def test_merge_gap_exceeds_threshold_splits() -> None:
    """两个同类窗口之间有 2s 空隙, merge_gap_sec=1.0 → 不合并, 中间填 unknown."""
    windows = [
        _wr(0, 0.0, 1.0, 1, "serve", 0.9),
        _wr(1, 3.0, 4.0, 1, "serve", 0.9),  # gap=2.0 > 1.0
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=1.0)
    # 因为 gap > merge_gap_sec, 不合并; 中间空隙由 _fill_gaps_with_unknown 填 unknown
    assert [s.label for s in segs] == ["serve", "unknown", "serve"]
    assert segs[1].start == pytest.approx(1.0)
    assert segs[1].end == pytest.approx(3.0)


# --------------------------------------------------------------------------------------
# 不同 label 不合并
# --------------------------------------------------------------------------------------


def test_different_labels_dont_merge() -> None:
    windows = [
        _wr(0, 0.0, 2.0, 1, "serve", 0.9),
        _wr(1, 1.0, 3.0, 2, "forehand", 0.85),
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=1.0)
    assert len(segs) == 2
    assert segs[0].label == "serve"
    assert segs[1].label == "forehand"


# --------------------------------------------------------------------------------------
# 段间无空隙、无重叠 (data-model.md 硬约束)
# --------------------------------------------------------------------------------------


def test_no_gaps_and_no_overlaps_invariant() -> None:
    """混合多种情况: 段之间必须满足 segments[i].end == segments[i+1].start."""
    windows = [
        _wr(0, 0.0, 2.0, 1, "serve", 0.9),
        _wr(1, 2.5, 4.5, 2, "forehand", 0.85),  # gap of 0.5s, different label
        _wr(2, 5.0, 7.0, 2, "forehand", 0.3),   # below threshold
        _wr(3, 7.5, 9.5, 1, "serve", 0.9),
    ]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=0.0)
    # 验证无空隙
    for i in range(len(segs) - 1):
        assert segs[i].end == pytest.approx(segs[i + 1].start), (
            f"segments[{i}].end={segs[i].end} != segments[{i+1}].start={segs[i+1].start}"
        )
    # 验证无重叠
    for s in segs:
        assert s.end > s.start


def test_first_segment_starts_at_zero_with_unknown_padding() -> None:
    """首窗 start>0 时, 应前置 unknown 段从 0 开始."""
    windows = [_wr(0, 5.0, 7.0, 1, "serve", 0.9)]
    segs = apply_threshold_and_merge(windows, conf_threshold=0.5, merge_gap_sec=0.0)
    assert segs[0].start == pytest.approx(0.0)
    assert segs[0].label == "unknown"
    assert segs[0].end == pytest.approx(5.0)
    assert segs[1].label == "serve"


def test_last_segment_padded_to_video_duration() -> None:
    """如果给了 video_duration_sec, 末端必须补到该时长."""
    windows = [_wr(0, 0.0, 2.0, 1, "serve", 0.9)]
    segs = apply_threshold_and_merge(
        windows, conf_threshold=0.5, merge_gap_sec=0.0,
        video_duration_sec=10.0,
    )
    assert segs[-1].end == pytest.approx(10.0)
    assert segs[-1].label == "unknown"


# --------------------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------------------


def test_conf_threshold_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_threshold_and_merge([], conf_threshold=1.5, merge_gap_sec=1.0)


def test_negative_merge_gap_raises() -> None:
    with pytest.raises(ValueError, match="merge_gap_sec"):
        apply_threshold_and_merge([], conf_threshold=0.5, merge_gap_sec=-0.1)


def test_empty_input_with_duration_returns_one_unknown_segment() -> None:
    segs = apply_threshold_and_merge([], conf_threshold=0.5, merge_gap_sec=1.0,
                                     video_duration_sec=10.0)
    assert len(segs) == 1
    assert segs[0].label == "unknown"
    assert segs[0].end == pytest.approx(10.0)


# --------------------------------------------------------------------------------------
# min_segment_sec
# --------------------------------------------------------------------------------------


def test_short_segment_absorbed_by_previous() -> None:
    """min_segment_sec=2.0 时, 中间 0.5s 的 forehand 段会被前段吸收."""
    windows = [
        _wr(0, 0.0, 3.0, 1, "serve", 0.9),
        _wr(1, 3.0, 3.5, 2, "forehand", 0.9),  # 0.5s 短段
        _wr(2, 3.5, 6.5, 1, "serve", 0.9),
    ]
    segs = apply_threshold_and_merge(
        windows, conf_threshold=0.5, merge_gap_sec=0.0,
        min_segment_sec=2.0,
    )
    # forehand 段应被吸收, 留下 serve 段(s)
    assert all(s.label == "serve" for s in segs), [s.label for s in segs]
