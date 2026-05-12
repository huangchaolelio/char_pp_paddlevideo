"""滑窗结果后处理: 阈值过滤 + 同类合并 (FR-014, research.md R5).

输入: :class:`WindowResult` 列表 (来自 :mod:`sliding_window`).
输出: :class:`TimelineSegment` 列表 (与 data-model.md ``video-timeline-v1`` 一致).

合并规则:
  1. ``confidence < conf_threshold`` 的窗口标签置为 ``"unknown"`` (label_id=-1);
  2. 相邻窗口若 (a) label 相同 且 (b) 时间间隔 ≤ ``merge_gap_sec``, 合并为一段;
  3. 段的 ``confidence`` = 该段所有底层窗口的平均;
  4. 输出的段必须**无空隙、无重叠** (data-model.md 硬约束):
     - 第一段从 0.0 开始 (若首窗 start>0, 前置一段 unknown);
     - 段之间无 gap (用 unknown 填空隙);
     - 最后一段补到视频末端.

不在本模块的范围:
- 滑窗采样 / 推理 (那是 :mod:`sliding_window`).
- JSON 写盘 (那是 :mod:`visualizer.write_timeline_json`).
"""

from __future__ import annotations

from dataclasses import dataclass

from pingpong_av.inference.sliding_window import WindowResult
from pingpong_av.utils.logging import get_logger

__all__ = ["TimelineSegment", "apply_threshold_and_merge"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class TimelineSegment:
    """data-model.md 中 TimelineSegment 实体的 Python 表示."""

    start: float       # 起始秒
    end: float         # 结束秒, 必须 > start
    label: str         # ActionClass.name 或 "unknown"
    label_id: int      # ActionClass.id; unknown 对应 -1
    confidence: float  # 段内所有底层窗口的均值 (∈ [0, 1])
    n_windows: int     # 该段合并自多少个滑窗 (≥1; 0 仅在合成的填充段)


def apply_threshold_and_merge(
    window_results: list[WindowResult],
    *,
    conf_threshold: float,
    merge_gap_sec: float,
    min_segment_sec: float = 0.0,
    video_duration_sec: float | None = None,
) -> list[TimelineSegment]:
    """阈值过滤 + 同类合并; 产出无空隙、无重叠的 :class:`TimelineSegment` 列表.

    参数:
        window_results: 已按时间顺序排好的窗口结果. 失败窗口 (error 非空 / label_id=-1)
                        视为 ``unknown``.
        conf_threshold: 单窗口的最高概率低于此值时, 标签置 ``unknown`` (R5).
        merge_gap_sec: 同 label 窗口间允许的最大时间间隔; 超过此间隔即使 label 相同
                        也单独成段 (默认应等于滑窗 stride 即可保证连续覆盖时不打散).
        min_segment_sec: 合并后短于此时长的段会被吸收到上一段; 默认 0 表示不丢.
        video_duration_sec: 若提供, 输出末端补齐到该时长 (避免最后一段 < duration).

    返回:
        按 ``start`` 升序的 :class:`TimelineSegment` 列表; 满足:
          - segments[0].start == 0.0 (若首窗 start > 0 自动前置 unknown 段)
          - segments[i].end == segments[i+1].start (无空隙)
          - 若 video_duration_sec 给定, segments[-1].end == video_duration_sec
    """
    if conf_threshold < 0 or conf_threshold > 1:
        raise ValueError(f"conf_threshold 应在 [0, 1], 实际 {conf_threshold}")
    if merge_gap_sec < 0:
        raise ValueError(f"merge_gap_sec 必须 ≥ 0, 实际 {merge_gap_sec}")

    # 1) 把 windows 按时间排序; 同时把低置信窗口归为 unknown
    sorted_results = sorted(window_results, key=lambda r: r.window.start_sec)
    items: list[tuple[float, float, int, str, float]] = []
    # tuple = (start, end, label_id, label_name, confidence)
    for r in sorted_results:
        if r.error or r.label_id < 0 or r.confidence < conf_threshold:
            label_id, label_name, conf = -1, "unknown", float(r.confidence) if not r.error else 0.0
        else:
            label_id, label_name, conf = r.label_id, r.label_name, float(r.confidence)
        items.append((r.window.start_sec, r.window.end_sec, label_id, label_name, conf))

    if not items:
        return [] if video_duration_sec is None else [
            TimelineSegment(
                start=0.0, end=float(video_duration_sec),
                label="unknown", label_id=-1, confidence=0.0, n_windows=0,
            )
        ]

    # 2) 同类合并: 连续 label 相同 + 间隔 ≤ merge_gap_sec → 一段
    segments: list[TimelineSegment] = []
    cur_start, cur_end, cur_id, cur_name, conf_sum, count = items[0][0], items[0][1], items[0][2], items[0][3], items[0][4], 1

    for s, e, lid, lname, conf in items[1:]:
        gap = s - cur_end
        if lid == cur_id and gap <= merge_gap_sec + 1e-6:
            # 合并: 推进 end, 累加 confidence
            cur_end = max(cur_end, e)
            conf_sum += conf
            count += 1
        else:
            segments.append(_emit_segment(cur_start, cur_end, cur_id, cur_name, conf_sum, count))
            cur_start, cur_end, cur_id, cur_name, conf_sum, count = s, e, lid, lname, conf, 1

    segments.append(_emit_segment(cur_start, cur_end, cur_id, cur_name, conf_sum, count))

    # 3) 填补空隙 (相邻段之间用 unknown 段衔接)
    segments = _fill_gaps_with_unknown(segments, video_duration_sec=video_duration_sec)

    # 4) 吸收过短段 (min_segment_sec > 0 时)
    if min_segment_sec > 0:
        segments = _absorb_short_segments(segments, min_segment_sec=min_segment_sec)

    _log.info(
        "post_process done",
        extra={
            "n_windows": len(window_results),
            "n_segments": len(segments),
            "n_unknown_segs": sum(1 for x in segments if x.label_id == -1),
        },
    )
    return segments


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _emit_segment(
    start: float, end: float, label_id: int, label: str, conf_sum: float, n: int,
) -> TimelineSegment:
    return TimelineSegment(
        start=float(start),
        end=float(end),
        label=label,
        label_id=int(label_id),
        confidence=float(conf_sum / max(n, 1)),
        n_windows=int(n),
    )


def _fill_gaps_with_unknown(
    segments: list[TimelineSegment],
    *,
    video_duration_sec: float | None,
) -> list[TimelineSegment]:
    """确保 segments 无空隙、无重叠; 必要时插入 unknown 填充段.

    硬规则:
        - segments[0].start == 0.0 (若 > 0 → 前置 unknown 段)
        - segments[i].end == segments[i+1].start (任意 gap > ε 则插入 unknown 填充)
        - 若 video_duration_sec 给定且 segments[-1].end < duration → 追加 unknown 段
        - 重叠 (a.end > b.start): 把 b.start 推到 a.end (滑窗重叠场景的常态; 不抛错)
    """
    if not segments:
        if video_duration_sec is not None:
            return [TimelineSegment(0.0, float(video_duration_sec), "unknown", -1, 0.0, 0)]
        return []

    out: list[TimelineSegment] = []
    eps = 1e-6

    # 前置 unknown
    first = segments[0]
    if first.start > eps:
        out.append(TimelineSegment(0.0, first.start, "unknown", -1, 0.0, 0))

    out.append(first)
    for nxt in segments[1:]:
        prev = out[-1]
        if nxt.start > prev.end + eps:
            # 有 gap: 插入 unknown 填充
            out.append(TimelineSegment(prev.end, nxt.start, "unknown", -1, 0.0, 0))
            out.append(nxt)
        elif nxt.start < prev.end - eps:
            # 重叠: 推 nxt.start 到 prev.end
            out.append(TimelineSegment(
                start=prev.end,
                end=max(nxt.end, prev.end),
                label=nxt.label,
                label_id=nxt.label_id,
                confidence=nxt.confidence,
                n_windows=nxt.n_windows,
            ))
        else:
            out.append(nxt)

    # 末端补齐
    if video_duration_sec is not None and out[-1].end < float(video_duration_sec) - eps:
        out.append(TimelineSegment(
            out[-1].end, float(video_duration_sec), "unknown", -1, 0.0, 0,
        ))

    return out


def _absorb_short_segments(
    segments: list[TimelineSegment], *, min_segment_sec: float,
) -> list[TimelineSegment]:
    """把 < min_segment_sec 的段吸收到前一段 (扩展前段 end). 首段则吸收到下一段."""
    if not segments:
        return segments
    out: list[TimelineSegment] = []
    pending_short: TimelineSegment | None = None

    for seg in segments:
        if (seg.end - seg.start) < min_segment_sec:
            if out:
                # 吸收到前一段
                prev = out[-1]
                # 加权置信度
                total_n = max(prev.n_windows + seg.n_windows, 1)
                merged_conf = (
                    (prev.confidence * prev.n_windows + seg.confidence * seg.n_windows) / total_n
                )
                out[-1] = TimelineSegment(
                    start=prev.start,
                    end=seg.end,
                    label=prev.label,
                    label_id=prev.label_id,
                    confidence=merged_conf,
                    n_windows=total_n,
                )
            else:
                pending_short = seg
        else:
            if pending_short is not None:
                # 把首段吸收进 seg
                seg = TimelineSegment(
                    start=pending_short.start,
                    end=seg.end,
                    label=seg.label,
                    label_id=seg.label_id,
                    confidence=seg.confidence,
                    n_windows=seg.n_windows + pending_short.n_windows,
                )
                pending_short = None
            out.append(seg)

    if pending_short is not None and not out:
        # 全部段都太短: 至少返回一个
        out.append(pending_short)
    return out
