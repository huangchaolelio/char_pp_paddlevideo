"""长视频推理产物: JSON 时间轴 + MP4 叠加 (FR-014/FR-015, data-model.md ``video-timeline-v1``).

两份产物在 ``pp infer-video`` 的同一次调用中**必须**一并产出 (FR-015):

- :func:`write_timeline_json` — 序列化 segments 列表为 ``video-timeline-v1`` schema
- :func:`render_mp4` — 在每帧上叠加当前命中的 ``label (confidence)`` 文本

不在本模块的范围:
- 滑窗 / 推理 (那是 :mod:`sliding_window`)
- 阈值合并 (那是 :mod:`post_process`)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pingpong_av.inference.post_process import TimelineSegment
from pingpong_av.utils.logging import get_logger

__all__ = ["write_timeline_json", "render_mp4"]

_log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# T061 — JSON 时间轴
# --------------------------------------------------------------------------------------


def write_timeline_json(
    *,
    segments: list[TimelineSegment],
    out_path: str | Path,
    input_meta: dict[str, Any],
    model_meta: dict[str, Any],
    inference_cfg: dict[str, Any],
    warnings: Iterable[str] = (),
) -> Path:
    """把 segments + 元信息序列化为 ``video-timeline-v1`` JSON, 写入 ``out_path``.

    输出 schema 严格对齐 data-model.md ``video-timeline-v1``:

    .. code-block:: json

        {
          "schema": "video-timeline-v1",
          "input": {"video_path": ..., "duration_sec": ..., "fps": ...},
          "model": {"checkpoint": ..., "config_hash": ...},
          "inference_config": {"window_sec": ..., "stride_sec": ..., ...},
          "segments": [{start, end, label, label_id, confidence, n_windows}, ...],
          "warnings": [...],
          "produced_at": "<ISO 8601 UTC>"
        }
    """
    payload: dict[str, Any] = {
        "schema": "video-timeline-v1",
        "input": dict(input_meta),
        "model": dict(model_meta),
        "inference_config": dict(inference_cfg),
        "segments": [_segment_to_dict(s) for s in segments],
        "warnings": list(warnings),
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8")
    _log.info(
        "timeline json written",
        extra={"path": str(out), "n_segments": len(segments), "n_warnings": len(payload["warnings"])},
    )
    return out


def _segment_to_dict(s: TimelineSegment) -> dict[str, Any]:
    return {
        "start": float(s.start),
        "end": float(s.end),
        "label": s.label,
        "label_id": int(s.label_id),
        "confidence": float(s.confidence),
        "n_windows": int(s.n_windows),
    }


# --------------------------------------------------------------------------------------
# T062 — MP4 渲染 (叠加文本)
# --------------------------------------------------------------------------------------


def render_mp4(
    *,
    video_path: str | Path,
    segments: list[TimelineSegment],
    out_path: str | Path,
    visualization_cfg: dict[str, Any] | None = None,
    class_display_names: dict[str, str] | None = None,
) -> Path:
    """在原视频每帧上叠加当前段的 ``label (confidence)``, 写入 ``out_path``.

    实现策略:
        - 用 OpenCV 逐帧解码 + 写入 (mp4v 编码; 避免对 ffmpeg 命令行的依赖, 但 ffmpeg
          底层仍由 OpenCV 链入).
        - 每帧根据时间戳查 segments 找到当前命中段, 调 ``cv2.putText`` 叠加文本.
        - 默认在左上角加半透明黑底白字, 提升可读性.

    参数:
        video_path: 输入视频.
        segments: 后处理产生的 :class:`TimelineSegment` 列表 (无空隙、按 start 升序).
        out_path: 输出 mp4 路径.
        visualization_cfg: 来自 ``configs/inference/sliding_window.yaml`` 的 visualization 段.
                            支持: font_scale, text_color, bg_color, bg_alpha, position,
                            show_confidence, unknown_label_text.
        class_display_names: ``ActionClass.name → display_name`` 映射, 用于把 "serve" 等英文
                             键替换成中文标签. 缺失时显示原 name.

    抛出:
        FileNotFoundError: 输入视频不存在.
        RuntimeError:      OpenCV 无法打开输入或写入输出.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"输入视频不存在: {src}")

    cfg = dict(visualization_cfg or {})
    font_scale = float(cfg.get("font_scale", 0.7))
    text_color = tuple(int(x) for x in cfg.get("text_color", [255, 255, 255]))[:3]
    bg_color = tuple(int(x) for x in cfg.get("bg_color", [0, 0, 0]))[:3]
    bg_alpha = float(cfg.get("bg_alpha", 0.6))
    position = str(cfg.get("position", "top_left"))
    show_confidence = bool(cfg.get("show_confidence", True))
    unknown_text = str(cfg.get("unknown_label_text", "unknown"))

    display_map = dict(class_display_names or {})

    # 延迟 import — opencv 启动较慢, 不渲染时不要付出代价
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"视频元信息异常 (fps={fps}, w={width}, h={height}): {src}")

    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV 无法创建输出 mp4: {out}. "
            "请确认 ffmpeg 已安装 (`apt install ffmpeg`)."
        )

    sorted_segs = sorted(segments, key=lambda s: s.start)
    seg_idx = 0
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_no / fps

            # 找出当前 ts 命中的段 (segments 已按 start 升序; 用游标加速)
            while seg_idx + 1 < len(sorted_segs) and ts >= sorted_segs[seg_idx + 1].start:
                seg_idx += 1
            cur = sorted_segs[seg_idx] if sorted_segs else None

            if cur is not None:
                label_text = unknown_text if cur.label_id == -1 else display_map.get(cur.label, cur.label)
                if show_confidence and cur.label_id != -1:
                    label_text = f"{label_text} ({cur.confidence:.2f})"
                _draw_label(
                    frame, label_text,
                    position=position,
                    font_scale=font_scale,
                    text_color=text_color,
                    bg_color=bg_color,
                    bg_alpha=bg_alpha,
                )

            writer.write(frame)
            frame_no += 1
    finally:
        cap.release()
        writer.release()

    _log.info(
        "mp4 rendered",
        extra={"out": str(out), "frames": frame_no, "expected": n_frames, "fps": fps},
    )
    return out


def _draw_label(
    frame, text: str, *,
    position: str,
    font_scale: float,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    bg_alpha: float,
) -> None:
    """在 frame 上画一个带半透明背景的文本 box."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = max(4, int(th * 0.4))

    h, w = frame.shape[:2]
    if position == "top_right":
        x = w - tw - pad - 8
        y = pad + th + 4
    elif position == "bottom_left":
        x = 8
        y = h - pad - 8
    elif position == "bottom_right":
        x = w - tw - pad - 8
        y = h - pad - 8
    else:  # top_left
        x = 8
        y = pad + th + 4

    # 半透明背景
    if bg_alpha > 0:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            bg_color,
            thickness=-1,
        )
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, dst=frame)

    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
