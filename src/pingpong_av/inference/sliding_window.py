"""长视频滑窗推理 (FR-014, research.md R5).

设计:
- :func:`iterate_windows` — 用 PyAV 解码视频流, 按 ``window_sec`` / ``stride_sec`` 切窗;
  每个窗口内按上游 PP-TSM 期望均匀采样 ``num_segments`` 帧.
- :func:`classify_windows` — 对每个窗口调用上游推理函数, 把 softmax/logits 向量收回;
  单窗失败追加 ``warnings[]``, **不**中断整体流程 (FR-016).

返回的 :class:`WindowResult` 是后处理阶段 :mod:`post_process` 的输入.

不在本模块的范围:
- 阈值过滤 + 同类合并 (那是 :mod:`pingpong_av.inference.post_process`, T059)
- JSON / MP4 落盘 (那是 :mod:`pingpong_av.inference.visualizer`, T061+)
- 单片段推理 (那是 :mod:`pingpong_av.inference.clip_runner`, T053)
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from pingpong_av.utils.logging import get_logger

__all__ = [
    "Window",
    "WindowResult",
    "iterate_windows",
    "classify_windows",
    "WindowError",
]

_log = get_logger(__name__)


class WindowError(RuntimeError):
    """单个窗口处理失败 (用于 classify_windows 的 warnings 通道)."""


@dataclass(frozen=True)
class Window:
    """滑窗的物理边界 (秒). 帧的实际采样在推理时由上游 pipeline 完成."""

    index: int          # 0-based
    start_sec: float
    end_sec: float


@dataclass
class WindowResult:
    """逐窗推理结果."""

    window: Window
    label_id: int        # argmax(scores)
    label_name: str      # class_names[label_id]
    confidence: float    # max(scores) — Top-1 概率
    scores: np.ndarray   # 完整 [num_classes] 概率向量
    error: str | None = None  # 失败时设置, label/confidence 等字段视为无效


# --------------------------------------------------------------------------------------
# 滑窗采样
# --------------------------------------------------------------------------------------


def iterate_windows(
    video_path: str | Path,
    *,
    window_sec: float,
    stride_sec: float,
    edge_policy: str = "truncate",
) -> Iterator[Window]:
    """按时长切窗 (单位: 秒). 不读视频内容, 只产出时间区间.

    参数:
        video_path: 输入视频文件.
        window_sec: 窗口长度.
        stride_sec: 步长 (≤ window_sec 表示重叠).
        edge_policy:
            ``"truncate"`` — 末尾不足 window_sec 的部分丢弃 (默认; 与 R5 一致);
            ``"pad"``      — 把最后窗口推后到 video_end - window_sec (保证覆盖到尾段).

    抛出:
        FileNotFoundError: video_path 不存在.
        ValueError: window_sec/stride_sec 非正; 或视频时长读取失败.
    """
    if window_sec <= 0:
        raise ValueError(f"window_sec 必须 > 0, 实际 {window_sec}")
    if stride_sec <= 0:
        raise ValueError(f"stride_sec 必须 > 0, 实际 {stride_sec}")
    if edge_policy not in ("truncate", "pad"):
        raise ValueError(f"未知 edge_policy: {edge_policy!r}")

    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"视频不存在: {video}")

    duration = _read_duration_sec(video)
    if duration is None or duration <= 0:
        raise ValueError(f"无法读取视频时长 (或时长为 0): {video}")
    if duration < window_sec:
        # 视频比窗口还短: 唯一的窗口就是 [0, duration]
        yield Window(index=0, start_sec=0.0, end_sec=float(duration))
        return

    idx = 0
    start = 0.0
    last_emitted_end = 0.0
    while True:
        end = start + window_sec
        if end > duration + 1e-6:
            break
        yield Window(index=idx, start_sec=float(start), end_sec=float(end))
        last_emitted_end = end
        idx += 1
        start += stride_sec

    if edge_policy == "pad" and last_emitted_end < duration - 1e-6:
        # 把最后窗口的末端贴到视频尾端
        yield Window(
            index=idx,
            start_sec=float(max(0.0, duration - window_sec)),
            end_sec=float(duration),
        )


# --------------------------------------------------------------------------------------
# 逐窗推理
# --------------------------------------------------------------------------------------


def classify_windows(
    windows: list[Window],
    *,
    video_path: str | Path,
    infer_fn: Callable[[Path], np.ndarray],
    class_names: list[str],
    skip_failed: bool = True,
) -> tuple[list[WindowResult], list[str]]:
    """对每个窗口调用 ``infer_fn`` 拿到概率向量, 整理为 :class:`WindowResult` 列表.

    参数:
        windows: :func:`iterate_windows` 产出的窗口列表 (强制实体化, 因为我们要先
                 切片再批量推理).
        video_path: 原视频路径; 内部把每个窗口剪成临时 mp4 喂给 ``infer_fn``.
        infer_fn: ``Path -> np.ndarray[num_classes]``. 通常是
                  ``functools.partial(run_upstream_infer, upstream_config, checkpoint)``.
        class_names: 类别名列表.
        skip_failed: True 时单窗失败追加 warning 后继续; False 时立刻抛错.

    返回:
        ``(results, warnings)``:
          - results — 与 windows 等长的 :class:`WindowResult` 列表;
            失败窗口的 result.error 非空, label_id=-1, confidence=0.
          - warnings — 人类可读的警告字符串列表 (与 result.error 一一对应).

    抛出:
        :class:`WindowError`: skip_failed=False 且某窗失败时立即抛.
        其他异常按上游原始类型透传 (例如 GPU OOM).
    """
    if not windows:
        return [], []

    n_classes = len(class_names)
    results: list[WindowResult] = []
    warnings: list[str] = []

    for w in windows:
        # 1) 把窗口剪成临时 mp4
        try:
            clip_path = _cut_window_to_tempfile(video_path, w)
        except Exception as exc:
            msg = f"window#{w.index} ({w.start_sec:.2f}-{w.end_sec:.2f}s): 切窗失败: {exc}"
            if not skip_failed:
                raise WindowError(msg) from exc
            warnings.append(msg)
            results.append(_failed_result(w, n_classes, msg))
            continue

        # 2) 推理 (此处任何异常被视为窗口级失败, 不致命)
        try:
            scores = np.asarray(infer_fn(clip_path)).reshape(-1)
            if scores.size != n_classes:
                raise WindowError(
                    f"模型输出维度 {scores.size} 与 class_names ({n_classes}) 不匹配"
                )
            scores = _ensure_probs(scores)
            top1 = int(scores.argmax())
            results.append(WindowResult(
                window=w,
                label_id=top1,
                label_name=class_names[top1],
                confidence=float(scores[top1]),
                scores=scores,
            ))
        except Exception as exc:
            msg = f"window#{w.index} ({w.start_sec:.2f}-{w.end_sec:.2f}s): 推理失败: {exc}"
            if not skip_failed:
                raise WindowError(msg) from exc
            warnings.append(msg)
            results.append(_failed_result(w, n_classes, msg))
        finally:
            # 清理临时文件
            try:
                Path(clip_path).unlink(missing_ok=True)
            except OSError:
                pass

    n_failed = sum(1 for r in results if r.error)
    _log.info(
        "classify_windows done",
        extra={"total": len(results), "failed": n_failed, "video": str(video_path)},
    )
    return results, warnings


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _read_duration_sec(video: Path) -> float | None:
    """优先 PyAV; 失败则 OpenCV."""
    try:
        import av
        with av.open(str(video)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration:
                return float(container.duration / 1_000_000)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and n > 0:
                return float(n / fps)
    except Exception:
        pass
    return None


def _cut_window_to_tempfile(video_path: str | Path, w: Window) -> Path:
    """用 PyAV 把指定时间段拷贝成 mp4 临时文件 (不重新编码, 极快).

    PyAV 的流复制在某些容器上会因起点不在 keyframe 而引入空白; 对动作识别推理来说
    可接受 (PP-TSM 用均匀采样, 少量起点偏差不影响结果). 若需精确帧级裁剪, 可改为
    全量解码再编码, 但代价高一个数量级.
    """
    import av
    src = Path(video_path)

    # 用 NamedTemporaryFile + delete=False 拿到一个我们自己 unlink 的 mp4 路径
    tmp = tempfile.NamedTemporaryFile(prefix=f"sw_{w.index:04d}_", suffix=".mp4", delete=False)
    tmp.close()
    out_path = Path(tmp.name)

    try:
        with av.open(str(src)) as in_c, av.open(str(out_path), mode="w") as out_c:
            in_stream = next((s for s in in_c.streams if s.type == "video"), None)
            if in_stream is None:
                raise WindowError(f"视频无视频流: {src}")
            # PyAV 12 API: add_stream(template=...); 旧版本 (<= 8) 用 add_stream_from_template
            try:
                out_stream = out_c.add_stream(template=in_stream)
            except TypeError:  # 极老 PyAV 兜底
                out_stream = out_c.add_stream_from_template(in_stream)  # type: ignore[attr-defined]

            # 跳到 start_sec
            try:
                in_c.seek(int(w.start_sec / float(in_stream.time_base)), stream=in_stream)
            except Exception:
                # 某些容器不支持 seek; 退化为顺序遍历
                pass

            for packet in in_c.demux(in_stream):
                if packet.dts is None:
                    continue
                ts_sec = float(packet.dts * in_stream.time_base) if in_stream.time_base else 0.0
                if ts_sec < w.start_sec - 1e-3:
                    continue
                if ts_sec >= w.end_sec - 1e-3:
                    break
                packet.stream = out_stream
                out_c.mux(packet)
    except Exception:
        # 出错时清理临时文件
        out_path.unlink(missing_ok=True)
        raise
    return out_path


def _ensure_probs(scores: np.ndarray) -> np.ndarray:
    """logits → softmax 兜底 (与 clip_runner 一致)."""
    if scores.size == 0:
        return scores
    if (scores >= 0).all() and 0.99 <= float(scores.sum()) <= 1.01:
        return scores
    s = scores - scores.max()
    exp = np.exp(s)
    return exp / exp.sum()


def _failed_result(w: Window, n_classes: int, error: str) -> WindowResult:
    """构造失败窗口的占位 WindowResult."""
    return WindowResult(
        window=w,
        label_id=-1,
        label_name="unknown",
        confidence=0.0,
        scores=np.zeros(n_classes, dtype=np.float32),
        error=error,
    )
