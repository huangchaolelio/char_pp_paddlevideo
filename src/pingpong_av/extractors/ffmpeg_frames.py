"""ffmpeg 抽帧 + fps 探测工具 (T204).

对应:
    - spec FR-035 (抽帧策略)
    - research.md R11 (fps 一致性: 强制 25 fps 与 BMN GT 对齐)
    - data-model.md RawVideo.fps_original / fps_used 字段

设计:
    - 走 subprocess 调 ffmpeg (不用 Python 绑定, 避免 decord/av-py 的 3.11 兼容麻烦)
    - ffprobe 探测原始 fps + 时长
    - 抽帧到 ``<output_dir>/%08d.jpg``, 与上游 ``applications/FootballAction/extractor/extract_feat.py`` 命名一致
    - 输出字典含足够元信息供 manifest.csv + timeline.json 审计

章程对齐:
    - III: fps / 质量参数从 yaml 读, 不硬编码
    - VII: 错误码与 FR-047 对齐 (2 = ffmpeg 缺失, 4 = 运行时失败)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pingpong_av.utils.logging import get_logger

__all__ = ["extract_frames_to_dir", "probe_video_metadata", "FramesResult", "FFmpegError"]

_log = get_logger(__name__)


class FFmpegError(RuntimeError):
    """ffmpeg / ffprobe 调用失败 (返回非 0 或无法解析输出)."""


@dataclass
class FramesResult:
    """抽帧结果. 字段覆盖 manifest.csv 的帧相关列."""

    frames_dir: Path          # 帧目录
    n_frames: int             # 实际抽出的 jpg 数
    fps_original: float       # 视频原 fps (ffprobe 探测)
    fps_used: int             # 实际抽帧 fps (= 入参, 默认 25)
    duration_sec: float       # 视频时长 (秒)
    container_format: str     # mp4 / mkv / flv / ...
    width: int                # 原视频宽 (抽帧后尺寸由后续 resize 决定)
    height: int               # 原视频高

    def to_manifest_fields(self) -> dict[str, Any]:
        """返回用于 manifest.csv 的子集字段. 其它字段 (clip_id / pkl_path / sha256 等) 由调用方补."""
        return {
            "n_frames":         self.n_frames,
            "fps_original":     round(float(self.fps_original), 3),
            "fps_used":         int(self.fps_used),
            "duration_sec":     round(float(self.duration_sec), 3),
        }


# --------------------------------------------------------------------------------------
# 公共 API
# --------------------------------------------------------------------------------------


def probe_video_metadata(video_path: Path | str) -> dict[str, Any]:
    """用 ffprobe 探测视频元信息.

    Returns:
        ``{"fps_original": float, "duration_sec": float, "width": int, "height": int,
           "container_format": str, "codec_name": str}``

    Raises:
        FFmpegError: ffprobe 不可用 / 视频损坏 / 输出无法解析.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FFmpegError(f"视频不存在: {video_path}")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FFmpegError(
            "ffprobe 不可用. 请安装 ffmpeg (含 ffprobe): "
            "`apt install ffmpeg` 或 `conda install -c conda-forge ffmpeg`."
        )

    # -show_format + -show_streams 拿全部元信息, JSON 输出便于解析
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height,codec_name,nb_frames,duration",
        "-show_entries", "format=format_name,duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(f"ffprobe 返回非 0: rc={exc.returncode}, stderr: {exc.stderr[:500]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffprobe 超时 (> 30s): {video_path}") from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe 输出非 JSON: {result.stdout[:500]}") from exc

    streams = data.get("streams", [])
    if not streams:
        raise FFmpegError(f"视频无 video 流: {video_path}")
    stream = streams[0]
    fmt = data.get("format", {})

    # r_frame_rate 格式是 "25/1" 这种分数, 需转 float
    fps_str = stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps_original = float(num) / float(den) if float(den) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps_original = 0.0

    # duration 优先用 stream, 回退到 format
    duration_sec = float(stream.get("duration") or fmt.get("duration") or 0.0)

    # container format 可能是 "mov,mp4,m4a,3gp,3g2,mj2" 这种复合, 取第一个
    container = (fmt.get("format_name") or "").split(",")[0]

    return {
        "fps_original":     fps_original,
        "duration_sec":     duration_sec,
        "width":            int(stream.get("width", 0)),
        "height":           int(stream.get("height", 0)),
        "container_format": container,
        "codec_name":       stream.get("codec_name", ""),
    }


def extract_frames_to_dir(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    fps: int = 25,
    quality: int = 2,
) -> FramesResult:
    """把视频抽帧到目录, 按 ``%08d.jpg`` 命名 (与上游约定一致).

    Args:
        video_path: 输入视频路径.
        output_dir: 输出目录 (必须已存在或可创建). 目录下如有旧 jpg 会被 ffmpeg 覆盖.
        fps: 强制重采样的目标 fps (默认 25, 与 BMN GT 一致, research.md R11).
             传 0 则保持原 fps (不加 ``-r`` 参数).
        quality: ffmpeg ``-q:v`` 值, 1 最好 2 次之, 2 较 0 更小但仍近无损. 默认 2.
                 上游 ``extract_feat.py`` 用 ``-q 0`` (最高质量但最大), 这里默认 2 更平衡.

    Returns:
        :class:`FramesResult` 含帧数 / fps / 时长等.

    Raises:
        FFmpegError: ffmpeg/ffprobe 不可用, 抽帧失败, 或帧数探测失败.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 先探测元信息 (如果视频损坏会在这一步直接报错, 避免白跑 ffmpeg)
    meta = probe_video_metadata(video_path)
    fps_original = float(meta["fps_original"])
    duration_sec = float(meta["duration_sec"])
    container = meta["container_format"]

    if duration_sec <= 0 or fps_original <= 0:
        raise FFmpegError(
            f"视频元信息异常: fps={fps_original}, duration={duration_sec}s ({video_path})"
        )

    # 2. 抽帧
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FFmpegError(
            "ffmpeg 不可用. 请安装 ffmpeg: "
            "`apt install ffmpeg` 或 `conda install -c conda-forge ffmpeg`."
        )

    cmd: list[str] = [
        ffmpeg, "-v", "error", "-y",
        "-i", str(video_path),
    ]
    if fps > 0:
        cmd += ["-r", str(int(fps))]
    cmd += [
        "-q:v", str(int(quality)),
        str(output_dir / "%08d.jpg"),
    ]

    _log.info(
        "ffmpeg extract start",
        extra={"video": str(video_path), "output_dir": str(output_dir),
               "fps_target": fps, "fps_original": fps_original},
    )

    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True, timeout=3600)
    except subprocess.CalledProcessError as exc:
        raise FFmpegError(
            f"ffmpeg 抽帧失败: rc={exc.returncode}\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {exc.stderr[-1000:]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg 抽帧超时 (> 1h): {video_path}") from exc

    # 3. 数帧数 (ffmpeg 不会告诉我们精确数字, 自己 glob)
    n_frames = sum(1 for p in output_dir.iterdir() if p.suffix == ".jpg")
    if n_frames == 0:
        raise FFmpegError(f"ffmpeg 返回 0 但未抽出任何帧: {output_dir}")

    fps_used = int(fps) if fps > 0 else int(round(fps_original))

    _log.info(
        "ffmpeg extract done",
        extra={"n_frames": n_frames, "fps_used": fps_used,
               "expected": int(round(duration_sec * fps_used)),
               "deviation_pct": round(
                   abs(n_frames - duration_sec * fps_used) / max(duration_sec * fps_used, 1) * 100,
                   2,
               )},
    )

    return FramesResult(
        frames_dir=output_dir,
        n_frames=n_frames,
        fps_original=fps_original,
        fps_used=fps_used,
        duration_sec=duration_sec,
        container_format=container,
        width=int(meta["width"]),
        height=int(meta["height"]),
    )
