"""``pp infer-clip`` 子命令 (FR-013, FR-016).

编排:
    1. 校验 checkpoint / video 文件存在 + 可读;
    2. 从 checkpoint 反推 run_dir, 加载 config snapshot 取 class_names + config_hash;
    3. 调用 :func:`pingpong_av.inference.clip_runner.infer_clip` 得到 PredictionResult dict;
    4. 写入 ``--output`` 文件 (若提供) + stdout 输出 JSON;

退出码 (contracts/cli.md):
    0  成功
    1  用户输入错 (文件不存在 / 不可读 / 过短)
    2  环境问题 (上游不可导)
    4  运行时失败 (上游推理异常)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(
    *,
    checkpoint: str,
    input_path: str,
    topk: int,
    output_path: str | None,
) -> int:
    """执行 infer-clip. 返回应当作为进程退出码使用的整数."""

    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        click.echo(f"ERROR: checkpoint 不存在: {ckpt}", err=True)
        return 1

    video = Path(input_path).resolve()
    if not video.is_file():
        click.echo(f"ERROR: 输入视频不存在: {video}", err=True)
        return 1

    if topk < 1:
        click.echo(f"ERROR: --topk 必须 >= 1, 实际 {topk}", err=True)
        return 1

    # ---- 反推 run_dir + 加载 snapshot ----
    run_dir = _find_run_dir(ckpt)
    if run_dir is None:
        click.echo(
            f"ERROR: 无法从 checkpoint 路径定位 experiments/<run_id>/ 目录: {ckpt}\n"
            "       infer-clip 需要从该目录读取 config.yaml 取 class_names 与 config_hash.",
            err=True,
        )
        return 1
    snapshot = run_dir / "config.yaml"
    upstream_config = run_dir / "upstream_config.yaml"
    if not snapshot.is_file() or not upstream_config.is_file():
        click.echo(
            f"ERROR: 缺失 snapshot ({snapshot}) 或 upstream_config ({upstream_config}). "
            "请确认该 checkpoint 是由 `pp train` 产出.",
            err=True,
        )
        return 1

    try:
        loaded = load_config(snapshot)
    except ConfigError as exc:
        click.echo(f"ERROR: snapshot 配置异常: {exc}", err=True)
        return 1

    class_names = [c["name"] for c in loaded.data["classes"]]
    config_hash = loaded.config_hash

    # ---- FR-016 边界: 视频可读性 / 过短 ----
    duration_check = _check_video_readable_and_long_enough(video, loaded.data)
    if duration_check is not None:
        click.echo(f"ERROR: {duration_check}", err=True)
        return 1

    # ---- 调用推理 ----
    try:
        from pingpong_av.inference.clip_runner import infer_clip
        from pingpong_av.upstream_adapter.trainer import UpstreamRuntimeError
    except ImportError as exc:
        click.echo(f"ERROR: 无法 import 推理模块: {exc}", err=True)
        return 2

    try:
        result = infer_clip(
            checkpoint=ckpt,
            upstream_config=upstream_config,
            video_path=video,
            class_names=class_names,
            config_hash=config_hash,
            topk=topk,
        )
    except FileNotFoundError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        return 1
    except UpstreamRuntimeError as exc:
        click.echo(f"ERROR: 上游推理失败: {exc}", err=True)
        return 4
    except ValueError as exc:
        # class_names 长度不一致 等
        click.echo(f"ERROR: {exc}", err=True)
        return 1
    except Exception as exc:
        click.echo(f"ERROR: 推理时出现意外错误: {type(exc).__name__}: {exc}", err=True)
        return 4

    # ---- 写文件 (可选) + stdout JSON ----
    if output_path:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        click.echo(f"✓ 推理结果已写入: {out}", err=True)

    click.echo(json.dumps(result, ensure_ascii=False, sort_keys=True))
    top1 = result["topk"][0]
    click.echo(
        f"✓ Top-1: {top1['name']} (id={top1['id']}, score={top1['score']:.4f})",
        err=True,
    )
    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _find_run_dir(checkpoint: Path) -> Path | None:
    parent = checkpoint.parent
    if parent.name != "checkpoints":
        return None
    candidate = parent.parent
    if not (candidate / "manifest.json").is_file():
        return None
    return candidate


def _check_video_readable_and_long_enough(video: Path, user_cfg: dict) -> str | None:
    """快速校验视频文件可读且长度足够覆盖一个采样窗口 (FR-016).

    返回 None 表示通过; 否则返回错误描述字符串.

    用 PyAV 探测; 失败则不阻拦 (让上游自己抛错). 这样避免在 PyAV 处理某些异常视频
    时误报.
    """
    if video.stat().st_size == 0:
        return f"视频文件为空: {video}"

    # 尝试用 PyAV 打开 + 读 1 帧
    try:
        import av
        with av.open(str(video)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return f"视频不含视频流: {video}"
            # 试图解码第一帧, 验证文件没有损坏
            for _ in container.decode(stream):
                break
            # 检查时长 ≥ pipeline.num_segments / fps (单段动作至少要够采样)
            num_segments = int((user_cfg.get("pipeline") or {}).get("num_segments", 8))
            duration_sec = None
            if stream.duration and stream.time_base:
                duration_sec = float(stream.duration * stream.time_base)
            if duration_sec is not None and duration_sec < 0.1:
                return (
                    f"视频时长 {duration_sec:.3f}s 过短 (FR-016), "
                    f"无法覆盖一个动作采样窗口. 至少需要 ≥0.1s, 推荐 ≥1s."
                )
            # FPS-based 帧数检查
            avg_rate = stream.average_rate
            n_frames = stream.frames
            if avg_rate and n_frames and n_frames < num_segments:
                return (
                    f"视频帧数 {n_frames} 少于 num_segments={num_segments} (FR-016), "
                    "无法均匀采样, 建议换更长的视频."
                )
    except Exception as exc:
        # PyAV 解码失败本身就是 FR-016 的"损坏文件"场景
        return f"视频无法解码 (可能已损坏或格式不支持): {type(exc).__name__}: {exc}"
    return None
