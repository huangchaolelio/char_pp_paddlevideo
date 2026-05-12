"""``pp infer-video`` 子命令 (FR-014/015/016, R5).

编排:
    1. 校验 checkpoint / video / inference_config 文件存在;
    2. 反推 run_dir + 加载 config snapshot 取 class_names + config_hash;
    3. :func:`iterate_windows` → :func:`classify_windows` → :func:`apply_threshold_and_merge`;
    4. :func:`write_timeline_json` 总是产出; :func:`render_mp4` 默认产出 (除非 --no-viz);
    5. warnings 比例 > robustness.max_fail_ratio → 退出 4 (运行时失败).

退出码 (contracts/cli.md):
    0  成功
    1  用户输入错 (文件不存在 / inference_config 非法)
    2  环境问题 (上游不可导)
    4  运行时失败 (失败窗口比例 > 阈值 / OpenCV 无法写 mp4)
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import click
import yaml

from pingpong_av.inference.post_process import apply_threshold_and_merge
from pingpong_av.inference.sliding_window import classify_windows, iterate_windows
from pingpong_av.inference.visualizer import render_mp4, write_timeline_json
from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(
    *,
    checkpoint: str,
    input_path: str,
    inference_config: str,
    output_dir: str,
    no_viz: bool,
) -> int:
    """执行 infer-video. 返回应当作为进程退出码使用的整数."""

    # ---- 1. 文件存在性 ----
    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_file():
        click.echo(f"ERROR: checkpoint 不存在: {ckpt}", err=True)
        return 1

    video = Path(input_path).resolve()
    if not video.is_file():
        click.echo(f"ERROR: 输入视频不存在: {video}", err=True)
        return 1

    infer_cfg_path = Path(inference_config).resolve()
    if not infer_cfg_path.is_file():
        click.echo(f"ERROR: inference-config 不存在: {infer_cfg_path}", err=True)
        return 1

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. 加载 inference 配置 ----
    try:
        infer_cfg_full = yaml.safe_load(infer_cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        click.echo(f"ERROR: inference-config YAML 解析失败: {exc}", err=True)
        return 1

    win_cfg = infer_cfg_full.get("window") or {}
    pp_cfg = infer_cfg_full.get("postprocess") or {}
    viz_cfg = infer_cfg_full.get("visualization") or {}
    rb_cfg = infer_cfg_full.get("robustness") or {}

    window_sec = float(win_cfg.get("duration_sec", 2.0))
    stride_sec = float(win_cfg.get("stride_sec", 1.0))
    edge_policy = str(win_cfg.get("edge_policy", "truncate"))
    conf_threshold = float(pp_cfg.get("conf_threshold", 0.5))
    merge_gap_sec = float(pp_cfg.get("merge_gap_sec", stride_sec))
    min_segment_sec = float(pp_cfg.get("min_segment_sec", 0.0))
    skip_failed = bool(rb_cfg.get("skip_failed_windows", True))
    max_fail_ratio = float(rb_cfg.get("max_fail_ratio", 0.5))

    # ---- 3. 反推 run_dir 取类别 + config_hash ----
    run_dir = _find_run_dir(ckpt)
    if run_dir is None:
        click.echo(
            f"ERROR: 无法从 checkpoint 路径定位 experiments/<run_id>/ 目录: {ckpt}",
            err=True,
        )
        return 1
    snapshot = run_dir / "config.yaml"
    upstream_config = run_dir / "upstream_config.yaml"
    if not snapshot.is_file() or not upstream_config.is_file():
        click.echo(
            f"ERROR: 缺失 snapshot ({snapshot}) 或 upstream_config ({upstream_config}). "
            "请确认 checkpoint 由 `pp train` 产出.",
            err=True,
        )
        return 1
    try:
        loaded = load_config(snapshot)
    except ConfigError as exc:
        click.echo(f"ERROR: snapshot 异常: {exc}", err=True)
        return 1
    class_names = [c["name"] for c in loaded.data["classes"]]
    class_display_names = {
        c["name"]: c.get("display_name") or c["name"]
        for c in loaded.data["classes"]
    }
    config_hash = loaded.config_hash

    # ---- 4. 切窗 ----
    try:
        windows = list(iterate_windows(
            video, window_sec=window_sec, stride_sec=stride_sec, edge_policy=edge_policy,
        ))
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"ERROR: 切窗失败: {exc}", err=True)
        return 1

    if not windows:
        click.echo(f"ERROR: 没有产生任何窗口 (视频可能太短): {video}", err=True)
        return 1

    click.echo(f"[infer-video] {len(windows)} 个窗口, 视频={video.name}", err=True)

    # ---- 5. 逐窗推理 ----
    try:
        from pingpong_av.upstream_adapter.trainer import (
            UpstreamRuntimeError,
            run_upstream_infer,
        )
    except ImportError as exc:
        click.echo(f"ERROR: 无法 import upstream_adapter: {exc}", err=True)
        return 2

    infer_fn = partial(run_upstream_infer, upstream_config, ckpt)

    try:
        results, warnings = classify_windows(
            windows,
            video_path=video,
            infer_fn=infer_fn,
            class_names=class_names,
            skip_failed=skip_failed,
        )
    except UpstreamRuntimeError as exc:
        click.echo(f"ERROR: 上游推理失败 (skip_failed=False 时整体退出): {exc}", err=True)
        return 4
    except Exception as exc:
        click.echo(f"ERROR: 滑窗推理异常: {type(exc).__name__}: {exc}", err=True)
        return 4

    n_failed = sum(1 for r in results if r.error)
    fail_ratio = n_failed / max(len(results), 1)
    if fail_ratio > max_fail_ratio:
        click.echo(
            f"ERROR: 失败窗口比例 {fail_ratio:.1%} > 阈值 {max_fail_ratio:.1%} (FR-016/SC-006). "
            "请检查输入视频或上游模型.",
            err=True,
        )
        return 4

    # ---- 6. 后处理 ----
    duration = _read_duration_sec(video)
    segments = apply_threshold_and_merge(
        results,
        conf_threshold=conf_threshold,
        merge_gap_sec=merge_gap_sec,
        min_segment_sec=min_segment_sec,
        video_duration_sec=duration,
    )

    # ---- 7. 写 JSON 时间轴 (始终产出) ----
    base = video.stem
    json_out = out_dir / f"{base}.timeline.json"
    write_timeline_json(
        segments=segments,
        out_path=json_out,
        input_meta={
            "video_path": str(video),
            "duration_sec": duration,
            "fps": _read_fps(video),
        },
        model_meta={
            "checkpoint": str(ckpt),
            "config_hash": config_hash,
        },
        inference_cfg={
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "conf_threshold": conf_threshold,
            "merge_gap_sec": merge_gap_sec,
        },
        warnings=warnings,
    )

    # ---- 8. 渲染 MP4 (除非 --no-viz) ----
    mp4_out: Path | None = None
    if not no_viz:
        mp4_out = out_dir / f"{base}.viz.mp4"
        try:
            render_mp4(
                video_path=video,
                segments=segments,
                out_path=mp4_out,
                visualization_cfg=viz_cfg,
                class_display_names=class_display_names,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            click.echo(f"WARN: MP4 渲染失败: {exc}; JSON 已产出.", err=True)
            mp4_out = None

    # ---- 9. stdout JSON 摘要 ----
    summary = {
        "input": str(video),
        "duration_sec": duration,
        "n_windows": len(results),
        "n_failed_windows": n_failed,
        "n_segments": len(segments),
        "n_unknown_segments": sum(1 for s in segments if s.label_id == -1),
        "timeline_json": str(json_out),
        "viz_mp4": str(mp4_out) if mp4_out else None,
    }
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        f"✓ 推理完成: {len(segments)} 段, "
        f"{n_failed}/{len(results)} 窗口失败 ({fail_ratio:.1%}).",
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


def _read_duration_sec(video: Path) -> float | None:
    """与 sliding_window 私有版本同语义, 但可被 cli 单独调用记录到 JSON."""
    try:
        import av
        with av.open(str(video)) as c:
            stream = next((s for s in c.streams if s.type == "video"), None)
            if stream and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if c.duration:
                return float(c.duration / 1_000_000)
    except Exception:
        pass
    return None


def _read_fps(video: Path) -> float | None:
    try:
        import av
        with av.open(str(video)) as c:
            stream = next((s for s in c.streams if s.type == "video"), None)
            if stream and stream.average_rate:
                return float(stream.average_rate)
    except Exception:
        pass
    return None
