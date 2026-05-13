"""``pp infer-rawvideo`` 子命令 (T211, FR-039/040/041/042 + contracts/cli.md).

端到端: raw mp4 → PP-TSM 特征 → BMN 滑窗 → BMN 推理 → timeline.json + 可视化 mp4.

内部流水线 (research.md R13, 必须复用现有 BMN 路径):
    1. extract_feat 等价: video → <out>/feature.pkl
    2. prepare_bmn_inputs_for_inference(...): → <out>/bmn_input/
    3. run_upstream_bmn_eval(gt_required=False): → <out>/bmn_eval/results/bmn_results_validation.json
    4. 解析 BMN proposals → 写 <out>/timeline.json (rawvideo-timeline-v1)
    5. (可选) visualize.py → <out>/<basename>_visualized.mp4

退出码 (FR-047): 0/1/2/3/4 与 extract-feat 一致.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(
    *,
    input_path: str,
    bmn_checkpoint: str,
    output_dir: str,
    threshold: float,
    min_duration: float,
    allow_dirty: bool,
    keep_frames: bool,
    keep_features: bool,
    no_visualize: bool,
    extractor_config: str,
    bmn_config: str,
) -> int:
    """`pp infer-rawvideo` 入口. 返回退出码."""
    t_start = time.time()

    # ---- 1. 校验输入 ----
    input_p = Path(input_path).resolve()
    if not input_p.is_file():
        click.echo(f"ERROR: 输入视频不存在: {input_p}", err=True)
        return 1
    ckpt_p = Path(bmn_checkpoint).resolve()
    if not ckpt_p.is_file():
        click.echo(f"ERROR: BMN checkpoint 不存在: {ckpt_p}", err=True)
        return 1

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. 加载业务 config ----
    try:
        extractor_loaded = load_config(Path(extractor_config))
    except ConfigError as exc:
        click.echo(f"ERROR: extractor 配置校验失败: {exc}", err=True)
        return 1

    classes_meta = extractor_loaded.data.get("classes") or []
    class_names = [c.get("display_name") or c.get("name") for c in classes_meta]

    repo_root = find_repo_root()

    # ---- 3. Stage A: 抽特征 (复用 cli.extract_feat.run, FR-042 强制复用) ----
    breakdown = {}
    feature_pkl = out_dir / "feature.pkl"

    stage_t0 = time.time()
    click.echo("[1/5] 抽特征 (extract_feat)...", err=True)
    from pingpong_av.cli import extract_feat as _extract
    rc_extract = _extract.run(
        input_path=str(input_p),
        output_path=str(feature_pkl),
        fps=None,
        batch_size=None,
        config_path=extractor_config,
        allow_dirty=allow_dirty,
        keep_frames=keep_frames,
    )
    breakdown["extract_feat"] = round(time.time() - stage_t0, 3)
    if rc_extract != 0:
        click.echo(f"ERROR: 抽特征失败 (rc={rc_extract})", err=True)
        return rc_extract

    # 读 meta.json 拿 clip_id + duration
    meta_p = feature_pkl.with_suffix(".meta.json")
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        click.echo(f"ERROR (运行时失败): 无法读 extract meta: {exc}", err=True)
        return 4

    clip_id = meta["clip_id"]
    duration_sec = float(meta["duration_sec"])
    fps_used = int(meta["fps_used"])

    # ---- 4. Stage B: 切 BMN 滑窗 + 占位 label_fixed.json ----
    stage_t0 = time.time()
    click.echo("[2/5] 切 BMN 滑窗...", err=True)

    # 把 scripts/ 加到 path 以导入 prepare_bmn_inputs
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from prepare_bmn_inputs import prepare_bmn_inputs_for_inference  # type: ignore[import]
    except ImportError as exc:
        click.echo(f"ERROR (环境问题): 无法导入 prepare_bmn_inputs: {exc}", err=True)
        return 2

    bmn_input_dir = out_dir / "bmn_input"
    try:
        bmn_inputs = prepare_bmn_inputs_for_inference(
            feature_pkl=feature_pkl,
            output_dir=bmn_input_dir,
            clip_id=clip_id,
            fps=fps_used,
        )
    except Exception as exc:
        click.echo(
            f"ERROR (运行时失败): prepare_bmn_inputs_for_inference 失败: {type(exc).__name__}: {exc}",
            err=True,
        )
        return 4
    breakdown["bmn_prepare"] = round(time.time() - stage_t0, 3)

    # ---- 5. Stage C: 加载 BMN config (写一份 upstream_yaml 使用本次 bmn_input) ----
    stage_t0 = time.time()
    click.echo("[3/5] 加载 BMN config...", err=True)

    try:
        bmn_loaded = load_config(Path(bmn_config))
    except ConfigError as exc:
        click.echo(f"ERROR: BMN 配置加载失败: {exc}", err=True)
        return 1

    # 用 load_bmn_config 产出 upstream yaml, 但 dataset 路径要覆盖到本次 bmn_input_dir
    from pingpong_av.models.bmn import load_bmn_config
    bmn_cfg_data = dict(bmn_loaded.data)
    bmn_cfg_data.setdefault("model", {})["bmn_inputs_dir"] = str(bmn_input_dir)
    try:
        upstream_yaml, _ = load_bmn_config(
            bmn_cfg_data,
            output_dir=out_dir / "bmn_eval",
            repo_root=repo_root,
        )
    except Exception as exc:
        click.echo(f"ERROR (运行时失败): 构造 BMN upstream yaml 失败: {exc}", err=True)
        return 4
    breakdown["bmn_config_materialize"] = round(time.time() - stage_t0, 3)

    # ---- 6. Stage D: BMN 推理 (gt_required=False, 002 feature) ----
    stage_t0 = time.time()
    click.echo("[4/5] BMN 推理 (gt_required=False)...", err=True)
    bmn_eval_dir = out_dir / "bmn_eval"
    bmn_results_dir = bmn_eval_dir / "results"
    bmn_intermediate_dir = bmn_eval_dir / "intermediate"

    try:
        from pingpong_av.upstream_adapter.trainer import (
            UpstreamRuntimeError,
            run_upstream_bmn_eval,
        )
    except ImportError as exc:
        click.echo(f"ERROR (环境问题): 无法 import upstream_adapter: {exc}", err=True)
        return 2

    try:
        bmn_metrics = run_upstream_bmn_eval(
            upstream_yaml,
            ckpt_p,
            result_path=bmn_results_dir,
            output_path=bmn_intermediate_dir,
            subset="validation",
            reuse_existing=False,
            gt_required=False,
        )
    except UpstreamRuntimeError as exc:
        click.echo(f"ERROR (运行时失败): BMN 推理失败: {exc}", err=True)
        return 4
    except Exception as exc:
        click.echo(
            f"ERROR (运行时失败): BMN 推理意外异常 {type(exc).__name__}: {exc}",
            err=True,
        )
        return 4
    breakdown["bmn_forward"] = round(time.time() - stage_t0, 3)

    # ---- 7. Stage E: 解析 BMN proposals → timeline.json (rawvideo-timeline-v1) ----
    stage_t0 = time.time()
    click.echo("[5/5] 写 timeline.json...", err=True)

    bmn_results_json = Path(bmn_metrics["bmn_results_json"])
    try:
        bmn_data = json.loads(bmn_results_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        click.echo(f"ERROR (运行时失败): 读 BMN 结果失败: {exc}", err=True)
        return 4

    # bmn_data["results"] 形如 {"<clip_id>_<start>_<end>": [{"segment":[s,e], "score":..., "label":...}, ...]}
    all_segments: list[dict[str, Any]] = []
    for window_key, props in (bmn_data.get("results") or {}).items():
        # window_key = "<clip_id>_<window_start_sec>_<window_end_sec>"
        parts = window_key.rsplit("_", 2)
        if len(parts) >= 3:
            try:
                win_start = float(parts[1])
            except ValueError:
                win_start = 0.0
        else:
            win_start = 0.0

        for rank, p in enumerate(props or []):
            seg = p.get("segment", [0, 0])
            if len(seg) < 2:
                continue
            # BMN 输出的 segment 是窗口内相对秒, 加上 win_start 得到全视频绝对时间
            start_sec = float(seg[0]) + win_start
            end_sec   = float(seg[1]) + win_start
            score     = float(p.get("score", 0.0))
            label_id  = int(p.get("label", -1))  # BMN 默认 0 (single proposal)
            label_name = (
                class_names[label_id]
                if 0 <= label_id < len(class_names)
                else "unknown"
            )
            # 过滤: score / duration
            if score < threshold:
                continue
            if (end_sec - start_sec) < min_duration:
                continue
            all_segments.append({
                "start_sec":      round(start_sec, 3),
                "end_sec":        round(end_sec, 3),
                "label_id":       label_id,
                "label_name":     label_name,
                "score":          round(score, 4),
                "rank_in_window": rank,
            })

    # 按 start_sec 排序, 同 start 按 score 降序
    all_segments.sort(key=lambda r: (r["start_sec"], -r["score"]))

    # 拼装 timeline.json (rawvideo-timeline-v1, data-model.md)
    import hashlib
    def _file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                h.update(b)
        return h.hexdigest()

    timeline = {
        "schema":                   "rawvideo-timeline-v1",
        "input_video":              str(input_p),
        "input_video_clip_id":      clip_id,
        "input_video_duration_sec": duration_sec,
        "extraction": {
            "fps_original":           meta.get("fps_original"),
            "fps_used":               fps_used,
            "n_frames":               meta.get("n_frames"),
            "n_samples":              meta.get("n_samples"),
            "feat_dim":               meta.get("feat_dim"),
            "pp_tsm_config_hash":     meta.get("pp_tsm_config_hash"),
            "feature_pkl_path":       str(feature_pkl),
            "feature_pkl_sha256":     meta.get("pkl_sha256"),
        },
        "bmn_inference": {
            "checkpoint":              str(ckpt_p),
            "checkpoint_sha256":       _file_sha256(ckpt_p),
            "subset":                  bmn_metrics.get("subset", "validation"),
            "n_proposals_before_filter": bmn_metrics.get("n_proposals", 0),
            "threshold":               threshold,
            "min_duration":            min_duration,
            "ar_at":                   None,
        },
        "produced_at":              datetime.now(timezone.utc).isoformat(),
        "results":                  all_segments,
    }

    timeline_p = out_dir / "timeline.json"
    timeline_p.write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    breakdown["write_timeline"] = round(time.time() - stage_t0, 3)

    # ---- 8. (可选) 可视化 ----
    visualized_mp4 = None
    if not no_visualize:
        stage_t0 = time.time()
        click.echo("[bonus] 可视化 (可 --no-visualize 关)...", err=True)
        try:
            from pingpong_av.inference.visualizer import render_mp4
            from pingpong_av.inference.post_process import TimelineSegment

            # 把 results 转 TimelineSegment 列表
            ts_list = [
                TimelineSegment(
                    start=s["start_sec"],
                    end=s["end_sec"],
                    label=s["label_name"],
                    label_id=s["label_id"],
                    confidence=s["score"],
                    n_windows=1,
                )
                for s in all_segments
            ]
            class_disp = {
                c.get("name"): c.get("display_name") or c.get("name")
                for c in classes_meta
            }
            visualized_mp4 = out_dir / f"{input_p.stem}_visualized.mp4"
            render_mp4(
                video_path=input_p,
                segments=ts_list,
                out_path=visualized_mp4,
                class_display_names=class_disp,
            )
            breakdown["visualize"] = round(time.time() - stage_t0, 3)
        except Exception as exc:
            # 可视化失败不阻塞整体输出 (FR-016 边界情况精神)
            click.echo(
                f"WARN: 可视化失败 ({type(exc).__name__}: {exc}); timeline.json 已正常产出",
                err=True,
            )
            visualized_mp4 = None

    # ---- 9. (可选) 清理中间产物 ----
    if not keep_features:
        feature_pkl.unlink(missing_ok=True)
        meta_p.unlink(missing_ok=True)

    elapsed = time.time() - t_start

    # ---- 10. stdout JSON 摘要 ----
    summary = {
        "schema":                   "infer-rawvideo-v1",
        "input_video":              str(input_p),
        "clip_id":                  clip_id,
        "duration_sec":             round(duration_sec, 3),
        "n_proposals":              bmn_metrics.get("n_proposals", 0),
        "n_proposals_after_filter": len(all_segments),
        "timeline_json":            str(timeline_p),
        "visualized_mp4":           str(visualized_mp4) if visualized_mp4 else None,
        "feature_pkl":              str(feature_pkl) if keep_features else None,
        "elapsed_sec":              round(elapsed, 3),
        "elapsed_breakdown":        breakdown,
    }
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        f"✓ 推理完成: {len(all_segments)} 候选区间 (n_total={bmn_metrics.get('n_proposals',0)}) → {timeline_p.name}",
        err=True,
    )
    return 0
