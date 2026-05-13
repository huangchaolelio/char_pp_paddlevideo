"""``pp extract-feat`` 子命令 (T210, FR-033 / contracts/cli.md).

单视频 → 单 .pkl, 与 ``Features_competition_train.tar.gz`` 内部 .pkl schema 100% 兼容.

退出码 (FR-047):
    0  成功
    1  用户输入错 (input/config 不存在 / 训练权重缺失给 curl 行)
    2  环境问题 (ffmpeg 缺失 / paddle 不可用 / 权重 sha256 不匹配)
    3  章程硬约束违反 (保留)
    4  运行时失败 (ffmpeg / 抽特征异常)

JSON stdout (extract-feat-v1 schema): 见 contracts/cli.md.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from pingpong_av.extractors import (
    ExtractorConfig,
    FFmpegError,
    PPTSMExtractor,
    PPTSMExtractorError,
    compute_clip_id,
    extract_frames_to_dir,
)
from pingpong_av.utils.config import ConfigError, load_config
from pingpong_av.utils.env import find_repo_root
from pingpong_av.utils.logging import get_logger

_log = get_logger(__name__)


def run(
    *,
    input_path: str,
    output_path: str | None,
    fps: int | None,
    batch_size: int | None,
    config_path: str,
    allow_dirty: bool,
    keep_frames: bool,
) -> int:
    """`pp extract-feat` 实际入口. 返回退出码."""
    t_start = time.time()

    # ---- 1. 加载业务 yaml ----
    cfg_p = Path(config_path)
    if not cfg_p.is_file():
        click.echo(f"ERROR: 配置不存在: {cfg_p}", err=True)
        return 1
    try:
        loaded = load_config(cfg_p)
    except ConfigError as exc:
        click.echo(f"ERROR: 配置校验失败: {exc}", err=True)
        return 1

    cfg = loaded.data
    config_hash = loaded.config_hash
    ext_cfg = ExtractorConfig.from_yaml_dict(cfg.get("extraction") or {})

    # 命令行 override
    if fps is not None:
        ext_cfg.fps = int(fps)
    if batch_size is not None:
        ext_cfg.batch_size = int(batch_size)

    # ---- 2. 检查 input ----
    input_p = Path(input_path).resolve()
    if not input_p.is_file():
        click.echo(f"ERROR: 输入视频不存在: {input_p}", err=True)
        return 1

    # ---- 3. 检查 ffmpeg ----
    if shutil.which("ffmpeg") is None:
        click.echo(
            "ERROR (环境问题): ffmpeg 命令不可用. 请安装:\n"
            "  apt install ffmpeg     # Debian/Ubuntu\n"
            "  conda install -c conda-forge ffmpeg     # conda",
            err=True,
        )
        return 2

    # ---- 4. 检查 PP-TSM 权重 (训练 + inference) ----
    repo_root = find_repo_root()
    pretrained = cfg.get("pretrained") or {}
    train_weight_path = Path(pretrained.get("train_weight_path", ""))
    if not train_weight_path.is_absolute():
        train_weight_path = repo_root / train_weight_path
    if not train_weight_path.is_file():
        url = pretrained.get("train_weight_url", "")
        click.echo(
            f"ERROR (用户输入错): PP-TSM 训练权重缺失: {train_weight_path}\n"
            f"  请下载 (~148MB):\n"
            f"    mkdir -p {train_weight_path.parent}\n"
            f"    curl -fL -o {train_weight_path} \\\n"
            f"      {url}",
            err=True,
        )
        return 1

    inf_dir = Path(pretrained.get("inference_dir", "data/raw/pretrained"))
    if not inf_dir.is_absolute():
        inf_dir = repo_root / inf_dir
    pdmodel = inf_dir / pretrained.get("inference_pdmodel", "ppTSM.pdmodel")
    pdiparams = inf_dir / pretrained.get("inference_pdiparams", "ppTSM.pdiparams")

    if not pdmodel.is_file() or not pdiparams.is_file():
        # 自动调用 export 脚本 (FR-038a)
        click.echo(
            f"INFO: PP-TSM inference 文件缺失, 自动运行 export_pptsm_inference.py...",
            err=True,
        )
        export_script = repo_root / "scripts" / "export_pptsm_inference.py"
        venv_py = repo_root / ".venv" / "bin" / "python"
        python_bin = str(venv_py) if venv_py.is_file() else sys.executable
        try:
            result = subprocess.run(
                [python_bin, str(export_script), "--config", str(cfg_p)],
                capture_output=True, text=True, check=True, timeout=600,
            )
            _log.info("export script done", extra={"stdout_tail": result.stdout[-200:]})
        except subprocess.CalledProcessError as exc:
            click.echo(
                f"ERROR (环境问题): export_pptsm_inference.py 失败 (rc={exc.returncode}):\n"
                f"  stderr: {exc.stderr[-500:]}",
                err=True,
            )
            return 2
        except subprocess.TimeoutExpired:
            click.echo("ERROR (环境问题): export_pptsm_inference.py 超时 (> 10 min)", err=True)
            return 2

        if not pdmodel.is_file() or not pdiparams.is_file():
            click.echo(
                f"ERROR (环境问题): export 后仍未找到 inference 文件:\n"
                f"  期望: {pdmodel} + {pdiparams}",
                err=True,
            )
            return 2

    # ---- 5. 计算 clip_id (内容 hash, 抗改名) ----
    clip_id = compute_clip_id(input_p)
    _log.info("clip_id computed", extra={"clip_id": clip_id, "video": str(input_p)})

    # ---- 6. 决定输出路径 ----
    if output_path:
        output_p = Path(output_path).resolve()
    else:
        # 默认: 输入视频同目录, clip_id.pkl
        output_p = input_p.parent / f"{clip_id}.pkl"
    output_p.parent.mkdir(parents=True, exist_ok=True)

    # ---- 7. 准备临时帧目录 ----
    tmp_root = Path(pretrained.get("frames_root", "data/raw/.tmp/"))
    tmp_cfg = cfg.get("tmp") or {}
    tmp_root_str = tmp_cfg.get("frames_root", "data/raw/.tmp/")
    tmp_root = Path(tmp_root_str)
    if not tmp_root.is_absolute():
        tmp_root = repo_root / tmp_root
    tmp_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    frames_dir = tmp_root / f"extract_{run_id}"

    # ---- 8. 抽帧 ----
    try:
        frames_result = extract_frames_to_dir(input_p, frames_dir, fps=ext_cfg.fps)
    except FFmpegError as exc:
        click.echo(f"ERROR (运行时失败): ffmpeg 抽帧失败: {exc}", err=True)
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return 4

    # ---- 9. PP-TSM 抽特征 ----
    try:
        extractor = PPTSMExtractor(
            pdmodel=pdmodel,
            pdiparams=pdiparams,
            config=ext_cfg,
            gpu_mem_mb=int(cfg.get("runtime", {}).get("gpu_mem_mb", 8000)),
        )
        features = extractor.extract_from_frames_dir(frames_dir)
    except PPTSMExtractorError as exc:
        click.echo(f"ERROR (运行时失败): PP-TSM 抽特征失败: {exc}", err=True)
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return 4
    except Exception as exc:
        click.echo(
            f"ERROR (运行时失败): 意外异常 {type(exc).__name__}: {exc}",
            err=True,
        )
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
        return 4

    # ---- 10. 写 pkl + meta ----
    pkl_data = {"image_feature": features}
    with output_p.open("wb") as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_sha256 = _sha256_of_file(output_p)

    # 元信息 (供下游 build-feature-pkls / infer-rawvideo 复用)
    meta_p = output_p.with_suffix(".meta.json")
    meta = {
        "schema":                "extract-feat-meta-v1",
        "video_path":            str(input_p),
        "clip_id":               clip_id,
        "n_frames":              frames_result.n_frames,
        "n_samples":             int(features.shape[0]),
        "feat_dim":              int(features.shape[1]),
        "fps_original":          frames_result.fps_original,
        "fps_used":              frames_result.fps_used,
        "duration_sec":          frames_result.duration_sec,
        "pkl_path":              str(output_p),
        "pkl_sha256":            pkl_sha256,
        "pp_tsm_config_hash":    config_hash,
        "pp_tsm_pdmodel":        str(pdmodel),
        "pp_tsm_pdiparams":      str(pdiparams),
        "extracted_at":          datetime.now(timezone.utc).isoformat(),
        "elapsed_sec":           round(time.time() - t_start, 3),
    }
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 11. 清理临时帧目录 ----
    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    # ---- 12. stdout JSON 摘要 ----
    elapsed = time.time() - t_start
    fps_throughput = frames_result.n_frames / elapsed if elapsed > 0 else 0.0
    summary = {
        "schema":          "extract-feat-v1",
        "input_video":     str(input_p),
        "clip_id":         clip_id,
        "n_frames":        frames_result.n_frames,
        "n_samples":       int(features.shape[0]),
        "fps_used":        frames_result.fps_used,
        "duration_sec":    round(frames_result.duration_sec, 3),
        "output_pkl":      str(output_p),
        "pkl_sha256":      pkl_sha256,
        "elapsed_sec":     round(elapsed, 3),
        "fps_throughput":  round(fps_throughput, 2),
    }
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    click.echo(
        f"✓ 抽特征完成: {features.shape[0]}×{features.shape[1]} float32 → {output_p.name}",
        err=True,
    )
    return 0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """流式 sha256."""
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
